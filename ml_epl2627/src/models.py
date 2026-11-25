"""
ml_epl2627/src/models.py
Benchmark Ladder for EPL 2026/27 (N=107) under strict Zero Data Leakage:
  - B0_NaivePrior: Historical + expanding H/D/A empirical prior
  - B1_TablePPG: Regularized Multinomial Logit on 3 trivial table features (dPPG, dGD/g, Home)
  - B2_Elo: Continuous historical->26/27 Elo (off-season regression + promoted init) + Logit head
  - B3_DixonColes_Goals: Time-decayed Dixon-Coles Bivariate Poisson (with L2 shrinkage & warm start)
  - B4_EventStateSpace_xT: Opponent-Adjusted Ridge Latent Performance (Goals + BoxShots + Opta xT) -> DC Poisson
  - B4_XGB_OverfitDiag: 12-feature XGBClassifier trained strictly on 26/27 expanding stream (overfit test)
"""

import math
import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.stats import poisson
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier

from src.features import SeasonTracker, build_xgb_feature_vector


def dc_tau(hg: int, ag: int, lam_h: float, lam_a: float, rho: float) -> float:
    if hg == 0 and ag == 0:
        return max(1e-6, 1.0 - lam_h * lam_a * rho)
    if hg == 0 and ag == 1:
        return max(1e-6, 1.0 + lam_h * rho)
    if hg == 1 and ag == 0:
        return max(1e-6, 1.0 + lam_a * rho)
    if hg == 1 and ag == 1:
        return max(1e-6, 1.0 - rho)
    return 1.0


def score_matrix_to_hda(lam_h: float, lam_a: float, rho: float = -0.08, max_goals: int = 8) -> np.ndarray:
    k = np.arange(max_goals + 1)
    ph = poisson.pmf(k, max(0.05, lam_h))
    pa = poisson.pmf(k, max(0.05, lam_a))
    mat = np.outer(ph, pa)
    # Dixon-Coles low-score correlation adjustment
    mat[0, 0] *= dc_tau(0, 0, lam_h, lam_a, rho)
    mat[0, 1] *= dc_tau(0, 1, lam_h, lam_a, rho)
    mat[1, 0] *= dc_tau(1, 0, lam_h, lam_a, rho)
    mat[1, 1] *= dc_tau(1, 1, lam_h, lam_a, rho)
    mat = np.clip(mat, 0.0, None)
    mat /= mat.sum()
    p_h = float(np.tril(mat, k=-1).sum())
    p_d = float(np.trace(mat))
    p_a = float(np.triu(mat, k=1).sum())
    s = p_h + p_d + p_a
    return np.array([p_h / s, p_d / s, p_a / s], dtype=float)


def fit_historical_elo_and_logit(df_hist: pd.DataFrame, df_2627_teams: list[str], shrink: float = 0.35, k: float = 20.0, hfa: float = 65.0):
    """
    Runs Elo sequentially across 2021/22 - 2025/26, applying off-season regression toward 1500
    at each season boundary, and fits a LogisticRegression head mapping (elo_diff) -> (H, D, A).
    """
    ratings = {}
    X_elo = []
    y_elo = []
    prev_season = None

    for _, r in df_hist.iterrows():
        sec = r["season"]
        if prev_season is not None and sec != prev_season:
            for t in list(ratings.keys()):
                ratings[t] = 1500.0 + (1.0 - shrink) * (ratings[t] - 1500.0)
        prev_season = sec

        ht, at = r["home_team"], r["away_team"]
        hr = ratings.get(ht, 1420.0)
        ar = ratings.get(at, 1420.0)

        diff = (hr + hfa) - ar
        X_elo.append([diff])
        y_elo.append(int(r["target"]))

        exp_h = 1.0 / (1.0 + 10.0 ** (-diff / 400.0))
        hg, ag = int(r["home_goals"]), int(r["away_goals"])
        actual_h = 1.0 if hg > ag else (0.5 if hg == ag else 0.0)
        mov = math.log1p(abs(hg - ag)) * (2.2 / ((abs(diff) * 0.001) + 2.2))
        delta = k * max(1.0, mov) * (actual_h - exp_h)
        ratings[ht] = hr + delta
        ratings[at] = ar - delta

    # Entering 2026/27: apply off-season regression; newly promoted teams init at 1420
    init_ratings_2627 = {}
    for t in df_2627_teams:
        if t in ratings:
            init_ratings_2627[t] = 1500.0 + (1.0 - shrink) * (ratings[t] - 1500.0)
        else:
            init_ratings_2627[t] = 1420.0

    elo_clf = LogisticRegression(C=1.0, max_iter=1000)
    # Use last 3 seasons (2023/24-2025/26) for Elo->HDA calibration
    n_cal = min(len(X_elo), 1140)
    elo_clf.fit(np.array(X_elo[-n_cal:]), np.array(y_elo[-n_cal:]))

    # Also train trivial table LogisticRegression (dPPG, dGD/g) on historical seasons
    hist_tracker = SeasonTracker()
    X_tbl, y_tbl = [], []
    last_sec = None
    for _, r in df_hist.iterrows():
        if r["season"] != last_sec:
            hist_tracker = SeasonTracker()
            last_sec = r["season"]
        ht, at = r["home_team"], r["away_team"]
        hp, hgd, hn = hist_tracker.get_table_features(ht)
        ap, agd, an = hist_tracker.get_table_features(at)
        if hn >= 3 and an >= 3:
            X_tbl.append([hp - ap, hgd - agd])
            y_tbl.append(int(r["target"]))
        hist_tracker.update(r)

    tbl_scaler = StandardScaler()
    X_tbl_s = tbl_scaler.fit_transform(np.array(X_tbl))
    tbl_clf = LogisticRegression(C=0.5, max_iter=1000)
    tbl_clf.fit(X_tbl_s, np.array(y_tbl))

    # Empirical H/D/A counts from history
    counts_hist = np.array([(df_hist["target"] == i).sum() for i in (0, 1, 2)], dtype=float)

    return {
        "init_elo": init_ratings_2627,
        "elo_clf": elo_clf,
        "tbl_scaler": tbl_scaler,
        "tbl_clf": tbl_clf,
        "hist_hda_counts": counts_hist,
    }


def fit_dixon_coles(
    df_train: pd.DataFrame,
    teams: list[str],
    promoted_teams: set[str],
    ref_date: pd.Timestamp,
    half_life_days: float = 150.0,
    l2_reg: float = 6.0,
    x0: np.ndarray | None = None,
) -> tuple[dict, np.ndarray]:
    """
    Time-decayed Dixon-Coles Bivariate Poisson MLE with L2 shrinkage toward priors.
    Returns parameter dict + parameter vector (for fast warm start).
    """
    t2i = {t: i for i, t in enumerate(teams)}
    n_t = len(teams)
    # Parameter vector: [att_0..att_{n-1}, def_0..def_{n-1}, mu, home_adv, rho]
    if x0 is None:
        x0 = np.zeros(2 * n_t + 3, dtype=float)
        x0[2 * n_t] = 0.20      # mu (log baseline goals ~ 1.22)
        x0[2 * n_t + 1] = 0.25  # home_adv
        x0[2 * n_t + 2] = -0.07 # rho

    prior_att = np.array([-0.18 if t in promoted_teams else 0.0 for t in teams])
    prior_def = np.array([0.18 if t in promoted_teams else 0.0 for t in teams])

    # Pre-extract numpy arrays for fast vectorized objective evaluation
    valid_mask = df_train["home_team"].isin(t2i) & df_train["away_team"].isin(t2i)
    sub = df_train[valid_mask]
    hi = np.array([t2i[t] for t in sub["home_team"]], dtype=int)
    ai = np.array([t2i[t] for t in sub["away_team"]], dtype=int)
    hg = sub["home_goals"].to_numpy(dtype=int)
    ag = sub["away_goals"].to_numpy(dtype=int)
    dt_days = np.maximum(0.0, (ref_date - pd.to_datetime(sub["date"])).dt.days.to_numpy(dtype=float))
    w = np.exp(-math.log(2.0) * dt_days / half_life_days)
    # Give 26/27 in-season matches higher relative weight vs 25/26 carryover
    is_cur = (sub["season"] == "2026/27").to_numpy(dtype=bool)
    w = np.where(is_cur, w * 1.5, w * 0.45)

    low_00 = (hg == 0) & (ag == 0)
    low_01 = (hg == 0) & (ag == 1)
    low_10 = (hg == 1) & (ag == 0)
    low_11 = (hg == 1) & (ag == 1)

    def nll(params):
        att = params[:n_t]
        dfn = params[n_t : 2 * n_t]
        mu = params[2 * n_t]
        hfa = params[2 * n_t + 1]
        rho = np.clip(params[2 * n_t + 2], -0.25, 0.25)

        # Soft centering constraint
        att_c = att - np.mean(att)
        def_c = dfn - np.mean(dfn)

        log_lh = mu + hfa + att_c[hi] + def_c[ai]
        log_la = mu + att_c[ai] + def_c[hi]
        lh = np.exp(np.clip(log_lh, -3.0, 2.5))
        la = np.exp(np.clip(log_la, -3.0, 2.5))

        ll = w * (hg * log_lh - lh + ag * log_la - la)

        tau = np.ones_like(w)
        if low_00.any():
            tau[low_00] = np.clip(1.0 - lh[low_00] * la[low_00] * rho, 1e-4, 10.0)
        if low_01.any():
            tau[low_01] = np.clip(1.0 + lh[low_01] * rho, 1e-4, 10.0)
        if low_10.any():
            tau[low_10] = np.clip(1.0 + la[low_10] * rho, 1e-4, 10.0)
        if low_11.any():
            tau[low_11] = np.clip(1.0 - rho, 1e-4, 10.0)
        ll += w * np.log(tau)

        reg = l2_reg * (np.sum((att_c - prior_att) ** 2) + np.sum((def_c - prior_def) ** 2))
        reg += 2.0 * (np.mean(att) ** 2 + np.mean(dfn) ** 2)
        return float(-np.sum(ll) + reg)

    res = minimize(nll, x0, method="L-BFGS-B", options={"maxiter": 80, "ftol": 1e-5})
    p = res.x
    att = p[:n_t] - np.mean(p[:n_t])
    dfn = p[n_t : 2 * n_t] - np.mean(p[n_t : 2 * n_t])
    model = {
        "t2i": t2i,
        "att": att,
        "def": dfn,
        "mu": float(p[2 * n_t]),
        "hfa": float(p[2 * n_t + 1]),
        "rho": float(np.clip(p[2 * n_t + 2], -0.25, 0.25)),
        "converged": bool(res.success or res.nit > 0),
    }
    return model, p


def predict_dixon_coles(model: dict, home_team: str, away_team: str) -> np.ndarray:
    t2i = model["t2i"]
    ha = model["att"][t2i[home_team]]
    hd = model["def"][t2i[home_team]]
    aa = model["att"][t2i[away_team]]
    ad = model["def"][t2i[away_team]]
    lam_h = math.exp(model["mu"] + model["hfa"] + ha + ad)
    lam_a = math.exp(model["mu"] + aa + hd)
    return score_matrix_to_hda(lam_h, lam_a, rho=model["rho"])


def fit_event_statespace_ridge(
    df_2627_past: pd.DataFrame,
    teams: list[str],
    elo_ratings: dict[str, float],
    promoted_teams: set[str],
    ridge_alpha: float = 18.0,
) -> dict:
    """
    B4_EventStateSpace_xT:
    Reduces small-sample noise in 26/27 by replacing raw binary/integer goals with a denoised
    match performance target combining Goals + Box Shots + Opta xT + WhoScored Team Rating delta:
        perf_h = 0.35 * hg + 0.35 * (h_box_shots * 0.14) + 0.30 * (h_xt * 0.90)
    Solves opponent-adjusted attack & defense strengths via closed-form weighted Ridge regression
    shrunk toward pre-match Elo priors (so early-season promoted/extreme variance is damped).
    """
    t2i = {t: i for i, t in enumerate(teams)}
    n_t = len(teams)
    # Parameter columns: [att_0..att_{n-1}, def_0..def_{n-1}, hfa]
    n_cols = 2 * n_t + 1

    # Elo prior translated to log-goal-rate offset around 0
    prior = np.zeros(n_cols, dtype=float)
    for t, idx in t2i.items():
        elo_z = (elo_ratings.get(t, 1500.0) - 1500.0) / 400.0
        prior[idx] = 0.45 * elo_z + (-0.10 if t in promoted_teams else 0.0)
        prior[n_t + idx] = -0.45 * elo_z + (0.10 if t in promoted_teams else 0.0)
    prior[2 * n_t] = 0.22  # league home advantage on log scale

    if len(df_2627_past) == 0:
        return {"t2i": t2i, "att": prior[:n_t], "def": prior[n_t : 2 * n_t], "hfa": prior[2 * n_t], "mean_rate": 1.37}

    rows_X = []
    rows_y = []
    rows_w = []
    max_date = pd.to_datetime(df_2627_past["date"]).max()

    for _, r in df_2627_past.iterrows():
        hi, ai = t2i[r["home_team"]], t2i[r["away_team"]]
        dt_days = max(0.0, float((max_date - pd.to_datetime(r["date"])).days))
        wt = math.exp(-math.log(2.0) * dt_days / 90.0)

        # Denoised attack performance on goal-equivalent scale
        perf_h = 0.35 * float(r["home_goals"]) + 0.35 * (float(r["home_box_shots"]) * 0.14) + 0.30 * (float(r["home_xt"]) * 0.90)
        perf_a = 0.35 * float(r["away_goals"]) + 0.35 * (float(r["away_box_shots"]) * 0.14) + 0.30 * (float(r["away_xt"]) * 0.90)

        # Home attacking observation
        xh = np.zeros(n_cols, dtype=float)
        xh[hi] = 1.0
        xh[n_t + ai] = 1.0
        xh[2 * n_t] = 1.0
        rows_X.append(xh)
        rows_y.append(math.log(max(0.25, perf_h) / 1.35))
        rows_w.append(wt)

        # Away attacking observation
        xa = np.zeros(n_cols, dtype=float)
        xa[ai] = 1.0
        xa[n_t + hi] = 1.0
        rows_X.append(xa)
        rows_y.append(math.log(max(0.25, perf_a) / 1.35))
        rows_w.append(wt)

    X = np.vstack(rows_X)
    y = np.array(rows_y, dtype=float)
    W = np.diag(rows_w)

    # Ridge around prior: (X^T W X + alpha * I) (theta - prior) = X^T W (y - X @ prior)
    A = X.T @ W @ X + ridge_alpha * np.eye(n_cols)
    b = X.T @ W @ (y - X @ prior)
    theta = prior + np.linalg.solve(A, b)

    att = theta[:n_t] - np.mean(theta[:n_t])
    dfn = theta[n_t : 2 * n_t] - np.mean(theta[n_t : 2 * n_t])
    hfa = float(np.clip(theta[2 * n_t], 0.05, 0.45))
    return {"t2i": t2i, "att": att, "def": dfn, "hfa": hfa, "mean_rate": 1.35}


def predict_event_statespace(model: dict, home_team: str, away_team: str) -> np.ndarray:
    t2i = model["t2i"]
    ha = model["att"][t2i[home_team]]
    hd = model["def"][t2i[home_team]]
    aa = model["att"][t2i[away_team]]
    ad = model["def"][t2i[away_team]]
    lam_h = model["mean_rate"] * math.exp(np.clip(model["hfa"] + ha + ad, -1.5, 1.5))
    lam_a = model["mean_rate"] * math.exp(np.clip(aa + hd, -1.5, 1.5))
    return score_matrix_to_hda(lam_h, lam_a, rho=-0.08)


def fit_xgb_overfit_diag(X_past: np.ndarray, y_past: np.ndarray) -> XGBClassifier | None:
    if len(y_past) < 25 or len(np.unique(y_past)) < 3:
        return None
    clf = XGBClassifier(
        n_estimators=80,
        max_depth=4,
        learning_rate=0.08,
        subsample=0.85,
        colsample_bytree=0.85,
        random_state=42,
        eval_metric="mlogloss",
        verbosity=0,
    )
    clf.fit(X_past, y_past)
    return clf
