"""
ml_epl2627/src/strict_models.py
Clean, leak-free implementations of:
  1. Sequential Walk-Forward Elo (goal-based & xG-blended) + Validation tuner (Train -> Val 2025/26)
  2. True Bayesian Poisson GLM with Exact Laplace Approximation (analytical gradient + Hessian)
     and Vectorized Monte Carlo predictive integration (supports Season-Only Flat Prior,
     Elo-Prior, and Multi-Season Time-Decayed Priorless History)
  3. Reference Bayesian Ridge (log-clipped) for apples-to-apples ablation against Poisson GLM
  4. Independent Stacking (Elo + Flat-Prior Bayes GLM with validated exp(-n/tau) blend)
"""

import math
import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.special import gammaln

from src.metrics import clip_probs, rps_per_match
from src.bayesian_models import (
    fit_ordered_logit_head,
    ordered_logit_probs,
)

_K_VEC = np.arange(9, dtype=float)[None, :]
_LGAMMA_K1 = gammaln(_K_VEC + 1.0)
_TRIL_MASK = np.tril(np.ones((9, 9), dtype=bool), k=-1)
_TRIU_MASK = np.triu(np.ones((9, 9), dtype=bool), k=1)
_DIAG_IDX = np.arange(9)


def fast_vectorized_score_hda(
    lam_h_vec: np.ndarray,
    lam_a_vec: np.ndarray,
    alpha_nb: float = 0.0,
    rho: float = -0.07,
) -> np.ndarray:
    lh = np.clip(lam_h_vec[:, None], 0.05, 6.0)
    la = np.clip(lam_a_vec[:, None], 0.05, 6.0)
    if alpha_nb > 1e-4:
        r = 1.0 / alpha_nb
        c_g = gammaln(_K_VEC + r) - gammaln(r) - _LGAMMA_K1
        ph = np.exp(c_g + r * np.log(r / (r + lh)) + _K_VEC * np.log(lh / (r + lh)))
        pa = np.exp(c_g + r * np.log(r / (r + la)) + _K_VEC * np.log(la / (r + la)))
    else:
        ph = np.exp(_K_VEC * np.log(lh) - lh - _LGAMMA_K1)
        pa = np.exp(_K_VEC * np.log(la) - la - _LGAMMA_K1)

    mats = ph[:, :, None] * pa[:, None, :]
    lh1, la1 = lh[:, 0], la[:, 0]
    mats[:, 0, 0] *= np.clip(1.0 - lh1 * la1 * rho, 1e-4, 5.0)
    mats[:, 0, 1] *= np.clip(1.0 + lh1 * rho, 1e-4, 5.0)
    mats[:, 1, 0] *= np.clip(1.0 + la1 * rho, 1e-4, 5.0)
    mats[:, 1, 1] *= np.clip(1.0 - rho, 1e-4, 5.0)
    mats = np.clip(mats, 0.0, None)
    mats /= mats.sum(axis=(1, 2), keepdims=True)
    return np.column_stack([
        mats[:, _TRIL_MASK].sum(axis=1),
        mats[:, _DIAG_IDX, _DIAG_IDX].sum(axis=1),
        mats[:, _TRIU_MASK].sum(axis=1),
    ])


def fast_predict_bayesian_posterior(
    model: dict,
    home_team: str,
    away_team: str,
    alpha_nb: float = 0.0,
    rho: float = -0.07,
    n_draws: int = 300,
    seed: int = 42,
) -> np.ndarray:
    t2i, n_t = model["t2i"], model["n_t"]
    hi, ai = t2i[home_team], t2i[away_team]
    th = model["theta"]
    sg = model["sigma"]
    h_idx = [hi, n_t + ai, 2 * n_t]
    a_idx = [ai, n_t + hi]
    mu_0 = float(th[h_idx[0]] + th[h_idx[1]] + th[h_idx[2]])
    mu_1 = float(th[a_idx[0]] + th[a_idx[1]])
    v00 = float(np.sum(sg[np.ix_(h_idx, h_idx)]))
    v11 = float(np.sum(sg[np.ix_(a_idx, a_idx)]))
    v01 = float(np.sum(sg[np.ix_(h_idx, a_idx)]))

    rng = np.random.default_rng(seed)
    draws = rng.multivariate_normal([mu_0, mu_1], [[v00, v01], [v01, v11]], size=n_draws)
    lam_h = model["mean_rate"] * np.exp(np.clip(draws[:, 0], -1.5, 1.5))
    lam_a = model["mean_rate"] * np.exp(np.clip(draws[:, 1], -1.5, 1.5))
    mean_p = fast_vectorized_score_hda(lam_h, lam_a, alpha_nb=alpha_nb, rho=rho).mean(axis=0)
    return mean_p / mean_p.sum()


def extract_unified_xg(df: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    """
    Returns comparable team-level xG vectors (home_xg_equiv, away_xg_equiv)
    across both history_1426.csv (home_xg/away_xg) and matches_2627.csv (opta_npxg + xT),
    even when historical and 2026/27 DataFrames are concatenated.
    """
    n = len(df)
    has_xg = "home_xg" in df.columns and "away_xg" in df.columns
    has_opta = "home_opta_npxg" in df.columns and "home_xt" in df.columns
    hx = df["home_xg"].to_numpy(dtype=float) if has_xg else np.full(n, np.nan, dtype=float)
    ax = df["away_xg"].to_numpy(dtype=float) if has_xg else np.full(n, np.nan, dtype=float)
    if has_opta:
        hx_opta = 0.65 * df["home_opta_npxg"].to_numpy(dtype=float) + 0.35 * (df["home_xt"].to_numpy(dtype=float) * 0.90)
        ax_opta = 0.65 * df["away_opta_npxg"].to_numpy(dtype=float) + 0.35 * (df["away_xt"].to_numpy(dtype=float) * 0.90)
        hx = np.where(np.isnan(hx), hx_opta, hx)
        ax = np.where(np.isnan(ax), ax_opta, ax)
    return hx, ax


def run_elo_walkforward(
    df: pd.DataFrame,
    k: float = 20.0,
    hfa: float = 65.0,
    shrink: float = 0.35,
    promo_init: float = 1420.0,
    w_xg: float = 0.0,
    initial_ratings: dict[str, float] | None = None,
    prev_season_teams: set[str] | None = None,
) -> tuple[np.ndarray, dict[str, float], set[str]]:
    """
    Causal sequential Elo across df (already sorted by date, match_id).
    Properly resets returning/promoted teams (not in previous completed season) to promo_init
    and shrinks continuing teams toward 1500.0 by `shrink` at each season boundary.
    When w_xg > 0, blends actual match outcome (0/0.5/1) with sigmoid(1.2 * (xg_h - xg_a)).
    """
    teams = sorted(set(df["home_team"]).union(df["away_team"]))
    if initial_ratings:
        teams = sorted(set(teams).union(initial_ratings.keys()))
    t2i = {t: i for i, t in enumerate(teams)}

    ratings = np.full(len(teams), 1500.0, dtype=float)
    if initial_ratings:
        for t, r in initial_ratings.items():
            ratings[t2i[t]] = float(r)

    home_names = df["home_team"].tolist()
    away_names = df["away_team"].tolist()
    seasons = df["season"].to_numpy()
    hi = np.array([t2i[t] for t in home_names], dtype=int)
    ai = np.array([t2i[t] for t in away_names], dtype=int)
    hg = df["home_goals"].to_numpy(dtype=int)
    ag = df["away_goals"].to_numpy(dtype=int)
    hx, ax = extract_unified_xg(df)

    diffs = np.empty(len(df), dtype=float)
    last_season_seen: set[str] = set(prev_season_teams) if prev_season_teams is not None else set()
    cur_season_seen: set[str] = set()
    prev_sec = None

    for idx in range(len(df)):
        sec = seasons[idx]
        if prev_sec is not None and sec != prev_sec:
            # Regress active teams from the just-finished season toward 1500;
            # teams absent in prev_sec will be initialized at promo_init upon first match.
            for t_name in cur_season_seen:
                ti = t2i[t_name]
                ratings[ti] = 1500.0 + (1.0 - shrink) * (ratings[ti] - 1500.0)
            last_season_seen = cur_season_seen
            cur_season_seen = set()
        elif prev_sec is None and initial_ratings is None:
            # First ever warmup season: everybody starts at 1500
            pass

        h_name = home_names[idx]
        a_name = away_names[idx]
        if prev_sec is not None or initial_ratings is not None:
            if h_name not in last_season_seen and h_name not in cur_season_seen:
                ratings[t2i[h_name]] = float(promo_init)
            if a_name not in last_season_seen and a_name not in cur_season_seen:
                ratings[t2i[a_name]] = float(promo_init)

        cur_season_seen.add(h_name)
        cur_season_seen.add(a_name)
        prev_sec = sec

        h_i, a_i = hi[idx], ai[idx]
        diff = (ratings[h_i] + hfa) - ratings[a_i]
        diffs[idx] = diff

        exp_h = 1.0 / (1.0 + 10.0 ** (-diff / 400.0))
        g_h, g_a = int(hg[idx]), int(ag[idx])
        actual_goal = 1.0 if g_h > g_a else (0.5 if g_h == g_a else 0.0)

        if w_xg > 0.0:
            xg_diff = float(hx[idx] - ax[idx])
            actual_xg = 1.0 / (1.0 + math.exp(-np.clip(1.25 * xg_diff, -10.0, 10.0)))
            actual_h = (1.0 - w_xg) * actual_goal + w_xg * actual_xg
            eff_gd = (1.0 - w_xg) * abs(g_h - g_a) + w_xg * abs(xg_diff)
        else:
            actual_h = actual_goal
            eff_gd = float(abs(g_h - g_a))

        mov = math.log1p(eff_gd) * (2.2 / (abs(diff) * 0.001 + 2.2))
        delta = k * (1.0 if mov < 1.0 else mov) * (actual_h - exp_h)
        ratings[h_i] += delta
        ratings[a_i] -= delta

    end_ratings = {t: float(ratings[t2i[t]]) for t in teams}
    return diffs, end_ratings, cur_season_seen


def prepare_next_season_ratings(
    end_ratings: dict[str, float],
    last_season_teams: set[str],
    next_teams: list[str],
    shrink: float,
    promo_init: float,
) -> dict[str, float]:
    out = {}
    for t in next_teams:
        if t in last_season_teams and t in end_ratings:
            out[t] = 1500.0 + (1.0 - shrink) * (end_ratings[t] - 1500.0)
        else:
            out[t] = float(promo_init)
    return out


def tune_elo_strict(
    df_train: pd.DataFrame,
    df_val: pd.DataFrame,
    allow_xg: bool = True,
) -> dict:
    """
    Strictly optimizes Elo hyperparameters (k, hfa, shrink, promo_init, w_xg) on df_val (2025/26)
    while fitting the Ordered Logit head ONLY on df_train (<=2024/25, excluding 1st warmup season).
    Zero leakage from df_val into logit head and zero leakage from 2026/27 test set.
    """
    df_tv = pd.concat([df_train, df_val], ignore_index=True)
    n_train = len(df_train)
    train_seasons = sorted(df_train["season"].unique())
    warmup_season = train_seasons[0]
    fit_head_mask = (df_train["season"] != warmup_season).to_numpy()
    y_train = df_train["target"].to_numpy(dtype=int)[fit_head_mask]
    y_val = df_val["target"].to_numpy(dtype=int)

    wxg_candidates = (0.0, 0.45, 0.70) if allow_xg else (0.0,)
    best = {"val_rps": 999.0}

    # Coarse grid on Validation 2025/26 RPS
    for w_xg in wxg_candidates:
        for k_g in (14.0, 22.0, 32.0):
            for hfa_g in (40.0, 60.0, 80.0):
                for sh_g in (0.20, 0.35, 0.50):
                    for pr_g in (1390.0, 1430.0):
                        diffs_tv, _, _ = run_elo_walkforward(df_tv, k=k_g, hfa=hfa_g, shrink=sh_g, promo_init=pr_g, w_xg=w_xg)
                        c1, c2, beta = fit_ordered_logit_head(diffs_tv[:n_train][fit_head_mask], y_train)
                        val_probs = clip_probs(ordered_logit_probs(diffs_tv[n_train:], c1, c2, beta))
                        val_rps = float(np.mean(rps_per_match(y_val, val_probs)))
                        if val_rps < best["val_rps"]:
                            best = {
                                "val_rps": val_rps,
                                "k": k_g,
                                "hfa": hfa_g,
                                "shrink": sh_g,
                                "promo_init": pr_g,
                                "w_xg": w_xg,
                                "head_train_only": (c1, c2, beta),
                                "val_probs": val_probs,
                                "val_diffs": diffs_tv[n_train:],
                            }

    # Local Powell refinement on Validation RPS keeping w_xg fixed
    w_xg_fixed = best["w_xg"]

    def obj(params: np.ndarray) -> float:
        k_v, hfa_v, sh_v, pr_v = params
        if not (5.0 <= k_v <= 60.0 and 0.0 <= hfa_v <= 140.0 and 0.0 <= sh_v <= 0.90 and 1250.0 <= pr_v <= 1550.0):
            return 1.0
        diffs_tv, _, _ = run_elo_walkforward(df_tv, k=float(k_v), hfa=float(hfa_v), shrink=float(sh_v), promo_init=float(pr_v), w_xg=w_xg_fixed)
        c1, c2, beta = fit_ordered_logit_head(diffs_tv[:n_train][fit_head_mask], y_train)
        val_probs = clip_probs(ordered_logit_probs(diffs_tv[n_train:], c1, c2, beta))
        return float(np.mean(rps_per_match(y_val, val_probs)))

    x0 = np.array([best["k"], best["hfa"], best["shrink"], best["promo_init"]])
    res = minimize(obj, x0, method="Powell", options={"maxiter": 18, "ftol": 1e-4})
    if float(res.fun) < best["val_rps"]:
        k_v, hfa_v, sh_v, pr_v = [float(x) for x in res.x]
    else:
        k_v, hfa_v, sh_v, pr_v = best["k"], best["hfa"], best["shrink"], best["promo_init"]

    diffs_tv, end_r_tv, last_teams_tv = run_elo_walkforward(df_tv, k=k_v, hfa=hfa_v, shrink=sh_v, promo_init=pr_v, w_xg=w_xg_fixed)
    c1_tr, c2_tr, beta_tr = fit_ordered_logit_head(diffs_tv[:n_train][fit_head_mask], y_train)
    val_probs = clip_probs(ordered_logit_probs(diffs_tv[n_train:], c1_tr, c2_tr, beta_tr))
    val_rps = float(np.mean(rps_per_match(y_val, val_probs)))

    # Frozen logit head fit on Train + Val (excluding 1st warmup season) for Locked Test 2026/27
    fit_tv_mask = (df_tv["season"] != warmup_season).to_numpy()
    c1_tv, c2_tv, beta_tv = fit_ordered_logit_head(diffs_tv[fit_tv_mask], df_tv["target"].to_numpy(dtype=int)[fit_tv_mask])

    return {
        "k": round(k_v, 3),
        "hfa": round(hfa_v, 3),
        "shrink": round(sh_v, 4),
        "promo_init": round(pr_v, 2),
        "w_xg": w_xg_fixed,
        "val_rps": val_rps,
        "val_probs": val_probs,
        "val_diffs": diffs_tv[n_train:],
        "head_train_only": (c1_tr, c2_tr, beta_tr),
        "head_tv": (c1_tv, c2_tv, beta_tv),
        "end_ratings_tv": end_r_tv,
        "last_teams_tv": last_teams_tv,
    }


def evaluate_fixed_elo(
    df_train: pd.DataFrame,
    df_val: pd.DataFrame,
    k: float,
    hfa: float,
    shrink: float,
    promo_init: float,
    w_xg: float = 0.0,
) -> dict:
    """Evaluates fixed/handcrafted Elo under the exact same strict Train->Val protocol."""
    df_tv = pd.concat([df_train, df_val], ignore_index=True)
    n_train = len(df_train)
    warmup_season = sorted(df_train["season"].unique())[0]
    fit_head_mask = (df_train["season"] != warmup_season).to_numpy()

    diffs_tv, end_r_tv, last_teams_tv = run_elo_walkforward(df_tv, k=k, hfa=hfa, shrink=shrink, promo_init=promo_init, w_xg=w_xg)
    c1_tr, c2_tr, beta_tr = fit_ordered_logit_head(diffs_tv[:n_train][fit_head_mask], df_train["target"].to_numpy(dtype=int)[fit_head_mask])
    val_probs = clip_probs(ordered_logit_probs(diffs_tv[n_train:], c1_tr, c2_tr, beta_tr))

    fit_tv_mask = (df_tv["season"] != warmup_season).to_numpy()
    c1_tv, c2_tv, beta_tv = fit_ordered_logit_head(diffs_tv[fit_tv_mask], df_tv["target"].to_numpy(dtype=int)[fit_tv_mask])

    return {
        "k": k,
        "hfa": hfa,
        "shrink": shrink,
        "promo_init": promo_init,
        "w_xg": w_xg,
        "val_probs": val_probs,
        "val_diffs": diffs_tv[n_train:],
        "head_train_only": (c1_tr, c2_tr, beta_tr),
        "head_tv": (c1_tv, c2_tv, beta_tv),
        "end_ratings_tv": end_r_tv,
        "last_teams_tv": last_teams_tv,
    }


def predict_elo_on_test(
    df_test: pd.DataFrame,
    elo_fit: dict,
) -> tuple[np.ndarray, np.ndarray]:
    """Walk-forward Elo on locked test season using frozen hyperparams and Train+Val logit head."""
    teams_test = sorted(set(df_test["home_team"]).union(df_test["away_team"]))
    init_test = prepare_next_season_ratings(
        elo_fit["end_ratings_tv"],
        elo_fit["last_teams_tv"],
        teams_test,
        elo_fit["shrink"],
        elo_fit["promo_init"],
    )
    diffs, _, _ = run_elo_walkforward(
        df_test,
        k=elo_fit["k"],
        hfa=elo_fit["hfa"],
        shrink=elo_fit["shrink"],
        promo_init=elo_fit["promo_init"],
        w_xg=elo_fit["w_xg"],
        initial_ratings=init_test,
        prev_season_teams=elo_fit["last_teams_tv"],
    )
    c1, c2, beta = elo_fit["head_tv"]
    return clip_probs(ordered_logit_probs(diffs, c1, c2, beta)), diffs


def fit_bayesian_poisson_glm_laplace(
    df_past: pd.DataFrame,
    teams: list[str],
    prior_scale: float = 0.28,
    half_life_days: float = 180.0,
    w_goals: float = 0.35,
    hfa_prior: float = 0.22,
    hfa_prior_std: float = 0.14,
    elo_priors: dict[str, float] | None = None,
    promoted_teams: set[str] | None = None,
    promo_offset: float = 0.0,
    season_gap_decay: float = 1.0,
    ref_date: pd.Timestamp | None = None,
    target_season: str | None = None,
) -> dict:
    """
    True Bayesian Poisson GLM via Damped Newton-Raphson + Exact Laplace Posterior Covariance:
      log(lambda_i) = log(mean_rate) + x_i^T theta
      J(theta) = sum_i w_i * (lambda_i - y_i * eta_i) + 0.5 * (theta - mu0)^T Lambda0 (theta - mu0)
      H(theta) = X^T diag(w * lambda) X + Lambda0   =>   Sigma = H(theta_MAP)^(-1)
    Zero ad-hoc log(max(0.25, perf)) clipping! Non-negative y_i works natively in Poisson GLM.
    """
    promoted_teams = promoted_teams or set()
    all_teams = sorted(set(teams).union(df_past["home_team"]).union(df_past["away_team"])) if len(df_past) > 0 else sorted(teams)
    t2i = {t: i for i, t in enumerate(all_teams)}
    n_t = len(all_teams)
    n_cols = 2 * n_t + 1

    mu0 = np.zeros(n_cols, dtype=float)
    if elo_priors is not None:
        for t, idx in t2i.items():
            elo_z = (elo_priors.get(t, 1500.0) - 1500.0) / 400.0
            mu0[idx] = 0.45 * elo_z
            mu0[n_t + idx] = -0.45 * elo_z
    if promo_offset != 0.0:
        for t in promoted_teams:
            if t in t2i:
                idx = t2i[t]
                mu0[idx] -= promo_offset
                mu0[n_t + idx] += promo_offset
    mu0[2 * n_t] = hfa_prior

    prec_vec = np.empty(n_cols, dtype=float)
    prec_vec[: 2 * n_t] = 1.0 / (prior_scale ** 2)
    prec_vec[2 * n_t] = 1.0 / (hfa_prior_std ** 2)

    if len(df_past) == 0:
        return {
            "t2i": t2i,
            "n_t": n_t,
            "theta": mu0,
            "sigma": np.diag(1.0 / prec_vec),
            "mean_rate": 1.35,
        }

    dates = pd.to_datetime(df_past["date"])
    max_d = dates.max() if ref_date is None else ref_date
    dt_days = np.maximum(0.0, (max_d - dates).dt.days.to_numpy(dtype=float))
    wt = np.exp(-math.log(2.0) * dt_days / half_life_days)
    if season_gap_decay < 1.0 and target_season is not None:
        same_sec = (df_past["season"].to_numpy() == target_season)
        wt = np.where(same_sec, wt, wt * season_gap_decay)

    hx, ax = extract_unified_xg(df_past)
    hg = df_past["home_goals"].to_numpy(dtype=float)
    ag = df_past["away_goals"].to_numpy(dtype=float)
    perf_h = np.maximum(0.0, w_goals * hg + (1.0 - w_goals) * hx)
    perf_a = np.maximum(0.0, w_goals * ag + (1.0 - w_goals) * ax)

    n_m = len(df_past)
    y = np.concatenate([perf_h, perf_a])
    w = np.concatenate([wt, wt])

    # Causal empirical scoring scale shrunk smoothly toward 1.35
    w_sum = float(np.sum(w))
    mean_rate = float(np.clip((np.sum(w * y) + 40.0 * 1.35) / (w_sum + 40.0), 1.05, 1.75))
    log_base = math.log(mean_rate)

    hi = np.array([t2i[t] for t in df_past["home_team"]], dtype=int)
    ai = np.array([t2i[t] for t in df_past["away_team"]], dtype=int)

    X = np.zeros((2 * n_m, n_cols), dtype=float)
    row_h = np.arange(n_m)
    row_a = n_m + np.arange(n_m)
    X[row_h, hi] = 1.0
    X[row_h, n_t + ai] = 1.0
    X[row_h, 2 * n_t] = 1.0
    X[row_a, ai] = 1.0
    X[row_a, n_t + hi] = 1.0

    # Damped Newton-Raphson for exact MAP and exact Laplace Hessian
    theta = mu0.copy()
    Lambda0 = np.diag(prec_vec)

    def nlp_and_grad(th: np.ndarray) -> tuple[float, np.ndarray, np.ndarray]:
        eta = np.clip(log_base + X @ th, -6.0, 4.0)
        lam = np.exp(eta)
        diff0 = th - mu0
        loss = float(np.sum(w * (lam - y * eta)) + 0.5 * np.sum(prec_vec * (diff0 ** 2)))
        grad = X.T @ (w * (lam - y)) + prec_vec * diff0
        return loss, grad, lam

    cur_loss, grad, lam = nlp_and_grad(theta)
    for _ in range(15):
        if np.max(np.abs(grad)) < 1e-6:
            break
        H = (X.T * (w * lam)) @ X + Lambda0
        step = np.linalg.solve(H, grad)
        alpha_ls = 1.0
        for _ in range(8):
            cand = theta - alpha_ls * step
            c_loss, c_grad, c_lam = nlp_and_grad(cand)
            if c_loss <= cur_loss + 1e-9:
                theta, cur_loss, grad, lam = cand, c_loss, c_grad, c_lam
                break
            alpha_ls *= 0.5
        if alpha_ls < 1e-2:
            break

    H_final = (X.T * (w * lam)) @ X + Lambda0
    Sigma = np.linalg.inv(H_final)
    # Soft-center attack and defense parameters (preserves relative strengths cleanly)
    theta[:n_t] -= np.mean(theta[:n_t])
    theta[n_t : 2 * n_t] -= np.mean(theta[n_t : 2 * n_t])

    return {
        "t2i": t2i,
        "n_t": n_t,
        "theta": theta,
        "sigma": Sigma,
        "mean_rate": mean_rate,
    }


def fit_bayesian_ridge_logclip(
    df_past: pd.DataFrame,
    teams: list[str],
    ridge_alpha: float = 18.0,
    half_life_days: float = 90.0,
    w_goals: float = 0.35,
    hfa_prior: float = 0.22,
    elo_priors: dict[str, float] | None = None,
    promoted_teams: set[str] | None = None,
    promo_offset: float = 0.10,
    ref_date: pd.Timestamp | None = None,
) -> dict:
    """
    Gaussian Bayesian Ridge with log(max(0.25, perf)) clipping (previous Champion Bayesian component).
    """
    promoted_teams = promoted_teams or set()
    all_teams = sorted(set(teams).union(df_past["home_team"]).union(df_past["away_team"])) if len(df_past) > 0 else sorted(teams)
    t2i = {t: i for i, t in enumerate(all_teams)}
    n_t = len(all_teams)
    n_cols = 2 * n_t + 1

    prior = np.zeros(n_cols, dtype=float)
    if elo_priors is not None:
        for t, idx in t2i.items():
            elo_z = (elo_priors.get(t, 1500.0) - 1500.0) / 400.0
            prior[idx] = 0.45 * elo_z
            prior[n_t + idx] = -0.45 * elo_z
    if promo_offset != 0.0:
        for t in promoted_teams:
            if t in t2i:
                idx = t2i[t]
                prior[idx] -= promo_offset
                prior[n_t + idx] += promo_offset
    prior[2 * n_t] = hfa_prior

    prior_sigma2 = 0.16
    if len(df_past) == 0:
        return {
            "t2i": t2i,
            "n_t": n_t,
            "theta": prior,
            "sigma": (prior_sigma2 / ridge_alpha) * np.eye(n_cols),
            "mean_rate": 1.35,
        }

    dates = pd.to_datetime(df_past["date"])
    max_d = dates.max() if ref_date is None else ref_date
    dt_days = np.maximum(0.0, (max_d - dates).dt.days.to_numpy(dtype=float))
    wt = np.exp(-math.log(2.0) * dt_days / half_life_days)

    hx, ax = extract_unified_xg(df_past)
    hg = df_past["home_goals"].to_numpy(dtype=float)
    ag = df_past["away_goals"].to_numpy(dtype=float)
    perf_h = np.maximum(0.25, w_goals * hg + (1.0 - w_goals) * hx)
    perf_a = np.maximum(0.25, w_goals * ag + (1.0 - w_goals) * ax)

    n_m = len(df_past)
    y = np.log(np.concatenate([perf_h, perf_a]) / 1.35)
    w = np.concatenate([wt, wt])

    hi = np.array([t2i[t] for t in df_past["home_team"]], dtype=int)
    ai = np.array([t2i[t] for t in df_past["away_team"]], dtype=int)
    X = np.zeros((2 * n_m, n_cols), dtype=float)
    row_h = np.arange(n_m)
    row_a = n_m + np.arange(n_m)
    X[row_h, hi] = 1.0
    X[row_h, n_t + ai] = 1.0
    X[row_h, 2 * n_t] = 1.0
    X[row_a, ai] = 1.0
    X[row_a, n_t + hi] = 1.0

    XTW = X.T * w
    A = XTW @ X + ridge_alpha * np.eye(n_cols)
    A_inv = np.linalg.inv(A)
    theta = prior + A_inv @ (XTW @ (y - X @ prior))

    resid = y - X @ theta
    eff_df = max(1.0, float(np.sum(w)) - float(np.trace((XTW @ X) @ A_inv)))
    emp_sigma2 = float(np.sum(w * (resid ** 2)) / eff_df)
    sigma2 = 0.5 * emp_sigma2 + 0.5 * prior_sigma2

    theta[:n_t] -= np.mean(theta[:n_t])
    theta[n_t : 2 * n_t] -= np.mean(theta[n_t : 2 * n_t])

    return {
        "t2i": t2i,
        "n_t": n_t,
        "theta": theta,
        "sigma": sigma2 * A_inv,
        "mean_rate": 1.35,
    }


def walkforward_bayesian_season(
    df_hist_before: pd.DataFrame,
    df_target: pd.DataFrame,
    fit_fn,
    fit_kwargs: dict,
    use_multi_season_history: bool = False,
    alpha_nb: float = 0.0,
    rho: float = -0.07,
    n_draws: int = 300,
    seed: int = 42,
    promoted_teams: set[str] | None = None,
) -> np.ndarray:
    """
    Runs strict causal matchday-by-matchday (date-by-date) posterior prediction over df_target.
    Refits once per unique match date using strictly matches BEFORE that date.
    """
    target_teams = sorted(set(df_target["home_team"]).union(df_target["away_team"]))
    target_season = str(df_target["season"].iloc[0])
    dates_target = pd.to_datetime(df_target["date"])
    unique_dates = sorted(dates_target.unique())

    probs_out = np.empty((len(df_target), 3), dtype=float)

    for d_val in unique_dates:
        mask_day = (dates_target == d_val).to_numpy()
        past_target = df_target[dates_target < d_val]
        if use_multi_season_history and len(df_hist_before) > 0:
            df_past = pd.concat([df_hist_before, past_target], ignore_index=True)
        else:
            df_past = past_target

        kw = dict(fit_kwargs)
        kw["ref_date"] = pd.Timestamp(d_val)
        if "target_season" in fit_fn.__code__.co_varnames:
            kw["target_season"] = target_season
        if promoted_teams is not None and "promoted_teams" in fit_fn.__code__.co_varnames:
            kw["promoted_teams"] = promoted_teams

        model = fit_fn(df_past, target_teams, **kw)
        idxs = np.where(mask_day)[0]
        for row_i in idxs:
            r = df_target.iloc[row_i]
            probs_out[row_i] = fast_predict_bayesian_posterior(
                model,
                r["home_team"],
                r["away_team"],
                alpha_nb=alpha_nb,
                rho=rho,
                n_draws=n_draws,
                seed=seed + int(row_i),
            )

    return clip_probs(probs_out)
