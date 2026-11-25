"""
ml_epl2627/src/bayesian_models.py
Distributional & Bayesian State-Space models + Historically-Tuned Elo:
  1. tune_elo_on_history(df_hist, teams_2627): learns (K, HFA, shrink, promoted_init)
     and Ordered Logit head strictly on 2021/22-2025/26 (Zero 26/27 leakage),
     plus historical Negative Binomial dispersion alpha_nb and Dixon-Coles rho.
  2. fit_bayesian_statespace & predict_bayesian_posterior: exact Bayesian Ridge
     posterior covariance Sigma and vectorized Monte Carlo integration over
     epistemic parameter uncertainty + aleatoric NegBin/Poisson score matrix.
"""

import math
import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.special import gammaln
from scipy.stats import poisson, nbinom

from src.metrics import rps_per_match, clip_probs


def ordered_logit_probs(diff: np.ndarray, c1: float, c2: float, beta: float) -> np.ndarray:
    """
    Proportional odds (Ordered Logistic) mapping scalar strength diff -> (P(H), P(D), P(A)).
    Note H=0, D=1, A=2 in ordered sequence; since higher diff favors Home (class 0):
      P(Y <= 0) = P(H)     = sigmoid(c1 + beta * diff)
      P(Y <= 1) = P(H | D) = sigmoid(c2 + beta * diff)   with c2 > c1
      P(D) = P(Y <= 1) - P(Y <= 0)
      P(A) = 1 - P(Y <= 1)
    """
    d = np.asarray(diff, dtype=float)
    z1 = np.clip(c1 + beta * d, -20.0, 20.0)
    z2 = np.clip(c2 + beta * d, -20.0, 20.0)
    s1 = 1.0 / (1.0 + np.exp(-z1))
    s2 = 1.0 / (1.0 + np.exp(-z2))
    p_h = s1
    p_d = np.maximum(1e-6, s2 - s1)
    p_a = np.maximum(1e-6, 1.0 - s2)
    probs = np.column_stack([p_h, p_d, p_a])
    return probs / probs.sum(axis=1, keepdims=True)


def fit_ordered_logit_head(diffs: np.ndarray, y_true: np.ndarray) -> tuple[float, float, float]:
    """Fit (c1, c2, beta) minimizing regularized LogLoss on ordered outcomes."""
    def obj(params):
        c1, gap, beta = params
        c2 = c1 + math.exp(np.clip(gap, -5.0, 5.0))
        probs = clip_probs(ordered_logit_probs(diffs, c1, c2, beta))
        ll = -np.mean(np.log(probs[np.arange(len(y_true)), y_true]))
        return float(ll + 1e-4 * (c1 ** 2 + beta ** 2))

    res = minimize(obj, np.array([-0.4, math.log(1.0), 0.004]), method="BFGS")
    c1, gap, beta = res.x
    c2 = c1 + math.exp(np.clip(gap, -5.0, 5.0))
    return float(c1), float(c2), float(beta)


def run_elo_sequence(
    hi: np.ndarray,
    ai: np.ndarray,
    hg: np.ndarray,
    ag: np.ndarray,
    season_idx: np.ndarray,
    n_teams: int,
    ever_seen_before: np.ndarray,
    k: float,
    hfa: float,
    shrink: float,
    promo_init: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Fast primitive loop running sequential Elo across matches, returning pre-match diffs & final ratings."""
    ratings = np.full(n_teams, 1500.0, dtype=float)
    active = np.zeros(n_teams, dtype=bool)
    diffs = np.empty(len(hi), dtype=float)
    prev_sec = -1

    for idx in range(len(hi)):
        sec = int(season_idx[idx])
        if prev_sec != -1 and sec != prev_sec:
            # Regress existing teams toward 1500
            for t in range(n_teams):
                if active[t]:
                    ratings[t] = 1500.0 + (1.0 - shrink) * (ratings[t] - 1500.0)
        prev_sec = sec

        h = int(hi[idx])
        a = int(ai[idx])
        if not active[h]:
            ratings[h] = promo_init if sec > 0 else 1500.0
            active[h] = True
        if not active[a]:
            ratings[a] = promo_init if sec > 0 else 1500.0
            active[a] = True

        diff = (ratings[h] + hfa) - ratings[a]
        diffs[idx] = diff

        exp_h = 1.0 / (1.0 + 10.0 ** (-diff / 400.0))
        g_h, g_a = int(hg[idx]), int(ag[idx])
        actual_h = 1.0 if g_h > g_a else (0.5 if g_h == g_a else 0.0)
        mov = math.log1p(abs(g_h - g_a)) * (2.2 / (abs(diff) * 0.001 + 2.2))
        delta = k * (1.0 if mov < 1.0 else mov) * (actual_h - exp_h)
        ratings[h] += delta
        ratings[a] -= delta

    return diffs, ratings


def compute_elo_dict_over_history(
    df_hist: pd.DataFrame,
    teams_2627: list[str],
    k: float,
    hfa: float,
    shrink: float,
    promo_init: float,
) -> tuple[dict[str, float], np.ndarray, np.ndarray]:
    """Run Elo across df_hist, return 26/27 initial ratings + out-of-sample (22/23-25/26) diffs & targets."""
    all_teams = sorted(set(df_hist["home_team"]).union(df_hist["away_team"]).union(teams_2627))
    t2i = {t: i for i, t in enumerate(all_teams)}
    seasons = sorted(df_hist["season"].unique())
    s2i = {s: i for i, s in enumerate(seasons)}

    hi = np.array([t2i[t] for t in df_hist["home_team"]], dtype=int)
    ai = np.array([t2i[t] for t in df_hist["away_team"]], dtype=int)
    hg = df_hist["home_goals"].to_numpy(dtype=int)
    ag = df_hist["away_goals"].to_numpy(dtype=int)
    sec_i = np.array([s2i[s] for s in df_hist["season"]], dtype=int)

    diffs, final_r = run_elo_sequence(hi, ai, hg, ag, sec_i, len(all_teams), np.zeros(len(all_teams)), k, hfa, shrink, promo_init)

    # Which teams played in 25/26?
    teams_2526 = set(df_hist[df_hist["season"] == "2025/26"]["home_team"]).union(
        df_hist[df_hist["season"] == "2025/26"]["away_team"]
    )
    init_2627 = {}
    for t in teams_2627:
        if t in teams_2526:
            init_2627[t] = float(1500.0 + (1.0 - shrink) * (final_r[t2i[t]] - 1500.0))
        else:
            init_2627[t] = float(promo_init)

    eval_mask = sec_i >= 1  # Exclude 2021/22 warmup season
    return init_2627, diffs[eval_mask], df_hist["target"].to_numpy(dtype=int)[eval_mask]


def estimate_historical_negbin_and_rho(df_hist: pd.DataFrame) -> tuple[float, float]:
    """
    Estimate Negative Binomial overdispersion alpha_nb (Var = mu + alpha_nb * mu^2)
    and Dixon-Coles low-score correlation rho strictly on historical EPL matches (2021/22-2025/26).
    Uses de-vigged odds or simple season team-mean goals as baseline expected lambda when odds exist.
    """
    # Fit quick Ridge on historical matches to get clean team expected goal rates (lam_h, lam_a)
    seasons = sorted(df_hist["season"].unique())
    lam_h_all, lam_a_all, hg_all, ag_all = [], [], [], []
    for sec in seasons:
        sub = df_hist[df_hist["season"] == sec]
        teams = sorted(set(sub["home_team"]).union(sub["away_team"]))
        t2i = {t: i for i, t in enumerate(teams)}
        n_t = len(teams)
        X_rows, y_rows = [], []
        for _, r in sub.iterrows():
            hi, ai = t2i[r["home_team"]], t2i[r["away_team"]]
            xh = np.zeros(2 * n_t + 1)
            xh[hi] = 1.0
            xh[n_t + ai] = 1.0
            xh[2 * n_t] = 1.0
            xa = np.zeros(2 * n_t + 1)
            xa[ai] = 1.0
            xa[n_t + hi] = 1.0
            X_rows.extend([xh, xa])
            y_rows.extend([math.log(max(0.35, float(r["home_goals"])) / 1.45), math.log(max(0.35, float(r["away_goals"])) / 1.45)])
        X = np.vstack(X_rows)
        y = np.array(y_rows)
        theta = np.linalg.solve(X.T @ X + 15.0 * np.eye(2 * n_t + 1), X.T @ y)
        for i in range(0, len(X_rows), 2):
            lam_h_all.append(1.45 * math.exp(np.clip(X_rows[i] @ theta, -1.2, 1.2)))
            lam_a_all.append(1.45 * math.exp(np.clip(X_rows[i + 1] @ theta, -1.2, 1.2)))
        hg_all.extend(sub["home_goals"].tolist())
        ag_all.extend(sub["away_goals"].tolist())

    lam = np.array(lam_h_all + lam_a_all, dtype=float)
    obs = np.array(hg_all + ag_all, dtype=int)

    # Fit NegBin alpha via MLE on (obs | lam)
    def nb_nll(alpha_arr):
        a = max(1e-5, float(alpha_arr[0]))
        r = 1.0 / a
        p = r / (r + lam)
        ll = gammaln(obs + r) - gammaln(r) - gammaln(obs + 1) + r * np.log(p) + obs * np.log(1.0 - p)
        return float(-np.sum(ll))

    res_nb = minimize(nb_nll, np.array([0.05]), bounds=[(1e-5, 0.5)], method="L-BFGS-B")
    alpha_nb = float(res_nb.x[0])

    # Fit Dixon-Coles rho on historical (lam_h, lam_a, hg, ag)
    lh = np.array(lam_h_all)
    la = np.array(lam_a_all)
    hg = np.array(hg_all)
    ag = np.array(ag_all)
    m00 = (hg == 0) & (ag == 0)
    m01 = (hg == 0) & (ag == 1)
    m10 = (hg == 1) & (ag == 0)
    m11 = (hg == 1) & (ag == 1)

    def rho_nll(rho_arr):
        rho = float(rho_arr[0])
        tau = np.ones_like(lh)
        tau[m00] = np.clip(1.0 - lh[m00] * la[m00] * rho, 1e-4, 5.0)
        tau[m01] = np.clip(1.0 + lh[m01] * rho, 1e-4, 5.0)
        tau[m10] = np.clip(1.0 + la[m10] * rho, 1e-4, 5.0)
        tau[m11] = np.clip(1.0 - rho, 1e-4, 5.0)
        return float(-np.sum(np.log(tau)))

    res_rho = minimize(rho_nll, np.array([-0.06]), bounds=[(-0.22, 0.22)], method="L-BFGS-B")
    rho_val = float(res_rho.x[0])
    return alpha_nb, rho_val


def tune_elo_on_history(df_hist: pd.DataFrame, teams_2627: list[str]) -> dict:
    """
    Learns optimal Elo hyperparameters (K, HFA, shrink, promoted_init) AND Ordered Logit head
    strictly on 2021/22-2025/26 (2021/22 warmup, 2022/23-2025/26 OOS RPS optimization).
    """
    assert df_hist["season"].max() == "2025/26", "Leakage check failed: df_hist contains >= 2026/27!"
    all_teams = sorted(set(df_hist["home_team"]).union(df_hist["away_team"]).union(teams_2627))
    t2i = {t: i for i, t in enumerate(all_teams)}
    seasons = sorted(df_hist["season"].unique())
    s2i = {s: i for i, s in enumerate(seasons)}

    hi = np.array([t2i[t] for t in df_hist["home_team"]], dtype=int)
    ai = np.array([t2i[t] for t in df_hist["away_team"]], dtype=int)
    hg = df_hist["home_goals"].to_numpy(dtype=int)
    ag = df_hist["away_goals"].to_numpy(dtype=int)
    sec_i = np.array([s2i[s] for s in df_hist["season"]], dtype=int)
    y_all = df_hist["target"].to_numpy(dtype=int)
    eval_mask = sec_i >= 1  # N = 1520 (2022/23-2025/26)

    def rps_objective(params):
        k, hfa, shrink, promo = params
        if not (8.0 <= k <= 40.0 and 20.0 <= hfa <= 110.0 and 0.10 <= shrink <= 0.70 and 1320.0 <= promo <= 1480.0):
            return 1.0
        diffs, _ = run_elo_sequence(hi, ai, hg, ag, sec_i, len(all_teams), np.zeros(len(all_teams)), k, hfa, shrink, promo)
        d_eval = diffs[eval_mask]
        y_eval = y_all[eval_mask]
        c1, c2, beta = fit_ordered_logit_head(d_eval, y_eval)
        probs = ordered_logit_probs(d_eval, c1, c2, beta)
        return float(np.mean(rps_per_match(y_eval, probs)))

    # Coarse grid seed + Powell refinement
    best_x = np.array([20.0, 65.0, 0.35, 1420.0])
    best_val = rps_objective(best_x)
    for k_g in (14.0, 20.0, 28.0):
        for sh_g in (0.20, 0.35, 0.50):
            for pr_g in (1370.0, 1420.0):
                x_try = np.array([k_g, 60.0, sh_g, pr_g])
                v = rps_objective(x_try)
                if v < best_val:
                    best_val = v
                    best_x = x_try

    res = minimize(rps_objective, best_x, method="Powell", options={"maxiter": 60, "ftol": 1e-5})
    k_opt, hfa_opt, shrink_opt, promo_opt = [float(x) for x in res.x]

    init_2627, d_eval, y_eval = compute_elo_dict_over_history(df_hist, teams_2627, k_opt, hfa_opt, shrink_opt, promo_opt)
    c1, c2, beta = fit_ordered_logit_head(d_eval, y_eval)

    # Also fit structural-only head (for Elo_B_StructHistory: cold start teams=1500, no shrinkage)
    _, d_cold, y_cold = compute_elo_dict_over_history(df_hist, teams_2627, k_opt, hfa_opt, 1.0, 1500.0)
    c1_s, c2_s, beta_s = fit_ordered_logit_head(d_cold, y_cold)

    # Hand-tuned reference head
    init_hand, d_hand, y_hand = compute_elo_dict_over_history(df_hist, teams_2627, 20.0, 65.0, 0.35, 1420.0)
    c1_h, c2_h, beta_h = fit_ordered_logit_head(d_hand, y_hand)

    # Naive no-shrinkage carryover (shrink=0.0)
    init_noshrink, d_ns, y_ns = compute_elo_dict_over_history(df_hist, teams_2627, 20.0, 65.0, 0.0, 1420.0)
    c1_ns, c2_ns, beta_ns = fit_ordered_logit_head(d_ns, y_ns)

    alpha_nb, rho_hist = estimate_historical_negbin_and_rho(df_hist)

    return {
        "k": round(k_opt, 3),
        "hfa": round(hfa_opt, 3),
        "shrink": round(shrink_opt, 4),
        "promo_init": round(promo_opt, 2),
        "hist_oos_rps": round(float(res.fun), 5),
        "ordered_head": (c1, c2, beta),
        "struct_head": (c1_s, c2_s, beta_s),
        "hand_head": (c1_h, c2_h, beta_h),
        "noshrink_head": (c1_ns, c2_ns, beta_ns),
        "init_2627_tuned": init_2627,
        "init_2627_hand": init_hand,
        "init_2627_noshrink": init_noshrink,
        "alpha_nb": round(alpha_nb, 5),
        "rho_hist": round(rho_hist, 5),
    }


def vectorized_score_hda(lam_h_vec: np.ndarray, lam_a_vec: np.ndarray, alpha_nb: float = 0.0, rho: float = -0.07, max_goals: int = 8) -> np.ndarray:
    """
    Vectorized score matrix conversion for M posterior draws -> (M, 3) array of [pH, pD, pA].
    Uses Negative Binomial if alpha_nb > 1e-4, else Poisson, with Dixon-Coles low-score rho.
    """
    k = np.arange(max_goals + 1)[None, :]  # (1, K)
    lh = np.clip(lam_h_vec[:, None], 0.05, 6.0)  # (M, 1)
    la = np.clip(lam_a_vec[:, None], 0.05, 6.0)

    if alpha_nb > 1e-4:
        r = 1.0 / alpha_nb
        ph = nbinom.pmf(k, r, r / (r + lh))
        pa = nbinom.pmf(k, r, r / (r + la))
    else:
        ph = poisson.pmf(k, lh)
        pa = poisson.pmf(k, la)

    # Outer product (M, K, K)
    mats = ph[:, :, None] * pa[:, None, :]
    lh1 = lh[:, 0]
    la1 = la[:, 0]
    mats[:, 0, 0] *= np.clip(1.0 - lh1 * la1 * rho, 1e-4, 5.0)
    mats[:, 0, 1] *= np.clip(1.0 + lh1 * rho, 1e-4, 5.0)
    mats[:, 1, 0] *= np.clip(1.0 + la1 * rho, 1e-4, 5.0)
    mats[:, 1, 1] *= np.clip(1.0 - rho, 1e-4, 5.0)
    mats = np.clip(mats, 0.0, None)
    mats /= mats.sum(axis=(1, 2), keepdims=True)

    tril_mask = np.tril(np.ones((max_goals + 1, max_goals + 1), dtype=bool), k=-1)
    triu_mask = np.triu(np.ones((max_goals + 1, max_goals + 1), dtype=bool), k=1)
    diag_idx = np.arange(max_goals + 1)

    p_h = mats[:, tril_mask].sum(axis=1)
    p_d = mats[:, diag_idx, diag_idx].sum(axis=1)
    p_a = mats[:, triu_mask].sum(axis=1)
    return np.column_stack([p_h, p_d, p_a])


def fit_bayesian_statespace(
    df_2627_past: pd.DataFrame,
    teams: list[str],
    elo_priors: dict[str, float] | None,
    promoted_teams: set[str],
    ridge_alpha: float = 18.0,
    hfa_prior: float = 0.22,
) -> dict:
    """
    Exact Bayesian Ridge Posterior over team attack/defense/HFA parameters theta ~ N(theta_hat, Sigma).
    When elo_priors is None (NoHistory ablation), prior = 0 for all teams (flat prior).
    """
    t2i = {t: i for i, t in enumerate(teams)}
    n_t = len(teams)
    n_cols = 2 * n_t + 1

    prior = np.zeros(n_cols, dtype=float)
    if elo_priors is not None:
        for t, idx in t2i.items():
            elo_z = (elo_priors.get(t, 1500.0) - 1500.0) / 400.0
            prior[idx] = 0.45 * elo_z + (-0.10 if t in promoted_teams else 0.0)
            prior[n_t + idx] = -0.45 * elo_z + (0.10 if t in promoted_teams else 0.0)
    prior[2 * n_t] = hfa_prior

    # Empirical residual variance of log-goal-equivalent denoised performance
    prior_sigma2 = 0.16

    if len(df_2627_past) == 0:
        A_inv = (1.0 / ridge_alpha) * np.eye(n_cols)
        return {
            "t2i": t2i,
            "n_t": n_t,
            "theta": prior,
            "sigma": prior_sigma2 * A_inv,
            "mean_rate": 1.35,
        }

    rows_X, rows_y, rows_w = [], [], []
    max_date = pd.to_datetime(df_2627_past["date"]).max()
    for _, r in df_2627_past.iterrows():
        hi, ai = t2i[r["home_team"]], t2i[r["away_team"]]
        dt_days = max(0.0, float((max_date - pd.to_datetime(r["date"])).days))
        wt = math.exp(-math.log(2.0) * dt_days / 90.0)

        perf_h = 0.35 * float(r["home_goals"]) + 0.35 * (float(r["home_box_shots"]) * 0.14) + 0.30 * (float(r["home_xt"]) * 0.90)
        perf_a = 0.35 * float(r["away_goals"]) + 0.35 * (float(r["away_box_shots"]) * 0.14) + 0.30 * (float(r["away_xt"]) * 0.90)

        xh = np.zeros(n_cols)
        xh[hi] = 1.0
        xh[n_t + ai] = 1.0
        xh[2 * n_t] = 1.0
        rows_X.append(xh)
        rows_y.append(math.log(max(0.25, perf_h) / 1.35))
        rows_w.append(wt)

        xa = np.zeros(n_cols)
        xa[ai] = 1.0
        xa[n_t + hi] = 1.0
        rows_X.append(xa)
        rows_y.append(math.log(max(0.25, perf_a) / 1.35))
        rows_w.append(wt)

    X = np.vstack(rows_X)
    y = np.array(rows_y)
    W = np.diag(rows_w)
    A = X.T @ W @ X + ridge_alpha * np.eye(n_cols)
    A_inv = np.linalg.inv(A)
    theta = prior + A_inv @ (X.T @ W @ (y - X @ prior))

    # Estimate residual noise scale shrunk toward prior_sigma2
    resid = y - X @ theta
    w_sum = float(np.sum(rows_w))
    eff_df = max(1.0, w_sum - float(np.trace(X.T @ W @ X @ A_inv)))
    emp_sigma2 = float(np.sum(np.array(rows_w) * (resid ** 2)) / eff_df)
    sigma2 = 0.5 * emp_sigma2 + 0.5 * prior_sigma2

    # Soft center att and def means in theta
    theta[:n_t] -= np.mean(theta[:n_t])
    theta[n_t : 2 * n_t] -= np.mean(theta[n_t : 2 * n_t])

    return {
        "t2i": t2i,
        "n_t": n_t,
        "theta": theta,
        "sigma": sigma2 * A_inv,
        "mean_rate": 1.35,
    }


def predict_bayesian_posterior(
    model: dict,
    home_team: str,
    away_team: str,
    alpha_nb: float = 0.0,
    rho: float = -0.07,
    n_draws: int = 300,
    seed: int = 42,
) -> dict:
    """
    Propagates full 2D epistemic posterior covariance Sigma into (pH, pD, pA) via Monte Carlo,
    returning marginalized predictive mean_probs AND natural confidence std_probs.
    """
    t2i, n_t = model["t2i"], model["n_t"]
    hi, ai = t2i[home_team], t2i[away_team]

    J = np.zeros((2, 2 * n_t + 1), dtype=float)
    J[0, hi] = 1.0
    J[0, n_t + ai] = 1.0
    J[0, 2 * n_t] = 1.0
    J[1, ai] = 1.0
    J[1, n_t + hi] = 1.0

    mu_2d = J @ model["theta"]
    cov_2d = J @ model["sigma"] @ J.T

    rng = np.random.default_rng(seed)
    draws = rng.multivariate_normal(mu_2d, cov_2d, size=n_draws)
    lam_h = model["mean_rate"] * np.exp(np.clip(draws[:, 0], -1.5, 1.5))
    lam_a = model["mean_rate"] * np.exp(np.clip(draws[:, 1], -1.5, 1.5))

    hda_draws = vectorized_score_hda(lam_h, lam_a, alpha_nb=alpha_nb, rho=rho)
    mean_probs = hda_draws.mean(axis=0)
    std_probs = hda_draws.std(axis=0)
    return {
        "mean_probs": mean_probs / mean_probs.sum(),
        "std_probs": std_probs,
        "epistemic_var": float(np.mean(std_probs)),
    }
