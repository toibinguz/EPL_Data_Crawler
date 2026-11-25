"""
ml_epl2627/experiments/run_greedy_retrain.py
Practical ML Experiment:
  Part A: Elo Update Formula Comparison (WDL vs Log-MOV vs Expected-Performance xG/xT Margin)
          with conservative params (K=20, HFA=65, shrink=0.35, promo=1420), zero 26/27 leakage.
  Part B: Greedy Rolling Retraining of Core Bayesian Event State-Space Model
          comparing Fixed vs Greedy_Every10 vs Greedy_Every5 vs Adaptive Blend (Elo + BayesSS).
  Part C: Post-MD11 Team Ratings Snapshot & Sample Matchup Predictions with Epistemic Confidence.
"""

import math
import sys
import time
from pathlib import Path
import numpy as np
import pandas as pd

ML_DIR = Path(__file__).resolve().parent.parent
sys.path.append(str(ML_DIR))

from src.metrics import evaluate_all, bootstrap_rps_diff, clip_probs, rps_per_match
from src.bayesian_models import (
    fit_ordered_logit_head,
    ordered_logit_probs,
    vectorized_score_hda,
    predict_bayesian_posterior,
)

DATA_DIR = ML_DIR / "data"
RES_DIR = ML_DIR / "results"


def run_elo_formula_history(
    df_hist: pd.DataFrame,
    teams_2627: list[str],
    formula: str,
    k: float = 20.0,
    hfa: float = 65.0,
    shrink: float = 0.35,
    promo_init: float = 1420.0,
) -> tuple[dict[str, float], tuple[float, float, float]]:
    """
    Runs Elo sequentially on 2021/22-2025/26 under one of 3 update formulas:
      - 'wdl': pure Win/Draw/Loss (1.0, 0.5, 0.0), mov=1.0
      - 'mov': Win/Draw/Loss + log Margin-of-Victory with autocorrelation damping
      - 'xg':  continuous expected-score target from 0.5*goals + 0.5*xg margin
    Returns initial 26/27 Elo dict + calibrated Ordered Logit head from 2022/23-2025/26.
    """
    assert df_hist["season"].max() == "2025/26"
    ratings = {}
    prev_sec = None
    diffs, tgts, is_eval = [], [], []

    for _, r in df_hist.iterrows():
        sec = r["season"]
        if prev_sec is not None and sec != prev_sec:
            for t in list(ratings.keys()):
                ratings[t] = 1500.0 + (1.0 - shrink) * (ratings[t] - 1500.0)
        prev_sec = sec

        ht, at = r["home_team"], r["away_team"]
        hr = ratings.get(ht, promo_init if sec != "2021/22" else 1500.0)
        ar = ratings.get(at, promo_init if sec != "2021/22" else 1500.0)
        diff = (hr + hfa) - ar
        diffs.append(diff)
        tgts.append(int(r["target"]))
        is_eval.append(sec != "2021/22")

        exp_h = 1.0 / (1.0 + 10.0 ** (-diff / 400.0))
        hg, ag = float(r["home_goals"]), float(r["away_goals"])
        if formula == "wdl":
            actual_h = 1.0 if hg > ag else (0.5 if hg == ag else 0.0)
            delta = k * (actual_h - exp_h)
        elif formula == "mov":
            actual_h = 1.0 if hg > ag else (0.5 if hg == ag else 0.0)
            mov = math.log1p(abs(hg - ag)) * (2.2 / (abs(diff) * 0.001 + 2.2))
            delta = k * max(1.0, mov) * (actual_h - exp_h)
        elif formula == "xg":
            h_xg = float(r["home_xg"]) if pd.notna(r.get("home_xg")) else hg
            a_xg = float(r["away_xg"]) if pd.notna(r.get("away_xg")) else ag
            perf_diff = (0.5 * hg + 0.5 * h_xg) - (0.5 * ag + 0.5 * a_xg)
            actual_h = 1.0 / (1.0 + math.exp(-1.1 * np.clip(perf_diff, -4.0, 4.0)))
            delta = k * 1.15 * (actual_h - exp_h)
        else:
            raise ValueError(f"Unknown Elo formula: {formula}")

        ratings[ht] = hr + delta
        ratings[at] = ar - delta

    teams_2526 = set(df_hist[df_hist["season"] == "2025/26"]["home_team"]).union(
        df_hist[df_hist["season"] == "2025/26"]["away_team"]
    )
    init_2627 = {
        t: (1500.0 + (1.0 - shrink) * (ratings[t] - 1500.0) if t in teams_2526 else promo_init)
        for t in teams_2627
    }
    mask = np.array(is_eval, dtype=bool)
    head = fit_ordered_logit_head(np.array(diffs)[mask], np.array(tgts)[mask])
    return init_2627, head


def update_elo_2627(ratings: dict, row: pd.Series, formula: str, k: float = 20.0, hfa: float = 65.0):
    ht, at = row["home_team"], row["away_team"]
    hr, ar = ratings[ht], ratings[at]
    diff = (hr + hfa) - ar
    exp_h = 1.0 / (1.0 + 10.0 ** (-diff / 400.0))
    hg, ag = float(row["home_goals"]), float(row["away_goals"])
    if formula == "wdl":
        actual_h = 1.0 if hg > ag else (0.5 if hg == ag else 0.0)
        delta = k * (actual_h - exp_h)
    elif formula == "mov":
        actual_h = 1.0 if hg > ag else (0.5 if hg == ag else 0.0)
        mov = math.log1p(abs(hg - ag)) * (2.2 / (abs(diff) * 0.001 + 2.2))
        delta = k * max(1.0, mov) * (actual_h - exp_h)
    else:
        h_perf = 0.5 * hg + 0.5 * (0.65 * float(row["home_opta_npxg"]) + 0.35 * float(row["home_xt"]) * 0.90)
        a_perf = 0.5 * ag + 0.5 * (0.65 * float(row["away_opta_npxg"]) + 0.35 * float(row["away_xt"]) * 0.90)
        actual_h = 1.0 / (1.0 + math.exp(-1.1 * np.clip(h_perf - a_perf, -4.0, 4.0)))
        delta = k * 1.15 * (actual_h - exp_h)
    ratings[ht] = hr + delta
    ratings[at] = ar - delta


def fit_param_bayesian_ss(
    df_past: pd.DataFrame,
    teams: list[str],
    elo_priors: dict[str, float],
    promoted: set[str],
    ridge_alpha: float = 18.0,
    weights: tuple[float, float, float] = (0.35, 0.35, 0.30),
    prior_scale: float = 0.45,
    half_life: float = 90.0,
) -> dict:
    """Parameterized Bayesian Event State-Space fit for greedy hyperparameter rolling tuning."""
    t2i = {t: i for i, t in enumerate(teams)}
    n_t = len(teams)
    n_cols = 2 * n_t + 1
    w_g, w_box, w_xt = weights

    prior = np.zeros(n_cols, dtype=float)
    for t, idx in t2i.items():
        elo_z = (elo_priors.get(t, 1500.0) - 1500.0) / 400.0
        prior[idx] = prior_scale * elo_z + (-0.10 if t in promoted else 0.0)
        prior[n_t + idx] = -prior_scale * elo_z + (0.10 if t in promoted else 0.0)
    prior[2 * n_t] = 0.22

    prior_sigma2 = 0.16
    if len(df_past) == 0:
        return {
            "t2i": t2i,
            "n_t": n_t,
            "theta": prior,
            "sigma": prior_sigma2 * (1.0 / ridge_alpha) * np.eye(n_cols),
            "mean_rate": 1.35,
        }

    rows_X, rows_y, rows_w = [], [], []
    max_date = pd.to_datetime(df_past["date"]).max()
    for _, r in df_past.iterrows():
        hi, ai = t2i[r["home_team"]], t2i[r["away_team"]]
        dt_days = max(0.0, float((max_date - pd.to_datetime(r["date"])).days))
        wt = math.exp(-math.log(2.0) * dt_days / half_life)

        perf_h = w_g * float(r["home_goals"]) + w_box * float(r["home_opta_npxg"]) + w_xt * (float(r["home_xt"]) * 0.90)
        perf_a = w_g * float(r["away_goals"]) + w_box * float(r["away_opta_npxg"]) + w_xt * (float(r["away_xt"]) * 0.90)

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

    resid = y - X @ theta
    eff_df = max(1.0, float(np.sum(rows_w)) - float(np.trace(X.T @ W @ X @ A_inv)))
    emp_sigma2 = float(np.sum(np.array(rows_w) * (resid ** 2)) / eff_df)
    sigma2 = 0.5 * emp_sigma2 + 0.5 * prior_sigma2

    theta[:n_t] -= np.mean(theta[:n_t])
    theta[n_t : 2 * n_t] -= np.mean(theta[n_t : 2 * n_t])

    return {"t2i": t2i, "n_t": n_t, "theta": theta, "sigma": sigma2 * A_inv, "mean_rate": 1.35}


def fast_point_predict(model: dict, ht: str, at: str, rho: float = -0.04) -> np.ndarray:
    t2i, n_t = model["t2i"], model["n_t"]
    hi, ai = t2i[ht], t2i[at]
    lh = model["mean_rate"] * math.exp(np.clip(model["theta"][2 * n_t] + model["theta"][hi] + model["theta"][n_t + ai], -1.5, 1.5))
    la = model["mean_rate"] * math.exp(np.clip(model["theta"][ai] + model["theta"][n_t + hi], -1.5, 1.5))
    return vectorized_score_hda(np.array([lh]), np.array([la]), alpha_nb=0.0, rho=rho)[0]


def greedy_select_params(
    df_past: pd.DataFrame,
    teams: list[str],
    elo_priors: dict[str, float],
    promoted: set[str],
) -> dict:
    """
    Strictly evaluates candidate hyperparameters via internal Walk-Forward RPS on the
    last min(30, len(df_past)-10) past matches (NEVER touching future test data).
    """
    candidates_alpha = [10.0, 18.0, 30.0]
    candidates_weights = [
        (0.35, 0.35, 0.30),
        (0.20, 0.40, 0.40),
        (0.50, 0.25, 0.25),
    ]
    candidates_pscale = [0.35, 0.45, 0.60]

    n_past = len(df_past)
    val_start = max(10, n_past - 30)
    unique_dates = sorted(df_past["date"].iloc[val_start:].unique())

    best_score = 999.0
    best_cfg = {"ridge_alpha": 18.0, "weights": (0.35, 0.35, 0.30), "prior_scale": 0.45}

    for alpha in candidates_alpha:
        for w in candidates_weights:
            for ps in candidates_pscale:
                probs_val, y_val = [], []
                for d_val in unique_dates:
                    tr_sub = df_past[df_past["date"] < d_val]
                    va_sub = df_past[df_past["date"] == d_val]
                    m = fit_param_bayesian_ss(tr_sub, teams, elo_priors, promoted, alpha, w, ps)
                    for _, r in va_sub.iterrows():
                        probs_val.append(fast_point_predict(m, r["home_team"], r["away_team"]))
                        y_val.append(int(r["target"]))
                score = float(np.mean(rps_per_match(np.array(y_val), np.array(probs_val))))
                if score < best_score:
                    best_score = score
                    best_cfg = {"ridge_alpha": alpha, "weights": w, "prior_scale": ps, "inner_rps": round(score, 5)}
    return best_cfg


def summarize_results(y_true: np.ndarray, preds: dict, base_key: str = "Elo_Formula_MOV") -> pd.DataFrame:
    b_probs = clip_probs(np.array(preds[base_key]))
    rows = []
    for name, plist in preds.items():
        parr = clip_probs(np.array(plist))
        assert np.allclose(parr.sum(axis=1), 1.0, atol=1e-4)
        res = evaluate_all(y_true, parr)
        boot = bootstrap_rps_diff(y_true, parr, b_probs, n_boot=2000, seed=42)
        res.update(
            {
                "model": name,
                "dRPS_vs_MOV": round(boot["mean_diff"], 4),
                "CI_vs_MOV": f"[{boot['ci_low']:+.4f}, {boot['ci_high']:+.4f}]",
                "p_beat_MOV": round(boot["p_better"], 3),
            }
        )
        rows.append(res)
    return pd.DataFrame(rows)[
        ["model", "n", "rps", "log_loss", "brier", "accuracy", "ece", "dRPS_vs_MOV", "CI_vs_MOV", "p_beat_MOV"]
    ].sort_values("rps")


def run():
    t0 = time.time()
    df_2627 = pd.read_csv(DATA_DIR / "matches_2627.csv")
    df_hist = pd.read_csv(DATA_DIR / "history_2126.csv")

    teams_2627 = sorted(set(df_2627["home_team"]).union(df_2627["away_team"]))
    teams_2526 = set(df_hist[df_hist["season"] == "2025/26"]["home_team"]).union(
        df_hist[df_hist["season"] == "2025/26"]["away_team"]
    )
    promoted = set(teams_2627) - teams_2526

    # Part A: Fit 3 Elo formulas on 2021/22-2025/26
    elo_init_wdl, head_wdl = run_elo_formula_history(df_hist, teams_2627, "wdl")
    elo_init_mov, head_mov = run_elo_formula_history(df_hist, teams_2627, "mov")
    elo_init_xg, head_xg = run_elo_formula_history(df_hist, teams_2627, "xg")

    elo_wdl = dict(elo_init_wdl)
    elo_mov = dict(elo_init_mov)
    elo_xg = dict(elo_init_xg)

    # Initial static hyperparameters
    cfg_10 = {"ridge_alpha": 18.0, "weights": (0.35, 0.35, 0.30), "prior_scale": 0.45, "inner_rps": None}
    cfg_5 = {"ridge_alpha": 18.0, "weights": (0.35, 0.35, 0.30), "prior_scale": 0.45, "inner_rps": None}
    cfg_5_xg = {"ridge_alpha": 18.0, "weights": (0.35, 0.35, 0.30), "prior_scale": 0.45, "inner_rps": None}
    param_traj = []

    model_names = [
        "Elo_Formula_WDL",
        "Elo_Formula_MOV",
        "Elo_Formula_xG",
        "BayesSS_Fixed",
        "BayesSS_Greedy_Every10",
        "BayesSS_Greedy_Every5",
        "BayesSS_Greedy5_xGPrior",
        "BayesSS_Blend_Elo",
        "Champion_Unified",
    ]
    preds = {k: [] for k in model_names}

    last_date = None
    m_fixed = None
    m_g10 = None
    m_g5 = None
    m_g5_xg = None

    for idx, row in df_2627.iterrows():
        ht, at = row["home_team"], row["away_team"]
        m_date = row["date"]

        # Greedy rolling hyperparameter search strictly on past 26/27 matches (< m_date)
        past_2627 = df_2627[df_2627["date"] < m_date]
        n_past = len(past_2627)

        if idx >= 20 and idx % 10 == 0 and n_past >= 20:
            cfg_10 = greedy_select_params(past_2627, teams_2627, elo_init_mov, promoted)
            param_traj.append({"strategy": "Greedy_Every10", "match_idx": idx, "date": m_date, **cfg_10})
            last_date = None  # Force refit with updated config

        if idx >= 20 and idx % 5 == 0 and n_past >= 20:
            cfg_5 = greedy_select_params(past_2627, teams_2627, elo_init_mov, promoted)
            cfg_5_xg = greedy_select_params(past_2627, teams_2627, elo_init_xg, promoted)
            param_traj.append({"strategy": "Greedy_Every5", "match_idx": idx, "date": m_date, **cfg_5})
            param_traj.append({"strategy": "Greedy5_xGPrior", "match_idx": idx, "date": m_date, **cfg_5_xg})
            last_date = None

        if last_date != m_date:
            m_fixed = fit_param_bayesian_ss(past_2627, teams_2627, elo_init_mov, promoted, 18.0, (0.35, 0.35, 0.30), 0.45)
            m_g10 = fit_param_bayesian_ss(
                past_2627, teams_2627, elo_init_mov, promoted, cfg_10["ridge_alpha"], cfg_10["weights"], cfg_10["prior_scale"]
            )
            m_g5 = fit_param_bayesian_ss(
                past_2627, teams_2627, elo_init_mov, promoted, cfg_5["ridge_alpha"], cfg_5["weights"], cfg_5["prior_scale"]
            )
            m_g5_xg = fit_param_bayesian_ss(
                past_2627, teams_2627, elo_init_xg, promoted, cfg_5_xg["ridge_alpha"], cfg_5_xg["weights"], cfg_5_xg["prior_scale"]
            )
            last_date = m_date

        # Elo formula predictions
        p_wdl = ordered_logit_probs(np.array([(elo_wdl[ht] + 65.0) - elo_wdl[at]]), *head_wdl)[0]
        p_mov = ordered_logit_probs(np.array([(elo_mov[ht] + 65.0) - elo_mov[at]]), *head_mov)[0]
        p_xg = ordered_logit_probs(np.array([(elo_xg[ht] + 65.0) - elo_xg[at]]), *head_xg)[0]
        preds["Elo_Formula_WDL"].append(p_wdl)
        preds["Elo_Formula_MOV"].append(p_mov)
        preds["Elo_Formula_xG"].append(p_xg)

        # Bayesian State-Space predictions (Monte Carlo posterior integration)
        p_fix = predict_bayesian_posterior(m_fixed, ht, at, alpha_nb=0.0, rho=-0.04, seed=100 + idx)["mean_probs"]
        p_g10 = predict_bayesian_posterior(m_g10, ht, at, alpha_nb=0.0, rho=-0.04, seed=200 + idx)["mean_probs"]
        p_g5 = predict_bayesian_posterior(m_g5, ht, at, alpha_nb=0.0, rho=-0.04, seed=300 + idx)["mean_probs"]
        p_g5_xg = predict_bayesian_posterior(m_g5_xg, ht, at, alpha_nb=0.0, rho=-0.04, seed=400 + idx)["mean_probs"]
        preds["BayesSS_Fixed"].append(p_fix)
        preds["BayesSS_Greedy_Every10"].append(p_g10)
        preds["BayesSS_Greedy_Every5"].append(p_g5)
        preds["BayesSS_Greedy5_xGPrior"].append(p_g5_xg)

        # Smooth adaptive blend between Elo_MOV (dominant early) and BayesSS_Fixed (dominant as N grows)
        w_elo = math.exp(-n_past / 40.0)
        p_blend = w_elo * p_mov + (1.0 - w_elo) * p_fix
        preds["BayesSS_Blend_Elo"].append(p_blend / p_blend.sum())

        # Unified Champion: Elo_Formula_xG (early blend exp(-n_past/25)) + BayesSS_Greedy5_xGPrior
        w_elo_u = math.exp(-n_past / 25.0)
        p_champ = w_elo_u * p_xg + (1.0 - w_elo_u) * p_g5_xg
        preds["Champion_Unified"].append(p_champ / p_champ.sum())

        # Strictly post-match Elo updates
        update_elo_2627(elo_wdl, row, "wdl")
        update_elo_2627(elo_mov, row, "mov")
        update_elo_2627(elo_xg, row, "xg")

    y_all = df_2627["target"].to_numpy(dtype=int)
    odds_mask = df_2627["odds_prob_home"].notna().to_numpy()
    if odds_mask.all():
        preds["B5_MarketOdds"] = df_2627[["odds_prob_home", "odds_prob_draw", "odds_prob_away"]].to_numpy(dtype=float).tolist()

    df_full = summarize_results(y_all, preds, base_key="B5_MarketOdds" if "B5_MarketOdds" in preds else "Elo_Formula_MOV")
    df_full["window"] = f"Full_N{len(y_all)}"

    preds_post = {k: v[30:] for k, v in preds.items()}
    df_post = summarize_results(y_all[30:], preds_post, base_key="B5_MarketOdds" if "B5_MarketOdds" in preds else "Elo_Formula_MOV")
    df_post["window"] = f"PostWarmup_N{len(y_all) - 30}"

    df_summary = pd.concat([df_full, df_post], ignore_index=True)
    df_summary.to_csv(RES_DIR / "greedy_retrain_summary.csv", index=False)
    pd.DataFrame(param_traj).to_csv(RES_DIR / "greedy_param_trajectory.csv", index=False)

    # Part C: Fit Final Core Model (xG Elo Prior + greedy weights) on all matches -> Team Ratings Snapshot
    final_model = fit_param_bayesian_ss(
        df_2627, teams_2627, elo_init_xg, promoted, cfg_5_xg["ridge_alpha"], cfg_5_xg["weights"], cfg_5_xg["prior_scale"]
    )
    print(f"[*] Final Greedy xG-Prior Config after N={len(df_2627)}: {cfg_5_xg}")
    t2i, n_t = final_model["t2i"], final_model["n_t"]
    rating_rows = []
    for t in teams_2627:
        ti = t2i[t]
        att = float(final_model["theta"][ti])
        dfn = float(final_model["theta"][n_t + ti])
        rating_rows.append(
            {
                "team": t,
                "att": round(att, 4),
                "def": round(dfn, 4),
                "net": round(att - dfn, 4),
                "elo_mov_md11": round(elo_mov[t], 1),
                "elo_xg_md11": round(elo_xg[t], 1),
                "exp_gf_vs_avg_home": round(1.35 * math.exp(final_model["theta"][2 * n_t] + att), 2),
                "exp_ga_vs_avg_home": round(1.35 * math.exp(dfn), 2),
            }
        )
    df_ratings = pd.DataFrame(rating_rows).sort_values("net", ascending=False).reset_index(drop=True)
    df_ratings.to_csv(RES_DIR / "team_ratings_md11.csv", index=False)

    print(f"[+] Completed in {time.time() - t0:.2f}s (N={len(y_all)}, odds_overlap={int(odds_mask.sum())})")
    print(f"\n=== FULL SEASON BENCHMARK vs MARKET ODDS (N = {len(y_all)}, Matches #1-#{len(y_all)}) ===")
    print(df_full.drop(columns=["window"]).to_string(index=False))

    print(f"\n=== POST-WARMUP BENCHMARK vs MARKET ODDS (N = {len(y_all)-30}, Matches #31-#{len(y_all)}) ===")
    print(df_post.drop(columns=["window"]).to_string(index=False))

    print("\n=== TOP-6 & BOTTOM-4 TEAM RATINGS (CORE BAYES STATE-SPACE POST-MD11) ===")
    print(pd.concat([df_ratings.head(6), df_ratings.tail(4)]).to_string(index=False))

    print("\n=== EXAMPLE UPCOMING-STYLE PREDICTIONS (NATURAL EPISTEMIC CONFIDENCE) ===")
    for h_ex, a_ex in [("ARS", "MCI"), ("LIV", "MUN"), ("CHE", "TOT")]:
        pr = predict_bayesian_posterior(final_model, h_ex, a_ex, alpha_nb=0.0, rho=-0.04, seed=99)
        mp, sp = pr["mean_probs"], pr["std_probs"]
        lh = 1.35 * math.exp(final_model["theta"][2 * n_t] + final_model["theta"][t2i[h_ex]] + final_model["theta"][n_t + t2i[a_ex]])
        la = 1.35 * math.exp(final_model["theta"][t2i[a_ex]] + final_model["theta"][n_t + t2i[h_ex]])
        print(
            f"  {h_ex} vs {a_ex} | xG=({lh:.2f}, {la:.2f}) | "
            f"H={mp[0]*100:4.1f}%±{sp[0]*100:3.1f}% | "
            f"D={mp[1]*100:4.1f}%±{sp[1]*100:3.1f}% | "
            f"A={mp[2]*100:4.1f}%±{sp[2]*100:3.1f}%"
        )


if __name__ == "__main__":
    run()
