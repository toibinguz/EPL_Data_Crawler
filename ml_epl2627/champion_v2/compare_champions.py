"""
ml_epl2627/champion_v2/compare_champions.py
Strict 3-way validation & comparison between:
  - Champion_v1_Legacy (handcrafted Elo -> EloPrior log-clipped Ridge + exp(-n/25))
  - Champion_v2_New    (Stabilized xG-Elo [promo bounded 1400..1460] + Independent Multi-Season Laplace Poisson GLM)
  - Champion_v2_BayesOnly (Pure Multi-Season Laplace Poisson GLM, Zero Elo)
  - Champion_v2_EloOnly   (Stabilized xG-Elo only)
  - Market_Odds           (De-vigged bookmaker odds benchmark)

Evaluates across Validation 2025/26 (N=380), Locked Test 2026/27 (N=110), and Post-MD3 (N=80),
reporting RPS, LogLoss, Brier, Top-1 Accuracy (argmax), Mean P(true outcome) [p . y_onehot], ECE,
and Paired Bootstrap significance vs Champion_v1_Legacy and vs Market_Odds.
"""

import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import minimize

ML_DIR = Path(__file__).resolve().parent.parent
if str(ML_DIR) not in sys.path:
    sys.path.insert(0, str(ML_DIR))

from src.bayesian_models import (  # noqa: E402
    estimate_historical_negbin_and_rho,
    fit_ordered_logit_head,
    ordered_logit_probs,
)
from src.metrics import bootstrap_rps_diff, clip_probs, evaluate_all, rps_per_match  # noqa: E402
from src.strict_models import (  # noqa: E402
    evaluate_fixed_elo,
    fit_bayesian_poisson_glm_laplace,
    fit_bayesian_ridge_logclip,
    predict_elo_on_test,
    prepare_next_season_ratings,
    run_elo_walkforward,
    walkforward_bayesian_season,
)

HIST_CSV = ML_DIR / "data" / "history_1426.csv"
TEST_CSV = ML_DIR / "data" / "matches_2627.csv"
OUT_CSV = ML_DIR / "results" / "champion_v2_comparison.csv"
META_JSON = ML_DIR / "results" / "champion_v2_meta.json"


def tune_stabilized_elo(df_train_ex: pd.DataFrame, df_val: pd.DataFrame) -> dict:
    """
    Tune xG-Elo strictly on Train<=24/25 (ExCOVID) -> Validation 2025/26,
    with promo_init constrained cleanly inside [1400, 1460] so it cannot overfit 3 promoted clubs.
    """
    df_tv = pd.concat([df_train_ex, df_val], ignore_index=True)
    n_train = len(df_train_ex)
    warmup_season = sorted(df_train_ex["season"].unique())[0]
    fit_head_mask = (df_train_ex["season"] != warmup_season).to_numpy()
    y_train = df_train_ex["target"].to_numpy(dtype=int)[fit_head_mask]
    y_val = df_val["target"].to_numpy(dtype=int)

    best = {"val_rps": 999.0}
    for w_xg in (0.45,):
        for k_g in (24.0, 32.0, 40.0):
            for hfa_g in (80.0, 110.0, 130.0):
                for sh_g in (0.10, 0.20, 0.30):
                    for pr_g in (1410.0, 1435.0, 1455.0):
                        diffs_tv, _, _ = run_elo_walkforward(df_tv, k=k_g, hfa=hfa_g, shrink=sh_g, promo_init=pr_g, w_xg=w_xg)
                        c1, c2, beta = fit_ordered_logit_head(diffs_tv[:n_train][fit_head_mask], y_train)
                        vp = clip_probs(ordered_logit_probs(diffs_tv[n_train:], c1, c2, beta))
                        vr = float(np.mean(rps_per_match(y_val, vp)))
                        if vr < best["val_rps"]:
                            best = {"val_rps": vr, "k": k_g, "hfa": hfa_g, "shrink": sh_g, "promo_init": pr_g, "w_xg": w_xg}

    def obj(params: np.ndarray) -> float:
        k_v, hfa_v, sh_v, pr_v = params
        if not (10.0 <= k_v <= 55.0 and 20.0 <= hfa_v <= 140.0 and 0.05 <= sh_v <= 0.55 and 1400.0 <= pr_v <= 1460.0):
            return 1.0
        diffs_tv, _, _ = run_elo_walkforward(df_tv, k=float(k_v), hfa=float(hfa_v), shrink=float(sh_v), promo_init=float(pr_v), w_xg=0.45)
        c1, c2, beta = fit_ordered_logit_head(diffs_tv[:n_train][fit_head_mask], y_train)
        vp = clip_probs(ordered_logit_probs(diffs_tv[n_train:], c1, c2, beta))
        return float(np.mean(rps_per_match(y_val, vp)))

    res = minimize(obj, np.array([best["k"], best["hfa"], best["shrink"], best["promo_init"]]), method="Powell", options={"maxiter": 18, "ftol": 1e-4})
    if float(res.fun) < best["val_rps"]:
        k_f, hfa_f, sh_f, pr_f = [round(float(x), 3) for x in res.x]
    else:
        k_f, hfa_f, sh_f, pr_f = best["k"], best["hfa"], best["shrink"], best["promo_init"]

    out = evaluate_fixed_elo(df_train_ex, df_val, k=k_f, hfa=hfa_f, shrink=sh_f, promo_init=pr_f, w_xg=0.45)
    return out


def mean_prob_true(y_true: np.ndarray, probs: np.ndarray) -> float:
    """Computes mean(p . y_onehot), i.e., average probability assigned to the true outcome."""
    p = clip_probs(probs)
    return float(np.mean(p[np.arange(len(y_true)), y_true]))


def match_played_counts(df: pd.DataFrame) -> np.ndarray:
    counts: dict[str, int] = {}
    out = np.empty(len(df), dtype=float)
    for idx, r in enumerate(df.itertuples(index=False)):
        h, a = r.home_team, r.away_team
        out[idx] = float(min(counts.get(h, 0), counts.get(a, 0)))
        counts[h] = counts.get(h, 0) + 1
        counts[a] = counts.get(a, 0) + 1
    return out


def eval_window(name: str, window: str, y_true: np.ndarray, p_mod: np.ndarray, p_v1: np.ndarray, p_mkt: np.ndarray) -> dict:
    m = evaluate_all(y_true, p_mod)
    b_v1 = bootstrap_rps_diff(y_true, p_mod, p_v1)
    b_mk = bootstrap_rps_diff(y_true, p_mod, p_mkt)
    return {
        "window": window,
        "model": name,
        "n": m["n"],
        "rps": round(m["rps"], 5),
        "log_loss": round(m["log_loss"], 5),
        "brier": round(m["brier"], 5),
        "top1_acc": round(m["accuracy"], 4),
        "mean_p_true": round(mean_prob_true(y_true, p_mod), 4),
        "ece": round(m["ece"], 4),
        "diff_vs_v1": round(b_v1["mean_diff"], 5),
        "p_better_v1": round(b_v1["p_better"], 4),
        "diff_vs_mkt": round(b_mk["mean_diff"], 5),
        "p_better_mkt": round(b_mk["p_better"], 4),
    }


def main() -> None:
    t0 = time.time()
    df_hist = pd.read_csv(HIST_CSV)
    df_2627 = pd.read_csv(TEST_CSV)

    df_val = df_hist[df_hist["season"] == "2025/26"].reset_index(drop=True)
    df_train_all = df_hist[df_hist["season"] <= "2024/25"].reset_index(drop=True)
    df_train_ex = df_train_all[df_train_all["season"] != "2020/21"].reset_index(drop=True)
    df_train_r4 = df_train_all[df_train_all["season"] >= "2021/22"].reset_index(drop=True)

    y_val = df_val["target"].to_numpy(dtype=int)
    y_test = df_2627["target"].to_numpy(dtype=int)
    md3_mask = np.arange(len(df_2627)) >= 30
    y_md3 = y_test[md3_mask]

    alpha_val, rho_val = estimate_historical_negbin_and_rho(df_train_r4)
    alpha_test, rho_test = estimate_historical_negbin_and_rho(df_hist[df_hist["season"] >= "2021/22"])

    # 1. Tune Champion v2 Stabilized xG-Elo strictly on Val 2025/26
    elo_v2 = tune_stabilized_elo(df_train_ex, df_val)
    print(f"[+] Stabilized xG-Elo tuned on Val 25/26: K={elo_v2['k']}, HFA={elo_v2['hfa']}, shrink={elo_v2['shrink']}, promo={elo_v2['promo_init']}, wxg={elo_v2['w_xg']}")
    p_elo_v2_val = elo_v2["val_probs"]
    p_elo_v2_test, _ = predict_elo_on_test(df_2627, elo_v2)

    # 2. Pure Multi-Season Bayesian Poisson GLM (Recent4 2021/22+, zero Elo)
    glm_kw = {"prior_scale": 0.35, "half_life_days": 360.0, "w_goals": 0.35, "hfa_prior": 0.22, "hfa_prior_std": 0.14, "elo_priors": None, "promo_offset": 0.0}
    p_glm_val = walkforward_bayesian_season(df_train_r4, df_val, fit_bayesian_poisson_glm_laplace, glm_kw, True, alpha_val, rho_val, 400, 42)
    p_glm_test = walkforward_bayesian_season(df_hist[df_hist["season"] >= "2021/22"], df_2627, fit_bayesian_poisson_glm_laplace, glm_kw, True, alpha_test, rho_test, 400, 42)

    # 3. Validate fusion weight w_elo strictly on 2025/26
    best_w, best_rps = 0.0, 999.0
    for w_try in np.linspace(0.0, 1.0, 21):
        p_try = clip_probs(w_try * p_elo_v2_val + (1.0 - w_try) * p_glm_val)
        r = float(np.mean(rps_per_match(y_val, p_try)))
        if r < best_rps:
            best_rps, best_w = r, round(float(w_try), 2)
    print(f"[+] Validated Champion v2 Pool Weight strictly on Val 25/26: w_elo={best_w:.2f} (Val RPS={best_rps:.5f})")

    p_v2_val = clip_probs(best_w * p_elo_v2_val + (1.0 - best_w) * p_glm_val)
    p_v2_test = clip_probs(best_w * p_elo_v2_test + (1.0 - best_w) * p_glm_test)

    # 4. Champion_v1_Legacy (handcrafted Elo -> EloPrior log-clipped Ridge + exp(-n/25))
    elo_v1 = evaluate_fixed_elo(df_train_r4, df_val, k=20.0, hfa=65.0, shrink=0.35, promo_init=1420.0, w_xg=0.0)
    _, end_v1_tr, last_v1_tr = run_elo_walkforward(df_train_r4, k=20.0, hfa=65.0, shrink=0.35, promo_init=1420.0, w_xg=0.0)
    teams_val = sorted(set(df_val["home_team"]).union(df_val["away_team"]))
    promo_val = set(teams_val) - set(df_train_r4[df_train_r4["season"] == "2024/25"]["home_team"]).union(df_train_r4[df_train_r4["season"] == "2024/25"]["away_team"])
    priors_v1_val = prepare_next_season_ratings(end_v1_tr, last_v1_tr, teams_val, 0.35, 1420.0)
    p_v1_bayes_val = walkforward_bayesian_season(
        pd.DataFrame(), df_val, fit_bayesian_ridge_logclip,
        {"ridge_alpha": 18.0, "half_life_days": 90.0, "w_goals": 0.35, "hfa_prior": 0.22, "elo_priors": priors_v1_val, "promo_offset": 0.10},
        False, alpha_val, rho_val, 400, 42, promo_val,
    )
    n_val = match_played_counts(df_val)
    w_v1_val = np.exp(-n_val / 25.0)[:, None]
    p_v1_val = clip_probs(w_v1_val * elo_v1["val_probs"] + (1.0 - w_v1_val) * p_v1_bayes_val)

    teams_test = sorted(set(df_2627["home_team"]).union(df_2627["away_team"]))
    promo_test = set(teams_test) - set(df_val["home_team"]).union(df_val["away_team"])
    priors_v1_test = prepare_next_season_ratings(elo_v1["end_ratings_tv"], elo_v1["last_teams_tv"], teams_test, 0.35, 1420.0)
    p_v1_elo_test, _ = predict_elo_on_test(df_2627, elo_v1)
    p_v1_bayes_test = walkforward_bayesian_season(
        pd.DataFrame(), df_2627, fit_bayesian_ridge_logclip,
        {"ridge_alpha": 18.0, "half_life_days": 90.0, "w_goals": 0.35, "hfa_prior": 0.22, "elo_priors": priors_v1_test, "promo_offset": 0.10},
        False, alpha_test, rho_test, 400, 42, promo_test,
    )
    n_test = match_played_counts(df_2627)
    w_v1_test = np.exp(-n_test / 25.0)[:, None]
    p_v1_test = clip_probs(w_v1_test * p_v1_elo_test + (1.0 - w_v1_test) * p_v1_bayes_test)

    p_mkt_val = clip_probs(df_val[["odds_prob_home", "odds_prob_draw", "odds_prob_away"]].to_numpy(dtype=float))
    p_mkt_test = clip_probs(df_2627[["odds_prob_home", "odds_prob_draw", "odds_prob_away"]].to_numpy(dtype=float))

    rows = []
    for m_name, pv, pt in [
        ("Champion_v1_Legacy", p_v1_val, p_v1_test),
        ("Champion_v2_EloOnly", p_elo_v2_val, p_elo_v2_test),
        ("Champion_v2_BayesOnly", p_glm_val, p_glm_test),
        ("Champion_v2_New", p_v2_val, p_v2_test),
        ("Market_Odds", p_mkt_val, p_mkt_test),
    ]:
        rows.append(eval_window(m_name, "Val_2025_26", y_val, pv, p_v1_val, p_mkt_val))
        rows.append(eval_window(m_name, "Test_2026_27_All", y_test, pt, p_v1_test, p_mkt_test))
        rows.append(eval_window(m_name, "Test_2026_27_PostMD3", y_md3, pt[md3_mask], p_v1_test[md3_mask], p_mkt_test[md3_mask]))

    df_out = pd.DataFrame(rows)
    df_out.to_csv(OUT_CSV, index=False)
    META_JSON.write_text(
        json.dumps(
            {
                "k": elo_v2["k"],
                "hfa": elo_v2["hfa"],
                "shrink": elo_v2["shrink"],
                "promo_init": elo_v2["promo_init"],
                "w_xg": elo_v2["w_xg"],
                "w_elo": best_w,
                "glm_kw": glm_kw,
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    for win in ("Val_2025_26", "Test_2026_27_All", "Test_2026_27_PostMD3"):
        print(f"\n=== {win} ===")
        sub = df_out[df_out["window"] == win][["model", "n", "rps", "log_loss", "brier", "top1_acc", "mean_p_true", "ece", "diff_vs_v1", "p_better_v1", "diff_vs_mkt", "p_better_mkt"]]
        print(sub.to_string(index=False))

    print(f"\n[+] Saved {OUT_CSV} & {META_JSON} in {time.time() - t0:.1f}s")


if __name__ == "__main__":
    main()
