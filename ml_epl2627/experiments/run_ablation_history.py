"""
ml_epl2627/experiments/run_ablation_history.py
Controlled Experiment comparing:
  1. Distributional / Bayesian State-Space models (with epistemic posterior sampling & NegBin vs Poisson)
  2. Historically-tuned Elo vs Cold-start Elo vs Naive No-Shrink Elo
  3. Full Season (N=107, Matches #1-#107) AND Post-Warmup (N=77, Matches #31-#107)
  4. Epistemic Uncertainty Calibration (High-Uncertainty vs Low-Uncertainty matches)
"""

import json
import math
import sys
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression

ML_DIR = Path(__file__).resolve().parent.parent
sys.path.append(str(ML_DIR))

from src.metrics import evaluate_all, bootstrap_rps_diff, clip_probs
from src.bayesian_models import (
    tune_elo_on_history,
    ordered_logit_probs,
    fit_bayesian_statespace,
    predict_bayesian_posterior,
)

DATA_DIR = ML_DIR / "data"
RES_DIR = ML_DIR / "results"


def update_elo_single(ratings: dict, ht: str, at: str, hg: int, ag: int, k: float, hfa: float):
    diff = (ratings[ht] + hfa) - ratings[at]
    exp_h = 1.0 / (1.0 + 10.0 ** (-diff / 400.0))
    actual_h = 1.0 if hg > ag else (0.5 if hg == ag else 0.0)
    mov = math.log1p(abs(hg - ag)) * (2.2 / (abs(diff) * 0.001 + 2.2))
    delta = k * max(1.0, mov) * (actual_h - exp_h)
    ratings[ht] += delta
    ratings[at] -= delta


def summarize_window(y_true: np.ndarray, preds_dict: dict, base_key: str) -> pd.DataFrame:
    b_probs = clip_probs(np.array(preds_dict[base_key]))
    rows = []
    for name, plist in preds_dict.items():
        parr = clip_probs(np.array(plist))
        assert np.allclose(parr.sum(axis=1), 1.0, atol=1e-4)
        res = evaluate_all(y_true, parr)
        boot = bootstrap_rps_diff(y_true, parr, b_probs, n_boot=2000, seed=42)
        res.update(
            {
                "model": name,
                "dRPS_vs_NoHist": round(boot["mean_diff"], 4),
                "CI_vs_NoHist": f"[{boot['ci_low']:+.4f}, {boot['ci_high']:+.4f}]",
                "p_beat_NoHist": round(boot["p_better"], 3),
            }
        )
        rows.append(res)
    return pd.DataFrame(rows)[
        ["model", "n", "rps", "log_loss", "brier", "accuracy", "ece", "dRPS_vs_NoHist", "CI_vs_NoHist", "p_beat_NoHist"]
    ].sort_values("rps")


def run():
    df_2627 = pd.read_csv(DATA_DIR / "matches_2627.csv")
    df_hist = pd.read_csv(DATA_DIR / "history_2126.csv")

    teams_2627 = sorted(set(df_2627["home_team"]).union(df_2627["away_team"]))
    teams_2526 = set(df_hist[df_hist["season"] == "2025/26"]["home_team"]).union(
        df_hist[df_hist["season"] == "2025/26"]["away_team"]
    )
    promoted = set(teams_2627) - teams_2526

    print("[*] Tuning Elo + Ordered Logit + NegBin dispersion strictly on 2021/22-2025/26...")
    tuned = tune_elo_on_history(df_hist, teams_2627)

    # Save learned historical parameters (without large dicts)
    params_out = {k: v for k, v in tuned.items() if not k.startswith("init_2627")}
    (RES_DIR / "elo_tuned_params.json").write_text(json.dumps(params_out, indent=2), encoding="utf-8")
    print(
        f"[+] Tuned Historical Elo: K={tuned['k']}, HFA={tuned['hfa']}, shrink={tuned['shrink']}, "
        f"promo={tuned['promo_init']} | OOS 22-26 RPS={tuned['hist_oos_rps']} | alpha_nb={tuned['alpha_nb']}, rho={tuned['rho_hist']}"
    )

    # Initialize the 5 Elo states
    elo_A = {t: 1500.0 for t in teams_2627}
    elo_B = {t: 1500.0 for t in teams_2627}
    elo_C_hand = dict(tuned["init_2627_hand"])
    elo_C_tuned = dict(tuned["init_2627_tuned"])
    elo_D_noshrink = dict(tuned["init_2627_noshrink"])

    eloA_X, eloA_y = [], []
    hda_counts = np.array([1.0, 1.0, 1.0])

    model_names = [
        "Elo_A_NoHistory",
        "Elo_B_StructHistory",
        "Elo_C_FullHistory_Hand",
        "Elo_C_FullHistory_Tuned",
        "Elo_D_NaiveNoShrink",
        "BayesSS_A_NoHistory",
        "BayesSS_C_FullHist_Poisson",
        "BayesSS_C_FullHist_NegBin",
    ]
    preds = {k: [] for k in model_names}
    unc_records = []

    last_date = None
    ss_nohist = None
    ss_hist = None

    for idx, row in df_2627.iterrows():
        ht, at = row["home_team"], row["away_team"]
        m_date = row["date"]
        hg, ag = int(row["home_goals"]), int(row["away_goals"])
        tgt = int(row["target"])

        # 1. Elo_A_NoHistory
        d_A = (elo_A[ht] + 65.0) - elo_A[at]
        if len(eloA_y) >= 18 and len(set(eloA_y)) == 3:
            clf_a = LogisticRegression(C=0.5, max_iter=500).fit(np.array(eloA_X), np.array(eloA_y))
            preds["Elo_A_NoHistory"].append(clf_a.predict_proba(np.array([[d_A]]))[0])
        else:
            preds["Elo_A_NoHistory"].append(hda_counts / hda_counts.sum())

        # 2. Elo_B_StructHistory
        d_B = (elo_B[ht] + tuned["hfa"]) - elo_B[at]
        c1s, c2s, bs = tuned["struct_head"]
        preds["Elo_B_StructHistory"].append(ordered_logit_probs(np.array([d_B]), c1s, c2s, bs)[0])

        # 3. Elo_C_FullHistory_Hand
        d_Ch = (elo_C_hand[ht] + 65.0) - elo_C_hand[at]
        c1h, c2h, bh = tuned["hand_head"]
        preds["Elo_C_FullHistory_Hand"].append(ordered_logit_probs(np.array([d_Ch]), c1h, c2h, bh)[0])

        # 4. Elo_C_FullHistory_Tuned
        d_Ct = (elo_C_tuned[ht] + tuned["hfa"]) - elo_C_tuned[at]
        c1t, c2t, bt = tuned["ordered_head"]
        preds["Elo_C_FullHistory_Tuned"].append(ordered_logit_probs(np.array([d_Ct]), c1t, c2t, bt)[0])

        # 5. Elo_D_NaiveNoShrink
        d_D = (elo_D_noshrink[ht] + 65.0) - elo_D_noshrink[at]
        c1ns, c2ns, bns = tuned["noshrink_head"]
        preds["Elo_D_NaiveNoShrink"].append(ordered_logit_probs(np.array([d_D]), c1ns, c2ns, bns)[0])

        # Refit Bayesian State-Space strictly on date < m_date
        if last_date != m_date:
            past_2627 = df_2627[df_2627["date"] < m_date]
            ss_nohist = fit_bayesian_statespace(past_2627, teams_2627, None, promoted, ridge_alpha=18.0, hfa_prior=0.22)
            ss_hist = fit_bayesian_statespace(past_2627, teams_2627, tuned["init_2627_tuned"], promoted, ridge_alpha=18.0, hfa_prior=0.22)
            last_date = m_date

        # 6. BayesSS_A_NoHistory (Poisson)
        res_nh = predict_bayesian_posterior(ss_nohist, ht, at, alpha_nb=0.0, rho=tuned["rho_hist"], seed=100 + idx)
        preds["BayesSS_A_NoHistory"].append(res_nh["mean_probs"])

        # 7. BayesSS_C_FullHist_Poisson
        res_hp = predict_bayesian_posterior(ss_hist, ht, at, alpha_nb=0.0, rho=tuned["rho_hist"], seed=200 + idx)
        preds["BayesSS_C_FullHist_Poisson"].append(res_hp["mean_probs"])

        # 8. BayesSS_C_FullHist_NegBin
        res_hnb = predict_bayesian_posterior(ss_hist, ht, at, alpha_nb=tuned["alpha_nb"], rho=tuned["rho_hist"], seed=300 + idx)
        preds["BayesSS_C_FullHist_NegBin"].append(res_hnb["mean_probs"])

        unc_records.append(
            {
                "idx": idx,
                "date": m_date,
                "home_team": ht,
                "away_team": at,
                "home_goals": hg,
                "away_goals": ag,
                "result": row["result"],
                "target": tgt,
                "pH": round(float(res_hnb["mean_probs"][0]), 4),
                "pD": round(float(res_hnb["mean_probs"][1]), 4),
                "pA": round(float(res_hnb["mean_probs"][2]), 4),
                "std_pH": round(float(res_hnb["std_probs"][0]), 4),
                "std_pD": round(float(res_hnb["std_probs"][1]), 4),
                "std_pA": round(float(res_hnb["std_probs"][2]), 4),
                "epistemic_var": round(res_hnb["epistemic_var"], 5),
            }
        )

        # Strictly post-match updates
        eloA_X.append([d_A])
        eloA_y.append(tgt)
        hda_counts[tgt] += 1.0
        update_elo_single(elo_A, ht, at, hg, ag, 20.0, 65.0)
        update_elo_single(elo_B, ht, at, hg, ag, tuned["k"], tuned["hfa"])
        update_elo_single(elo_C_hand, ht, at, hg, ag, 20.0, 65.0)
        update_elo_single(elo_C_tuned, ht, at, hg, ag, tuned["k"], tuned["hfa"])
        update_elo_single(elo_D_noshrink, ht, at, hg, ag, 20.0, 65.0)

    y_all = df_2627["target"].to_numpy(dtype=int)

    # Window 1: Full Season (N = 107)
    df_full = summarize_window(y_all, preds, base_key="Elo_A_NoHistory")
    df_full["window"] = "Full_N107"

    # Window 2: Early Cold-Start Only (Matches #1-#30, N = 30)
    preds_early = {k: v[:30] for k, v in preds.items()}
    df_early = summarize_window(y_all[:30], preds_early, base_key="Elo_A_NoHistory")
    df_early["window"] = "Early_MD1_3_N30"

    # Window 3: Post-Warmup (Matches #31-#107, N = 77)
    preds_post = {k: v[30:] for k, v in preds.items()}
    df_post = summarize_window(y_all[30:], preds_post, base_key="Elo_A_NoHistory")
    df_post["window"] = "PostWarmup_N77"

    df_all_summary = pd.concat([df_full, df_early, df_post], ignore_index=True)
    df_all_summary.to_csv(RES_DIR / "ablation_history_summary.csv", index=False)

    df_unc = pd.DataFrame(unc_records)
    df_unc.to_csv(RES_DIR / "bayesian_uncertainty_2627.csv", index=False)

    # Epistemic Uncertainty Split Evaluation (Low vs High Epistemic Std)
    med_unc = df_unc["epistemic_var"].median()
    low_m = df_unc["epistemic_var"] <= med_unc
    high_m = df_unc["epistemic_var"] > med_unc
    p_hnb_arr = df_unc[["pH", "pD", "pA"]].to_numpy()
    m_low = evaluate_all(y_all[low_m], p_hnb_arr[low_m])
    m_high = evaluate_all(y_all[high_m], p_hnb_arr[high_m])

    print("\n=== WINDOW 1: FULL SEASON (N = 107, Matches #1-#107) ===")
    print(df_full.drop(columns=["window"]).to_string(index=False))

    print("\n=== WINDOW 2: EARLY SEASON COLD-START (N = 30, Matches #1-#30) ===")
    print(df_early.drop(columns=["window"]).to_string(index=False))

    print("\n=== WINDOW 3: POST-WARMUP (N = 77, Matches #31-#107) ===")
    print(df_post.drop(columns=["window"]).to_string(index=False))

    print("\n=== EPISTEMIC UNCERTAINTY CHECK (BayesSS_C_FullHist_NegBin, N=107) ===")
    print(f"  Low-Uncertainty  (n={m_low['n']}, mean std={df_unc.loc[low_m,'epistemic_var'].mean():.4f}): RPS={m_low['rps']:.4f}, LogLoss={m_low['log_loss']:.4f}, Acc={m_low['accuracy']:.3f}")
    print(f"  High-Uncertainty (n={m_high['n']}, mean std={df_unc.loc[high_m,'epistemic_var'].mean():.4f}): RPS={m_high['rps']:.4f}, LogLoss={m_high['log_loss']:.4f}, Acc={m_high['accuracy']:.3f}")

    # Print 4 sample matches showing natural confidence intervals (2 early, 2 late)
    print("\n=== SAMPLE PREDICTIONS WITH NATURAL CONFIDENCE (p ± epistemic std) ===")
    for s_idx in [0, 7, 68, 106]:
        r = df_unc.iloc[s_idx]
        print(
            f"  #{s_idx+1:3d} {r['date']} {r['home_team']} vs {r['away_team']} ({r['home_goals']}-{r['away_goals']}): "
            f"H={r['pH']*100:4.1f}%±{r['std_pH']*100:3.1f}% | "
            f"D={r['pD']*100:4.1f}%±{r['std_pD']*100:3.1f}% | "
            f"A={r['pA']*100:4.1f}%±{r['std_pA']*100:3.1f}% (avg_std={r['epistemic_var']*100:.2f}%)"
        )


if __name__ == "__main__":
    run()
