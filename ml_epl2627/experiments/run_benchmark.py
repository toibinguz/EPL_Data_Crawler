"""
ml_epl2627/experiments/run_benchmark.py
Executes strict Zero-Leakage Walk-Forward Validation across EPL 2026/27 (N=107, eval N=77 from match #31),
compares Benchmark Ladder (B0..B5 + XGB Overfit Diag), runs paired Bootstrap CI tests,
and evaluates post-MD11 team strength correlation against opta_power_ranking_261124.json.
"""

import json
import math
import sys
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.stats import spearmanr, pearsonr

ML_DIR = Path(__file__).resolve().parent.parent
sys.path.append(str(ML_DIR))

from src.metrics import evaluate_all, bootstrap_rps_diff, clip_probs
from src.features import SeasonTracker, build_xgb_feature_vector
from src.models import (
    fit_historical_elo_and_logit,
    fit_dixon_coles,
    predict_dixon_coles,
    fit_event_statespace_ridge,
    predict_event_statespace,
    fit_xgb_overfit_diag,
)

DATA_DIR = ML_DIR / "data"
RES_DIR = ML_DIR / "results"
OPTA_PATH = ML_DIR.parent / "mapped_epl" / "opta_power_ranking_261124.json"


def run():
    df_2627 = pd.read_csv(DATA_DIR / "matches_2627.csv")
    df_hist = pd.read_csv(DATA_DIR / "history_2126.csv")

    teams_2627 = sorted(set(df_2627["home_team"]).union(df_2627["away_team"]))
    teams_2526 = set(df_hist[df_hist["season"] == "2025/26"]["home_team"]).union(
        df_hist[df_hist["season"] == "2025/26"]["away_team"]
    )
    promoted = set(teams_2627) - teams_2526
    print(f"[*] EPL 2026/27 teams={len(teams_2627)}, promoted={sorted(promoted)}")

    hist_priors = fit_historical_elo_and_logit(df_hist, teams_2627)
    elo_ratings = dict(hist_priors["init_elo"])
    elo_clf = hist_priors["elo_clf"]
    tbl_scaler = hist_priors["tbl_scaler"]
    tbl_clf = hist_priors["tbl_clf"]
    hda_counts = hist_priors["hist_hda_counts"].copy()

    # Keep only 2025/26 history for Dixon-Coles carryover speed
    df_2526 = df_hist[df_hist["season"] == "2025/26"][["season", "date", "home_team", "away_team", "home_goals", "away_goals"]].copy()

    tracker = SeasonTracker()
    xgb_X_hist, xgb_y_hist = [], []

    eval_start_idx = 30  # Evaluate matches 31..107 (N_eval = 77) so teams have ~3 games of 26/27 form
    preds = {k: [] for k in ["B0_NaivePrior", "B1_TablePPG", "B2_Elo", "B3_DixonColes", "B4_EventStateSpace_xT", "B4_XGB_OverfitDiag"]}
    eval_rows = []

    dc_model = None
    dc_x0 = None
    ss_model = None
    last_refit_date = None
    xgb_clf = None

    for idx, row in df_2627.iterrows():
        ht, at = row["home_team"], row["away_team"]
        m_date = row["date"]
        m_dt = pd.to_datetime(m_date)

        # Build pre-match feature vectors strictly BEFORE updating state with match idx
        hp, hgd, _ = tracker.get_table_features(ht)
        ap, agd, _ = tracker.get_table_features(at)
        xgb_vec = build_xgb_feature_vector(tracker, ht, at)

        if idx >= eval_start_idx:
            # 1. B0_NaivePrior
            p_b0 = hda_counts / hda_counts.sum()
            preds["B0_NaivePrior"].append(p_b0)

            # 2. B1_TablePPG
            x_tbl = tbl_scaler.transform(np.array([[hp - ap, hgd - agd]]))
            preds["B1_TablePPG"].append(tbl_clf.predict_proba(x_tbl)[0])

            # 3. B2_Elo
            elo_diff = (elo_ratings[ht] + 65.0) - elo_ratings[at]
            preds["B2_Elo"].append(elo_clf.predict_proba(np.array([[elo_diff]]))[0])

            # Refit Dixon-Coles, EventStateSpace & XGB per new matchday (strictly using date < m_date)
            if last_refit_date != m_date:
                past_2627 = df_2627[df_2627["date"] < m_date]
                dc_train = pd.concat(
                    [df_2526, past_2627[["season", "date", "home_team", "away_team", "home_goals", "away_goals"]]],
                    ignore_index=True,
                )
                dc_model, dc_x0 = fit_dixon_coles(dc_train, teams_2627, promoted, m_dt, x0=dc_x0)
                ss_model = fit_event_statespace_ridge(past_2627, teams_2627, hist_priors["init_elo"], promoted)

                # Fit XGB strictly on past 26/27 matches before m_date
                past_n = len(past_2627)
                if past_n >= 25:
                    xgb_clf = fit_xgb_overfit_diag(np.array(xgb_X_hist[:past_n]), np.array(xgb_y_hist[:past_n]))
                last_refit_date = m_date

            # 4. B3_DixonColes
            preds["B3_DixonColes"].append(predict_dixon_coles(dc_model, ht, at))

            # 5. B4_EventStateSpace_xT
            preds["B4_EventStateSpace_xT"].append(predict_event_statespace(ss_model, ht, at))

            # 6. B4_XGB_OverfitDiag
            if xgb_clf is not None:
                preds["B4_XGB_OverfitDiag"].append(xgb_clf.predict_proba(xgb_vec.reshape(1, -1))[0])
            else:
                preds["B4_XGB_OverfitDiag"].append(p_b0)

            eval_rows.append(row)

        # Strictly AFTER prediction: record feature+target for future XGB fits & update states
        xgb_X_hist.append(xgb_vec)
        xgb_y_hist.append(int(row["target"]))
        tracker.update(row)
        hda_counts[int(row["target"])] += 1.0

        # Sequential Elo update
        diff_post = (elo_ratings[ht] + 65.0) - elo_ratings[at]
        exp_h = 1.0 / (1.0 + 10.0 ** (-diff_post / 400.0))
        hg, ag = int(row["home_goals"]), int(row["away_goals"])
        actual_h = 1.0 if hg > ag else (0.5 if hg == ag else 0.0)
        mov = math.log1p(abs(hg - ag)) * (2.2 / ((abs(diff_post) * 0.001) + 2.2))
        delta = 20.0 * max(1.0, mov) * (actual_h - exp_h)
        elo_ratings[ht] += delta
        elo_ratings[at] -= delta

    df_eval = pd.DataFrame(eval_rows).reset_index(drop=True)
    y_eval = df_eval["target"].to_numpy(dtype=int)

    # Validate probabilities sum to 1.0 & assemble benchmark table
    summary_rows = []
    b2_probs = clip_probs(np.array(preds["B2_Elo"]))
    b0_probs = clip_probs(np.array(preds["B0_NaivePrior"]))

    for name, plist in preds.items():
        parr = clip_probs(np.array(plist))
        assert np.allclose(parr.sum(axis=1), 1.0, atol=1e-4)
        res = evaluate_all(y_eval, parr)
        boot_b0 = bootstrap_rps_diff(y_eval, parr, b0_probs)
        boot_b2 = bootstrap_rps_diff(y_eval, parr, b2_probs)
        res.update(
            {
                "model": name,
                "dRPS_vs_B0": round(boot_b0["mean_diff"], 4),
                "dRPS_vs_B0_CI": f"[{boot_b0['ci_low']:+.4f}, {boot_b0['ci_high']:+.4f}]",
                "dRPS_vs_B2": round(boot_b2["mean_diff"], 4),
                "p_beat_B2": round(boot_b2["p_better"], 3),
            }
        )
        summary_rows.append(res)

    df_sum = pd.DataFrame(summary_rows)[
        ["model", "n", "rps", "log_loss", "brier", "accuracy", "ece", "dRPS_vs_B0", "dRPS_vs_B0_CI", "dRPS_vs_B2", "p_beat_B2"]
    ].sort_values("rps")
    df_sum.to_csv(RES_DIR / "benchmark_summary.csv", index=False)

    # Subset comparison against B5_MarketOdds where odds are present in eval window
    odds_mask = df_eval["odds_prob_home"].notna().to_numpy()
    odds_rows = []
    if odds_mask.any():
        y_o = y_eval[odds_mask]
        p_mkt = df_eval.loc[odds_mask, ["odds_prob_home", "odds_prob_draw", "odds_prob_away"]].to_numpy(dtype=float)
        r_mkt = evaluate_all(y_o, p_mkt)
        r_mkt["model"] = "B5_MarketOdds"
        odds_rows.append(r_mkt)
        for name, plist in preds.items():
            r_sub = evaluate_all(y_o, np.array(plist)[odds_mask])
            r_sub["model"] = name
            odds_rows.append(r_sub)
    df_odds_sub = pd.DataFrame(odds_rows).sort_values("rps") if odds_rows else pd.DataFrame()
    if not df_odds_sub.empty:
        df_odds_sub.to_csv(RES_DIR / "benchmark_odds_overlap.csv", index=False)

    # Save per-match predictions
    pred_out = df_eval[["ws_match_id", "date", "home_team", "away_team", "home_goals", "away_goals", "result"]].copy()
    for name, plist in preds.items():
        parr = clip_probs(np.array(plist))
        pred_out[f"{name}_pH"] = np.round(parr[:, 0], 4)
        pred_out[f"{name}_pD"] = np.round(parr[:, 1], 4)
        pred_out[f"{name}_pA"] = np.round(parr[:, 2], 4)
    pred_out.to_csv(RES_DIR / "predictions_2627.csv", index=False)

    # Post-hoc MD11 correlation vs opta_power_ranking_261124.json (NEVER used in training!)
    dc_final, _ = fit_dixon_coles(
        pd.concat([df_2526, df_2627[["season", "date", "home_team", "away_team", "home_goals", "away_goals"]]], ignore_index=True),
        teams_2627,
        promoted,
        pd.to_datetime(df_2627["date"].max()),
        x0=dc_x0,
    )
    ss_final = fit_event_statespace_ridge(df_2627, teams_2627, hist_priors["init_elo"], promoted)

    opta_raw = json.loads(OPTA_PATH.read_text(encoding="utf-8"))
    opta_list = opta_raw["division"][0]["ranking"]
    opta_map = {x["contestantCode"]: float(x["currentRating"]) for x in opta_list}

    corr_rows = []
    for t in teams_2627:
        ppg, gd_pg, _ = tracker.get_table_features(t)
        dc_net = float(dc_final["att"][dc_final["t2i"][t]] - dc_final["def"][dc_final["t2i"][t]])
        ss_net = float(ss_final["att"][ss_final["t2i"][t]] - ss_final["def"][ss_final["t2i"][t]])
        corr_rows.append(
            {
                "team": t,
                "opta_rating": opta_map[t],
                "table_ppg": round(ppg, 3),
                "elo": round(elo_ratings[t], 1),
                "dc_net": round(dc_net, 3),
                "event_ss_net": round(ss_net, 3),
            }
        )
    df_corr = pd.DataFrame(corr_rows).sort_values("opta_rating", ascending=False).reset_index(drop=True)
    df_corr.to_csv(RES_DIR / "opta_correlation.csv", index=False)

    corr_stats = []
    for col in ["table_ppg", "elo", "dc_net", "event_ss_net"]:
        sp, sp_p = spearmanr(df_corr["opta_rating"], df_corr[col])
        pe, pe_p = pearsonr(df_corr["opta_rating"], df_corr[col])
        corr_stats.append((col, sp, sp_p, pe, pe_p))

    print("\n=== WALK-FORWARD BENCHMARK (EPL 2026/27, Matches #31-#107, N=77) ===")
    print(df_sum.to_string(index=False))
    if not df_odds_sub.empty:
        print(f"\n=== MARKET ODDS OVERLAP SUBSET (N={int(odds_mask.sum())}) ===")
        print(df_odds_sub[["model", "n", "rps", "log_loss", "brier", "accuracy", "ece"]].to_string(index=False))
    print("\n=== POST-HOC CORRELATION vs OPTA POWER RANKING (261124, NEVER USED IN TRAIN) ===")
    for col, sp, sp_p, pe, pe_p in corr_stats:
        print(f"  {col:14s} -> Spearman r={sp:.3f} (p={sp_p:.2e}) | Pearson r={pe:.3f} (p={pe_p:.2e})")


if __name__ == "__main__":
    run()
