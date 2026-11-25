"""
ml_epl2627/src/shift_analysis.py
Quantifies distribution shift between historical EPL seasons (2021/22-2025/26)
and current EPL 2026/27 (N=107) to validate cross-season prior usability.
"""

import json
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.stats import chi2_contingency, ks_2samp

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
OUT_JSON = Path(__file__).resolve().parent.parent / "results" / "shift_report.json"


def summarize_split(df: pd.DataFrame) -> dict:
    total_goals = df["home_goals"] + df["away_goals"]
    total_shots = df["home_shots"] + df["away_shots"]
    goal_diff = df["home_goals"] - df["away_goals"]
    return {
        "n": int(len(df)),
        "home_win_rate": round(float((df["result"] == "H").mean()), 4),
        "draw_rate": round(float((df["result"] == "D").mean()), 4),
        "away_win_rate": round(float((df["result"] == "A").mean()), 4),
        "goals_per_match": round(float(total_goals.mean()), 4),
        "home_goal_adv": round(float(goal_diff.mean()), 4),
        "shots_per_match": round(float(total_shots.dropna().mean()), 4) if total_shots.notna().any() else None,
    }


def main():
    df_2627 = pd.read_csv(DATA_DIR / "matches_2627.csv")
    df_hist = pd.read_csv(DATA_DIR / "history_2126.csv")

    s_hist = summarize_split(df_hist)
    s_2627 = summarize_split(df_2627)

    # Chi-square test on H/D/A outcome counts
    c_hist = [int((df_hist["result"] == k).sum()) for k in ("H", "D", "A")]
    c_2627 = [int((df_2627["result"] == k).sum()) for k in ("H", "D", "A")]
    chi2, p_hda, _, _ = chi2_contingency([c_hist, c_2627])

    # KS tests on Goals/Match, Home-Away Goal Diff, Shots/Match
    ks_goals = ks_2samp(df_hist["home_goals"] + df_hist["away_goals"], df_2627["home_goals"] + df_2627["away_goals"])
    ks_gd = ks_2samp(df_hist["home_goals"] - df_hist["away_goals"], df_2627["home_goals"] - df_2627["away_goals"])
    ks_shots = ks_2samp(
        (df_hist["home_shots"] + df_hist["away_shots"]).dropna(),
        (df_2627["home_shots"] + df_2627["away_shots"]).dropna(),
    )

    per_season = {sec: summarize_split(g) for sec, g in df_hist.groupby("season")}
    per_season["2026/27"] = s_2627

    report = {
        "history_2126": s_hist,
        "epl_2627": s_2627,
        "per_season": per_season,
        "tests": {
            "hda_chi2_p": round(float(p_hda), 4),
            "goals_ks_p": round(float(ks_goals.pvalue), 4),
            "home_adv_ks_p": round(float(ks_gd.pvalue), 4),
            "shots_ks_p": round(float(ks_shots.pvalue), 4),
        },
    }
    OUT_JSON.write_text(json.dumps(report, indent=2), encoding="utf-8")

    print("=== DISTRIBUTION SHIFT: EPL 2021/22-2025/26 vs 2026/27 (N=107) ===")
    print(f"Hist (N={s_hist['n']}): H={s_hist['home_win_rate']:.3f}, D={s_hist['draw_rate']:.3f}, A={s_hist['away_win_rate']:.3f} | Goals/M={s_hist['goals_per_match']:.2f} | H-Adv={s_hist['home_goal_adv']:+.2f} | Shots/M={s_hist['shots_per_match']:.2f}")
    print(f"2627 (N={s_2627['n']}): H={s_2627['home_win_rate']:.3f}, D={s_2627['draw_rate']:.3f}, A={s_2627['away_win_rate']:.3f} | Goals/M={s_2627['goals_per_match']:.2f} | H-Adv={s_2627['home_goal_adv']:+.2f} | Shots/M={s_2627['shots_per_match']:.2f}")
    print(f"Shift p-values -> H/D/A chi2 p={p_hda:.4f} | Goals KS p={ks_goals.pvalue:.4f} | Home-Adv KS p={ks_gd.pvalue:.4f} | Shots KS p={ks_shots.pvalue:.4f}")


if __name__ == "__main__":
    main()
