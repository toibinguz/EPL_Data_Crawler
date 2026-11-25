"""
ml_epl2627/champion/predict.py
Standalone CLI runner for the locked EPL 2026/27 Champion Model (Champion_Unified).
Usage:
  python ml_epl2627/champion/predict.py
  python ml_epl2627/champion/predict.py ARS MCI
"""

import sys
from pathlib import Path
import pandas as pd

CHAMP_DIR = Path(__file__).resolve().parent
ML_DIR = CHAMP_DIR.parent
if str(ML_DIR) not in sys.path:
    sys.path.insert(0, str(ML_DIR))

from champion.champion_model import EPLChampionModel  # noqa: E402

DATA_DIR = ML_DIR / "data"


def print_prediction(pred: dict):
    scores_str = ", ".join(f"{h}-{a} ({p:.1f}%)" for h, a, p in pred["top_scores"])
    print(
        f"  {pred['home_team']} vs {pred['away_team']} | "
        f"xG=({pred['xg_home']:.2f}, {pred['xg_away']:.2f}) | "
        f"Win={pred['pH']*100:4.1f}%±{pred['std_pH']*100:3.1f}% | "
        f"Draw={pred['pD']*100:4.1f}%±{pred['std_pD']*100:3.1f}% | "
        f"Away={pred['pA']*100:4.1f}%±{pred['std_pA']*100:3.1f}% | "
        f"Top Scores: {scores_str}"
    )


def main():
    df_hist = pd.read_csv(DATA_DIR / "history_2126.csv")
    df_2627 = pd.read_csv(DATA_DIR / "matches_2627.csv")

    model = EPLChampionModel().fit(df_hist, df_2627)
    print(f"[+] Locked EPLChampionModel fitted on N={model.n_matches_2627} matches (2026/27) + history (2021/22-2025/26)")
    print(f"[+] Learned Greedy State-Space Config: {model.latest_cfg} | Elo Blend Weight: {model.predict_match('ARS', 'MCI')['w_elo_blend']:.4f}")

    if len(sys.argv) >= 3:
        ht, at = sys.argv[1].upper(), sys.argv[2].upper()
        print(f"\n=== MATCH PREDICTION: {ht} vs {at} ===")
        print_prediction(model.predict_match(ht, at))
    else:
        print("\n=== LOCKED POWER RANKINGS (20 TEAMS AFTER 110 MATCHES) ===")
        print(model.get_rankings_df().to_string(index=False))
        print("\n=== SAMPLE PREDICTIONS ===")
        for h, a in [("ARS", "MCI"), ("LIV", "MUN"), ("CHE", "TOT")]:
            print_prediction(model.predict_match(h, a))


if __name__ == "__main__":
    main()
