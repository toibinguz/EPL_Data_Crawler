"""
ml_epl2627/champion_v2/predict.py
Standalone CLI runner for the new Champion v2 Model (`EPLChampionV2`).
Usage:
  python ml_epl2627/champion_v2/predict.py
  python ml_epl2627/champion_v2/predict.py ARS MCI
"""

import sys
from pathlib import Path
import pandas as pd

V2_DIR = Path(__file__).resolve().parent
ML_DIR = V2_DIR.parent
if str(ML_DIR) not in sys.path:
    sys.path.insert(0, str(ML_DIR))

from champion_v2.champion_v2_model import EPLChampionV2  # noqa: E402

DATA_DIR = ML_DIR / "data"


def print_prediction(pred: dict) -> None:
    scores_str = ", ".join(f"{h}-{a} ({p:.1f}%)" for h, a, p in pred["top_scores"])
    print(
        f"  {pred['home_team']} vs {pred['away_team']} | "
        f"xG=({pred['xg_home']:.2f}, {pred['xg_away']:.2f}) | "
        f"Win={pred['pH']*100:4.1f}%±{pred['std_pH']*100:3.1f}% | "
        f"Draw={pred['pD']*100:4.1f}%±{pred['std_pD']*100:3.1f}% | "
        f"Away={pred['pA']*100:4.1f}%±{pred['std_pA']*100:3.1f}% | "
        f"Top Scores: {scores_str}"
    )


def main() -> None:
    df_hist = pd.read_csv(DATA_DIR / "history_1426.csv")
    df_2627 = pd.read_csv(DATA_DIR / "matches_2627.csv")

    model = EPLChampionV2().fit(df_hist, df_2627)
    print(
        f"[+] EPLChampionV2 fitted on N_2627={model.n_matches_2627} + N_recent={model.n_hist_matches} "
        f"| Elo(K={model.k}, HFA={model.hfa}, sh={model.shrink}, promo={model.promo_init}, wxg={model.w_xg}) "
        f"| Pool w_elo={model.w_elo:.2f} | Dixon-Coles rho={model.rho:.4f}"
    )

    if len(sys.argv) >= 3:
        ht, at = sys.argv[1].upper(), sys.argv[2].upper()
        print(f"\n=== CHAMPION V2 PREDICTION: {ht} vs {at} ===")
        print_prediction(model.predict_match(ht, at))
    else:
        print("\n=== CHAMPION V2 POWER RANKINGS (20 TEAMS AFTER 110 MATCHES) ===")
        print(model.get_rankings_df().to_string(index=False))
        print("\n=== SAMPLE PREDICTIONS ===")
        for h, a in [("ARS", "MCI"), ("LIV", "MUN"), ("CHE", "TOT")]:
            print_prediction(model.predict_match(h, a))


if __name__ == "__main__":
    main()
