"""
ml_epl2627/src/features.py
Strict Zero-Leakage sequential state & feature builders.
Every feature for match t uses strictly matches before date(t) (or within the sequential stream
with update occurring strictly AFTER predicting match t).
"""

import numpy as np
import pandas as pd


class SeasonTracker:
    """Tracks per-team expanding season table & Opta spatial stats strictly prior to each match."""

    def __init__(self):
        self.stats = {}  # team -> list of match dicts

    def _hist(self, team: str) -> list[dict]:
        return self.stats.get(team, [])

    def get_table_features(self, team: str) -> tuple[float, float, int]:
        h = self._hist(team)
        if not h:
            return 1.35, 0.0, 0
        pts = sum(m["pts"] for m in h)
        gd = sum(m["gf"] - m["ga"] for m in h)
        n = len(h)
        return pts / n, gd / n, n

    def get_rolling_features(self, team: str, window: int = 5) -> dict:
        h = self._hist(team)[-window:]
        if not h:
            return {
                "ppg": 1.35,
                "gd": 0.0,
                "shots": 12.0,
                "sot": 4.2,
                "box_shots": 7.5,
                "xt": 1.2,
                "prog_passes": 42.0,
                "poss": 50.0,
                "rating": 6.60,
                "shots_ag": 12.0,
                "xt_ag": 1.2,
            }
        n = len(h)
        return {
            "ppg": sum(m["pts"] for m in h) / n,
            "gd": sum(m["gf"] - m["ga"] for m in h) / n,
            "shots": sum(m["shots"] for m in h) / n,
            "sot": sum(m["sot"] for m in h) / n,
            "box_shots": sum(m["box_shots"] for m in h) / n,
            "xt": sum(m["xt"] for m in h) / n,
            "prog_passes": sum(m["prog_passes"] for m in h) / n,
            "poss": sum(m["poss"] for m in h) / n,
            "rating": sum(m["rating"] for m in h) / n,
            "shots_ag": sum(m["shots_ag"] for m in h) / n,
            "xt_ag": sum(m["xt_ag"] for m in h) / n,
        }

    def update(self, row: pd.Series):
        hg, ag = int(row["home_goals"]), int(row["away_goals"])
        h_pts = 3 if hg > ag else (1 if hg == ag else 0)
        a_pts = 3 if ag > hg else (1 if hg == ag else 0)

        h_rec = {
            "pts": h_pts,
            "gf": hg,
            "ga": ag,
            "shots": float(row.get("home_shots", 12.0)),
            "sot": float(row.get("home_sot", 4.0)),
            "box_shots": float(row.get("home_box_shots", 7.0)),
            "xt": float(row.get("home_xt", 1.2)),
            "prog_passes": float(row.get("home_prog_passes", 40.0)),
            "poss": float(row.get("home_poss", 50.0)),
            "rating": float(row.get("home_rating", 6.6)),
            "shots_ag": float(row.get("away_shots", 12.0)),
            "xt_ag": float(row.get("away_xt", 1.2)),
        }
        a_rec = {
            "pts": a_pts,
            "gf": ag,
            "ga": hg,
            "shots": float(row.get("away_shots", 12.0)),
            "sot": float(row.get("away_sot", 4.0)),
            "box_shots": float(row.get("away_box_shots", 7.0)),
            "xt": float(row.get("away_xt", 1.2)),
            "prog_passes": float(row.get("away_prog_passes", 40.0)),
            "poss": float(row.get("away_poss", 50.0)),
            "rating": float(row.get("away_rating", 6.6)),
            "shots_ag": float(row.get("home_shots", 12.0)),
            "xt_ag": float(row.get("home_xt", 1.2)),
        }
        self.stats.setdefault(row["home_team"], []).append(h_rec)
        self.stats.setdefault(row["away_team"], []).append(a_rec)


def build_xgb_feature_vector(tracker: SeasonTracker, home_team: str, away_team: str) -> np.ndarray:
    """12 raw rolling + table features for the B4_XGB_OverfitDiag benchmark."""
    hf = tracker.get_rolling_features(home_team, window=5)
    af = tracker.get_rolling_features(away_team, window=5)
    h_ppg, h_gd, _ = tracker.get_table_features(home_team)
    a_ppg, a_gd, _ = tracker.get_table_features(away_team)
    return np.array(
        [
            h_ppg - a_ppg,
            h_gd - a_gd,
            hf["ppg"] - af["ppg"],
            hf["gd"] - af["gd"],
            hf["shots"] - af["shots"],
            hf["sot"] - af["sot"],
            hf["box_shots"] - af["box_shots"],
            hf["xt"] - af["xt"],
            hf["prog_passes"] - af["prog_passes"],
            hf["poss"] - af["poss"],
            hf["rating"] - af["rating"],
            hf["xt_ag"] - af["xt_ag"],
        ],
        dtype=float,
    )
