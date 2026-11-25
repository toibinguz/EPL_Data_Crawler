"""
ml_epl2627/src/shot_xg_model.py
Self-contained Shot-level non-penalty Expected Goals (npxG) model
trained strictly on historical clean_shots (season < '2026/27', Zero 26/27 Leakage).
Scores every shot in WhoScored 2026/27 matchCentreData.json cleanly without needing FotMob/Understat.
"""

import math
import sqlite3
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss, roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parent.parent.parent
CLEAN_DB = ROOT / "mapped_epl" / "epl_clean.db"
PITCH_L = 105.0
PITCH_W = 68.0
GOAL_W = 7.32
HALF_GW = GOAL_W / 2.0


def compute_shot_geom(x_m: np.ndarray, y_m: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    dx = np.maximum(0.25, PITCH_L - x_m)
    dy = y_m - (PITCH_W / 2.0)
    dist_m = np.sqrt(dx * dx + dy * dy)
    denom = dx * dx + dy * dy - HALF_GW * HALF_GW
    angle_rad = np.where(denom > 0, np.arctan2(GOAL_W * dx, denom), math.pi / 2.0)
    return dist_m, angle_rad


def train_historical_shot_xg_model() -> tuple[Pipeline, dict]:
    """
    Trains calibrated LogisticRegression on historical Understat shots (< '2026/27'),
    excluding penalties & own goals, and validates OOS AUC/Brier on 2025/26.
    """
    conn = sqlite3.connect(str(CLEAN_DB))
    df = pd.read_sql_query(
        """
        SELECT season, x, y, outcome, situation, shot_type
        FROM clean_shots
        WHERE source = 'understat'
          AND season < '2026/27'
          AND situation != 'Penalty'
          AND outcome != 'OwnGoal'
          AND x IS NOT NULL AND y IS NOT NULL
        """,
        conn,
    )
    conn.close()
    assert len(df) > 50000 and df["season"].max() == "2025/26"

    x_m = df["x"].to_numpy(dtype=float) * PITCH_L
    y_m = df["y"].to_numpy(dtype=float) * PITCH_W
    dist_m, angle_rad = compute_shot_geom(x_m, y_m)

    sit = df["situation"].fillna("").astype(str)
    st = df["shot_type"].fillna("").astype(str)
    is_header = st.str.contains("Head", case=False).astype(float).to_numpy()
    is_set_piece = sit.str.contains("Corner|Free|SetPiece|ThrowIn", case=False).astype(float).to_numpy()
    is_fast_break = sit.str.contains("FastBreak|Counter", case=False).astype(float).to_numpy()

    X = np.column_stack([dist_m, angle_rad, np.log1p(dist_m), is_header, is_set_piece, is_fast_break])
    y = (df["outcome"] == "Goal").astype(int).to_numpy()

    val_mask = (df["season"] == "2025/26").to_numpy()
    tr_mask = ~val_mask

    pipe_val = Pipeline([("sc", StandardScaler()), ("lr", LogisticRegression(C=2.0, max_iter=500))])
    pipe_val.fit(X[tr_mask], y[tr_mask])
    p_val = pipe_val.predict_proba(X[val_mask])[:, 1]
    val_auc = float(roc_auc_score(y[val_mask], p_val))
    val_brier = float(brier_score_loss(y[val_mask], p_val))

    # Refit on ALL historical seasons (< '2026/27')
    pipe_full = Pipeline([("sc", StandardScaler()), ("lr", LogisticRegression(C=2.0, max_iter=500))])
    pipe_full.fit(X, y)

    metrics = {
        "n_train_total": int(len(df)),
        "n_val_2526": int(val_mask.sum()),
        "val_2526_auc": round(val_auc, 4),
        "val_2526_brier": round(val_brier, 5),
        "val_2526_goal_rate": round(float(y[val_mask].mean()), 4),
        "val_2526_mean_pred_xg": round(float(p_val.mean()), 4),
    }
    return pipe_full, metrics


def score_whoscored_match_npxg(events: list, home_tid: int, model: Pipeline) -> tuple[float, float]:
    """Compute (home_opta_npxg, away_opta_npxg) from WhoScored Opta event stream."""
    shots_feats = []
    sides = []
    for e in events:
        etype = (e.get("type") or {}).get("displayName", "")
        if etype not in ("Goal", "SavedShot", "MissedShots", "ShotOnPost"):
            continue
        quals = {(q.get("type") or {}).get("displayName") for q in e.get("qualifiers", [])}
        if "Penalty" in quals or "OwnGoal" in quals:
            continue
        x_m = (float(e.get("x", 0.0)) / 100.0) * PITCH_L
        y_m = (float(e.get("y", 0.0)) / 100.0) * PITCH_W
        d_m, ang = compute_shot_geom(np.array([x_m]), np.array([y_m]))
        is_hdr = 1.0 if "Head" in quals else 0.0
        is_sp = 1.0 if any(k in quals for k in ("FromCorner", "DirectFreekick", "SetPiece", "ThrowinSetPiece")) else 0.0
        is_fb = 1.0 if "FastBreak" in quals else 0.0
        shots_feats.append([float(d_m[0]), float(ang[0]), math.log1p(float(d_m[0])), is_hdr, is_sp, is_fb])
        sides.append(e.get("teamId") == home_tid)

    if not shots_feats:
        return 0.0, 0.0
    probs = model.predict_proba(np.array(shots_feats, dtype=float))[:, 1]
    sides_arr = np.array(sides, dtype=bool)
    h_npxg = float(probs[sides_arr].sum())
    a_npxg = float(probs[~sides_arr].sum())
    return round(h_npxg, 4), round(a_npxg, 4)


if __name__ == "__main__":
    m, info = train_historical_shot_xg_model()
    print("[+] Shot-xG Historical Model:", info)
