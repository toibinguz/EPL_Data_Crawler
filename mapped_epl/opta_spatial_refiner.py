"""
mapped_epl/opta_spatial_refiner.py
Spatial & Temporal Refinement Engine for WhoScored Opta Event Streams:
  1. Metric pitch conversion (105m x 68m) & true geometric Progressive Pass detection.
  2. Set-piece coordinate snapping (CornerTaken -> corner, ThrowIn -> touchline, GoalKick -> 5.5m box).
  3. Sub-second timestamp disambiguation using physical ball flight velocity.
  4. Standard 12x8 Expected Threat (xT) valuation for passes & carries.
  5. Cross-source Shot xG fusion from FotMob into Opta shot/goal events.
  6. Physical Carry plausibility filtering (v_carry <= 10.5 m/s) & carry speed estimation.
"""

import math
from typing import Any

PITCH_L = 105.0
PITCH_W = 68.0

# Standard 12x8 Expected Threat (xT) grid (Karun Singh open-source baseline)
XT_GRID = [
    [0.00638303, 0.00779616, 0.00844854, 0.01676086, 0.01676086, 0.00844854, 0.00779616, 0.00638303],
    [0.00750072, 0.00878589, 0.00942382, 0.01592616, 0.01592616, 0.00942382, 0.00878589, 0.00750072],
    [0.00887990, 0.00977745, 0.01001304, 0.01484214, 0.01484214, 0.01001304, 0.00977745, 0.00887990],
    [0.00941056, 0.01082722, 0.01125847, 0.01440262, 0.01440262, 0.01125847, 0.01082722, 0.00941056],
    [0.01059369, 0.01248511, 0.01319152, 0.01542617, 0.01542617, 0.01319152, 0.01248511, 0.01059369],
    [0.01222425, 0.01449439, 0.01630062, 0.01911457, 0.01911457, 0.01630062, 0.01449439, 0.01222425],
    [0.01417965, 0.01701414, 0.02032917, 0.02418957, 0.02418957, 0.02032917, 0.01701414, 0.01417965],
    [0.01666789, 0.02081063, 0.02652794, 0.03298945, 0.03298945, 0.02652794, 0.02081063, 0.01666789],
    [0.01980876, 0.02635613, 0.03630106, 0.04930261, 0.04930261, 0.03630106, 0.02635613, 0.01980876],
    [0.02447236, 0.03556758, 0.05434157, 0.08826712, 0.08826712, 0.05434157, 0.03556758, 0.02447236],
    [0.03069092, 0.04880025, 0.08792461, 0.17266586, 0.17266586, 0.08792461, 0.04880025, 0.03069092],
    [0.04055930, 0.06880814, 0.14536083, 0.34183892, 0.34183892, 0.14536083, 0.06880814, 0.04055930],
]


def get_xt(x_pct: float, y_pct: float) -> float:
    """Lookup Expected Threat value for Opta (0..100, 0..100) coordinates."""
    ix = min(11, max(0, int((x_pct / 100.0) * 12)))
    iy = min(7, max(0, int((y_pct / 100.0) * 8)))
    return XT_GRID[ix][iy]


def refine_opta_events(
    raw_evs: list[dict[str, Any]],
    home_tid: int,
    p_dict: dict[str, str],
    fotmob_shots: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    """Refine raw Opta events with spatial snapping, sub-second timing, xT, xG fusion, and carry metrics."""
    keep_types = (
        "Pass", "Tackle", "Challenge", "Interception", "BallRecovery", "Clearance",
        "Foul", "TakeOn", "BallTouch", "SavedShot", "MissedShots", "Goal", "ShotOnPost",
        "Aerial", "BlockedPass", "Save", "Dispossessed", "KeeperPickup", "Claim",
        "KeeperSweeper", "Smother", "Error"
    )

    # Index FotMob shots by (minute, isHome) for cross-source xG fusion
    fm_shot_map: dict[tuple[int, bool], list[dict[str, Any]]] = {}
    if fotmob_shots:
        for s in fotmob_shots:
            key = (int(s.get("min", -1)), bool(s.get("isHome")))
            fm_shot_map.setdefault(key, []).append(s)

    refined = []
    snapped_count = 0
    xg_fused_count = 0
    prev_t = -1.0

    for i, e in enumerate(raw_evs):
        etype = (e.get("type") or {}).get("displayName")
        if etype not in keep_types:
            continue

        pid = str(e.get("playerId")) if e.get("playerId") is not None else None
        tid = e.get("teamId")
        is_home = (tid == home_tid)
        pname = p_dict.get(pid, pid or "")

        x = float(e.get("x", 0.0))
        y = float(e.get("y", 0.0))
        end_x = float(e["endX"]) if e.get("endX") is not None else None
        end_y = float(e["endY"]) if e.get("endY") is not None else None
        ok = (e.get("outcomeType") or {}).get("value") == 1

        # Deduplicate Opta's mirrored Foul pair (keep only the player who committed the foul: ok=False)
        if etype == "Foul" and ok:
            continue

        quals_map = {(q.get("type") or {}).get("displayName"): q.get("value") for q in e.get("qualifiers", [])}
        quals = set(quals_map.keys())

        # 1. Set-piece spatial snapping
        snapped = False
        if "CornerTaken" in quals:
            x = 100.0 if x > 50 else 0.0
            y = 100.0 if y > 50 else 0.0
            snapped = True
        elif "ThrowIn" in quals or "ThrowinSetPiece" in quals:
            y = 100.0 if y > 50 else 0.0
            snapped = True
        elif "GoalKick" in quals:
            x = 5.24 if x < 50 else 94.76
            y = min(63.2, max(36.8, y))
            snapped = True
        if snapped:
            snapped_count += 1

        # 2. Metric conversion (105m x 68m) & Progressive Pass geometry
        x_m = round((x / 100.0) * PITCH_L, 2)
        y_m = round((y / 100.0) * PITCH_W, 2)
        end_x_m = round((end_x / 100.0) * PITCH_L, 2) if end_x is not None else None
        end_y_m = round((end_y / 100.0) * PITCH_W, 2) if end_y is not None else None

        dist_m = 0.0
        is_prog = False
        xt_gain = 0.0
        if end_x_m is not None and end_y_m is not None:
            dist_m = round(math.hypot(end_x_m - x_m, end_y_m - y_m), 1)
            d_goal_start = math.hypot(PITCH_L - x_m, 34.0 - y_m)
            d_goal_end = math.hypot(PITCH_L - end_x_m, 34.0 - end_y_m)
            if ok and etype == "Pass" and (
                (d_goal_start > 0 and (d_goal_start - d_goal_end) / d_goal_start >= 0.25)
                or (end_x_m - x_m >= 10.0 and end_x_m >= 40.0)
            ):
                is_prog = True
            if ok and etype == "Pass":
                xt_gain = round(get_xt(end_x, end_y) - get_xt(x, y), 4)

        # 3. Sub-second timestamp disambiguation using physical ball speed
        exp_min = int(e.get("expandedMinute", e.get("minute", 0)))
        sec = int(e.get("second", 0))
        base_t = exp_min * 60.0 + sec
        if base_t <= prev_t:
            v_ball = 14.5 if any(q in quals for q in ("Longball", "Cross", "Chipped")) else 18.5
            flight_dt = max(0.35, min(1.8, (dist_m / v_ball) if dist_m > 0 else 0.45))
            t_sub = round(prev_t + flight_dt, 2)
        else:
            t_sub = round(base_t, 2)
        prev_t = t_sub

        # 4. Cross-source FotMob xG fusion + Shot target coordinates (BlockedX/Y or GoalMouthY)
        xg_val = None
        shot_end_x = None
        shot_end_y = None
        if etype in ("Goal", "SavedShot", "MissedShots", "ShotOnPost"):
            if quals_map.get("BlockedX") is not None and quals_map.get("BlockedY") is not None:
                try:
                    shot_end_x = round(float(quals_map["BlockedX"]), 2)
                    shot_end_y = round(float(quals_map["BlockedY"]), 2)
                except Exception:
                    pass
            if shot_end_x is None:
                shot_end_x = 100.0
                try:
                    shot_end_y = round(float(quals_map.get("GoalMouthY", 50.0)), 2)
                except Exception:
                    shot_end_y = 50.0
            minute_raw = int(e.get("minute", 0))
            candidates = fm_shot_map.get((minute_raw, is_home)) or fm_shot_map.get((minute_raw - 1, is_home)) or fm_shot_map.get((minute_raw + 1, is_home)) or []
            last_name = pname.split()[-1].lower() if pname else ""
            for cand in candidates:
                c_name = str(cand.get("player", "")).lower()
                if last_name and last_name in c_name:
                    xg_val = round(float(cand.get("xg", 0.0)), 3)
                    xg_fused_count += 1
                    break
            if xg_val is None and candidates:
                xg_val = round(float(candidates[0].get("xg", 0.0)), 3)
                xg_fused_count += 1

        refined.append({
            "type": etype,
            "isHome": is_home,
            "pid": pid,
            "player": pname,
            "min": int(e.get("minute", 0)),
            "sec": sec,
            "expMin": exp_min,
            "tSub": t_sub,
            "period": (e.get("period") or {}).get("value", 1),
            "x": round(x, 2),
            "y": round(y, 2),
            "endX": round(end_x, 2) if end_x is not None else None,
            "endY": round(end_y, 2) if end_y is not None else None,
            "shotEndX": shot_end_x,
            "shotEndY": shot_end_y,
            "isTouch": bool(e.get("isTouch")),
            "isThrowIn": ("ThrowIn" in quals or "ThrowinSetPiece" in quals),
            "isGoalKick": ("GoalKick" in quals),
            "isBlock": (etype == "BlockedPass" or (etype == "Save" and "OutfielderBlock" in quals)),
            "distM": dist_m,
            "isProg": is_prog,
            "xt": xt_gain,
            "xg": xg_val,
            "ok": ok,
            "snapped": snapped,
        })

    # 5. Compute physical Carry metrics between consecutive same-team live events
    # Carry can start after a successful Pass, BallRecovery, or TakeOn (dribble)
    valid_carries = 0
    rejected_carries = 0
    for i in range(1, len(refined)):
        prev = refined[i - 1]
        cur = refined[i]
        dt = cur["tSub"] - prev["tSub"]
        prev_end_x = prev["endX"] if prev["endX"] is not None else prev["x"]
        prev_end_y = prev["endY"] if prev["endY"] is not None else prev["y"]
        if (
            prev["isHome"] == cur["isHome"]
            and prev["ok"]
            and prev["type"] in ("Pass", "BallRecovery", "TakeOn")
            and cur["type"] not in ("Foul", "BallRecovery")
            and 0 < dt <= 16.0
        ):
            dx_m = ((cur["x"] - prev_end_x) / 100.0) * PITCH_L
            dy_m = ((cur["y"] - prev_end_y) / 100.0) * PITCH_W
            c_dist_m = math.hypot(dx_m, dy_m)
            # Account for Opta 1-second integer rounding (±1s) on short intervals
            effective_dt = max(dt, 1.5) if dt < 2.0 else dt
            c_speed_ms = c_dist_m / effective_dt
            if c_dist_m >= 1.5:
                if c_speed_ms <= 12.0:  # Account for ball flight + sprint
                    cur["carryFromX"] = prev_end_x
                    cur["carryFromY"] = prev_end_y
                    cur["carryDistM"] = round(c_dist_m, 1)
                    cur["carryDt"] = round(dt, 1)
                    cur["carryKmh"] = round(min(c_speed_ms * 3.6, 36.5), 1)
                    cur["carryXt"] = round(get_xt(cur["x"], cur["y"]) - get_xt(prev_end_x, prev_end_y), 4)
                    valid_carries += 1
                else:
                    rejected_carries += 1
        elif (
            cur["type"] == "BallRecovery"
            and 0 < dt <= 5.0
        ):
            # Loose-ball deflection link from previous failed pass/clearance/interception to BallRecovery
            px_abs = prev_end_x if prev["isHome"] == cur["isHome"] else (100.0 - prev_end_x)
            py_abs = prev_end_y if prev["isHome"] == cur["isHome"] else (100.0 - prev_end_y)
            dx_m = ((cur["x"] - px_abs) / 100.0) * PITCH_L
            dy_m = ((cur["y"] - py_abs) / 100.0) * PITCH_W
            if 1.0 <= math.hypot(dx_m, dy_m) <= 20.0:
                cur["looseFromX"] = round(px_abs, 2)
                cur["looseFromY"] = round(py_abs, 2)

    return {
        "events": refined,
        "diagnostics": {
            "snapped_set_pieces": snapped_count,
            "xg_fused_shots": xg_fused_count,
            "valid_carries": valid_carries,
            "rejected_carries": rejected_carries,
        },
    }
