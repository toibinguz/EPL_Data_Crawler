"""
mapped_epl/sync_2627_to_clean_db.py
Incrementally integrates ALL completed EPL 2026/27 matches from:
  - mapped_epl/whoscored_2627/*_matchCentreData.json (110 matches)
  - 2026_2027_new/football_data_raw_2627.csv (110 matches + odds)
  - 2026_2027_new/matches/fotmob/*.json (110 matches + FotMob xG/npxG/stats)
into mapped_epl/epl_clean.db WITHOUT altering historical seasons (< 2026/27) or clean/.
"""

import json
import math
import sqlite3
import sys
from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.append(str(ROOT / "mapped_epl"))
from opta_spatial_refiner import PITCH_L, PITCH_W  # noqa: E402

CLEAN_DB = ROOT / "mapped_epl" / "epl_clean.db"
WS_DIR = ROOT / "mapped_epl" / "whoscored_2627"
FD_CSV = ROOT / "2026_2027_new" / "football_data_raw_2627.csv"
FM_DIR = ROOT / "2026_2027_new" / "matches" / "fotmob"

WS_TO_CODE = {
    "Arsenal": "ARS",
    "Aston Villa": "AVL",
    "Bournemouth": "BOU",
    "Brentford": "BRE",
    "Brighton": "BHA",
    "Chelsea": "CHE",
    "Coventry": "COV",
    "Crystal Palace": "CRY",
    "Everton": "EVE",
    "Fulham": "FUL",
    "Hull": "HUL",
    "Ipswich": "IPS",
    "Leeds": "LEE",
    "Liverpool": "LIV",
    "Man City": "MCI",
    "Man Utd": "MUN",
    "Newcastle": "NEW",
    "Nottingham Forest": "NFO",
    "Sunderland": "SUN",
    "Tottenham": "TOT",
}

FD_TO_CODE = {
    "Arsenal": "ARS",
    "Aston Villa": "AVL",
    "Bournemouth": "BOU",
    "Brentford": "BRE",
    "Brighton": "BHA",
    "Chelsea": "CHE",
    "Coventry": "COV",
    "Crystal Palace": "CRY",
    "Everton": "EVE",
    "Fulham": "FUL",
    "Hull": "HUL",
    "Ipswich": "IPS",
    "Leeds": "LEE",
    "Liverpool": "LIV",
    "Man City": "MCI",
    "Man United": "MUN",
    "Newcastle": "NEW",
    "Nott'm Forest": "NFO",
    "Sunderland": "SUN",
    "Tottenham": "TOT",
}


def parse_fd_date(s: str) -> str:
    return pd.to_datetime(s, dayfirst=True).strftime("%Y-%m-%d")


def parse_fotmob_file(path: Path) -> dict | None:
    d = json.loads(path.read_text(encoding="utf-8"))
    gen = d.get("general") or {}
    if not (gen.get("started") and gen.get("finished")):
        return None
    dt_str = (gen.get("matchTimeUTCDate") or "")[:10]
    ht_id = (gen.get("homeTeam") or {}).get("id")
    at_id = (gen.get("awayTeam") or {}).get("id")

    stats_all = (((d.get("content") or {}).get("stats") or {}).get("Periods") or {}).get("All", {}).get("stats", [])
    flat = {}
    for grp in stats_all:
        for it in grp.get("stats", []):
            k = it.get("key")
            vals = it.get("stats")
            if k and isinstance(vals, list) and len(vals) == 2 and vals[0] is not None:
                flat[k] = vals

    def get_pair(k):
        vals = flat.get(k)
        if not vals:
            return None, None
        def to_f(v):
            if v is None:
                return None
            try:
                return float(str(v).split()[0].replace("%", ""))
            except Exception:
                return None
        return to_f(vals[0]), to_f(vals[1])

    xg_h, xg_a = get_pair("expected_goals")
    npxg_h, npxg_a = get_pair("expected_goals_non_penalty")
    xgot_h, xgot_a = get_pair("expected_goals_on_target")
    poss_h, poss_a = get_pair("BallPossesion")
    box_h, box_a = get_pair("touches_in_opp_box")

    return {
        "date": dt_str,
        "home_fm_id": int(ht_id) if ht_id is not None else None,
        "away_fm_id": int(at_id) if at_id is not None else None,
        "fm_xg": (xg_h, xg_a),
        "fm_npxg": (npxg_h, npxg_a),
        "fm_xgot": (xgot_h, xgot_a),
        "fm_poss": (poss_h, poss_a),
        "fm_opp_box": (box_h, box_a),
    }


def devig_three(h, d, a):
    if pd.isna(h) or pd.isna(d) or pd.isna(a) or h <= 1.0 or d <= 1.0 or a <= 1.0:
        return None, None, None
    ih, id_, ia = 1.0 / float(h), 1.0 / float(d), 1.0 / float(a)
    s = ih + id_ + ia
    return round(ih / s, 6), round(id_ / s, 6), round(ia / s, 6)


def count_prog_passes(events: list, home_tid: int, is_home: bool) -> int:
    cnt = 0
    for e in events:
        if (e.get("teamId") == home_tid) != is_home:
            continue
        if (e.get("type") or {}).get("displayName") != "Pass":
            continue
        if (e.get("outcomeType") or {}).get("value") != 1:
            continue
        quals = {(q.get("type") or {}).get("displayName") for q in e.get("qualifiers", [])}
        if "ThrowIn" in quals or "ThrowinSetPiece" in quals:
            continue
        ex, ey = e.get("endX"), e.get("endY")
        if ex is None or ey is None:
            continue
        xm, ym = (float(e.get("x", 0.0)) / 100.0) * PITCH_L, (float(e.get("y", 0.0)) / 100.0) * PITCH_W
        exm, eym = (float(ex) / 100.0) * PITCH_L, (float(ey) / 100.0) * PITCH_W
        d0 = math.hypot(PITCH_L - xm, 34.0 - ym)
        d1 = math.hypot(PITCH_L - exm, 34.0 - eym)
        if (d0 > 0 and (d0 - d1) / d0 >= 0.25) or (exm - xm >= 10.0 and exm >= 40.0):
            cnt += 1
    return cnt


def main():
    conn = sqlite3.connect(str(CLEAN_DB))
    dim_club = pd.read_sql_query("SELECT club_id, fotmob_team_id FROM dim_club WHERE fotmob_team_id IS NOT NULL", conn)
    fm_id_to_code = {int(r["fotmob_team_id"]): r["club_id"] for _, r in dim_club.iterrows()}

    # Preserve existing 26/27 understat_xg / kickoff / etc. if present
    old_m = pd.read_sql_query("SELECT match_id, kickoff_time FROM clean_matches WHERE season='2026/27'", conn)
    old_kickoff = dict(zip(old_m["match_id"], old_m["kickoff_time"]))
    old_ts = pd.read_sql_query("SELECT match_id, club_id, understat_xg FROM clean_match_team_stats WHERE season='2026/27'", conn)
    old_us_xg = {(r["match_id"], r["club_id"]): r["understat_xg"] for _, r in old_ts.iterrows()}

    # Load FotMob 26/27 matches keyed by (home_code, away_code)
    fm_map = {}
    for jf in sorted(FM_DIR.glob("*.json")):
        parsed = parse_fotmob_file(jf)
        if not parsed:
            continue
        hc = fm_id_to_code.get(parsed["home_fm_id"])
        ac = fm_id_to_code.get(parsed["away_fm_id"])
        if hc and ac:
            fm_map[(hc, ac)] = parsed

    # Load Football-Data 26/27 CSV keyed by (home_code, away_code)
    df_fd = pd.read_csv(FD_CSV)
    fd_map = {}
    for _, r in df_fd.iterrows():
        hc = FD_TO_CODE.get(r["HomeTeam"])
        ac = FD_TO_CODE.get(r["AwayTeam"])
        if hc and ac:
            fd_map[(hc, ac)] = r

    ws_files = sorted(WS_DIR.glob("*_matchCentreData.json"))
    match_rows = []
    odds_rows = []
    ts_rows = []

    for jf in ws_files:
        mc = json.loads(jf.read_text(encoding="utf-8"))
        date_str = jf.name.split("_")[0]
        hc = WS_TO_CODE[mc["home"]["name"]]
        ac = WS_TO_CODE[mc["away"]["name"]]
        mid = f"EPL_2026_2027_{hc}_{ac}"

        sc = (mc.get("ftScore") or mc.get("score") or "0 : 0").split(":")
        hg, ag = int(sc[0].strip()), int(sc[1].strip())
        res = "H" if hg > ag else ("A" if ag > hg else "D")

        ht_sc = (mc.get("htScore") or "").split(":")
        ht_hg = float(ht_sc[0].strip()) if len(ht_sc) == 2 and ht_sc[0].strip().isdigit() else None
        ht_ag = float(ht_sc[1].strip()) if len(ht_sc) == 2 and ht_sc[1].strip().isdigit() else None
        ht_res = ("H" if ht_hg > ht_ag else ("A" if ht_ag > ht_hg else "D")) if (ht_hg is not None and ht_ag is not None) else None

        fd_r = fd_map.get((hc, ac))
        kickoff = str(fd_r["Time"]) if fd_r is not None and pd.notna(fd_r.get("Time")) else old_kickoff.get(mid)
        ref_name = (mc.get("referee") or {}).get("name") or (str(fd_r["Referee"]) if fd_r is not None and pd.notna(fd_r.get("Referee")) else None)
        h_form = ((mc["home"].get("formations") or [{}])[0]).get("formationName")
        a_form = ((mc["away"].get("formations") or [{}])[0]).get("formationName")
        if h_form and str(h_form).isdigit():
            h_form = "-".join(list(str(h_form)))
        if a_form and str(a_form).isdigit():
            a_form = "-".join(list(str(a_form)))

        match_rows.append({
            "match_id": mid,
            "season": "2026/27",
            "date": date_str,
            "kickoff_time": kickoff,
            "home_club_id": hc,
            "away_club_id": ac,
            "home_goals": hg,
            "away_goals": ag,
            "result": res,
            "ht_home_goals": ht_hg,
            "ht_away_goals": ht_ag,
            "ht_result": ht_res,
            "referee_canonical": ref_name,
            "attendance": mc.get("attendance"),
            "stadium": mc.get("venueName"),
            "home_manager": mc["home"].get("managerName"),
            "away_manager": mc["away"].get("managerName"),
            "home_formation": h_form,
            "away_formation": a_form,
            "status": "played",
            "primary_source": "whoscored_sync",
            "attendance_imputed": 0,
        })

        # Odds from Football-Data
        if fd_r is not None:
            # Prefer closing avg (AvgCH/CD/CA) or opening avg (AvgH/D/A), fallback B365
            oh = fd_r.get("AvgCH") if pd.notna(fd_r.get("AvgCH")) else fd_r.get("AvgH")
            od = fd_r.get("AvgCD") if pd.notna(fd_r.get("AvgCD")) else fd_r.get("AvgD")
            oa = fd_r.get("AvgCA") if pd.notna(fd_r.get("AvgCA")) else fd_r.get("AvgA")
            if pd.isna(oh):
                oh, od, oa = fd_r.get("B365H"), fd_r.get("B365D"), fd_r.get("B365A")
            ph, pd_, pa = devig_three(oh, od, oa)
            odds_rows.append({
                "match_id": mid,
                "season": "2026/27",
                "b365_home": float(fd_r["B365H"]) if pd.notna(fd_r.get("B365H")) else None,
                "b365_draw": float(fd_r["B365D"]) if pd.notna(fd_r.get("B365D")) else None,
                "b365_away": float(fd_r["B365A"]) if pd.notna(fd_r.get("B365A")) else None,
                "ps_home": float(fd_r["PSH"]) if pd.notna(fd_r.get("PSH")) else None,
                "ps_draw": float(fd_r["PSD"]) if pd.notna(fd_r.get("PSD")) else None,
                "ps_away": float(fd_r["PSA"]) if pd.notna(fd_r.get("PSA")) else None,
                "avg_home": float(fd_r["AvgH"]) if pd.notna(fd_r.get("AvgH")) else None,
                "avg_draw": float(fd_r["AvgD"]) if pd.notna(fd_r.get("AvgD")) else None,
                "avg_away": float(fd_r["AvgA"]) if pd.notna(fd_r.get("AvgA")) else None,
                "max_home": float(fd_r["MaxH"]) if pd.notna(fd_r.get("MaxH")) else None,
                "max_draw": float(fd_r["MaxD"]) if pd.notna(fd_r.get("MaxD")) else None,
                "max_away": float(fd_r["MaxA"]) if pd.notna(fd_r.get("MaxA")) else None,
                "prob_home": ph,
                "prob_draw": pd_,
                "prob_away": pa,
            })

        # Team stats (WhoScored + FotMob + FD)
        fm_r = fm_map.get((hc, ac)) or {}
        home_tid = mc["home"]["teamId"]
        evs = mc.get("events", [])

        for is_h, cid, side_obj, goals in [(1, hc, mc["home"], hg), (0, ac, mc["away"], ag)]:
            st = side_obj.get("stats") or {}
            def s_sum(k):
                o = st.get(k)
                return float(sum(v for v in o.values() if isinstance(v, (int, float)))) if isinstance(o, dict) else None

            idx_side = 0 if is_h else 1
            shots_ws = s_sum("shotsTotal")
            sot_ws = s_sum("shotsOnTarget")
            corners_ws = s_sum("cornersTotal")
            fouls_ws = s_sum("foulsTotal") or (float(fd_r["HF" if is_h else "AF"]) if fd_r is not None and pd.notna(fd_r.get("HF" if is_h else "AF")) else None)
            prog_p = count_prog_passes(evs, home_tid, bool(is_h))

            ts_rows.append({
                "match_id": mid,
                "club_id": cid,
                "season": "2026/27",
                "is_home": is_h,
                "goals": goals,
                "shots": shots_ws,
                "shots_on_target": sot_ws,
                "corners": corners_ws,
                "fouls": fouls_ws,
                "yellow_cards": float(fd_r["HY" if is_h else "AY"]) if fd_r is not None and pd.notna(fd_r.get("HY" if is_h else "AY")) else None,
                "red_cards": float(fd_r["HR" if is_h else "AR"]) if fd_r is not None and pd.notna(fd_r.get("HR" if is_h else "AR")) else None,
                "understat_xg": old_us_xg.get((mid, cid)),
                "fm_possession": (fm_r.get("fm_poss") or (None, None))[idx_side],
                "fm_xg": (fm_r.get("fm_xg") or (None, None))[idx_side],
                "fm_npxg": (fm_r.get("fm_npxg") or (None, None))[idx_side],
                "fm_xgot": (fm_r.get("fm_xgot") or (None, None))[idx_side],
                "fm_big_chances": None,
                "fm_big_chances_missed": None,
                "fm_shots_inside_box": None,
                "fm_shots_outside_box": None,
                "fm_passes": s_sum("passesTotal"),
                "fm_accurate_passes": s_sum("passesAccurate"),
                "fm_pass_accuracy": None,
                "fm_touches_opp_box": (fm_r.get("fm_opp_box") or (None, None))[idx_side],
                "fm_tackles": s_sum("tacklesTotal"),
                "fm_interceptions": s_sum("interceptions"),
                "fm_clearances": s_sum("clearances"),
                "fm_duels_won": s_sum("aerialsWon"),
                "fm_saves": None,
                "fb_touches": s_sum("touches"),
                "fb_sca": None,
                "fb_prog_passes": prog_p,
            })

    # Replace ONLY season = '2026/27' in clean_matches, clean_match_odds, clean_match_team_stats
    with conn:
        for tbl in ("clean_matches", "clean_match_odds", "clean_match_team_stats"):
            conn.execute(f"DELETE FROM {tbl} WHERE season = ?", ("2026/27",))
        pd.DataFrame(match_rows).to_sql("clean_matches", conn, if_exists="append", index=False)
        pd.DataFrame(odds_rows).to_sql("clean_match_odds", conn, if_exists="append", index=False)
        pd.DataFrame(ts_rows).to_sql("clean_match_team_stats", conn, if_exists="append", index=False)

    # Verification
    n_m = conn.execute("SELECT COUNT(*) FROM clean_matches WHERE season='2026/27'").fetchone()[0]
    n_o = conn.execute("SELECT COUNT(*) FROM clean_match_odds WHERE season='2026/27' AND prob_home IS NOT NULL").fetchone()[0]
    n_ts = conn.execute("SELECT COUNT(*) FROM clean_match_team_stats WHERE season='2026/27'").fetchone()[0]
    n_25 = conn.execute("SELECT COUNT(*) FROM clean_matches WHERE season='2025/26'").fetchone()[0]
    conn.close()
    print(f"[+] Updated epl_clean.db 2026/27: clean_matches={n_m}, clean_match_odds={n_o}, clean_match_team_stats={n_ts} (2025/26 untouched={n_25})")


if __name__ == "__main__":
    main()
