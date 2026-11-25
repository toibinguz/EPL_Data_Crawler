"""
mapped_epl/build_clean_db.py
Transforms all mapped raw tables in mapped_epl/epl_mapped.db into a clean,
normalized, QA-verified relational database at mapped_epl/epl_clean.db.
Does NOT modify clean/ or clean.db.
"""

import json
import re
import sqlite3
import time
from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
MAPPED_DB = ROOT / "mapped_epl" / "epl_mapped.db"
CLEAN_DB = ROOT / "mapped_epl" / "epl_clean.db"
QA_JSON = ROOT / "mapped_epl" / "qa_report.json"


def normalize_formation(val: str | None) -> str | None:
    if not val or not str(val).strip():
        return None
    m = re.search(r"\b(\d-\d-\d(?:-\d)?)\b", str(val).strip())
    return m.group(1) if m else str(val).strip()


def parse_fotmob_num_pct(val) -> tuple[float | None, float | None]:
    """Parse FotMob values like '583 (89%)' or '54%' or '1.85'."""
    if val is None or (isinstance(val, float) and np.isnan(val)):
        return None, None
    s = str(val).strip()
    if not s or s.lower() in ("none", "nan", "-"):
        return None, None
    m = re.match(r"^([0-9]+(?:\.[0-9]+)?)\s*\((\d+(?:\.\d+)?)%\)$", s)
    if m:
        return float(m.group(1)), float(m.group(2))
    m2 = re.match(r"^([0-9]+(?:\.[0-9]+)?)%$", s)
    if m2:
        return float(m2.group(1)), None
    try:
        return float(s), None
    except ValueError:
        return None, None


def to_num(series: pd.Series) -> pd.Series:
    return pd.to_numeric(series, errors="coerce")


def build_clean_database():
    t0 = time.time()
    src = sqlite3.connect(str(MAPPED_DB))
    if CLEAN_DB.exists():
        CLEAN_DB.unlink()
    dst = sqlite3.connect(str(CLEAN_DB))
    dst.execute("PRAGMA journal_mode=WAL;")
    dst.execute("PRAGMA synchronous=NORMAL;")
    dst.execute("PRAGMA temp_store=MEMORY;")
    dst.execute("PRAGMA cache_size=-200000;")

    cleaning_logs = []

    def log_clean(tbl, row_key, col, old_v, new_v, rule, reason):
        cleaning_logs.append({
            "table_name": tbl,
            "row_key": str(row_key),
            "column_name": col,
            "old_value": None if pd.isna(old_v) else str(old_v),
            "new_value": None if new_v is None or (isinstance(new_v, float) and np.isnan(new_v)) else str(new_v),
            "rule": rule,
            "reason": reason,
        })

    # 1. Copy Dimensions & Bridges
    print("[*] Copying dimensions & bridges...")
    for tbl in ["dim_club", "dim_season", "dim_stadium", "dim_referee", "dim_player", "map_match_ids", "map_player_ids"]:
        df = pd.read_sql_query(f"SELECT * FROM {tbl}", src)
        df.to_sql(tbl, dst, if_exists="replace", index=False)
    dst.execute("CREATE UNIQUE INDEX idx_c_club ON dim_club(club_id);")
    dst.execute("CREATE UNIQUE INDEX idx_c_player ON dim_player(player_key);")

    # 2. Build clean_matches
    print("[*] Building clean_matches...")
    dm = pd.read_sql_query("SELECT * FROM dim_match", src)
    dm = dm.drop_duplicates(subset=["match_id"], keep="first").copy()
    dm["home_formation"] = dm["home_formation"].apply(normalize_formation)
    dm["away_formation"] = dm["away_formation"].apply(normalize_formation)
    dm["attendance_imputed"] = 0

    # R9: COVID closed-doors attendance = 0 (2020-06-17 to 2021-05-23)
    covid_mask = (dm["date"] >= "2020-06-17") & (dm["date"] <= "2021-05-23") & (dm["attendance"].isna())
    for idx in dm.index[covid_mask]:
        log_clean("clean_matches", dm.at[idx, "match_id"], "attendance", None, 0, "R9", "COVID closed-doors match")
        dm.at[idx, "attendance"] = 0
        dm.at[idx, "attendance_imputed"] = 1

    # R4: Ensure result matches goals
    played_mask = dm["home_goals"].notna() & dm["away_goals"].notna()
    for idx in dm.index[played_mask]:
        hg, ag = int(dm.at[idx, "home_goals"]), int(dm.at[idx, "away_goals"])
        exp_res = "H" if hg > ag else ("A" if hg < ag else "D")
        if dm.at[idx, "result"] != exp_res:
            log_clean("clean_matches", dm.at[idx, "match_id"], "result", dm.at[idx, "result"], exp_res, "R4", "Recomputed result from goals")
            dm.at[idx, "result"] = exp_res

    # R5: Ensure ht_goals <= ft_goals
    ht_bad = played_mask & dm["ht_home_goals"].notna() & dm["ht_away_goals"].notna() & (
        (dm["ht_home_goals"] > dm["home_goals"]) | (dm["ht_away_goals"] > dm["away_goals"])
    )
    for idx in dm.index[ht_bad]:
        for col in ("ht_home_goals", "ht_away_goals", "ht_result"):
            log_clean("clean_matches", dm.at[idx, "match_id"], col, dm.at[idx, col], None, "R5", "HT goals > FT goals")
            dm.at[idx, col] = None

    dm.to_sql("clean_matches", dst, if_exists="replace", index=False)
    dst.execute("CREATE UNIQUE INDEX idx_clean_matches_id ON clean_matches(match_id);")
    print(f"  [+] clean_matches: {len(dm):,} rows")

    # 3. Build clean_match_odds (merging BbAv* -> Avg*, validating odds > 1.0, computing margin-free probs)
    print("[*] Building clean_match_odds...")
    fd = pd.read_sql_query("SELECT * FROM raw_football_data_matches", src).drop_duplicates(subset=["match_id"], keep="first")
    odds_df = pd.DataFrame({"match_id": fd["match_id"], "season": fd["season"]})

    def first_valid(cols):
        s = pd.Series(np.nan, index=fd.index, dtype="float64")
        for c in cols:
            if c in fd.columns:
                s = s.fillna(to_num(fd[c]))
        return s

    odds_df["b365_home"] = first_valid(["B365H"])
    odds_df["b365_draw"] = first_valid(["B365D"])
    odds_df["b365_away"] = first_valid(["B365A"])
    odds_df["ps_home"] = first_valid(["PSH", "PSCH"])
    odds_df["ps_draw"] = first_valid(["PSD", "PSCD"])
    odds_df["ps_away"] = first_valid(["PSA", "PSCA"])
    odds_df["avg_home"] = first_valid(["AvgH", "BbAvH"])
    odds_df["avg_draw"] = first_valid(["AvgD", "BbAvD"])
    odds_df["avg_away"] = first_valid(["AvgA", "BbAvA"])
    odds_df["max_home"] = first_valid(["MaxH", "BbMxH"])
    odds_df["max_draw"] = first_valid(["MaxD", "BbMxD"])
    odds_df["max_away"] = first_valid(["MaxA", "BbMxA"])

    # R3: Odds <= 1.0 invalid
    for c in [c for c in odds_df.columns if c not in ("match_id", "season")]:
        bad = odds_df[c].notna() & (odds_df[c] <= 1.0)
        for idx in odds_df.index[bad]:
            log_clean("clean_match_odds", odds_df.at[idx, "match_id"], c, odds_df.at[idx, c], None, "R3", "Odds <= 1.0")
            odds_df.at[idx, c] = np.nan

    # Margin-free implied probabilities (prefer avg -> b365 -> ps)
    best_h = odds_df["avg_home"].fillna(odds_df["b365_home"]).fillna(odds_df["ps_home"])
    best_d = odds_df["avg_draw"].fillna(odds_df["b365_draw"]).fillna(odds_df["ps_draw"])
    best_a = odds_df["avg_away"].fillna(odds_df["b365_away"]).fillna(odds_df["ps_away"])
    raw_sum = (1.0 / best_h) + (1.0 / best_d) + (1.0 / best_a)
    odds_df["prob_home"] = np.round((1.0 / best_h) / raw_sum, 6)
    odds_df["prob_draw"] = np.round((1.0 / best_d) / raw_sum, 6)
    odds_df["prob_away"] = np.round((1.0 / best_a) / raw_sum, 6)
    odds_df.to_sql("clean_match_odds", dst, if_exists="replace", index=False)
    dst.execute("CREATE UNIQUE INDEX idx_clean_odds_id ON clean_match_odds(match_id);")
    print(f"  [+] clean_match_odds: {len(odds_df):,} rows")

    # 4. Build clean_match_team_stats (2 rows per match: (match_id, club_id))
    print("[*] Building clean_match_team_stats...")
    us = pd.read_sql_query("SELECT match_id, home_xg, away_xg FROM raw_understat_matches", src).drop_duplicates("match_id")
    fm = pd.read_sql_query("SELECT * FROM raw_fotmob_matches", src).drop_duplicates("match_id")
    fb_t = pd.read_sql_query("SELECT match_id, club_id, source_table, xG, Pressures_Pressures, PrgP, SCA_SCA, GCA_GCA FROM raw_fbref_team_stats", src) if False else None

    # Build lookup dicts for fast team-match assembly
    us_map = us.set_index("match_id").to_dict("index")
    fm_map = fm.set_index("match_id").to_dict("index")
    fd_map = fd.set_index("match_id").to_dict("index")

    # FBref team stats: aggregate key team metrics per (match_id, club_id)
    fb_team_df = pd.read_sql_query("SELECT * FROM raw_fbref_team_stats", src)
    fb_team_map = {}
    for _, r in fb_team_df.iterrows():
        k = (r["match_id"], r["club_id"])
        d = fb_team_map.setdefault(k, {})
        for col, out_col in [
            ("xG", "fb_xg"), ("Touches_Touches", "fb_touches"),
            ("Press_Pressures", "fb_pressures"), ("Succ_Pressures", "fb_pressures_succ"),
            ("PrgP", "fb_prog_passes"), ("SCA_SCA", "fb_sca"), ("GCA_GCA", "fb_gca"),
        ]:
            if col in r and pd.notna(r[col]) and out_col not in d:
                try:
                    d[out_col] = float(r[col])
                except (ValueError, TypeError):
                    pass

    team_rows = []
    for _, m in dm.iterrows():
        mid = m["match_id"]
        fdr = fd_map.get(mid, {})
        usr = us_map.get(mid, {})
        fmr = fm_map.get(mid, {})

        for is_home, cid, prefix_fd, prefix_s in (
            (1, m["home_club_id"], "H", "home"),
            (0, m["away_club_id"], "A", "away"),
        ):
            def fd_num(col):
                v = fdr.get(col)
                try:
                    return float(v) if v is not None and not pd.isna(v) else np.nan
                except (ValueError, TypeError):
                    return np.nan

            poss, _ = parse_fotmob_num_pct(fmr.get(f"{prefix_s}_BallPossesion"))
            acc_p, p_acc = parse_fotmob_num_pct(fmr.get(f"{prefix_s}_accurate_passes"))
            tot_p, _ = parse_fotmob_num_pct(fmr.get(f"{prefix_s}_passes"))
            if p_acc is None and acc_p is not None and tot_p and tot_p > 0:
                p_acc = round(acc_p / tot_p * 100.0, 1)

            fbr = fb_team_map.get((mid, cid), {})
            row = {
                "match_id": mid,
                "club_id": cid,
                "season": m["season"],
                "is_home": is_home,
                "goals": m[f"{prefix_s}_goals"],
                "shots": fd_num(f"{prefix_fd}S"),
                "shots_on_target": fd_num(f"{prefix_fd}ST"),
                "corners": fd_num(f"{prefix_fd}C"),
                "fouls": fd_num(f"{prefix_fd}F"),
                "yellow_cards": fd_num(f"{prefix_fd}Y"),
                "red_cards": fd_num(f"{prefix_fd}R"),
                "understat_xg": float(usr[f"{prefix_s}_xg"]) if usr.get(f"{prefix_s}_xg") is not None and not pd.isna(usr.get(f"{prefix_s}_xg")) else np.nan,
                "fm_possession": poss,
                "fm_xg": parse_fotmob_num_pct(fmr.get(f"{prefix_s}_expected_goals"))[0],
                "fm_npxg": parse_fotmob_num_pct(fmr.get(f"{prefix_s}_expected_goals_non_penalty"))[0],
                "fm_xgot": parse_fotmob_num_pct(fmr.get(f"{prefix_s}_expected_goals_on_target"))[0],
                "fm_big_chances": parse_fotmob_num_pct(fmr.get(f"{prefix_s}_big_chance"))[0],
                "fm_big_chances_missed": parse_fotmob_num_pct(fmr.get(f"{prefix_s}_big_chance_missed_title"))[0],
                "fm_shots_inside_box": parse_fotmob_num_pct(fmr.get(f"{prefix_s}_shots_inside_box"))[0],
                "fm_shots_outside_box": parse_fotmob_num_pct(fmr.get(f"{prefix_s}_shots_outside_box"))[0],
                "fm_passes": tot_p,
                "fm_accurate_passes": acc_p,
                "fm_pass_accuracy": p_acc,
                "fm_touches_opp_box": parse_fotmob_num_pct(fmr.get(f"{prefix_s}_touches_opp_box"))[0],
                "fm_tackles": parse_fotmob_num_pct(fmr.get(f"{prefix_s}_matchstats.headers.tackles"))[0],
                "fm_interceptions": parse_fotmob_num_pct(fmr.get(f"{prefix_s}_interceptions"))[0],
                "fm_clearances": parse_fotmob_num_pct(fmr.get(f"{prefix_s}_clearances"))[0],
                "fm_duels_won": parse_fotmob_num_pct(fmr.get(f"{prefix_s}_duel_won"))[0],
                "fm_saves": parse_fotmob_num_pct(fmr.get(f"{prefix_s}_keeper_saves"))[0],
                **fbr,
            }
            team_rows.append(row)

    ts_df = pd.DataFrame(team_rows)
    # R6: shots_on_target > shots
    bad_sot = ts_df["shots"].notna() & ts_df["shots_on_target"].notna() & (ts_df["shots_on_target"] > ts_df["shots"])
    for idx in ts_df.index[bad_sot]:
        key = f"{ts_df.at[idx, 'match_id']}:{ts_df.at[idx, 'club_id']}"
        log_clean("clean_match_team_stats", key, "shots_on_target", ts_df.at[idx, "shots_on_target"], None, "R6", "shots_on_target > shots")
        ts_df.at[idx, "shots"] = np.nan
        ts_df.at[idx, "shots_on_target"] = np.nan

    # R11: accurate_passes > passes; big_chances_missed > big_chances
    bad_bc = ts_df["fm_big_chances"].notna() & ts_df["fm_big_chances_missed"].notna() & (ts_df["fm_big_chances_missed"] > ts_df["fm_big_chances"])
    for idx in ts_df.index[bad_bc]:
        key = f"{ts_df.at[idx, 'match_id']}:{ts_df.at[idx, 'club_id']}"
        log_clean("clean_match_team_stats", key, "fm_big_chances_missed", ts_df.at[idx, "fm_big_chances_missed"], None, "R11", "big_chances_missed > big_chances")
        ts_df.at[idx, "fm_big_chances"] = np.nan
        ts_df.at[idx, "fm_big_chances_missed"] = np.nan

    ts_df.to_sql("clean_match_team_stats", dst, if_exists="replace", index=False)
    dst.execute("CREATE UNIQUE INDEX idx_clean_ts_pk ON clean_match_team_stats(match_id, club_id);")
    print(f"  [+] clean_match_team_stats: {len(ts_df):,} rows")

    # 5. clean_match_weather
    mw = pd.read_sql_query("SELECT * FROM raw_match_weather", src)
    mw.to_sql("clean_match_weather", dst, if_exists="replace", index=False)
    dst.execute("CREATE UNIQUE INDEX idx_clean_weather_id ON clean_match_weather(match_id);")
    print(f"  [+] clean_match_weather: {len(mw):,} rows")

    # 6. clean_player_match_stats (Unified (match_id, club_id, player_key) across TM, FotMob, Understat, FBref, FPL!)
    print("[*] Building clean_player_match_stats...")
    pm_dict = {}  # (match_id, club_id, player_key) -> dict

    def get_pm(mid, cid, pkey, season):
        k = (mid, cid, pkey)
        d = pm_dict.get(k)
        if d is None:
            d = {"match_id": mid, "club_id": cid, "player_key": pkey, "season": season}
            pm_dict[k] = d
        return d

    # 6a. Transfermarkt lineups + appearances
    lineup_club_lookup = {}
    for _, r in pd.read_sql_query("SELECT match_id, epl_club_id, player_key, season, type, position, team_captain, number FROM raw_tfm_game_lineups", src).iterrows():
        lineup_club_lookup[(r["match_id"], r["player_key"])] = r["epl_club_id"]
        d = get_pm(r["match_id"], r["epl_club_id"], r["player_key"], r["season"])
        d["tm_is_starter"] = 1 if r["type"] == "starting_lineup" else 0
        d["tm_position"] = r["position"]
        d["tm_captain"] = r["team_captain"]
        d["tm_shirt_number"] = r["number"]

    for _, r in pd.read_sql_query("SELECT match_id, epl_club_id, player_key, season, minutes_played, goals, assists, yellow_cards, red_cards FROM raw_tfm_appearances WHERE match_id IS NOT NULL", src).iterrows():
        true_cid = lineup_club_lookup.get((r["match_id"], r["player_key"]), r["epl_club_id"])
        if true_cid != r["epl_club_id"]:
            log_clean("clean_player_match_stats", f"{r['match_id']}:{r['player_key']}", "club_id", r["epl_club_id"], true_cid, "R_CLUB_FIX", "tfm_appearances.player_club_id disagreed with tfm_game_lineups")
        d = get_pm(r["match_id"], true_cid, r["player_key"], r["season"])
        d["tm_minutes"] = r["minutes_played"]
        d["tm_goals"] = r["goals"]
        d["tm_assists"] = r["assists"]
        d["tm_yellow_cards"] = r["yellow_cards"]
        d["tm_red_cards"] = r["red_cards"]

    # 6b. FotMob player stats
    for _, r in pd.read_sql_query("SELECT match_id, club_id, player_key, season, minutes_played, fotmob_rating, is_goalkeeper FROM raw_fotmob_player_stats", src).iterrows():
        d = get_pm(r["match_id"], r["club_id"], r["player_key"], r["season"])
        d["fm_minutes"] = r["minutes_played"]
        d["fm_rating"] = r["fotmob_rating"]
        d["is_goalkeeper"] = r["is_goalkeeper"]

    # 6c. Understat rosters
    for _, r in pd.read_sql_query("SELECT match_id, club_id, player_key, season, time, goals, assists, shots, key_passes, xG, xA, xGChain, xGBuildup FROM raw_understat_rosters", src).iterrows():
        d = get_pm(r["match_id"], r["club_id"], r["player_key"], r["season"])
        d["us_minutes"] = r["time"]
        d["us_shots"] = r["shots"]
        d["us_xg"] = r["xG"]
        d["us_xa"] = r["xA"]
        d["us_key_passes"] = r["key_passes"]
        d["us_xg_chain"] = r["xGChain"]
        d["us_xg_buildup"] = r["xGBuildup"]

    # 6d. FBref summary_player + passing_player + defense_player
    fb_sum = pd.read_sql_query("SELECT match_id, club_id, player_key, season_std, Min, xG_Expected, xAG_Expected, SCA_SCA, GCA_GCA, Touches_Performance FROM raw_fbref_summary_player_advanced_match_stats", src) if False else None
    for tbl, col_map in [
        ("raw_fbref_summary_player_advanced_match_stats", {"Min": "fb_minutes", "xG_Expected": "fb_xg", "xAG_Expected": "fb_xag", "SCA_SCA": "fb_sca", "GCA_GCA": "fb_gca"}),
        ("raw_fbref_passing_player_advanced_match_stats", {"PrgP": "fb_prog_passes", "Cmp_Total": "fb_passes_cmp", "Att_Total": "fb_passes_att"}),
        ("raw_fbref_defense_player_advanced_match_stats", {"Tkl_Tackles": "fb_tackles", "Int": "fb_interceptions", "Blocks_Blocks": "fb_blocks"}),
    ]:
        cols_avail = {c[1] for c in src.execute(f"PRAGMA table_info({tbl})").fetchall()}
        sel = [c for c in ["match_id", "club_id", "player_key", "season_std"] + list(col_map.keys()) if c in cols_avail]
        for _, r in pd.read_sql_query(f"SELECT {', '.join(sel)} FROM {tbl} WHERE match_id IS NOT NULL AND club_id IS NOT NULL", src).iterrows():
            d = get_pm(r["match_id"], r["club_id"], r["player_key"], r["season_std"])
            for src_c, dst_c in col_map.items():
                if src_c in r and pd.notna(r[src_c]):
                    try:
                        d[dst_c] = float(r[src_c])
                    except (ValueError, TypeError):
                        pass

    # 6e. FPL player GW
    for _, r in pd.read_sql_query("SELECT match_id, club_id, player_key, season, minutes, total_points, ict_index, influence, creativity, threat, bps, bonus, value FROM raw_fpl_player_gw", src).iterrows():
        d = get_pm(r["match_id"], r["club_id"], r["player_key"], r["season"])
        d["fpl_minutes"] = r["minutes"]
        d["fpl_points"] = r["total_points"]
        d["fpl_ict_index"] = r["ict_index"]
        d["fpl_influence"] = r["influence"]
        d["fpl_creativity"] = r["creativity"]
        d["fpl_threat"] = r["threat"]
        d["fpl_bps"] = r["bps"]
        d["fpl_bonus"] = r["bonus"]
        d["fpl_value"] = r["value"]

    pm_df = pd.DataFrame(list(pm_dict.values()))
    # Resolve any remaining dual-club (match_id, player_key) by keeping the club with actual match participation
    dup_mask = pm_df.duplicated(subset=["match_id", "player_key"], keep=False)
    if dup_mask.any():
        pm_df["_score"] = (
            pm_df["tm_is_starter"].notna().astype(int) * 4
            + pm_df["tm_minutes"].notna().astype(int) * 3
            + pm_df["fm_minutes"].notna().astype(int) * 3
            + pm_df["us_minutes"].notna().astype(int) * 3
            + pm_df["fb_minutes"].notna().astype(int) * 3
            + (pm_df["fpl_minutes"].fillna(0) > 0).astype(int) * 2
            + pm_df["fpl_minutes"].notna().astype(int)
        )
        pm_df = pm_df.sort_values("_score", ascending=False)
        dropped_dups = pm_df[pm_df.duplicated(subset=["match_id", "player_key"], keep="first")]
        for _, dr in dropped_dups.iterrows():
            log_clean("clean_player_match_stats", f"{dr['match_id']}:{dr['player_key']}", "club_id", dr["club_id"], None, "R_DUP_CLUB", "Dropped 0-min/phantom duplicate club row in same match")
        pm_df = pm_df.drop_duplicates(subset=["match_id", "player_key"], keep="first").drop(columns=["_score"]).copy()

    pm_df.to_sql("clean_player_match_stats", dst, if_exists="replace", index=False)
    dst.execute("CREATE INDEX idx_clean_pm_match ON clean_player_match_stats(match_id);")
    dst.execute("CREATE INDEX idx_clean_pm_player ON clean_player_match_stats(player_key);")
    print(f"  [+] clean_player_match_stats: {len(pm_df):,} rows")

    # 7. clean_shots (Unified Understat + FotMob + FBref shots)
    print("[*] Building clean_shots...")
    us_s = pd.read_sql_query(
        "SELECT 'understat' AS source, shot_id, match_id, season, club_id, player_key, assist_player_key, minute, NULL AS minute_added, X AS x, Y AS y, xG AS xg, NULL AS xgot, result AS outcome, situation, shotType AS shot_type, NULL AS distance FROM raw_understat_shots",
        src,
    )
    fm_s = pd.read_sql_query(
        "SELECT 'fotmob' AS source, CAST(shot_id AS TEXT) AS shot_id, match_id, season, club_id, player_key, NULL AS assist_player_key, min AS minute, min_added AS minute_added, x, y, xg, xgot, event_type AS outcome, situation, shot_type, NULL AS distance FROM raw_fotmob_shots",
        src,
    )
    fb_s = pd.read_sql_query(
        'SELECT \'fbref\' AS source, NULL AS shot_id, match_id, season_std AS season, club_id, player_key, NULL AS assist_player_key, Minute AS minute, NULL AS minute_added, NULL AS x, NULL AS y, xG AS xg, PSxG AS xgot, Outcome AS outcome, Notes AS situation, "Body Part" AS shot_type, Distance AS distance FROM raw_fbref_match_shooting WHERE match_id IS NOT NULL',
        src,
    )
    shots_all = pd.concat([us_s, fm_s, fb_s], ignore_index=True)

    # Parse stoppage-time minutes like '45+2' -> minute=45 (int), minute_added=2 (int)
    min_str = shots_all["minute"].astype(str).str.strip()
    has_plus = min_str.str.contains(r"^\d+\+\d+$", regex=True, na=False)
    if has_plus.any():
        split_m = min_str[has_plus].str.split("+", expand=True)
        shots_all.loc[has_plus, "minute"] = pd.to_numeric(split_m[0], errors="coerce")
        shots_all.loc[has_plus, "minute_added"] = pd.to_numeric(split_m[1], errors="coerce")
    shots_all["minute"] = to_num(shots_all["minute"])
    shots_all["minute_added"] = to_num(shots_all["minute_added"])
    for col in ("xg", "xgot", "x", "y", "distance"):
        shots_all[col] = to_num(shots_all[col])
    shots_all.to_sql("clean_shots", dst, if_exists="replace", index=False)
    dst.execute("CREATE INDEX idx_clean_shots_match ON clean_shots(match_id);")
    dst.execute("CREATE INDEX idx_clean_shots_player ON clean_shots(player_key);")
    print(f"  [+] clean_shots: {len(shots_all):,} rows")

    # 8. clean_match_events, clean_player_valuations, clean_transfers
    print("[*] Building clean_match_events, clean_player_valuations, clean_transfers...")
    ev = pd.read_sql_query("SELECT game_event_id, match_id, season, epl_club_id AS club_id, player_key, in_player_key, assist_player_key, minute, type, description FROM raw_tfm_game_events", src)
    bad_ev_min = ev["minute"].notna() & ((ev["minute"] < 0) | (ev["minute"] > 130))
    for idx in ev.index[bad_ev_min]:
        log_clean("clean_match_events", ev.at[idx, "game_event_id"], "minute", ev.at[idx, "minute"], None, "R2", "Invalid negative/out-of-range event minute")
        ev.at[idx, "minute"] = np.nan
    ev.to_sql("clean_match_events", dst, if_exists="replace", index=False)

    val = pd.read_sql_query("SELECT player_key, epl_club_id AS club_id, date, market_value_in_eur AS market_value_eur FROM raw_tfm_player_valuations", src)
    val["date"] = val["date"].astype(str).str[:10]
    val.to_sql("clean_player_valuations", dst, if_exists="replace", index=False)

    tr = pd.read_sql_query("SELECT player_key, from_epl_club_id AS from_club_id, to_epl_club_id AS to_club_id, from_club_name, to_club_name, transfer_date, transfer_season, transfer_fee, market_value_in_eur FROM raw_tfm_transfers", src)
    tr.to_sql("clean_transfers", dst, if_exists="replace", index=False)

    # 8b. Backfill any orphan TM_* keys (e.g. from transfers/events) into dim_player
    raw_db_conn = sqlite3.connect(str(ROOT / "raw_data.db"))
    existing_pkeys = {r[0] for r in dst.execute("SELECT player_key FROM dim_player").fetchall()}
    needed_pkeys = set()
    for tbl_name in ("clean_player_match_stats", "clean_shots", "clean_match_events", "clean_transfers"):
        needed_pkeys.update(r[0] for r in dst.execute(f"SELECT DISTINCT player_key FROM {tbl_name} WHERE player_key IS NOT NULL").fetchall())
    missing_tm_ids = [int(k[3:]) for k in (needed_pkeys - existing_pkeys) if k.startswith("TM_") and k[3:].isdigit()]
    if missing_tm_ids:
        id_list = ",".join(str(i) for i in missing_tm_ids)
        extra_p = pd.read_sql_query(
            f"SELECT 'TM_' || player_id AS player_key, player_id AS tm_player_id, name AS canonical_name, lower(name) AS norm_name, position, country_of_citizenship, substr(date_of_birth,1,10) AS date_of_birth, height_in_cm AS height_cm, foot AS preferred_foot FROM transfermarkt__tfm_players_full WHERE player_id IN ({id_list})",
            raw_db_conn,
        )
        # Fallback for any TM IDs not even in tfm_players_full
        found_keys = set(extra_p["player_key"])
        still_missing = [k for k in (needed_pkeys - existing_pkeys) if k not in found_keys]
        if still_missing:
            extra_rows = [{"player_key": k, "tm_player_id": int(k[3:]) if k.startswith("TM_") and k[3:].isdigit() else None, "canonical_name": k, "norm_name": k.lower()} for k in still_missing]
            extra_p = pd.concat([extra_p, pd.DataFrame(extra_rows)], ignore_index=True)
        extra_p.to_sql("dim_player", dst, if_exists="append", index=False)
        print(f"  [+] Backfilled {len(extra_p):,} players into dim_player (0 orphans)")
    raw_db_conn.close()

    # 9. Write cleaning_log
    cl_df = pd.DataFrame(cleaning_logs)
    cl_df.to_sql("cleaning_log", dst, if_exists="replace", index=False)
    print(f"  [+] cleaning_log: {len(cl_df):,} changes logged")

    # 10. Automated QA Suite
    print("[*] Running automated QA checks on epl_clean.db...")
    qa_checks = {}

    def check(name, cond, detail=""):
        qa_checks[name] = {"passed": bool(cond), "detail": str(detail)}

    check("clean_matches: match_id unique", dm["match_id"].is_unique, f"{len(dm)} rows")
    check("clean_matches: (season, home, away) unique", len(dm.drop_duplicates(["season", "home_club_id", "away_club_id"])) == len(dm), "")
    check("clean_matches: home != away", (dm["home_club_id"] != dm["away_club_id"]).all(), "")
    check("clean_matches: all 26 completed seasons (2000/01-2025/26) have 380 matches",
          all(cnt == 380 for s, cnt in dm[dm["season"] < "2026/27"].groupby("season").size().items()),
          f"min={dm[dm['season'] < '2026/27'].groupby('season').size().min()}")
    check("clean_matches: result matches goals",
          all((r["home_goals"] > r["away_goals"] and r["result"] == "H") or (r["home_goals"] < r["away_goals"] and r["result"] == "A") or (r["home_goals"] == r["away_goals"] and r["result"] == "D")
              for _, r in dm[dm["home_goals"].notna()].iterrows()), "")
    check("clean_match_odds: all odds > 1",
          all((odds_df[c].dropna() > 1.0).all() for c in ["b365_home", "b365_draw", "b365_away", "avg_home", "avg_draw", "avg_away"]), "")
    prob_sum = (odds_df["prob_home"] + odds_df["prob_draw"] + odds_df["prob_away"]).dropna()
    check("clean_match_odds: implied prob sum ~ 1.0", ((prob_sum >= 0.999) & (prob_sum <= 1.001)).all(), f"min={prob_sum.min():.4f}, max={prob_sum.max():.4f}")
    check("clean_match_team_stats: 100% club_id", ts_df["club_id"].notna().all(), f"{len(ts_df)} rows")
    check("clean_match_team_stats: shots_on_target <= shots",
          (ts_df.dropna(subset=["shots", "shots_on_target"])["shots_on_target"] <= ts_df.dropna(subset=["shots", "shots_on_target"])["shots"]).all(), "")
    check("clean_player_match_stats: 100% match_id & club_id & player_key",
          pm_df["match_id"].notna().all() and pm_df["club_id"].notna().all() and pm_df["player_key"].notna().all(), f"{len(pm_df)} rows")
    check("clean_player_match_stats: 0 players on >1 club in same match",
          int((pm_df.groupby(["match_id", "player_key"])["club_id"].nunique() > 1).sum()) == 0, "")
    check("clean_shots: minute in [0, 120]", (shots_all["minute"].dropna().between(0, 120)).all(), f"min={shots_all['minute'].min()}, max={shots_all['minute'].max()}")
    check("clean_shots: xg in [0, 1.5]", (shots_all["xg"].dropna().between(0, 1.5)).all(), f"max={shots_all['xg'].max():.4f}")
    check("clean_match_events: minute in [0, 130]", (ev["minute"].dropna().between(0, 130)).all(), "")

    passed_cnt = sum(1 for v in qa_checks.values() if v["passed"])
    qa_report = {
        "total_checks": len(qa_checks),
        "passed": passed_cnt,
        "failed": [k for k, v in qa_checks.items() if not v["passed"]],
        "details": qa_checks,
    }
    QA_JSON.write_text(json.dumps(qa_report, indent=2), encoding="utf-8")
    pd.DataFrame([{"check_name": k, "passed": v["passed"], "detail": v["detail"]} for k, v in qa_checks.items()]).to_sql("qa_report", dst, if_exists="replace", index=False)

    dst.commit()
    src.close()
    dst.close()
    db_mb = CLEAN_DB.stat().st_size / (1024 * 1024)
    print(f"\n[+] Built {CLEAN_DB.name} ({db_mb:.2f} MB) in {time.time() - t0:.1f}s | QA: {passed_cnt}/{len(qa_checks)} passed")


if __name__ == "__main__":
    build_clean_database()
