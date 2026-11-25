"""
mapped_epl/build_core_and_matches.py
Builds the core dimensions, match bridge, and match/team-level mapped tables in mapped_epl/epl_mapped.db:
- dim_club, dim_season, dim_stadium, dim_referee, dim_match, map_match_ids
- raw_football_data_matches (all 27 seasons, fixing 2003/04 & 2004/05)
- raw_understat_matches
- raw_tfm_games (GB1)
- raw_fotmob_matches (from sofascore/cache_matches/*.json)
- raw_fbref_team_stats (all 7 team-level FBref tables)
- raw_fpl_teams, raw_fpl_fixtures
- raw_weather_archive, raw_weather_stadium_hourly, raw_match_weather
"""

import json
import re
import sqlite3
import time
from datetime import datetime
from zoneinfo import ZoneInfo
from pathlib import Path
import pandas as pd

from mapped_epl.mappings import (
    build_club_lookups,
    export_dimension_csvs,
    normalize_referee,
)

ROOT = Path(__file__).resolve().parent.parent
RAW_DB = ROOT / "raw_data.db"
OUT_DIR = ROOT / "mapped_epl"
MAPPED_DB = OUT_DIR / "epl_mapped.db"
UK_TZ = ZoneInfo("Europe/London")
UTC_TZ = ZoneInfo("UTC")


def infer_season_from_date(dt_str: str) -> str:
    """Infer EPL season 'YYYY/YY' from YYYY-MM-DD (month < 8 -> previous start year)."""
    y = int(dt_str[:4])
    m = int(dt_str[5:7])
    start_y = y - 1 if m < 8 else y
    return f"{start_y}/{str(start_y + 1)[-2:]}"


def make_match_id(season: str, home_cid: str, away_cid: str) -> str:
    y1 = int(season[:4])
    y2 = y1 + 1
    return f"EPL_{y1}_{y2}_{home_cid}_{away_cid}"


def parse_fd_date(val: str) -> str | None:
    if not val or not str(val).strip():
        return None
    s = str(val).strip().lstrip("\ufeff")
    for fmt in ("%d/%m/%Y", "%d/%m/%y", "%Y-%m-%d"):
        try:
            return datetime.strptime(s, fmt).strftime("%Y-%m-%d")
        except ValueError:
            continue
    return None


def utc_to_uk(dt_str: str) -> tuple[str | None, str | None]:
    if not dt_str or not str(dt_str).strip():
        return None, None
    s = str(dt_str).strip().replace("Z", "+00:00")
    for fmt in ("%Y-%m-%d %H:%M:%S", "%Y-%m-%dT%H:%M:%S.%f%z", "%Y-%m-%dT%H:%M:%S%z", "%Y-%m-%dT%H:%M:%S"):
        try:
            dt = datetime.strptime(s, fmt)
            if dt.tzinfo is None:
                dt = dt.replace(tzinfo=UTC_TZ)
            uk = dt.astimezone(UK_TZ)
            return uk.strftime("%Y-%m-%d"), uk.strftime("%H:%M")
        except ValueError:
            continue
    try:
        dt = datetime.fromisoformat(s)
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=UTC_TZ)
        uk = dt.astimezone(UK_TZ)
        return uk.strftime("%Y-%m-%d"), uk.strftime("%H:%M")
    except Exception:
        return None, None


def init_mapped_db() -> sqlite3.Connection:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    if MAPPED_DB.exists():
        MAPPED_DB.unlink()
    conn = sqlite3.connect(str(MAPPED_DB))
    conn.execute("PRAGMA journal_mode=WAL;")
    conn.execute("PRAGMA synchronous=NORMAL;")
    conn.execute("PRAGMA temp_store=MEMORY;")
    conn.execute("PRAGMA cache_size=-200000;")
    conn.execute(
        """
        CREATE TABLE _mapping_audit (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            stage TEXT,
            target_table TEXT,
            source_name TEXT,
            total_rows INTEGER,
            mapped_rows INTEGER,
            unmapped_rows INTEGER,
            notes TEXT
        )
        """
    )
    return conn


def log_audit(conn, stage, target_table, source_name, total, mapped, notes=""):
    conn.execute(
        "INSERT INTO _mapping_audit (stage, target_table, source_name, total_rows, mapped_rows, unmapped_rows, notes) VALUES (?, ?, ?, ?, ?, ?, ?)",
        (stage, target_table, source_name, int(total), int(mapped), int(total - mapped), notes),
    )


def run_core_and_matches():
    t0 = time.time()
    raw = sqlite3.connect(str(RAW_DB))
    conn = init_mapped_db()
    lookups = build_club_lookups()

    # 1. Export & load dim_club, dim_season
    dim_club_df, dim_season_df = export_dimension_csvs(OUT_DIR / "mappings")
    dim_club_df.to_sql("dim_club", conn, if_exists="replace", index=False)
    dim_season_df.to_sql("dim_season", conn, if_exists="replace", index=False)
    print(f"[+] dim_club ({len(dim_club_df)}), dim_season ({len(dim_season_df)})")

    # 2. Football-Data (all 27 seasons, including 380/380 for 03/04 & 04/05)
    fd_tables = raw.execute(
        "SELECT table_name, rel_path FROM _catalog WHERE rel_path LIKE 'football_data_co_uk/%' AND status='ok' ORDER BY rel_path"
    ).fetchall()
    fd_frames = []
    unmapped_fd = set()
    for tbl, rel in fd_tables:
        m = re.search(r"(\d{2})(\d{2})\.csv$", rel)
        y1 = 2000 + int(m.group(1))
        season = f"{y1}/{m.group(2)}"
        df = pd.read_sql_query(f"SELECT * FROM {tbl}", raw)
        df = df[df["HomeTeam"].notna() & (df["HomeTeam"].astype(str).str.strip() != "")].copy()
        drop_cols = [c for c in df.columns if c.startswith("extra_col_")]
        if drop_cols:
            df.drop(columns=drop_cols, inplace=True)
        df["season"] = season
        df["date_iso"] = df["Date"].apply(parse_fd_date)
        df["home_club_id"] = df["HomeTeam"].astype(str).str.strip().map(lookups["fd"])
        df["away_club_id"] = df["AwayTeam"].astype(str).str.strip().map(lookups["fd"])
        for t in df.loc[df["home_club_id"].isna(), "HomeTeam"].unique():
            unmapped_fd.add(t)
        for t in df.loc[df["away_club_id"].isna(), "AwayTeam"].unique():
            unmapped_fd.add(t)
        df["match_id"] = [
            make_match_id(s, h, a) if pd.notna(h) and pd.notna(a) else None
            for s, h, a in zip(df["season"], df["home_club_id"], df["away_club_id"])
        ]
        if "Referee" in df.columns:
            df["referee_canonical"] = df["Referee"].apply(normalize_referee)
        fd_frames.append(df)

    assert not unmapped_fd, f"Unmapped FD teams: {unmapped_fd}"
    fd_all = pd.concat(fd_frames, ignore_index=True)
    fd_all.to_sql("raw_football_data_matches", conn, if_exists="replace", index=False)
    log_audit(conn, "matches", "raw_football_data_matches", "football_data_co_uk (27 seasons)", len(fd_all), fd_all["match_id"].notna().sum(), "Includes full 380 rows for 2003/04 & 2004/05")
    print(f"[+] raw_football_data_matches: {len(fd_all):,} rows (0 unmapped)")

    # Build base match dict keyed by match_id
    matches_map = {}
    for _, r in fd_all.iterrows():
        mid = r["match_id"]
        matches_map[mid] = {
            "match_id": mid,
            "season": r["season"],
            "date": r["date_iso"],
            "kickoff_time": str(r["Time"]).strip() if "Time" in r and pd.notna(r["Time"]) and str(r["Time"]).strip() else None,
            "home_club_id": r["home_club_id"],
            "away_club_id": r["away_club_id"],
            "home_goals": int(r["FTHG"]) if pd.notna(r.get("FTHG")) else None,
            "away_goals": int(r["FTAG"]) if pd.notna(r.get("FTAG")) else None,
            "result": r.get("FTR"),
            "ht_home_goals": int(r["HTHG"]) if pd.notna(r.get("HTHG")) else None,
            "ht_away_goals": int(r["HTAG"]) if pd.notna(r.get("HTAG")) else None,
            "ht_result": r.get("HTR"),
            "referee_canonical": r.get("referee_canonical"),
            "attendance": None,
            "stadium": None,
            "home_manager": None,
            "away_manager": None,
            "home_formation": None,
            "away_formation": None,
            "status": "played" if pd.notna(r.get("FTHG")) else "upcoming",
            "primary_source": "football_data",
        }

    # Bridge table map_match_ids
    match_bridge = {mid: {"match_id": mid, "season": m["season"], "home_club_id": m["home_club_id"], "away_club_id": m["away_club_id"]} for mid, m in matches_map.items()}

    # 3. Understat matches
    us_df = pd.read_sql_query("SELECT * FROM understat__understat_matches_all_2014_2027", raw)
    us_df["season_std"] = us_df["season"].astype(str).str[:4].astype(int).apply(lambda y: f"{y}/{str(y+1)[-2:]}")
    us_df["home_club_id"] = us_df["home_team"].astype(str).str.strip().map(lookups["understat"])
    us_df["away_club_id"] = us_df["away_team"].astype(str).str.strip().map(lookups["understat"])
    assert us_df["home_club_id"].isna().sum() == 0 and us_df["away_club_id"].isna().sum() == 0, "Unmapped Understat teams!"
    us_df["match_id"] = [make_match_id(s, h, a) for s, h, a in zip(us_df["season_std"], us_df["home_club_id"], us_df["away_club_id"])]
    uk_dt = us_df["datetime"].apply(utc_to_uk)
    us_df["date_uk"] = [x[0] for x in uk_dt]
    us_df["kickoff_uk"] = [x[1] for x in uk_dt]
    us_df.to_sql("raw_understat_matches", conn, if_exists="replace", index=False)

    for _, r in us_df.iterrows():
        mid = r["match_id"]
        is_res = str(r.get("is_result")).lower() in ("1", "true")
        if mid not in matches_map and is_res:
            hg = int(float(r["home_goals"])) if pd.notna(r["home_goals"]) else None
            ag = int(float(r["away_goals"])) if pd.notna(r["away_goals"]) else None
            res = ("H" if hg > ag else ("A" if hg < ag else "D")) if hg is not None and ag is not None else None
            matches_map[mid] = {
                "match_id": mid,
                "season": r["season_std"],
                "date": r["date_uk"],
                "kickoff_time": r["kickoff_uk"],
                "home_club_id": r["home_club_id"],
                "away_club_id": r["away_club_id"],
                "home_goals": hg,
                "away_goals": ag,
                "result": res,
                "ht_home_goals": None,
                "ht_away_goals": None,
                "ht_result": None,
                "referee_canonical": None,
                "attendance": None,
                "stadium": None,
                "home_manager": None,
                "away_manager": None,
                "home_formation": None,
                "away_formation": None,
                "status": "played",
                "primary_source": "understat",
            }
            match_bridge[mid] = {"match_id": mid, "season": r["season_std"], "home_club_id": r["home_club_id"], "away_club_id": r["away_club_id"]}
        if mid in matches_map and not matches_map[mid]["kickoff_time"] and r["kickoff_uk"]:
            matches_map[mid]["kickoff_time"] = r["kickoff_uk"]
        if mid in match_bridge:
            match_bridge[mid]["understat_id"] = str(r["understat_id"])

    log_audit(conn, "matches", "raw_understat_matches", "understat_matches_all_2014_2027", len(us_df), us_df["match_id"].isin(match_bridge).sum())
    print(f"[+] raw_understat_matches: {len(us_df):,} rows")

    # 4. Transfermarkt GB1 games
    tm_df = pd.read_sql_query("SELECT * FROM transfermarkt__tfm_games_full WHERE competition_id = 'GB1'", raw)
    tm_df["season_std"] = tm_df["season"].astype(int).apply(lambda y: f"{y}/{str(y+1)[-2:]}")
    tm_df["home_club_id"] = tm_df["home_club_id"].astype(int).map(lookups["tm_id"])
    tm_df["away_club_id"] = tm_df["away_club_id"].astype(int).map(lookups["tm_id"])
    assert tm_df["home_club_id"].isna().sum() == 0 and tm_df["away_club_id"].isna().sum() == 0, "Unmapped TM GB1 teams!"
    tm_df["match_id"] = [make_match_id(s, h, a) for s, h, a in zip(tm_df["season_std"], tm_df["home_club_id"], tm_df["away_club_id"])]
    tm_df.to_sql("raw_tfm_games", conn, if_exists="replace", index=False)

    for _, r in tm_df.iterrows():
        mid = r["match_id"]
        if mid in matches_map:
            if pd.notna(r.get("attendance")):
                matches_map[mid]["attendance"] = int(r["attendance"])
            if pd.notna(r.get("stadium")):
                matches_map[mid]["stadium"] = r["stadium"]
            if pd.notna(r.get("home_club_manager_name")):
                matches_map[mid]["home_manager"] = r["home_club_manager_name"]
            if pd.notna(r.get("away_club_manager_name")):
                matches_map[mid]["away_manager"] = r["away_club_manager_name"]
            if pd.notna(r.get("home_club_formation")):
                matches_map[mid]["home_formation"] = r["home_club_formation"]
            if pd.notna(r.get("away_club_formation")):
                matches_map[mid]["away_formation"] = r["away_club_formation"]
            if not matches_map[mid]["referee_canonical"] and pd.notna(r.get("referee")):
                matches_map[mid]["referee_canonical"] = normalize_referee(r["referee"])
        if mid in match_bridge:
            match_bridge[mid]["tfm_game_id"] = int(r["game_id"])

    log_audit(conn, "matches", "raw_tfm_games", "tfm_games_full (GB1)", len(tm_df), tm_df["match_id"].isin(match_bridge).sum())
    print(f"[+] raw_tfm_games (GB1): {len(tm_df):,} rows")

    # 5. FotMob / Sofascore cache_matches/*.json (extract match metadata & team stats)
    fm_rows = []
    fm_total = 0
    fm_skipped_non_epl = 0
    for (content,) in raw.execute("SELECT content FROM raw_json_files WHERE rel_path LIKE 'sofascore/cache_matches/%'").fetchall():
        fm_total += 1
        d = json.loads(content)
        g = d.get("general", {})
        league_id = g.get("leagueId")
        parent_id = g.get("parentLeagueId")
        if league_id != 47 and parent_id != 47:
            fm_skipped_non_epl += 1
            continue
        ht = g.get("homeTeam", {})
        at = g.get("awayTeam", {})
        h_cid = lookups["fm_id"].get(ht.get("id")) or lookups["fm_name"].get(ht.get("name"))
        a_cid = lookups["fm_id"].get(at.get("id")) or lookups["fm_name"].get(at.get("name"))
        date_uk, ko_uk = utc_to_uk(g.get("matchTimeUTCDate"))
        if not (h_cid and a_cid and date_uk):
            fm_skipped_non_epl += 1
            continue
        season = infer_season_from_date(date_uk)
        mid = make_match_id(season, h_cid, a_cid)
        if mid not in matches_map:
            # Check if it's a valid 2026/27 played match not in FD yet
            header_teams = d.get("header", {}).get("teams", [])
            finished = g.get("finished", False)
            if season == "2026/27" and finished and len(header_teams) == 2:
                hg = header_teams[0].get("score")
                ag = header_teams[1].get("score")
                res = ("H" if hg > ag else ("A" if hg < ag else "D")) if hg is not None and ag is not None else None
                matches_map[mid] = {
                    "match_id": mid, "season": season, "date": date_uk, "kickoff_time": ko_uk,
                    "home_club_id": h_cid, "away_club_id": a_cid, "home_goals": hg, "away_goals": ag,
                    "result": res, "ht_home_goals": None, "ht_away_goals": None, "ht_result": None,
                    "referee_canonical": None, "attendance": None, "stadium": None,
                    "home_manager": None, "away_manager": None, "home_formation": None, "away_formation": None,
                    "status": "played", "primary_source": "fotmob",
                }
                match_bridge[mid] = {"match_id": mid, "season": season, "home_club_id": h_cid, "away_club_id": a_cid}
            else:
                fm_skipped_non_epl += 1
                continue

        if not matches_map[mid]["kickoff_time"] and ko_uk:
            matches_map[mid]["kickoff_time"] = ko_uk
        match_bridge[mid]["fotmob_match_id"] = int(g["matchId"])

        # Extract flat team stats from content.stats.Periods.All.stats
        stat_dict = {
            "match_id": mid,
            "fotmob_match_id": int(g["matchId"]),
            "season": season,
            "date_uk": date_uk,
            "kickoff_uk": ko_uk,
            "home_club_id": h_cid,
            "away_club_id": a_cid,
        }
        content_obj = d.get("content") or {}
        stats_obj = content_obj.get("stats") or {}
        periods = stats_obj.get("Periods") or {}
        all_p = periods.get("All", {}).get("stats", []) if isinstance(periods, dict) else []
        for group in all_p:
            for item in group.get("stats", []):
                key = item.get("key")
                vals = item.get("stats", [])
                if key and isinstance(vals, list) and len(vals) == 2:
                    stat_dict[f"home_{key}"] = str(vals[0]) if vals[0] is not None else None
                    stat_dict[f"away_{key}"] = str(vals[1]) if vals[1] is not None else None
        fm_rows.append(stat_dict)

    fm_df = pd.DataFrame(fm_rows)
    fm_df.to_sql("raw_fotmob_matches", conn, if_exists="replace", index=False)
    log_audit(conn, "matches", "raw_fotmob_matches", "sofascore/cache_matches/*.json", fm_total, len(fm_df), f"Skipped {fm_skipped_non_epl} non-EPL or unmatched")
    print(f"[+] raw_fotmob_matches: {len(fm_df):,} rows (skipped {fm_skipped_non_epl} non-EPL)")

    # 6. FBref team-level tables (7 tables)
    fb_team_tables = raw.execute(
        "SELECT table_name, rel_path FROM _catalog WHERE rel_path LIKE 'fbref/%team%' AND status='ok' ORDER BY rel_path"
    ).fetchall()
    fb_team_frames = []
    for tbl, rel in fb_team_tables:
        df = pd.read_sql_query(f"SELECT * FROM {tbl}", raw)
        df["source_table"] = tbl.replace("fbref__", "")
        df["date_iso"] = df["Match_Date"].astype(str).str[:10]
        df["season_std"] = df["date_iso"].apply(infer_season_from_date)
        df["home_club_id"] = df["Home_Team"].astype(str).str.strip().map(lookups["fbref"])
        df["away_club_id"] = df["Away_Team"].astype(str).str.strip().map(lookups["fbref"])
        assert df["home_club_id"].isna().sum() == 0 and df["away_club_id"].isna().sum() == 0, f"Unmapped FBref team in {tbl}"
        df["match_id"] = [make_match_id(s, h, a) for s, h, a in zip(df["season_std"], df["home_club_id"], df["away_club_id"])]
        if "Team" in df.columns:
            df["club_id"] = df["Team"].astype(str).str.strip().map(lookups["fbref"])
        for _, r in df[["match_id", "Game_URL"]].dropna().drop_duplicates().iterrows():
            if r["match_id"] in match_bridge:
                match_bridge[r["match_id"]]["fbref_game_url"] = r["Game_URL"]
        fb_team_frames.append(df)
    fb_team_all = pd.concat(fb_team_frames, ignore_index=True)
    fb_team_all.to_sql("raw_fbref_team_stats", conn, if_exists="replace", index=False)
    log_audit(conn, "matches", "raw_fbref_team_stats", "fbref (7 team tables)", len(fb_team_all), fb_team_all["match_id"].isin(match_bridge).sum())
    print(f"[+] raw_fbref_team_stats: {len(fb_team_all):,} rows")

    # 7. FPL teams & fixtures
    fpl_team_tables = raw.execute(
        "SELECT table_name, rel_path FROM _catalog WHERE rel_path LIKE 'fpl_api/fpl_teams_%' AND status='ok' ORDER BY rel_path"
    ).fetchall()
    fpl_t_frames = []
    fpl_id_map = {}  # (season, fpl_team_id) -> club_id
    for tbl, rel in fpl_team_tables:
        m = re.search(r"(\d{4})_(\d{4})", rel)
        season = f"{m.group(1)}/{m.group(2)[-2:]}"
        df = pd.read_sql_query(f"SELECT * FROM {tbl}", raw)
        df["season"] = season
        df["club_id"] = df["short_name"].map(lookups["fpl"]).fillna(df["name"].map(lookups["fpl"]))
        assert df["club_id"].isna().sum() == 0, f"Unmapped FPL team in {tbl}"
        for _, r in df.iterrows():
            fpl_id_map[(season, int(r["id"]))] = r["club_id"]
        fpl_t_frames.append(df)
    fpl_t_all = pd.concat(fpl_t_frames, ignore_index=True)
    fpl_t_all.to_sql("raw_fpl_teams", conn, if_exists="replace", index=False)

    # Note: for 2018/19, 2019/20, 2020/21 where fpl_teams_* doesn't exist, infer team mapping from match dates or fpl_merged_gw
    fpl_fix_tables = raw.execute(
        "SELECT table_name, rel_path FROM _catalog WHERE rel_path LIKE 'fpl_api/fpl_fixtures_%' AND status='ok' ORDER BY rel_path"
    ).fetchall()
    fpl_f_frames = []
    # Build (date, home_goals, away_goals) -> match_id lookup for older FPL fixtures without fpl_teams
    date_score_to_mid = {}
    for mid, m in matches_map.items():
        if m["date"] and m["home_goals"] is not None and m["away_goals"] is not None:
            date_score_to_mid.setdefault((m["season"], m["date"], m["home_goals"], m["away_goals"]), []).append(mid)

    for tbl, rel in fpl_fix_tables:
        m = re.search(r"(\d{4})_(\d{4})", rel)
        season = f"{m.group(1)}/{m.group(2)[-2:]}"
        df = pd.read_sql_query(f"SELECT * FROM {tbl}", raw)
        df["season"] = season

        # Pass 1: discover (season, fpl_team_id) -> club_id from matched date+score if fpl_teams_* is missing
        for _, r in df.iterrows():
            th = int(r["team_h"]) if pd.notna(r.get("team_h")) else None
            ta = int(r["team_a"]) if pd.notna(r.get("team_a")) else None
            if (season, th) in fpl_id_map and (season, ta) in fpl_id_map:
                continue
            d_uk, _ = utc_to_uk(r.get("kickoff_time"))
            hg = int(r["team_h_score"]) if pd.notna(r.get("team_h_score")) else None
            ag = int(r["team_a_score"]) if pd.notna(r.get("team_a_score")) else None
            cands = date_score_to_mid.get((season, d_uk, hg, ag), [])
            if len(cands) == 1 and cands[0] in matches_map:
                mm = matches_map[cands[0]]
                if th is not None:
                    fpl_id_map.setdefault((season, th), mm["home_club_id"])
                if ta is not None:
                    fpl_id_map.setdefault((season, ta), mm["away_club_id"])

        # Pass 2: assign home_club_id, away_club_id, match_id using complete fpl_id_map
        mids = []
        h_cids = []
        a_cids = []
        for _, r in df.iterrows():
            hc = fpl_id_map.get((season, int(r["team_h"]))) if pd.notna(r.get("team_h")) else None
            ac = fpl_id_map.get((season, int(r["team_a"]))) if pd.notna(r.get("team_a")) else None
            mid = make_match_id(season, hc, ac) if (hc and ac) else None
            mids.append(mid)
            h_cids.append(hc)
            a_cids.append(ac)
            if mid and mid in match_bridge and pd.notna(r.get("id")):
                match_bridge[mid]["fpl_fixture_id"] = int(r["id"])
        df["match_id"] = mids
        df["home_club_id"] = h_cids
        df["away_club_id"] = a_cids
        fpl_f_frames.append(df)
    fpl_f_all = pd.concat(fpl_f_frames, ignore_index=True)
    fpl_f_all.to_sql("raw_fpl_fixtures", conn, if_exists="replace", index=False)
    log_audit(conn, "matches", "raw_fpl_fixtures", "fpl_fixtures (2018-2026)", len(fpl_f_all), fpl_f_all["match_id"].notna().sum())
    print(f"[+] raw_fpl_fixtures: {len(fpl_f_all):,} rows ({fpl_f_all['match_id'].notna().sum():,} mapped to match_id)")

    # 8. Open-Meteo Weather (Archive + Stadium JSONs + Match Weather join)
    w_arch = pd.read_sql_query("SELECT * FROM open_meteo__epl_weather_archive_2000_2021", raw)
    w_arch.to_sql("raw_weather_archive", conn, if_exists="replace", index=False)
    log_audit(conn, "weather", "raw_weather_archive", "epl_weather_archive_2000_2021", len(w_arch), len(w_arch))

    stadium_dims = []
    w_hourly_frames = []
    for fname, content in raw.execute("SELECT file_name, content FROM raw_json_files WHERE source='open_meteo'").fetchall():
        oj = json.loads(content)
        cid = lookups["stadium"].get(fname)
        assert cid, f"Unmapped stadium JSON: {fname}"
        skey = fname.replace(".json", "")
        stadium_dims.append({
            "stadium_key": skey,
            "club_id": cid,
            "weather_region": lookups["region"][cid],
            "latitude": oj.get("latitude"),
            "longitude": oj.get("longitude"),
            "elevation": oj.get("elevation"),
        })
        hr = oj.get("hourly", {})
        if hr and "time" in hr:
            hdf = pd.DataFrame(hr)
            hdf["stadium_key"] = skey
            hdf["club_id"] = cid
            hdf["date"] = hdf["time"].str[:10]
            hdf["hour"] = hdf["time"].str[11:13].astype(int)
            w_hourly_frames.append(hdf)

    dim_stadium_df = pd.DataFrame(stadium_dims)
    dim_stadium_df.to_csv(OUT_DIR / "mappings" / "dim_stadium.csv", index=False)
    dim_stadium_df.to_sql("dim_stadium", conn, if_exists="replace", index=False)

    w_hourly_all = pd.concat(w_hourly_frames, ignore_index=True)
    w_hourly_all.to_sql("raw_weather_stadium_hourly", conn, if_exists="replace", index=False)
    conn.execute("CREATE INDEX idx_weather_club_date_hr ON raw_weather_stadium_hourly(club_id, date, hour);")
    log_audit(conn, "weather", "raw_weather_stadium_hourly", "open_meteo/stadiums/*.json (27 stadiums)", len(w_hourly_all), len(w_hourly_all))
    print(f"[+] raw_weather_stadium_hourly: {len(w_hourly_all):,} rows across {len(dim_stadium_df)} stadiums")

    # 9. Write dim_match, dim_referee, map_match_ids
    dim_match_df = pd.DataFrame(list(matches_map.values())).sort_values(["season", "date", "match_id"])
    dim_match_df.to_sql("dim_match", conn, if_exists="replace", index=False)
    conn.execute("CREATE UNIQUE INDEX idx_dim_match_id ON dim_match(match_id);")

    map_match_df = pd.DataFrame(list(match_bridge.values()))
    map_match_df.to_sql("map_match_ids", conn, if_exists="replace", index=False)
    conn.execute("CREATE UNIQUE INDEX idx_map_match_id ON map_match_ids(match_id);")

    # dim_referee
    ref_rows = []
    if "Referee" in fd_all.columns:
        for raw_r, can_r in fd_all[["Referee", "referee_canonical"]].dropna().drop_duplicates().itertuples(index=False):
            ref_rows.append({"raw_name": str(raw_r).strip(), "canonical_name": can_r, "source": "football_data"})
    dim_ref_df = pd.DataFrame(ref_rows).drop_duplicates()
    dim_ref_df.to_sql("dim_referee", conn, if_exists="replace", index=False)

    # 10. Build raw_match_weather (joining dim_match with stadium hourly (2021+) or archive (2000-2021))
    conn.execute(
        """
        CREATE TABLE raw_match_weather AS
        SELECT
            m.match_id,
            m.season,
            m.date,
            m.kickoff_time,
            m.home_club_id,
            COALESCE(h.temperature_2m, a.temperature_c) AS temperature_c,
            COALESCE(h.precipitation, a.precipitation_mm) AS precipitation_mm,
            COALESCE(h.wind_speed_10m, a.wind_speed_kmh) AS wind_speed_kmh,
            h.relative_humidity_2m,
            h.weather_code,
            CASE WHEN h.temperature_2m IS NOT NULL THEN 'open_meteo_stadium_hourly'
                 WHEN a.temperature_c IS NOT NULL THEN 'open_meteo_archive_daily'
                 ELSE NULL END AS weather_source
        FROM dim_match m
        LEFT JOIN dim_club c ON m.home_club_id = c.club_id
        LEFT JOIN raw_weather_stadium_hourly h
            ON h.club_id = m.home_club_id
            AND h.date = m.date
            AND h.hour = COALESCE(CAST(substr(m.kickoff_time, 1, 2) AS INTEGER), 15)
        LEFT JOIN raw_weather_archive a
            ON a.region = c.weather_region
            AND a.date = m.date
        """
    )
    mw_mapped = conn.execute("SELECT COUNT(*) FROM raw_match_weather WHERE temperature_c IS NOT NULL").fetchone()[0]
    log_audit(conn, "weather", "raw_match_weather", "joined dim_match + open_meteo", len(dim_match_df), mw_mapped)
    print(f"[+] dim_match: {len(dim_match_df):,} rows | raw_match_weather mapped: {mw_mapped:,}/{len(dim_match_df):,}")

    conn.commit()
    raw.close()
    conn.close()
    print(f"\nCore & Match mapping finished in {time.time() - t0:.1f}s -> {MAPPED_DB}")


if __name__ == "__main__":
    run_core_and_matches()
