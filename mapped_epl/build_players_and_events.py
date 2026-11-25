"""
mapped_epl/build_players_and_events.py
Ultra-careful Player Entity Resolution + mapping of all player-level, event-level,
shot-level, and valuation/transfer tables from raw_data.db into mapped_epl/epl_mapped.db.
"""

import json
import re
import sqlite3
import time
from collections import defaultdict
from difflib import SequenceMatcher
from pathlib import Path
import pandas as pd

from mapped_epl.mappings import (
    build_club_lookups,
    normalize_person_name,
    name_tokens,
)
from mapped_epl.build_core_and_matches import (
    infer_season_from_date,
    make_match_id,
    utc_to_uk,
    log_audit,
)

ROOT = Path(__file__).resolve().parent.parent
RAW_DB = ROOT / "raw_data.db"
MAPPED_DB = ROOT / "mapped_epl" / "epl_mapped.db"


class PlayerResolver:
    """
    Collision-safe cross-source player entity resolver.
    Anchors on Transfermarkt EPL players (TM_<player_id>) and resolves
    Understat, FotMob, FBref, and FPL players via:
      1. Match + Club co-occurrence (exact normalized name)
      2. Match + Club co-occurrence (unambiguous token-subset / initial+lastname)
      3. Season + Club (exact normalized name, unique in squad)
      4. Global unique normalized name (unique across all EPL TM players)
      5. Standalone fallback key (zero false merges!)
    """

    def __init__(self):
        self.tm_players = {}          # tm_id -> dict
        self.by_match_club = defaultdict(list)   # (match_id, club_id) -> [(tm_id, norm_name, tokens)]
        self.by_season_club = defaultdict(dict)  # (season, club_id) -> {norm_name: [tm_id, ...]}
        self.by_global_name = defaultdict(list)  # norm_name -> [tm_id, ...]

        self.cache = {}               # (source, cache_id) -> (player_key, method)
        self.mappings_log = {}        # (source, cache_id, raw_name) -> dict
        self.standalone_players = {}  # player_key -> dict

    def register_tm_player(self, tm_id: int, raw_name: str, meta: dict):
        norm = normalize_person_name(raw_name)
        if tm_id not in self.tm_players:
            self.tm_players[tm_id] = {
                "player_key": f"TM_{tm_id}",
                "tm_player_id": tm_id,
                "canonical_name": raw_name,
                "norm_name": norm,
                **meta,
            }
            if norm:
                self.by_global_name[norm].append(tm_id)

    def add_tm_appearance(self, match_id: str, season: str, club_id: str, tm_id: int, raw_name: str):
        norm = normalize_person_name(raw_name) or self.tm_players.get(tm_id, {}).get("norm_name", "")
        toks = name_tokens(norm)
        if match_id and club_id:
            self.by_match_club[(match_id, club_id)].append((tm_id, norm, toks))
        if season and club_id and norm:
            lst = self.by_season_club[(season, club_id)].setdefault(norm, [])
            if tm_id not in lst:
                lst.append(tm_id)

    def resolve(
        self,
        source: str,
        src_id: str | None,
        raw_name: str | None,
        match_id: str | None = None,
        season: str | None = None,
        club_id: str | None = None,
    ) -> str:
        norm = normalize_person_name(raw_name)
        cache_key = (source, str(src_id)) if src_id is not None else (source, f"name:{norm}:{club_id}")
        if cache_key in self.cache:
            return self.cache[cache_key][0]

        resolved_key = None
        method = "standalone"

        # 1. Match + Club exact normalized name
        if match_id and club_id and norm:
            cands = self.by_match_club.get((match_id, club_id), [])
            exact = {tid for tid, nm, _ in cands if nm == norm}
            if len(exact) == 1:
                tid = next(iter(exact))
                resolved_key = f"TM_{tid}"
                method = "match_club_exact"

            # 2. Match + Club token-subset or initial+lastname (strictly unambiguous)
            if not resolved_key and cands:
                q_toks = name_tokens(norm)
                q_set = set(q_toks)
                matched = set()
                for tid, nm, toks in cands:
                    if not toks:
                        continue
                    t_set = set(toks)
                    # Subset match (handles middle names / "Son Heung-Min" vs "Heung-Min Son")
                    if len(q_set) >= 2 and (q_set.issubset(t_set) or t_set.issubset(q_set)):
                        matched.add(tid)
                    # Single-name / nickname exact token match (e.g. "Fernandinho", "Ederson", "Alisson", "Joelinton")
                    elif len(q_toks) == 1 and len(q_toks[0]) >= 5 and q_toks[0] in t_set:
                        matched.add(tid)
                    # Initial + last name (e.g. "K De Bruyne" vs "Kevin De Bruyne", or "J Ward-Prowse")
                    elif len(q_toks) >= 2 and len(toks) >= 2 and q_toks[0][0] == toks[0][0] and q_toks[-1] == toks[-1]:
                        matched.add(tid)
                if len(matched) == 1:
                    tid = next(iter(matched))
                    resolved_key = f"TM_{tid}"
                    method = "match_club_token"

            # 2b. Match + Club fuzzy match (strictly unique candidate >= 0.80)
            if not resolved_key and cands and len(norm) >= 5:
                fuzzy_hits = set()
                for tid, nm, _ in cands:
                    if nm and SequenceMatcher(None, norm, nm).ratio() >= 0.80:
                        fuzzy_hits.add(tid)
                if len(fuzzy_hits) == 1:
                    tid = next(iter(fuzzy_hits))
                    resolved_key = f"TM_{tid}"
                    method = "match_club_fuzzy"

        # 3. Season + Club exact normalized name (unique in squad)
        if not resolved_key and season and club_id and norm:
            sc_map = self.by_season_club.get((season, club_id), {})
            tids = sc_map.get(norm, [])
            if len(tids) == 1:
                resolved_key = f"TM_{tids[0]}"
                method = "season_club_exact"

        # 4. Global unique normalized name across all EPL TM players
        if not resolved_key and norm:
            tids = self.by_global_name.get(norm, [])
            if len(tids) == 1:
                resolved_key = f"TM_{tids[0]}"
                method = "global_unique_name"

        # 5. Standalone fallback (never falsely merge)
        if not resolved_key:
            safe_id = str(src_id) if src_id is not None else re.sub(r"\s+", "_", norm or "unknown")
            prefix = {"understat": "US", "fotmob": "FM", "fpl": "FPL", "fbref": "FB"}.get(source, source.upper())
            resolved_key = f"{prefix}_{safe_id}"
            if resolved_key not in self.standalone_players:
                self.standalone_players[resolved_key] = {
                    "player_key": resolved_key,
                    "tm_player_id": None,
                    "canonical_name": raw_name or safe_id,
                    "norm_name": norm,
                    "position": None,
                    "country_of_citizenship": None,
                    "date_of_birth": None,
                    "height_cm": None,
                    "preferred_foot": None,
                }

        self.cache[cache_key] = (resolved_key, method)
        self.mappings_log[(source, str(src_id) if src_id is not None else norm)] = {
            "source": source,
            "source_player_id": str(src_id) if src_id is not None else None,
            "source_player_name": raw_name,
            "norm_name": norm,
            "player_key": resolved_key,
            "match_method": method,
        }
        return resolved_key


def run_players_and_events():
    t0 = time.time()
    raw = sqlite3.connect(str(RAW_DB))
    conn = sqlite3.connect(str(MAPPED_DB))
    conn.execute("PRAGMA journal_mode=WAL;")
    conn.execute("PRAGMA synchronous=NORMAL;")
    conn.execute("PRAGMA temp_store=MEMORY;")
    conn.execute("PRAGMA cache_size=-200000;")

    lookups = build_club_lookups()
    tm_id_to_cid = lookups["tm_id"]

    # Load match bridge & dim_match lookups
    bridge_df = pd.read_sql_query("SELECT * FROM map_match_ids", conn)
    dim_match_df = pd.read_sql_query("SELECT match_id, season, date, home_club_id, away_club_id FROM dim_match", conn)

    tm_gid_to_match = {}
    us_id_to_match = {}
    fm_id_to_match = {}
    for _, r in bridge_df.iterrows():
        info = (r["match_id"], r["season"], r["home_club_id"], r["away_club_id"])
        if pd.notna(r.get("tfm_game_id")):
            tm_gid_to_match[int(r["tfm_game_id"])] = info
        if pd.notna(r.get("understat_id")):
            us_id_to_match[str(r["understat_id"])] = info
        if pd.notna(r.get("fotmob_match_id")):
            fm_id_to_match[int(r["fotmob_match_id"])] = info

    valid_match_ids = set(dim_match_df["match_id"])
    gb1_game_ids = sorted(tm_gid_to_match.keys())
    gb1_id_set = set(gb1_game_ids)

    resolver = PlayerResolver()

    # =========================================================================
    # 1. TRANSFERMARKT (Clubs, Players, Lineups, Appearances, Events, Valuations, Transfers)
    # =========================================================================
    print("[*] Processing Transfermarkt GB1 tables...")
    tfm_clubs = pd.read_sql_query("SELECT * FROM transfermarkt__tfm_clubs_full", raw)
    tfm_clubs["epl_club_id"] = tfm_clubs["club_id"].astype(int).map(tm_id_to_cid)
    tfm_clubs_epl = tfm_clubs[tfm_clubs["epl_club_id"].notna()].copy()
    tfm_clubs_epl.to_sql("raw_tfm_clubs", conn, if_exists="replace", index=False)
    log_audit(conn, "tfm", "raw_tfm_clubs", "tfm_clubs_full (EPL)", len(tfm_clubs_epl), len(tfm_clubs_epl))

    # Club games (GB1)
    tfm_cg = pd.read_sql_query("SELECT * FROM transfermarkt__tfm_club_games_full", raw)
    tfm_cg = tfm_cg[tfm_cg["game_id"].isin(gb1_id_set)].copy()
    tfm_cg["match_id"] = tfm_cg["game_id"].map(lambda g: tm_gid_to_match[int(g)][0])
    tfm_cg["season"] = tfm_cg["game_id"].map(lambda g: tm_gid_to_match[int(g)][1])
    tfm_cg["epl_club_id"] = tfm_cg["club_id"].astype(int).map(tm_id_to_cid)
    tfm_cg.to_sql("raw_tfm_club_games", conn, if_exists="replace", index=False)
    log_audit(conn, "tfm", "raw_tfm_club_games", "tfm_club_games_full (GB1)", len(tfm_cg), tfm_cg["epl_club_id"].notna().sum())

    # Lineups (GB1)
    tfm_lu = pd.read_sql_query("SELECT * FROM transfermarkt__tfm_game_lineups_full", raw)
    tfm_lu = tfm_lu[tfm_lu["game_id"].isin(gb1_id_set)].copy()
    tfm_lu["match_id"] = tfm_lu["game_id"].map(lambda g: tm_gid_to_match[int(g)][0])
    tfm_lu["season"] = tfm_lu["game_id"].map(lambda g: tm_gid_to_match[int(g)][1])
    tfm_lu["epl_club_id"] = tfm_lu["club_id"].astype(int).map(tm_id_to_cid)
    tfm_lu["player_key"] = "TM_" + tfm_lu["player_id"].astype(int).astype(str)
    tfm_lu.to_sql("raw_tfm_game_lineups", conn, if_exists="replace", index=False)
    log_audit(conn, "tfm", "raw_tfm_game_lineups", "tfm_game_lineups_full (GB1)", len(tfm_lu), len(tfm_lu))

    # Appearances (GB1)
    tfm_app = pd.read_sql_query("SELECT * FROM transfermarkt__tfm_appearances_full WHERE competition_id = 'GB1'", raw)
    tfm_app["match_id"] = tfm_app["game_id"].map(lambda g: tm_gid_to_match.get(int(g), (None,))[0])
    tfm_app["season"] = tfm_app["game_id"].map(lambda g: tm_gid_to_match.get(int(g), (None, None))[1])
    tfm_app["epl_club_id"] = tfm_app["player_club_id"].astype(int).map(tm_id_to_cid)
    tfm_app["player_key"] = "TM_" + tfm_app["player_id"].astype(int).astype(str)
    tfm_app.to_sql("raw_tfm_appearances", conn, if_exists="replace", index=False)
    log_audit(conn, "tfm", "raw_tfm_appearances", "tfm_appearances_full (GB1)", len(tfm_app), tfm_app["match_id"].notna().sum())

    # Game Events (GB1)
    tfm_ev = pd.read_sql_query("SELECT * FROM transfermarkt__tfm_game_events_full", raw)
    tfm_ev = tfm_ev[tfm_ev["game_id"].isin(gb1_id_set)].copy()
    tfm_ev["match_id"] = tfm_ev["game_id"].map(lambda g: tm_gid_to_match[int(g)][0])
    tfm_ev["season"] = tfm_ev["game_id"].map(lambda g: tm_gid_to_match[int(g)][1])
    tfm_ev["epl_club_id"] = tfm_ev["club_id"].astype(int).map(tm_id_to_cid)
    tfm_ev["player_key"] = "TM_" + tfm_ev["player_id"].astype(int).astype(str)
    tfm_ev["in_player_key"] = tfm_ev["player_in_id"].map(lambda x: f"TM_{int(x)}" if pd.notna(x) else None)
    tfm_ev["assist_player_key"] = tfm_ev["player_assist_id"].map(lambda x: f"TM_{int(x)}" if pd.notna(x) else None)
    tfm_ev.to_sql("raw_tfm_game_events", conn, if_exists="replace", index=False)
    log_audit(conn, "tfm", "raw_tfm_game_events", "tfm_game_events_full (GB1)", len(tfm_ev), len(tfm_ev))

    # Collect all EPL Transfermarkt player IDs
    epl_tm_pids = set(tfm_lu["player_id"].dropna().astype(int)) | set(tfm_app["player_id"].dropna().astype(int)) | set(tfm_ev["player_id"].dropna().astype(int))

    # Register TM players & appearances in resolver
    tfm_pl = pd.read_sql_query("SELECT * FROM transfermarkt__tfm_players_full", raw)
    tfm_pl_epl = tfm_pl[tfm_pl["player_id"].astype(int).isin(epl_tm_pids) | tfm_pl["current_club_id"].isin(tm_id_to_cid.keys())].copy()
    tfm_pl_epl["player_key"] = "TM_" + tfm_pl_epl["player_id"].astype(int).astype(str)
    tfm_pl_epl["epl_current_club_id"] = tfm_pl_epl["current_club_id"].map(lambda x: tm_id_to_cid.get(int(x)) if pd.notna(x) else None)
    tfm_pl_epl.to_sql("raw_tfm_players", conn, if_exists="replace", index=False)
    log_audit(conn, "tfm", "raw_tfm_players", "tfm_players_full (EPL)", len(tfm_pl_epl), len(tfm_pl_epl))

    for _, r in tfm_pl_epl.iterrows():
        resolver.register_tm_player(
            int(r["player_id"]),
            r["name"],
            {
                "position": r.get("position"),
                "country_of_citizenship": r.get("country_of_citizenship"),
                "date_of_birth": str(r["date_of_birth"])[:10] if pd.notna(r.get("date_of_birth")) else None,
                "height_cm": r.get("height_in_cm"),
                "preferred_foot": r.get("foot"),
            },
        )
    for _, r in tfm_lu[["match_id", "season", "epl_club_id", "player_id", "player_name"]].iterrows():
        resolver.add_tm_appearance(r["match_id"], r["season"], r["epl_club_id"], int(r["player_id"]), r["player_name"])
    for _, r in tfm_app[["match_id", "season", "epl_club_id", "player_id", "player_name"]].dropna(subset=["match_id", "epl_club_id"]).iterrows():
        resolver.add_tm_appearance(r["match_id"], r["season"], r["epl_club_id"], int(r["player_id"]), r["player_name"])

    # Valuations & Transfers for EPL players/clubs
    epl_tm_pids_all = set(tfm_pl_epl["player_id"].astype(int))
    tfm_val = pd.read_sql_query("SELECT * FROM transfermarkt__tfm_player_valuations_full", raw)
    tfm_val = tfm_val[tfm_val["player_id"].astype(int).isin(epl_tm_pids_all)].copy()
    tfm_val["player_key"] = "TM_" + tfm_val["player_id"].astype(int).astype(str)
    tfm_val["epl_club_id"] = tfm_val["current_club_id"].map(lambda x: tm_id_to_cid.get(int(x)) if pd.notna(x) else None)
    tfm_val.to_sql("raw_tfm_player_valuations", conn, if_exists="replace", index=False)
    log_audit(conn, "tfm", "raw_tfm_player_valuations", "tfm_player_valuations_full (EPL players)", len(tfm_val), len(tfm_val))

    tfm_tr = pd.read_sql_query("SELECT * FROM transfermarkt__tfm_transfers_full", raw)
    tfm_tr = tfm_tr[
        tfm_tr["from_club_id"].isin(tm_id_to_cid.keys())
        | tfm_tr["to_club_id"].isin(tm_id_to_cid.keys())
        | tfm_tr["player_id"].astype(int).isin(epl_tm_pids_all)
    ].copy()
    tfm_tr["player_key"] = "TM_" + tfm_tr["player_id"].astype(int).astype(str)
    tfm_tr["from_epl_club_id"] = tfm_tr["from_club_id"].map(lambda x: tm_id_to_cid.get(int(x)) if pd.notna(x) else None)
    tfm_tr["to_epl_club_id"] = tfm_tr["to_club_id"].map(lambda x: tm_id_to_cid.get(int(x)) if pd.notna(x) else None)
    tfm_tr.to_sql("raw_tfm_transfers", conn, if_exists="replace", index=False)
    log_audit(conn, "tfm", "raw_tfm_transfers", "tfm_transfers_full (EPL)", len(tfm_tr), len(tfm_tr))
    print(f"[+] Transfermarkt done: lineups={len(tfm_lu):,}, apps={len(tfm_app):,}, events={len(tfm_ev):,}, vals={len(tfm_val):,}, transfers={len(tfm_tr):,}")

    # =========================================================================
    # 2. UNDERSTAT SHOTS & PLAYERS (from cache_matches/*.json + CSVs)
    # =========================================================================
    print("[*] Processing Understat cache_matches/*.json (rosters & shots) + player CSVs...")
    us_shots_rows = []
    us_roster_rows = []
    for rel_path, content in raw.execute("SELECT rel_path, content FROM raw_json_files WHERE rel_path LIKE 'understat/cache_matches/%'").fetchall():
        uid = Path(rel_path).stem
        minfo = us_id_to_match.get(uid)
        if not minfo:
            continue
        mid, season, h_cid, a_cid = minfo
        d = json.loads(content)
        # Rosters
        rosters = d.get("rosters") or {}
        for side, cid in (("h", h_cid), ("a", a_cid)):
            rdict = rosters.get(side) or {}
            items = rdict.values() if isinstance(rdict, dict) else rdict
            for item in items:
                pid = str(item.get("player_id") or "")
                pname = item.get("player")
                pkey = resolver.resolve("understat", pid or None, pname, match_id=mid, season=season, club_id=cid)
                us_roster_rows.append({
                    "match_id": mid, "season": season, "club_id": cid, "side": side,
                    "understat_player_id": pid, "player_name": pname, "player_key": pkey,
                    "position": item.get("position"), "time": item.get("time"),
                    "goals": item.get("goals"), "own_goals": item.get("own_goals"),
                    "shots": item.get("shots"), "xG": item.get("xG"),
                    "key_passes": item.get("key_passes"), "assists": item.get("assists"),
                    "xA": item.get("xA"), "xGChain": item.get("xGChain"), "xGBuildup": item.get("xGBuildup"),
                    "yellow_card": item.get("yellow_card"), "red_card": item.get("red_card"),
                })
        # Shots
        shots = d.get("shots") or {}
        for side, cid in (("h", h_cid), ("a", a_cid)):
            for s in shots.get(side, []):
                pid = str(s.get("player_id") or "")
                pname = s.get("player")
                pkey = resolver.resolve("understat", pid or None, pname, match_id=mid, season=season, club_id=cid)
                assist_name = s.get("player_assisted")
                assist_key = resolver.resolve("understat", None, assist_name, match_id=mid, season=season, club_id=cid) if assist_name else None
                us_shots_rows.append({
                    "shot_id": s.get("id"), "match_id": mid, "understat_match_id": uid,
                    "season": season, "club_id": cid, "side": side,
                    "understat_player_id": pid, "player_name": pname, "player_key": pkey,
                    "assist_player_name": assist_name, "assist_player_key": assist_key,
                    "minute": int(s["minute"]) if s.get("minute") is not None else None,
                    "result": s.get("result"), "X": float(s["X"]), "Y": float(s["Y"]),
                    "xG": float(s["xG"]), "situation": s.get("situation"),
                    "shotType": s.get("shotType"), "lastAction": s.get("lastAction"),
                })

    us_roster_df = pd.DataFrame(us_roster_rows)
    us_shots_df = pd.DataFrame(us_shots_rows)
    us_roster_df.to_sql("raw_understat_rosters", conn, if_exists="replace", index=False)
    us_shots_df.to_sql("raw_understat_shots", conn, if_exists="replace", index=False)
    log_audit(conn, "understat", "raw_understat_rosters", "understat/cache_matches/*.json", len(us_roster_df), len(us_roster_df))
    log_audit(conn, "understat", "raw_understat_shots", "understat/cache_matches/*.json", len(us_shots_df), len(us_shots_df))

    # Understat player season CSVs
    us_p_all = pd.read_sql_query("SELECT * FROM understat__understat_players_all_2014_2027", raw)
    us_p_all["season_std"] = us_p_all["season"].astype(str).str[:4].astype(int).apply(lambda y: f"{y}/{str(y+1)[-2:]}")
    us_p_all["player_key"] = [
        resolver.resolve("understat", str(pid), pname, season=s)
        for pid, pname, s in zip(us_p_all["id"], us_p_all["player_name"], us_p_all["season_std"])
    ]
    us_p_all.to_sql("raw_understat_players", conn, if_exists="replace", index=False)
    log_audit(conn, "understat", "raw_understat_players", "understat_players_all_2014_2027", len(us_p_all), len(us_p_all))
    print(f"[+] Understat done: rosters={len(us_roster_df):,}, shots={len(us_shots_df):,}, season_players={len(us_p_all):,}")

    # =========================================================================
    # 3. FOTMOB SHOTS, PLAYER STATS & 47_DETAILS
    # =========================================================================
    print("[*] Processing FotMob shots, player stats, and fotmob_epl_47_details...")
    fm_shots_rows = []
    fm_pstats_rows = []
    for (content,) in raw.execute("SELECT content FROM raw_json_files WHERE rel_path LIKE 'sofascore/cache_matches/%'").fetchall():
        d = json.loads(content)
        g = d.get("general", {})
        fmid = g.get("matchId")
        if fmid is None or int(fmid) not in fm_id_to_match:
            continue
        mid, season, h_cid, a_cid = fm_id_to_match[int(fmid)]
        ht_id = g.get("homeTeam", {}).get("id")
        at_id = g.get("awayTeam", {}).get("id")
        cobj = d.get("content") or {}

        # Shotmap
        shots_list = (cobj.get("shotmap") or {}).get("shots") or []
        for s in shots_list:
            tid = s.get("teamId")
            cid = h_cid if tid == ht_id else (a_cid if tid == at_id else lookups["fm_id"].get(tid))
            pid = s.get("playerId")
            pname = s.get("playerName")
            pkey = resolver.resolve("fotmob", str(pid) if pid is not None else None, pname, match_id=mid, season=season, club_id=cid)
            fm_shots_rows.append({
                "shot_id": s.get("id"), "match_id": mid, "fotmob_match_id": int(fmid),
                "season": season, "club_id": cid, "fotmob_player_id": pid,
                "player_name": pname, "player_key": pkey,
                "min": s.get("min"), "min_added": s.get("minAdded"),
                "x": s.get("x"), "y": s.get("y"), "xg": s.get("expectedGoals"),
                "xgot": s.get("expectedGoalsOnTarget"), "event_type": s.get("eventType"),
                "situation": s.get("situation"), "shot_type": s.get("shotType"),
                "is_blocked": s.get("isBlocked"), "is_on_target": s.get("isOnTarget"),
            })

        # PlayerStats
        pstats = cobj.get("playerStats") or {}
        if isinstance(pstats, dict):
            for pid_str, pdata in pstats.items():
                if not isinstance(pdata, dict):
                    continue
                tid = pdata.get("teamId")
                cid = h_cid if tid == ht_id else (a_cid if tid == at_id else lookups["fm_id"].get(tid))
                pname = pdata.get("name")
                pkey = resolver.resolve("fotmob", str(pid_str), pname, match_id=mid, season=season, club_id=cid)
                # Extract key summary stats if present
                rating = None
                mins = None
                for st_group in pdata.get("stats", []):
                    st_map = st_group.get("stats", {})
                    if "FotMob rating" in st_map:
                        rating = (st_map["FotMob rating"].get("stat") or {}).get("value")
                    if "Minutes played" in st_map:
                        mins = (st_map["Minutes played"].get("stat") or {}).get("value")
                fm_pstats_rows.append({
                    "match_id": mid, "fotmob_match_id": int(fmid), "season": season,
                    "club_id": cid, "fotmob_player_id": pid_str, "player_name": pname,
                    "player_key": pkey, "position_id": pdata.get("positionId"),
                    "is_goalkeeper": pdata.get("isGoalkeeper"), "minutes_played": mins,
                    "fotmob_rating": rating,
                })

    pd.DataFrame(fm_shots_rows).to_sql("raw_fotmob_shots", conn, if_exists="replace", index=False)
    pd.DataFrame(fm_pstats_rows).to_sql("raw_fotmob_player_stats", conn, if_exists="replace", index=False)
    log_audit(conn, "fotmob", "raw_fotmob_shots", "sofascore/cache_matches/*.json", len(fm_shots_rows), len(fm_shots_rows))
    log_audit(conn, "fotmob", "raw_fotmob_player_stats", "sofascore/cache_matches/*.json", len(fm_pstats_rows), len(fm_pstats_rows))

    # fotmob_epl_47_details (fix season via match_id!)
    fm47 = pd.read_sql_query("SELECT * FROM sofascore__fotmob_epl_47_details", raw)
    fm47["canonical_match_id"] = fm47["match_id"].map(lambda x: fm_id_to_match.get(int(x), (None,))[0] if pd.notna(x) else None)
    fm47["canonical_season"] = fm47["match_id"].map(lambda x: fm_id_to_match.get(int(x), (None, None))[1] if pd.notna(x) else None)
    fm47["club_id"] = fm47["team_id"].map(lambda x: lookups["fm_id"].get(int(x)) if pd.notna(x) else None)
    fm47["player_key"] = [
        resolver.resolve("fotmob", str(int(pid)) if pd.notna(pid) else None, pname, match_id=mid, season=s, club_id=cid)
        for pid, pname, mid, s, cid in zip(fm47["player_id"], fm47["player_name"], fm47["canonical_match_id"], fm47["canonical_season"], fm47["club_id"])
    ]
    fm47.to_sql("raw_fotmob_47_details", conn, if_exists="replace", index=False)
    log_audit(conn, "fotmob", "raw_fotmob_47_details", "fotmob_epl_47_details.csv", len(fm47), fm47["canonical_match_id"].notna().sum(), "Fixed season via fotmob_match_id bridge")
    print(f"[+] FotMob done: shots={len(fm_shots_rows):,}, player_stats={len(fm_pstats_rows):,}, 47_details={len(fm47):,}")

    # =========================================================================
    # 4. FBREF PLAYER-LEVEL TABLES (9 tables)
    # =========================================================================
    print("[*] Processing 9 FBref player-level tables...")
    fb_url_df = pd.read_sql_query(
        "SELECT DISTINCT MatchURL, match_id, season_std, home_club_id, away_club_id FROM raw_fbref_team_stats WHERE MatchURL IS NOT NULL",
        conn,
    )
    fb_url_map = {
        r["MatchURL"]: (r["match_id"], r["season_std"], r["home_club_id"], r["away_club_id"])
        for _, r in fb_url_df.iterrows()
    }
    fb_player_tables = raw.execute(
        "SELECT table_name, rel_path FROM _catalog WHERE rel_path LIKE 'fbref/%' AND rel_path NOT LIKE '%team%' AND status='ok' ORDER BY rel_path"
    ).fetchall()
    for tbl, rel in fb_player_tables:
        df = pd.read_sql_query(f"SELECT * FROM {tbl}", raw)
        if "Match_Date" in df.columns and "Home_Team" in df.columns:
            df["date_iso"] = df["Match_Date"].astype(str).str[:10]
            df["season_std"] = df["date_iso"].apply(infer_season_from_date)
            df["home_club_id"] = df["Home_Team"].astype(str).str.strip().map(lookups["fbref"])
            df["away_club_id"] = df["Away_Team"].astype(str).str.strip().map(lookups["fbref"])
            df["match_id"] = [make_match_id(s, h, a) if pd.notna(h) and pd.notna(a) else None for s, h, a in zip(df["season_std"], df["home_club_id"], df["away_club_id"])]
            for url, mid, s, h, a in zip(df.get("MatchURL", []), df["match_id"], df["season_std"], df["home_club_id"], df["away_club_id"]):
                if url and url not in fb_url_map and mid:
                    fb_url_map[url] = (mid, s, h, a)
        else:
            df["date_iso"] = df["Date"].astype(str).str[:10] if "Date" in df.columns else None
            df["match_id"] = df["MatchURL"].map(lambda u: fb_url_map.get(u, (None,))[0])
            df["season_std"] = df["MatchURL"].map(lambda u: fb_url_map.get(u, (None, None))[1])
            df["home_club_id"] = df["MatchURL"].map(lambda u: fb_url_map.get(u, (None, None, None))[2])
            df["away_club_id"] = df["MatchURL"].map(lambda u: fb_url_map.get(u, (None, None, None, None))[3])

        team_col = "Team" if "Team" in df.columns else ("Squad" if "Squad" in df.columns else None)
        if team_col:
            df["club_id"] = df[team_col].astype(str).str.strip().map(lookups["fbref"])
        elif "Home_Away" in df.columns:
            df["club_id"] = [h if ha == "Home" else a for ha, h, a in zip(df["Home_Away"], df["home_club_id"], df["away_club_id"])]
        else:
            df["club_id"] = None
        p_col = "Player" if "Player" in df.columns else ("Player_Name" if "Player_Name" in df.columns else None)
        if p_col:
            pid_col = "Player_Href" if "Player_Href" in df.columns else None
            df["player_key"] = [
                resolver.resolve("fbref", str(pid) if pid_col and pd.notna(pid) else None, pname, match_id=mid, season=s, club_id=cid)
                for pid, pname, mid, s, cid in zip(
                    df[pid_col] if pid_col else [None] * len(df),
                    df[p_col],
                    df["match_id"],
                    df["season_std"],
                    df["club_id"],
                )
            ]
        out_tbl = "raw_" + tbl.replace("fbref__ENG_M_1st_", "fbref_")
        df.to_sql(out_tbl, conn, if_exists="replace", index=False)
        log_audit(conn, "fbref", out_tbl, rel, len(df), df["match_id"].isin(valid_match_ids).sum())
        print(f"  [+] {out_tbl}: {len(df):,} rows")

    # =========================================================================
    # 5. FPL MERGED GW (10 seasons) & CLEANED PLAYERS (5 seasons)
    # =========================================================================
    print("[*] Processing FPL player GW & season tables...")
    fpl_gw_tables = raw.execute(
        "SELECT table_name, rel_path FROM _catalog WHERE rel_path LIKE 'fpl_api/fpl_merged_gw_%' AND status='ok' ORDER BY rel_path"
    ).fetchall()
    # Build (season, fpl_fixture_id) -> (match_id, home_club_id, away_club_id) lookup
    fpl_fix_df = pd.read_sql_query("SELECT season, id, match_id, home_club_id, away_club_id FROM raw_fpl_fixtures WHERE match_id IS NOT NULL", conn)
    fix_map = {(r["season"], int(r["id"])): (r["match_id"], r["home_club_id"], r["away_club_id"]) for _, r in fpl_fix_df.iterrows()}

    # For 2016/17 & 2017/18 (no fpl_fixtures files), build lookup from dim_match
    dm_all = pd.read_sql_query("SELECT match_id, season, date, home_club_id, away_club_id, home_goals, away_goals FROM dim_match", conn)
    dm_date_score = defaultdict(list)
    dm_by_opp_date = {}
    for _, mr in dm_all.iterrows():
        dm_date_score[(mr["season"], mr["date"], mr["home_goals"], mr["away_goals"])].append((mr["match_id"], mr["home_club_id"], mr["away_club_id"]))
        dm_by_opp_date[(mr["season"], mr["away_club_id"], True, mr["date"])] = (mr["match_id"], mr["home_club_id"])
        dm_by_opp_date[(mr["season"], mr["home_club_id"], False, mr["date"])] = (mr["match_id"], mr["away_club_id"])

    fpl_gw_frames = []
    for tbl, rel in fpl_gw_tables:
        m = re.search(r"(\d{4})_(\d{4})", rel)
        season = f"{m.group(1)}/{m.group(2)[-2:]}"
        df = pd.read_sql_query(f"SELECT * FROM {tbl}", raw)
        df["season"] = season
        if "team" in df.columns:
            df["club_id"] = df["team"].astype(str).str.strip().map(lookups["fpl"])
        else:
            df["club_id"] = None

        # Infer opponent_team -> club_id for seasons without fpl_fixtures (2016/17, 2017/18)
        opp_id_to_cid = {}
        if season in ("2016/17", "2017/18") and "opponent_team" in df.columns:
            for _, r in df.iterrows():
                ot = int(r["opponent_team"]) if pd.notna(r.get("opponent_team")) else None
                if ot is None or ot in opp_id_to_cid:
                    continue
                d_uk, _ = utc_to_uk(r.get("kickoff_time"))
                hg = int(r["team_h_score"]) if pd.notna(r.get("team_h_score")) else None
                ag = int(r["team_a_score"]) if pd.notna(r.get("team_a_score")) else None
                cands = dm_date_score.get((season, d_uk, hg, ag), [])
                if len(cands) == 1:
                    _, hc, ac = cands[0]
                    wh = r.get("was_home") in (True, 1, "True", "true")
                    opp_id_to_cid[ot] = ac if wh else hc

        mids = []
        cids = []
        for idx, r in df.iterrows():
            fid = r.get("fixture")
            finfo = fix_map.get((season, int(fid))) if pd.notna(fid) else None
            wh = r.get("was_home") in (True, 1, "True", "true")
            mid = finfo[0] if finfo else None
            cid = r["club_id"]
            if not cid and finfo:
                cid = finfo[1] if wh else finfo[2]
            if (not mid or not cid) and opp_id_to_cid and pd.notna(r.get("opponent_team")):
                opp_cid = opp_id_to_cid.get(int(r["opponent_team"]))
                d_uk, _ = utc_to_uk(r.get("kickoff_time"))
                hit = dm_by_opp_date.get((season, opp_cid, wh, d_uk))
                if hit:
                    mid, cid = hit[0], hit[1]
            mids.append(mid)
            cids.append(cid)
        df["match_id"] = mids
        df["club_id"] = cids
        df["player_key"] = [
            resolver.resolve(
                "fpl",
                f"{season}_{int(el)}" if pd.notna(el) else None,
                re.sub(r"_\d+$", "", str(pname)).replace("_", " "),
                match_id=mid,
                season=season,
                club_id=cid,
            )
            for el, pname, mid, cid in zip(df["element"], df["name"], df["match_id"], df["club_id"])
        ]
        fpl_gw_frames.append(df)

    fpl_gw_all = pd.concat(fpl_gw_frames, ignore_index=True)
    fpl_gw_all.to_sql("raw_fpl_player_gw", conn, if_exists="replace", index=False)
    log_audit(conn, "fpl", "raw_fpl_player_gw", "fpl_merged_gw (10 seasons)", len(fpl_gw_all), fpl_gw_all["match_id"].notna().sum())
    print(f"[+] raw_fpl_player_gw: {len(fpl_gw_all):,} rows ({fpl_gw_all['match_id'].notna().sum():,} mapped to match_id)")

    fpl_cp_tables = raw.execute(
        "SELECT table_name, rel_path FROM _catalog WHERE rel_path LIKE 'fpl_api/fpl_cleaned_players_%' AND status='ok' ORDER BY rel_path"
    ).fetchall()
    fpl_cp_frames = []
    for tbl, rel in fpl_cp_tables:
        m = re.search(r"(\d{4})_(\d{4})", rel)
        season = f"{m.group(1)}/{m.group(2)[-2:]}"
        df = pd.read_sql_query(f"SELECT * FROM {tbl}", raw)
        df["season"] = season
        full_names = (df["first_name"].fillna("") + " " + df["second_name"].fillna("")).str.strip()
        df["player_key"] = [
            resolver.resolve("fpl", None, fn, season=season)
            for fn in full_names
        ]
        fpl_cp_frames.append(df)
    fpl_cp_all = pd.concat(fpl_cp_frames, ignore_index=True)
    fpl_cp_all.to_sql("raw_fpl_players_season", conn, if_exists="replace", index=False)
    log_audit(conn, "fpl", "raw_fpl_players_season", "fpl_cleaned_players (5 seasons)", len(fpl_cp_all), len(fpl_cp_all))

    # =========================================================================
    # 6. EXPORT dim_player & map_player_ids
    # =========================================================================
    all_dim_players = list(resolver.tm_players.values()) + list(resolver.standalone_players.values())
    dim_player_df = pd.DataFrame(all_dim_players)
    dim_player_df.to_sql("dim_player", conn, if_exists="replace", index=False)
    conn.execute("CREATE UNIQUE INDEX idx_dim_player_key ON dim_player(player_key);")

    map_player_df = pd.DataFrame(list(resolver.mappings_log.values()))
    map_player_df.to_sql("map_player_ids", conn, if_exists="replace", index=False)
    conn.execute("CREATE INDEX idx_map_player_key ON map_player_ids(player_key);")
    dim_player_df.to_csv(ROOT / "mapped_epl" / "mappings" / "dim_player.csv", index=False)
    map_player_df.to_csv(ROOT / "mapped_epl" / "mappings" / "map_player_ids.csv", index=False)

    # Print player resolution breakdown
    print("\n=== PLAYER ENTITY RESOLUTION SUMMARY ===")
    for r in conn.execute("SELECT source, match_method, COUNT(*) FROM map_player_ids GROUP BY source, match_method ORDER BY source, COUNT(*) DESC"):
        print(f"  {r[0]:10s} | {r[1]:20s} : {r[2]:,}")

    conn.commit()
    raw.close()
    conn.close()
    print(f"\nPlayers & Events mapping finished in {time.time() - t0:.1f}s -> {MAPPED_DB}")


if __name__ == "__main__":
    run_players_and_events()
