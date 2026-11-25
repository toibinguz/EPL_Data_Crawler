#!/usr/bin/env python3
"""
Ingest WhoScored / Opta match events into mapped_epl/epl_clean.db.
Creates table `clean_match_events_opta` with full spatial event coordinates
(x, y, end_x, end_y, qualifiers, pass vectors, defensive actions).
"""

import json
import sqlite3
import sys
from pathlib import Path

# Add scripts to sys.path
sys.path.append(str(Path(__file__).parent.parent / "scripts"))
from whoscored_downloader import extract_match_centre_data


DB_PATH = Path(__file__).parent / "epl_clean.db"


def init_db(conn: sqlite3.Connection):
    """Ensure clean_match_events_opta table exists with appropriate schema."""
    conn.execute("""
    CREATE TABLE IF NOT EXISTS clean_match_events_opta (
        opta_event_id BIGINT PRIMARY KEY,
        match_id TEXT NOT NULL,
        season TEXT NOT NULL,
        period TEXT,
        minute INTEGER,
        second INTEGER,
        expanded_minute INTEGER,
        club_id TEXT NOT NULL,
        player_id INTEGER,
        player_key TEXT,
        player_name TEXT,
        event_type TEXT NOT NULL,
        outcome TEXT,
        x REAL,
        y REAL,
        end_x REAL,
        end_y REAL,
        is_touch INTEGER,
        is_shot INTEGER,
        is_goal INTEGER,
        qualifiers TEXT,
        FOREIGN KEY (match_id) REFERENCES clean_matches(match_id),
        FOREIGN KEY (club_id) REFERENCES dim_club(club_id),
        FOREIGN KEY (player_key) REFERENCES dim_player(player_key)
    );
    """)
    conn.execute("CREATE INDEX IF NOT EXISTS idx_opta_match ON clean_match_events_opta(match_id);")
    conn.execute("CREATE INDEX IF NOT EXISTS idx_opta_player ON clean_match_events_opta(player_key);")
    conn.execute("CREATE INDEX IF NOT EXISTS idx_opta_type ON clean_match_events_opta(event_type);")
    conn.commit()


def build_player_resolver(conn: sqlite3.Connection):
    """Build a lookup table from normalized player name to player_key."""
    cur = conn.cursor()
    cur.execute("SELECT player_key, canonical_name, norm_name FROM dim_player")
    name_map = {}
    for pkey, cname, nname in cur.fetchall():
        if cname:
            name_map[cname.strip().lower()] = pkey
        if nname:
            name_map[nname.strip().lower()] = pkey
    return name_map


def ingest_file(file_path: Path):
    if not file_path.exists():
        raise FileNotFoundError(f"File not found: {file_path}")

    print(f"[*] Reading file: {file_path.name}")
    content = file_path.read_text(encoding="utf-8", errors="ignore")

    if file_path.suffix == ".json":
        data = json.loads(content)
        if "matchCentreData" in data:
            data = data["matchCentreData"]
    else:
        data = extract_match_centre_data(content)

    if not data or "events" not in data:
        raise ValueError("Could not extract matchCentreData from file.")

    events = data["events"]
    player_dict = data.get("playerIdNameDictionary", {})
    home_meta = data.get("home", {})
    away_meta = data.get("away", {})

    home_name = home_meta.get("name", "Home")
    away_name = away_meta.get("name", "Away")
    home_id = home_meta.get("teamId")
    away_id = away_meta.get("teamId")

    print(f"[*] Match detected: {home_name} vs {away_name} ({len(events):,} events)")

    # Resolve club_id
    team_to_club = {}
    if "City" in home_name:
        team_to_club[home_id] = "MCI"
    elif "Fulham" in home_name:
        team_to_club[home_id] = "FUL"

    if "Fulham" in away_name:
        team_to_club[away_id] = "FUL"
    elif "City" in away_name:
        team_to_club[away_id] = "MCI"

    # Default fallback
    if home_id not in team_to_club:
        team_to_club[home_id] = "MCI" if "City" in home_name else "HOME"
    if away_id not in team_to_club:
        team_to_club[away_id] = "FUL" if "Fulham" in away_name else "AWAY"

    match_id = f"EPL_2026_2027_{team_to_club[home_id]}_{team_to_club[away_id]}"
    season = "2026/27"

    conn = sqlite3.connect(DB_PATH)
    init_db(conn)
    player_resolver = build_player_resolver(conn)

    # Ingest events
    rows = []
    unresolved_players = set()

    for e in events:
        eid = int(e["id"])
        period = e.get("period", {}).get("displayName", "Regular") if isinstance(e.get("period"), dict) else str(e.get("period"))
        minute = int(e.get("minute", 0))
        second = int(e.get("second", 0))
        exp_minute = int(e.get("expandedMinute", minute))

        team_id = e.get("teamId")
        club_id = team_to_club.get(team_id, "UNK")

        pid = e.get("playerId")
        pname = player_dict.get(str(pid)) or player_dict.get(pid)
        pkey = None
        if pname:
            norm_pname = pname.strip().lower()
            pkey = player_resolver.get(norm_pname)
            if not pkey:
                unresolved_players.add(pname)

        etype = e.get("type", {}).get("displayName", "Unknown") if isinstance(e.get("type"), dict) else str(e.get("type"))
        outcome = e.get("outcomeType", {}).get("displayName", "Successful") if isinstance(e.get("outcomeType"), dict) else str(e.get("outcomeType"))

        x = e.get("x")
        y = e.get("y")
        end_x = e.get("endX")
        end_y = e.get("endY")

        quals = e.get("qualifiers", [])
        # Check qualifiers for endX / endY if missing
        if end_x is None:
            for q in quals:
                if q.get("type", {}).get("displayName") == "PassEndX":
                    try:
                        end_x = float(q.get("value"))
                    except Exception:
                        pass
        if end_y is None:
            for q in quals:
                if q.get("type", {}).get("displayName") == "PassEndY":
                    try:
                        end_y = float(q.get("value"))
                    except Exception:
                        pass

        is_touch = 1 if e.get("isTouch") else 0
        is_shot = 1 if e.get("isShot") or etype in ["Miss", "Post", "AttemptSaved", "Goal"] else 0
        is_goal = 1 if e.get("isGoal") or etype == "Goal" else 0

        quals_json = json.dumps(quals, ensure_ascii=False)

        rows.append((
            eid, match_id, season, period, minute, second, exp_minute,
            club_id, pid, pkey, pname, etype, outcome,
            x, y, end_x, end_y, is_touch, is_shot, is_goal, quals_json
        ))

    cur = conn.cursor()
    cur.executemany("""
    INSERT OR REPLACE INTO clean_match_events_opta (
        opta_event_id, match_id, season, period, minute, second, expanded_minute,
        club_id, player_id, player_key, player_name, event_type, outcome,
        x, y, end_x, end_y, is_touch, is_shot, is_goal, qualifiers
    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?);
    """, rows)
    conn.commit()

    print(f"[+] Successfully inserted/updated {len(rows):,} Opta events into `clean_match_events_opta`.")
    print(f"[+] Match ID: {match_id}")
    cur.execute("SELECT COUNT(*), COUNT(DISTINCT player_key) FROM clean_match_events_opta WHERE match_id = ?", (match_id,))
    total_ev, resolved_p = cur.fetchone()
    print(f"[+] Total in DB: {total_ev:,} events | Resolved to dim_player: {resolved_p} unique players")
    if unresolved_players:
        print(f"[!] Unresolved player names ({len(unresolved_players)}): {list(unresolved_players)[:5]}")
    conn.close()


if __name__ == "__main__":
    file_path = Path("Manchester City 1-1 Fulham - Premier League 2026_2027 Live.html")
    if len(sys.argv) > 1:
        file_path = Path(sys.argv[1])
    ingest_file(file_path)
