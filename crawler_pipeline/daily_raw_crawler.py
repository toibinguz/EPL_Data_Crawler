"""
crawler_pipeline/daily_raw_crawler.py
Performs daily incremental raw crawling for the active Premier League season (2026/27).
Architecture: Raw Atomic Only (No layer 2 processing).
Source of Truth: Hugging Face Dataset Hub (Toibinguz/epl-premier-league-dataset).

Flow:
1. Query Hugging Face metadata (0.4s) to discover match IDs already archived.
2. Calculate delta (matches finished - matches on HF).
3. Only crawl individual JSON files for NEW matches in delta.
4. Overwrite daily cumulative raw snapshot feeds (football_data_raw_2627.csv, overviews).
5. Sync 2026_2027_new folder directly to Hugging Face Hub in seconds.
"""

import os
import sys
import json
import gzip
import time
import shutil
import datetime
import urllib.request
import urllib.error
from pathlib import Path
from huggingface_hub import HfApi

ROOT_DIR = Path(__file__).resolve().parent.parent
RAW_DIR = ROOT_DIR / "raw_data"
SEASON_NEW_DIR = ROOT_DIR / "2026_2027_new"
TOKEN_FILE = ROOT_DIR / "crawler_pipeline" / "hf_access_key.txt"
FALLBACK_TOKEN_FILE = ROOT_DIR / "access_token_hf.txt"
REPO_ID = "Toibinguz/epl-premier-league-dataset"

# Ensure directories exist
SEASON_NEW_DIR.mkdir(parents=True, exist_ok=True)
(SEASON_NEW_DIR / "matches" / "fotmob").mkdir(parents=True, exist_ok=True)
(SEASON_NEW_DIR / "matches" / "understat").mkdir(parents=True, exist_ok=True)

HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36",
    "Accept": "application/json, text/plain, */*",
}


def log(msg: str):
    now = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"[{now}] {msg}", flush=True)


def get_token():
    if "HF_TOKEN" in os.environ and os.environ["HF_TOKEN"].strip():
        return os.environ["HF_TOKEN"].strip()
    for f in [TOKEN_FILE, FALLBACK_TOKEN_FILE]:
        if f.exists():
            t = f.read_text(encoding="utf-8").strip()
            if t:
                return t
    return None


def get_hf_existing_match_ids(api: HfApi, path_in_repo: str) -> set:
    """Query Hugging Face metadata tree to get all match IDs already saved on the Hub.
    Executes in <0.5s without downloading any file content.
    """
    if not api:
        return set()
    try:
        items = api.list_repo_tree(repo_id=REPO_ID, repo_type="dataset", path_in_repo=path_in_repo)
        existing_ids = set()
        for item in items:
            name = item.path.split("/")[-1]
            if name.endswith(".json"):
                existing_ids.add(name[:-5])
        return existing_ids
    except Exception as e:
        log(f"[*] Note: Could not query HF metadata for {path_in_repo} ({e}). Falling back to local check.")
        return set()


def update_football_data(season_code="2627"):
    """Download the latest 2026/27 cumulative match & odds CSV from Football-Data.co.uk.
    Since Football-Data maintains the entire season in E0.csv, overwriting provides
    an atomic snapshot of all matches played so far with zero merge overhead.
    """
    log("=" * 70)
    log("[1/5] Updating Football-Data.co.uk (Cumulative CSV)...")
    url = f"https://www.football-data.co.uk/mmz4281/{season_code}/E0.csv"
    target_file = SEASON_NEW_DIR / f"football_data_raw_{season_code}.csv"

    try:
        req = urllib.request.Request(url, headers=HEADERS)
        with urllib.request.urlopen(req, timeout=20) as resp:
            content = resp.read()
            if len(content) > 100:
                with open(target_file, "wb") as f:
                    f.write(content)
                log(f"[+] Successfully updated {target_file.name} ({len(content)/1024:.1f} KB)")
                # If local raw_data exists, keep a mirror copy
                raw_dest = RAW_DIR / "football_data_co_uk" / f"football_data_raw_{season_code}.csv"
                if RAW_DIR.exists() and raw_dest.parent.exists():
                    shutil.copy2(target_file, raw_dest)
            else:
                log("[!] Football-Data returned empty response.")
    except Exception as e:
        log(f"[!] Error updating Football-Data: {e}")


def update_fpl_api():
    """Update latest Fantasy Premier League bootstrap-static and fixtures snapshots."""
    log("=" * 70)
    log("[2/5] Updating Fantasy Premier League (FPL API)...")

    endpoints = [
        ("fpl_bootstrap_latest.json", "https://fantasy.premierleague.com/api/bootstrap-static/"),
        ("fpl_fixtures_latest.json", "https://fantasy.premierleague.com/api/fixtures/"),
    ]

    for fname, url in endpoints:
        try:
            req = urllib.request.Request(url, headers=HEADERS)
            with urllib.request.urlopen(req, timeout=25) as resp:
                data = json.loads(resp.read().decode("utf-8"))
                target = SEASON_NEW_DIR / fname
                with open(target, "w", encoding="utf-8") as f:
                    json.dump(data, f)
                log(f"[+] Successfully updated {fname}")
                # Mirror to raw_data if present
                raw_dest = RAW_DIR / "fpl_api" / fname
                if RAW_DIR.exists() and raw_dest.parent.exists():
                    shutil.copy2(target, raw_dest)
        except Exception as e:
            log(f"[!] Error updating {fname}: {e}")


def update_understat(api: HfApi = None, year="2026", season_tag="2026_2027"):
    """Update Understat league overview and fetch any new match details.
    Uses Hugging Face Hub as the single source of truth for existing matches.
    """
    log("=" * 70)
    log("[3/5] Updating Understat (Overview + Atomic Match Details)...")
    target_match_dir = SEASON_NEW_DIR / "matches" / "understat"
    target_match_dir.mkdir(parents=True, exist_ok=True)

    url = f"https://understat.com/getLeagueData/EPL/{year}"
    understat_headers = {
        **HEADERS,
        "X-Requested-With": "XMLHttpRequest",
        "Referer": f"https://understat.com/league/EPL/{year}",
        "Accept-Encoding": "gzip",
    }

    overview_file = SEASON_NEW_DIR / f"understat_raw_{season_tag}.json"
    matches_to_fetch = []
    overview_ok = False

    try:
        # Step A: Query existing matches on Hugging Face and local disk
        existing_hf_ids = get_hf_existing_match_ids(api, "2026_2027_new/matches/understat")
        existing_local_ids = {f.stem for f in target_match_dir.glob("*.json")}
        already_saved_ids = existing_hf_ids | existing_local_ids
        log(f"[*] Understat matches already archived: {len(already_saved_ids)} ({len(existing_hf_ids)} on HF, {len(existing_local_ids)} on disk)")

        # Step B: Fetch current league overview
        req = urllib.request.Request(url, headers=understat_headers)
        with urllib.request.urlopen(req, timeout=30) as resp:
            raw = resp.read()
            if raw[:2] == b"\x1f\x8b":
                raw = gzip.decompress(raw)
            txt = raw.decode("utf-8", errors="replace").strip()
            if not txt.startswith("{"):
                raise ValueError("Understat returned empty or non-JSON response (blocked/changed upstream)")
            data = json.loads(txt)
            overview_ok = True
            with open(overview_file, "w", encoding="utf-8") as f:
                json.dump(data, f)
            log(f"[+] Successfully updated {overview_file.name}")
            # Mirror to raw_data if present
            raw_dest = RAW_DIR / "understat" / f"understat_raw_{season_tag}.json"
            if RAW_DIR.exists() and raw_dest.parent.exists():
                shutil.copy2(overview_file, raw_dest)

            # Step C: Identify finished matches not yet archived
            for m in data.get("dates", []):
                if m.get("isResult") is True and "id" in m:
                    m_id = str(m["id"])
                    if m_id not in already_saved_ids:
                        matches_to_fetch.append(m_id)
    except Exception as e:
        log(f"[!] Understat overview skipped ({e}). Keeping existing archived matches.")

    # Step D: Fetch only the new delta matches
    if not overview_ok:
        return
    if matches_to_fetch:
        log(f"[*] Found {len(matches_to_fetch)} NEW completed Understat matches to crawl: {matches_to_fetch}")
        for m_id in matches_to_fetch:
            m_url = f"https://understat.com/getMatchData/{m_id}"
            m_req = urllib.request.Request(m_url, headers={**understat_headers, "Accept-Encoding": "gzip"})
            try:
                with urllib.request.urlopen(m_req, timeout=15) as resp:
                    raw = resp.read()
                    if raw[:2] == b"\x1f\x8b":
                        raw = gzip.decompress(raw)
                    m_data = json.loads(raw.decode("utf-8"))
                    target = target_match_dir / f"{m_id}.json"
                    with open(target, "w", encoding="utf-8") as f:
                        json.dump(m_data, f)
                    log(f"    [+] Saved Understat Match {m_id}.json")
                    # Mirror to raw_data cache if present
                    raw_cache = RAW_DIR / "understat" / "cache_matches" / f"{m_id}.json"
                    if RAW_DIR.exists() and raw_cache.parent.exists():
                        shutil.copy2(target, raw_cache)
                time.sleep(0.3)
            except Exception as e:
                log(f"    [!] Failed to fetch Understat match {m_id}: {e}")
    else:
        log("[+] All completed Understat matches are already archived on Hugging Face (delta = 0).")


def update_fotmob(api: HfApi = None, season_tag="2026_2027"):
    """Update FotMob fixtures and fetch any new match details.
    Uses Hugging Face Hub as the single source of truth for existing matches.
    """
    log("=" * 70)
    log("[4/5] Updating FotMob (Fixtures + Atomic Match Details)...")
    target_match_dir = SEASON_NEW_DIR / "matches" / "fotmob"
    target_match_dir.mkdir(parents=True, exist_ok=True)

    url = "https://www.fotmob.com/api/data/leagues?id=47&season=2026%2F2027"
    overview_file = SEASON_NEW_DIR / f"fotmob_fixtures_{season_tag}.json"
    matches_to_fetch = []

    try:
        # Step A: Query existing matches on Hugging Face and local disk
        existing_hf_ids = get_hf_existing_match_ids(api, "2026_2027_new/matches/fotmob")
        existing_local_ids = {f.stem for f in target_match_dir.glob("*.json")}
        already_saved_ids = existing_hf_ids | existing_local_ids
        log(f"[*] FotMob matches already archived: {len(already_saved_ids)} ({len(existing_hf_ids)} on HF, {len(existing_local_ids)} on disk)")

        # Step B: Fetch current fixtures
        req = urllib.request.Request(url, headers=HEADERS)
        with urllib.request.urlopen(req, timeout=25) as resp:
            data = json.loads(resp.read().decode("utf-8"))
            with open(overview_file, "w", encoding="utf-8") as f:
                json.dump(data, f)
            log(f"[+] Successfully updated {overview_file.name}")
            # Mirror to raw_data if present
            raw_dest = RAW_DIR / "sofascore" / f"fotmob_fixtures_{season_tag}.json"
            if RAW_DIR.exists() and raw_dest.parent.exists():
                shutil.copy2(overview_file, raw_dest)

            # Step C: Identify finished matches not yet archived
            fixtures = data.get("fixtures", {}).get("allMatches", [])
            for m in fixtures:
                m_id = str(m.get("id"))
                status = m.get("status", {})
                if status.get("finished", False) and m_id:
                    if m_id not in already_saved_ids:
                        matches_to_fetch.append(m_id)
    except Exception as e:
        log(f"[!] Error fetching FotMob fixtures: {e}")

    # Step D: Fetch only the new delta matches
    if matches_to_fetch:
        log(f"[*] Found {len(matches_to_fetch)} NEW completed FotMob matches to crawl: {matches_to_fetch}")
        for m_id in matches_to_fetch:
            m_url = f"https://www.fotmob.com/api/data/matchDetails?matchId={m_id}"
            m_req = urllib.request.Request(m_url, headers=HEADERS)
            try:
                with urllib.request.urlopen(m_req, timeout=15) as resp:
                    m_data = json.loads(resp.read().decode("utf-8"))
                    target = target_match_dir / f"{m_id}.json"
                    with open(target, "w", encoding="utf-8") as f:
                        json.dump(m_data, f)
                    log(f"    [+] Saved FotMob Match {m_id}.json")
                    # Mirror to raw_data cache if present
                    raw_cache = RAW_DIR / "sofascore" / "cache_matches" / f"{m_id}.json"
                    if RAW_DIR.exists() and raw_cache.parent.exists():
                        shutil.copy2(target, raw_cache)
                time.sleep(0.3)
            except Exception as e:
                log(f"    [!] Failed to fetch FotMob match {m_id}: {e}")
    else:
        log("[+] All completed FotMob matches are already archived on Hugging Face (delta = 0).")


EPL2627_STADIUM_COORDS = {
    "Arsenal": (51.55, -0.11),
    "Aston Villa": (52.51, -1.88),
    "Bournemouth": (50.74, -1.84),
    "Brentford": (51.49, -0.29),
    "Brighton": (50.86, -0.08),
    "Chelsea": (51.48, -0.19),
    "Coventry": (52.45, -1.50),
    "Crystal Palace": (51.40, -0.09),
    "Everton": (53.44, -2.97),
    "Fulham": (51.47, -0.22),
    "Hull": (53.75, -0.37),
    "Ipswich": (52.06, 1.14),
    "Leeds": (53.78, -1.57),
    "Liverpool": (53.43, -2.96),
    "Man City": (53.48, -2.20),
    "Man United": (53.46, -2.29),
    "Newcastle": (54.98, -1.62),
    "Nott'm Forest": (52.94, -1.13),
    "Sunderland": (54.91, -1.39),
    "Tottenham": (51.60, -0.07),
}


def update_open_meteo(season_code="2627", season_tag="2026_2027"):
    """Fetch missing hourly match weather from Open-Meteo Archive (with forecast past_days fallback)."""
    import csv

    log("=" * 70)
    log("[5/5] Updating Open-Meteo Weather Cache (Incremental)...")
    w_target = SEASON_NEW_DIR / f"weather_cache_{season_tag}.json"
    fd_file = SEASON_NEW_DIR / f"football_data_raw_{season_code}.csv"

    cache = {}
    if w_target.exists():
        try:
            cache = json.loads(w_target.read_text(encoding="utf-8"))
        except Exception:
            cache = {}

    if not fd_file.exists():
        log(f"[!] Missing {fd_file.name}, skipping weather update.")
        return

    needed_by_coord = {}
    with open(fd_file, "r", encoding="utf-8-sig", newline="") as f:
        for r in csv.DictReader(f):
            ht = (r.get("HomeTeam") or "").strip()
            dt_raw = (r.get("Date") or "").strip()
            tm_raw = (r.get("Time") or "15:00").strip()
            coords = EPL2627_STADIUM_COORDS.get(ht)
            if not coords or not dt_raw:
                continue
            try:
                dt_iso = datetime.datetime.strptime(dt_raw, "%d/%m/%Y").strftime("%Y-%m-%d")
            except ValueError:
                try:
                    dt_iso = datetime.datetime.strptime(dt_raw, "%Y-%m-%d").strftime("%Y-%m-%d")
                except ValueError:
                    continue
            hour = int(tm_raw.split(":")[0]) if ":" in tm_raw else 15
            lat, lon = coords
            wkey = f"{lat:.2f}_{lon:.2f}_{dt_iso}_{hour:02d}"
            if wkey not in cache:
                needed_by_coord.setdefault((lat, lon), []).append((wkey, dt_iso, hour))

    total_missing = sum(len(v) for v in needed_by_coord.values())
    if total_missing == 0:
        log(f"[+] Weather cache already up to date ({len(cache)} entries).")
        return

    log(f"[*] Fetching weather for {total_missing} new matches across {len(needed_by_coord)} stadiums...")
    added = 0
    for (lat, lon), items in needed_by_coord.items():
        d_min = min(x[1] for x in items)
        d_max = max(x[1] for x in items)
        hourly_lookup = {}

        # Query Archive first, then Forecast past_days=10 for very recent matches
        urls = [
            f"https://archive-api.open-meteo.com/v1/archive?latitude={lat:.2f}&longitude={lon:.2f}&start_date={d_min}&end_date={d_max}&hourly=temperature_2m,relative_humidity_2m,precipitation,wind_speed_10m,weather_code&timezone=Europe%2FLondon",
            f"https://api.open-meteo.com/v1/forecast?latitude={lat:.2f}&longitude={lon:.2f}&past_days=10&forecast_days=1&hourly=temperature_2m,relative_humidity_2m,precipitation,wind_speed_10m,weather_code&timezone=Europe%2FLondon",
        ]
        for api_url in urls:
            try:
                req = urllib.request.Request(api_url, headers=HEADERS)
                with urllib.request.urlopen(req, timeout=15) as resp:
                    payload = json.loads(resp.read().decode("utf-8"))
                hr = payload.get("hourly") or {}
                times = hr.get("time") or []
                for idx_t, ts in enumerate(times):
                    t_v = hr["temperature_2m"][idx_t]
                    h_v = hr["relative_humidity_2m"][idx_t]
                    p_v = hr["precipitation"][idx_t]
                    w_v = hr["wind_speed_10m"][idx_t]
                    c_v = hr["weather_code"][idx_t]
                    if t_v is not None:
                        hourly_lookup[ts] = {
                            "temp": t_v,
                            "humidity": h_v,
                            "precipitation": p_v,
                            "wind": w_v,
                            "weather_code": c_v,
                        }
            except Exception as e:
                log(f"    [!] Open-Meteo query warning ({lat:.2f},{lon:.2f}): {e}")
            time.sleep(0.1)

        for wkey, dt_iso, hour in items:
            ts_key = f"{dt_iso}T{hour:02d}:00"
            if ts_key in hourly_lookup:
                cache[wkey] = hourly_lookup[ts_key]
                added += 1

    w_target.write_text(json.dumps(cache), encoding="utf-8")
    raw_dest = RAW_DIR / "open_meteo" / f"weather_cache_{season_tag}.json"
    if RAW_DIR.exists() and raw_dest.parent.exists():
        shutil.copy2(w_target, raw_dest)
    log(f"[+] Updated weather_cache_{season_tag}.json: +{added} new entries (total={len(cache)}).")


def sync_season_to_hf(api: HfApi):
    """Sync the 2026_2027_new folder (active season raw atomic feeds) to Hugging Face."""
    if not api:
        log("[!] No Hugging Face client authenticated, skipping upload.")
        return

    log("=" * 70)
    log(f"[*] Syncing 2026_2027_new folder to Hugging Face: {REPO_ID}...")
    try:
        t0 = time.time()
        api.upload_folder(
            folder_path=str(SEASON_NEW_DIR),
            path_in_repo="2026_2027_new",
            repo_id=REPO_ID,
            repo_type="dataset",
            commit_message=f"Daily raw update: Season 2026_2027 matches and feeds ({datetime.date.today()})",
        )
        elapsed = time.time() - t0
        log(f"[+] Successfully synced 2026_2027_new to Hugging Face in {elapsed:.1f}s! [OK]")
    except Exception as e:
        log(f"[!] Warning on HF season sync: {e}")


def run_daily_raw_pipeline(sync_hf: bool = True):
    log("=" * 80)
    log("STARTING DAILY PREMIER LEAGUE RAW ATOMIC PIPELINE (2026/27)")
    log(f"Target Current Season Folder: {SEASON_NEW_DIR}")
    log("=" * 80)
    start_time = time.time()

    token = get_token()
    api = HfApi(token=token) if token else None
    if api:
        log(f"[+] Authenticated with Hugging Face Hub (Repo: {REPO_ID})")
    else:
        log("[!] Running in offline / unauthenticated mode (local check only)")

    update_football_data("2627")
    update_fpl_api()
    update_understat(api=api, year="2026", season_tag="2026_2027")
    update_fotmob(api=api, season_tag="2026_2027")
    update_open_meteo()

    if sync_hf and api:
        sync_season_to_hf(api)

    elapsed = time.time() - start_time
    log("=" * 80)
    log(f"DAILY RAW CRAWL COMPLETED IN {elapsed:.1f} SECONDS [OK]")
    log("=" * 80)


if __name__ == "__main__":
    do_sync = "--no-sync" not in sys.argv
    run_daily_raw_pipeline(sync_hf=do_sync)
