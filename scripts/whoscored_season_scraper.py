#!/usr/bin/env python3
"""
WhoScored 2026/27 Premier League Season Batch Downloader.

Inherits browser session, stealth persistent context, and DOM extraction from
`scripts/whoscored_downloader.py`.

Workflow:
  1. Reuses the authenticated Chrome persistent profile (`mapped_epl/downloaded_matches/.browser_profile`).
  2. Navigates to Premier League 2026/27 on WhoScored to resolve `stage_id` and `wsCalendar`.
  3. Fetches monthly fixture JSONs (`/tournaments/{stage_id}/data/?d=YYYYMM`) directly inside the browser session.
  4. Filters to completed matches (`FT` / finished) and saves `schedule_2627.json`.
  5. Downloads every match's `.html` and extracts `_matchCentreData.json` into `mapped_epl/whoscored_2627/`,
     automatically skipping already-downloaded matches for seamless resume.
"""

import argparse
import json
import random
import re
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

from playwright.sync_api import sync_playwright

sys.path.append(str(Path(__file__).parent))
from whoscored_downloader import create_browser_context, fetch_match_on_page

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8")

EPL_URL = "https://www.whoscored.com/Regions/252/Tournaments/2/England-Premier-League"
DEFAULT_OUTPUT_DIR = Path("mapped_epl/whoscored_2627")
DEFAULT_PROFILE_DIR = Path("mapped_epl/downloaded_matches/.browser_profile")


def sanitize_slug(text: str) -> str:
    return re.sub(r"[^a-zA-Z0-9]+", "_", text).strip("_")


def wait_for_page_ready(page, timeout_sec: int = 45):
    start = time.time()
    while time.time() - start < timeout_sec:
        title = page.title()
        if "Just a moment..." in title or "Attention Required!" in title or "Cloudflare" in title:
            print(f"[!] Cloudflare challenge detected ('{title}'). Waiting for completion in browser UI...")
            time.sleep(3)
            continue
        html = page.content()
        if "wsCalendar" in html or "Fixtures" in html:
            return html
        time.sleep(1.5)
    return page.content()


def discover_season_schedule(page, output_dir: Path, timeout_sec: int = 45) -> list[dict]:
    """Navigate to EPL page, extract stage_id + wsCalendar, and fetch all monthly fixtures."""
    print(f"[*] Navigating to EPL Tournament page: {EPL_URL}")
    page.goto(EPL_URL, wait_until="commit", timeout=timeout_sec * 1000)
    html = wait_for_page_ready(page, timeout_sec=timeout_sec)

    # Extract stage_id from Fixtures link or URL
    stage_match = re.search(r"/Stages/(\d+)/", html, re.IGNORECASE)
    if not stage_match:
        # Try finding via DOM
        stage_id = page.evaluate("""() => {
            const a = Array.from(document.querySelectorAll('a')).find(el => el.textContent.trim() === 'Fixtures');
            if (a && a.href) {
                const m = a.href.match(/Stages\\/(\\d+)/i);
                return m ? m[1] : null;
            }
            return null;
        }""")
    else:
        stage_id = stage_match.group(1)

    if not stage_id:
        raise RuntimeError("Could not locate Premier League 2026/27 stage_id on WhoScored.")
    print(f"[+] Resolved Premier League stage_id: {stage_id}")

    # Navigate to Fixtures / Stage page to get wsCalendar
    fixtures_url = f"https://www.whoscored.com/Regions/252/Tournaments/2/Seasons/10743/Stages/{stage_id}/Fixtures/England-Premier-League-2026-2027"
    # More reliably, extract the exact Fixtures href from the page!
    exact_fixtures_href = page.evaluate("""() => {
        const a = Array.from(document.querySelectorAll('a')).find(el => el.textContent.trim() === 'Fixtures');
        return a ? a.href : null;
    }""")
    target_cal_url = exact_fixtures_href or f"https://www.whoscored.com/Regions/252/Tournaments/2/Stages/{stage_id}/Fixtures"
    print(f"[*] Loading calendar from: {target_cal_url}")
    page.goto(target_cal_url, wait_until="commit", timeout=timeout_sec * 1000)
    cal_html = wait_for_page_ready(page, timeout_sec=timeout_sec)

    # Extract wsCalendar from JS window or HTML
    ws_cal = page.evaluate("() => window.wsCalendar || null")
    if not ws_cal:
        m = re.search(r"wsCalendar\s*=\s*(\{.+?\});", cal_html, re.DOTALL)
        if m:
            ws_cal = json.loads(m.group(1))

    if not ws_cal or "mask" not in ws_cal:
        # Fallback: generate months from Aug 2026 to current month (Nov 2026)
        print("[!] wsCalendar.mask not found directly, falling back to Aug-Nov 2026...")
        ym_pairs = [("2026", 7), ("2026", 8), ("2026", 9), ("2026", 10)]  # 0-indexed months: 7=Aug, 10=Nov
    else:
        mask = ws_cal["mask"]
        ym_pairs = [(year, int(month)) for year in sorted(mask.keys()) for month in sorted(mask[year], key=int)]

    print(f"[*] Calendar months discovered: {[(y, m + 1) for y, m in ym_pairs]}")

    all_matches = {}
    now_utc = datetime.now(timezone.utc)

    for year, month_zero in ym_pairs:
        d_param = f"{year}{(month_zero + 1):02d}"
        data_url = f"https://www.whoscored.com/tournaments/{stage_id}/data/?d={d_param}"
        print(f"[*] Fetching fixtures for {d_param} ...")

        resp = page.evaluate("""async (url) => {
            const r = await fetch(url, { credentials: 'include' });
            if (!r.ok) return { error: r.status };
            return await r.json();
        }""", data_url)

        if not resp or "error" in resp:
            print(f"[!] Failed fetching {d_param}: {resp}")
            continue

        tournaments = resp.get("tournaments", [])
        count_month = 0
        for t in tournaments:
            for m in t.get("matches", []):
                mid = m.get("id")
                if not mid:
                    continue
                all_matches[mid] = m
                count_month += 1
        print(f"    -> Found {count_month} matches in {d_param}")
        time.sleep(0.8)

    # Filter to played / finished matches
    played = []
    for mid, m in all_matches.items():
        status = m.get("status")
        elapsed = m.get("elapsed", "")
        has_score = m.get("homeScore") is not None and m.get("awayScore") is not None
        start_str = m.get("startTimeUtc", "")
        is_past = False
        if start_str:
            try:
                dt = datetime.fromisoformat(start_str.replace("Z", "+00:00"))
                if dt.tzinfo is None:
                    dt = dt.replace(tzinfo=timezone.utc)
                is_past = dt < now_utc
            except Exception:
                pass
        if has_score or str(elapsed).upper() == "FT" or status == 6 or is_past:
            if has_score:
                played.append(m)

    played.sort(key=lambda x: x.get("startTimeUtc", ""))
    schedule_file = output_dir / "schedule_2627.json"
    schedule_file.write_text(json.dumps(played, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n[+] Total season matches discovered: {len(all_matches)} | Completed matches: {len(played)}")
    print(f"[+] Saved schedule manifest: {schedule_file.resolve()}\n")
    return played


def run_season_scrape(
    output_dir: Path = DEFAULT_OUTPUT_DIR,
    profile_dir: Path = DEFAULT_PROFILE_DIR,
    cdp_port: int | None = None,
    headless: bool = False,
    limit: int | None = None,
    delay_min: float = 2.0,
    delay_max: float = 3.5,
    timeout_sec: int = 45,
):
    output_dir.mkdir(parents=True, exist_ok=True)
    schedule_file = output_dir / "schedule_2627.json"

    with sync_playwright() as p:
        context, _, page = create_browser_context(
            p, profile_dir=profile_dir, cdp_port=cdp_port, headless=headless
        )

        try:
            if schedule_file.exists():
                played = json.loads(schedule_file.read_text(encoding="utf-8"))
                print(f"[*] Loaded cached schedule from {schedule_file} ({len(played)} completed matches)")
                # Check if schedule is fresh; if user wants refresh they can delete schedule_2627.json
            else:
                played = discover_season_schedule(page, output_dir, timeout_sec=timeout_sec)

            if limit is not None:
                targets = played[:limit]
                print(f"[*] Limit applied: processing first {len(targets)} of {len(played)} matches")
            else:
                targets = played

            downloaded = 0
            skipped = 0
            failed = 0

            for idx, m in enumerate(targets, 1):
                mid = m["id"]
                home = sanitize_slug(m.get("homeTeamName", "Home"))
                away = sanitize_slug(m.get("awayTeamName", "Away"))
                date_str = (m.get("startTimeUtc") or "")[:10]
                prefix = f"{date_str}_{mid}_{home}_vs_{away}" if date_str else f"{mid}_{home}_vs_{away}"

                html_path = output_dir / f"{prefix}.html"
                json_path = output_dir / f"{prefix}_matchCentreData.json"

                if html_path.exists() and json_path.exists() and json_path.stat().st_size > 1000:
                    skipped += 1
                    print(f"[{idx}/{len(targets)}] SKIP (already exists): {prefix}")
                    continue

                match_url = f"https://www.whoscored.com/matches/{mid}/live"
                print(f"\n[{idx}/{len(targets)}] Fetching {m.get('homeTeamName')} vs {m.get('awayTeamName')} ({date_str}, ID={mid})")
                try:
                    res = fetch_match_on_page(
                        page,
                        match_url=match_url,
                        output_dir=output_dir,
                        timeout_sec=timeout_sec,
                        file_prefix=prefix,
                    )
                    if res["events_count"] > 0:
                        downloaded += 1
                    else:
                        failed += 1
                except Exception as e:
                    print(f"[!] Error downloading match {mid}: {e}")
                    failed += 1

                if idx < len(targets):
                    wait_s = random.uniform(delay_min, delay_max)
                    time.sleep(wait_s)

            print("\n========================================================")
            print(f"[+] Season 26/27 Scrape Summary")
            print(f"    Output Dir : {output_dir.resolve()}")
            print(f"    Total      : {len(targets)}")
            print(f"    Downloaded : {downloaded}")
            print(f"    Skipped    : {skipped}")
            print(f"    Failed     : {failed}")
            print("========================================================")
        finally:
            try:
                if not cdp_port and context:
                    context.close()
            except Exception:
                pass


def main():
    parser = argparse.ArgumentParser(description="WhoScored 2026/27 EPL Season Batch Scraper")
    parser.add_argument("--output-dir", type=str, default=str(DEFAULT_OUTPUT_DIR), help="Folder to save 26/27 matches")
    parser.add_argument("--profile-dir", type=str, default=str(DEFAULT_PROFILE_DIR), help="Persistent browser profile dir")
    parser.add_argument("--cdp", type=int, default=None, help="Optional CDP port")
    parser.add_argument("--headless", action="store_true", help="Headless mode")
    parser.add_argument("--limit", type=int, default=None, help="Limit number of matches to download")
    parser.add_argument("--refresh-schedule", action="store_true", help="Re-fetch schedule_2627.json from WhoScored")
    parser.add_argument("--timeout", type=int, default=45, help="Timeout per page in seconds")
    args = parser.parse_args()

    out_dir = Path(args.output_dir)
    if args.refresh_schedule and (out_dir / "schedule_2627.json").exists():
        (out_dir / "schedule_2627.json").unlink()

    run_season_scrape(
        output_dir=out_dir,
        profile_dir=Path(args.profile_dir),
        cdp_port=args.cdp,
        headless=args.headless,
        limit=args.limit,
        timeout_sec=args.timeout,
    )


if __name__ == "__main__":
    main()
