"""
crawler_pipeline/extractors/whoscored_ext.py
Consolidated WhoScored stealth browser session, schedule discovery, and Opta matchCentreData extractor.

Note: WhoScored uses Cloudflare Turnstile/Bot-Management that blocks GitHub Actions/cloud runners.
This module uses a real local Chrome persistent profile (`WHOSCORED_PROFILE_DIR`), hides
`navigator.webdriver`, pauses for human Cloudflare verification if challenged, and saves both
raw `.html` and extracted `_matchCentreData.json` per match.
"""

import json
import random
import re
import time
from datetime import datetime, timezone
from pathlib import Path

from crawler_pipeline.config import WHOSCORED_2627_DIR, WHOSCORED_PROFILE_DIR

EPL_URL = "https://www.whoscored.com/Regions/252/Tournaments/2/England-Premier-League"
_WS_ID_RE = re.compile(r"(?:^|_)(\d{6,8})_")


def sanitize_slug(text: str) -> str:
    return re.sub(r"[^a-zA-Z0-9]+", "_", text).strip("_")


def build_match_prefix(match_obj: dict) -> str:
    mid = match_obj["id"]
    home = sanitize_slug(match_obj.get("homeTeamName", "Home"))
    away = sanitize_slug(match_obj.get("awayTeamName", "Away"))
    date_str = (match_obj.get("startTimeUtc") or "")[:10]
    return f"{date_str}_{mid}_{home}_vs_{away}" if date_str else f"{mid}_{home}_vs_{away}"


def scan_local_whoscored_dir(output_dir: Path = WHOSCORED_2627_DIR) -> dict:
    """
    Scan local WhoScored archive folder and return:
      - complete_ids: set of match_ids with BOTH .html (>1KB) and _matchCentreData.json (>1KB)
      - html_ids: set of match_ids with .html
      - json_ids: set of match_ids with valid _matchCentreData.json
      - total_files: number of files in output_dir (excluding hidden dirs)
      - filenames: set of basenames
    """
    html_ids: set[str] = set()
    json_ids: set[str] = set()
    filenames: set[str] = set()

    if not output_dir.exists():
        return {
            "complete_ids": set(),
            "html_ids": set(),
            "json_ids": set(),
            "total_files": 0,
            "filenames": set(),
        }

    for f in output_dir.iterdir():
        if not f.is_file():
            continue
        name = f.name
        filenames.add(name)
        m = _WS_ID_RE.search(name)
        if not m:
            continue
        mid = m.group(1)
        size = f.stat().st_size
        if name.endswith(".html") and size > 1000:
            html_ids.add(mid)
        elif name.endswith("_matchCentreData.json") and size > 1000:
            json_ids.add(mid)

    return {
        "complete_ids": html_ids & json_ids,
        "html_ids": html_ids,
        "json_ids": json_ids,
        "total_files": len(filenames),
        "filenames": filenames,
    }


def extract_match_centre_data(html_content: str) -> dict | None:
    """Extract matchCentreData JSON object from raw HTML."""
    patterns = [
        r"require\.config\.params\[['\"]args['\"]\]\s*=\s*(\{.+?\});\s*(?:require|var|let|const|<)",
        r"var\s+matchCentreData\s*=\s*(\{.+?\});\s*(?:var|let|const|<)",
        r"matchCentreData\s*:\s*(\{.+?\})\s*,\s*matchId",
    ]
    for pattern in patterns:
        match = re.search(pattern, html_content, re.DOTALL)
        if match:
            raw_json = match.group(1).strip()
            try:
                parsed = json.loads(raw_json)
                if "matchCentreData" in parsed:
                    return parsed["matchCentreData"]
                if "events" in parsed:
                    return parsed
            except json.JSONDecodeError:
                pass

    idx = html_content.find("matchCentreData :")
    if idx == -1:
        idx = html_content.find("matchCentreData:")
    if idx != -1:
        start = html_content.find("{", idx)
        if start != -1:
            depth = 0
            in_str = False
            escape = False
            for i in range(start, len(html_content)):
                c = html_content[i]
                if escape:
                    escape = False
                    continue
                if c == "\\":
                    escape = True
                    continue
                if c == '"':
                    in_str = not in_str
                    continue
                if not in_str:
                    if c == "{":
                        depth += 1
                    elif c == "}":
                        depth -= 1
                        if depth == 0:
                            try:
                                return json.loads(html_content[start : i + 1])
                            except Exception:
                                break
    return None


def create_browser_context(
    playwright,
    profile_dir: Path = WHOSCORED_PROFILE_DIR,
    cdp_port: int | None = None,
    headless: bool = False,
):
    """Create or attach to a Playwright real-Chrome context and return (context, browser, page)."""
    browser = None
    if cdp_port:
        print(f"[*] Connecting via Chrome DevTools Protocol (CDP) on port {cdp_port}...", flush=True)
        browser = playwright.chromium.connect_over_cdp(f"http://localhost:{cdp_port}")
        context = browser.contexts[0] if browser.contexts else browser.new_context()
        page = context.new_page()
        return context, browser, page

    profile_dir.mkdir(parents=True, exist_ok=True)
    print(f"[*] Launching persistent Chrome context: {profile_dir.resolve()} (headless={headless})", flush=True)

    args = [
        "--disable-blink-features=AutomationControlled",
        "--no-sandbox",
        "--disable-infobars",
        "--disable-dev-shm-usage",
        "--window-size=1920,1080",
    ]
    ua = "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/131.0.0.0 Safari/537.36"

    try:
        context = playwright.chromium.launch_persistent_context(
            user_data_dir=str(profile_dir.resolve()),
            channel="chrome",
            headless=headless,
            args=args,
            viewport={"width": 1920, "height": 1080},
            user_agent=ua,
        )
    except Exception as exc:
        print(f"[!] System Chrome channel failed ({exc}), falling back to bundled Chromium...", flush=True)
        context = playwright.chromium.launch_persistent_context(
            user_data_dir=str(profile_dir.resolve()),
            headless=headless,
            args=args,
            viewport={"width": 1920, "height": 1080},
            user_agent=ua,
        )

    context.add_init_script("""
        Object.defineProperty(navigator, 'webdriver', {
            get: () => undefined
        });
    """)
    page = context.pages[0] if context.pages else context.new_page()
    return context, browser, page


def _wait_for_page_ready(page, timeout_sec: int = 60) -> str:
    start = time.time()
    while time.time() - start < timeout_sec:
        title = page.title()
        if "Just a moment..." in title or "Attention Required!" in title or "Cloudflare" in title:
            print(f"[!] Cloudflare challenge ('{title}'). Please verify in the visible Chrome window...", flush=True)
            time.sleep(3)
            continue
        html = page.content()
        if "wsCalendar" in html or "Fixtures" in html:
            return html
        time.sleep(1.5)
    return page.content()


def discover_season_schedule(
    page,
    output_dir: Path = WHOSCORED_2627_DIR,
    timeout_sec: int = 60,
) -> list[dict]:
    """
    Navigate to WhoScored EPL page, resolve stage_id + wsCalendar, fetch monthly fixture feeds,
    filter to completed (`FT` / scored) matches, and save `schedule_2627.json`.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"[*] Navigating to EPL Tournament page: {EPL_URL}", flush=True)
    page.goto(EPL_URL, wait_until="commit", timeout=timeout_sec * 1000)
    html = _wait_for_page_ready(page, timeout_sec=timeout_sec)

    stage_match = re.search(r"/Stages/(\d+)/", html, re.IGNORECASE)
    if stage_match:
        stage_id = stage_match.group(1)
    else:
        stage_id = page.evaluate("""() => {
            const a = Array.from(document.querySelectorAll('a')).find(el => el.textContent.trim() === 'Fixtures');
            if (a && a.href) {
                const m = a.href.match(/Stages\\/(\\d+)/i);
                return m ? m[1] : null;
            }
            return null;
        }""")
    if not stage_id:
        raise RuntimeError("Could not locate Premier League 2026/27 stage_id on WhoScored.")
    print(f"[+] Resolved Premier League stage_id: {stage_id}", flush=True)

    exact_fixtures_href = page.evaluate("""() => {
        const a = Array.from(document.querySelectorAll('a')).find(el => el.textContent.trim() === 'Fixtures');
        return a ? a.href : null;
    }""")
    target_cal_url = exact_fixtures_href or f"https://www.whoscored.com/Regions/252/Tournaments/2/Stages/{stage_id}/Fixtures"
    print(f"[*] Loading calendar: {target_cal_url}", flush=True)
    page.goto(target_cal_url, wait_until="commit", timeout=timeout_sec * 1000)
    cal_html = _wait_for_page_ready(page, timeout_sec=timeout_sec)

    ws_cal = page.evaluate("() => window.wsCalendar || null")
    if not ws_cal:
        m = re.search(r"wsCalendar\s*=\s*(\{.+?\});", cal_html, re.DOTALL)
        if m:
            ws_cal = json.loads(m.group(1))

    if not ws_cal or "mask" not in ws_cal:
        now = datetime.now(timezone.utc)
        ym_pairs = []
        y, mo = 2026, 8
        while (y < now.year) or (y == now.year and mo <= now.month):
            ym_pairs.append((str(y), mo - 1))
            mo += 1
            if mo > 12:
                mo = 1
                y += 1
        print(f"[!] wsCalendar.mask not found directly; falling back to {ym_pairs}", flush=True)
    else:
        mask = ws_cal["mask"]
        ym_pairs = [(year, int(month)) for year in sorted(mask.keys()) for month in sorted(mask[year], key=int)]

    print(f"[*] Calendar months discovered: {[(y, m + 1) for y, m in ym_pairs]}", flush=True)
    all_matches: dict[int, dict] = {}
    now_utc = datetime.now(timezone.utc)

    for year, month_zero in ym_pairs:
        d_param = f"{year}{(month_zero + 1):02d}"
        data_url = f"https://www.whoscored.com/tournaments/{stage_id}/data/?d={d_param}"
        print(f"[*] Fetching fixtures for {d_param} ...", flush=True)
        resp = page.evaluate("""async (url) => {
            const r = await fetch(url, { credentials: 'include' });
            if (!r.ok) return { error: r.status };
            return await r.json();
        }""", data_url)
        if not resp or "error" in resp:
            print(f"[!] Failed fetching {d_param}: {resp}", flush=True)
            continue

        count_month = 0
        for t in resp.get("tournaments", []):
            for m in t.get("matches", []):
                mid = m.get("id")
                if mid:
                    all_matches[mid] = m
                    count_month += 1
        print(f"    -> {count_month} matches in {d_param}", flush=True)
        time.sleep(0.6)

    played = []
    for m in all_matches.values():
        status = m.get("status")
        elapsed = str(m.get("elapsed", "")).upper()
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
        if has_score and (elapsed == "FT" or status == 6 or is_past):
            played.append(m)

    played.sort(key=lambda x: x.get("startTimeUtc", ""))
    schedule_file = output_dir / "schedule_2627.json"
    schedule_file.write_text(json.dumps(played, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[+] Schedule updated: {len(all_matches)} total fixtures | {len(played)} completed matches -> {schedule_file.name}", flush=True)
    return played


def fetch_match_on_page(
    page,
    match_url: str,
    output_dir: Path = WHOSCORED_2627_DIR,
    timeout_sec: int = 60,
    file_prefix: str | None = None,
) -> dict:
    """Navigate Playwright page to match_url, save raw .html and extract _matchCentreData.json."""
    output_dir.mkdir(parents=True, exist_ok=True)
    m_id = re.search(r"/matches/(\d+)(?:/|$)", match_url, re.IGNORECASE)
    match_id = m_id.group(1) if m_id else "unknown_match"
    stem = file_prefix or f"whoscored_{match_id}"

    html_file = output_dir / f"{stem}.html"
    json_file = output_dir / f"{stem}_matchCentreData.json"

    print(f"[*] Navigating to: {match_url}", flush=True)
    page.goto(match_url, wait_until="commit", timeout=timeout_sec * 1000)

    start_time = time.time()
    match_data = None
    while time.time() - start_time < timeout_sec:
        html_content = page.content()
        title = page.title()
        if "Just a moment..." in title or "Attention Required!" in title or "Cloudflare" in title:
            print(f"[!] Cloudflare challenge ('{title}'). Waiting for manual verification in Chrome...", flush=True)
            time.sleep(3)
            continue

        try:
            js_data = page.evaluate(
                "() => (window.require && window.require.config && window.require.config.params && "
                "window.require.config.params['args']) ? window.require.config.params['args'].matchCentreData : null"
            )
            if js_data and "events" in js_data:
                match_data = js_data
                break
        except Exception:
            pass

        if "matchCentreData" in html_content:
            match_data = extract_match_centre_data(html_content)
            if match_data:
                break

        time.sleep(1.5)

    html_content = page.content()
    html_file.write_text(html_content, encoding="utf-8")
    if not match_data:
        match_data = extract_match_centre_data(html_content)

    if match_data:
        json_file.write_text(json.dumps(match_data, ensure_ascii=False, indent=2), encoding="utf-8")
        events_count = len(match_data.get("events", []))
        home_team = (match_data.get("home") or {}).get("name", "Home")
        away_team = (match_data.get("away") or {}).get("name", "Away")
        score = match_data.get("score", "N/A")
        print(f"[+] Saved {stem}: {home_team} {score} {away_team} ({events_count:,} events)", flush=True)
    else:
        print(f"[!] Warning: matchCentreData missing for match {match_id}", flush=True)

    return {
        "match_id": match_id,
        "html_path": str(html_file.resolve()),
        "json_path": str(json_file.resolve()) if match_data else None,
        "events_count": len(match_data.get("events", [])) if match_data else 0,
    }


def download_match(
    match_url: str,
    output_dir: Path = WHOSCORED_2627_DIR,
    profile_dir: Path = WHOSCORED_PROFILE_DIR,
    cdp_port: int | None = None,
    headless: bool = False,
    timeout_sec: int = 60,
) -> dict:
    """Download a single WhoScored match HTML + matchCentreData JSON."""
    from playwright.sync_api import sync_playwright

    with sync_playwright() as p:
        context, _, page = create_browser_context(
            p, profile_dir=profile_dir, cdp_port=cdp_port, headless=headless
        )
        try:
            return fetch_match_on_page(page, match_url, output_dir=output_dir, timeout_sec=timeout_sec)
        finally:
            try:
                if not cdp_port and context:
                    context.close()
            except Exception:
                pass


def download_missing_matches(
    matches_to_fetch: list[dict],
    output_dir: Path = WHOSCORED_2627_DIR,
    profile_dir: Path = WHOSCORED_PROFILE_DIR,
    cdp_port: int | None = None,
    headless: bool = False,
    delay_min: float = 2.0,
    delay_max: float = 3.5,
    timeout_sec: int = 60,
    existing_page=None,
) -> dict:
    """Download each match dict in `matches_to_fetch` using real Chrome context."""
    downloaded = 0
    failed = 0

    def _run_loop(page):
        nonlocal downloaded, failed
        total = len(matches_to_fetch)
        for idx, m in enumerate(matches_to_fetch, 1):
            mid = m["id"]
            prefix = build_match_prefix(m)
            date_str = (m.get("startTimeUtc") or "")[:10]
            url = f"https://www.whoscored.com/matches/{mid}/live"
            print(f"\n[{idx}/{total}] Downloading {m.get('homeTeamName')} vs {m.get('awayTeamName')} ({date_str}, ID={mid})", flush=True)
            try:
                res = fetch_match_on_page(
                    page,
                    match_url=url,
                    output_dir=output_dir,
                    timeout_sec=timeout_sec,
                    file_prefix=prefix,
                )
                if res["events_count"] > 0:
                    downloaded += 1
                else:
                    failed += 1
            except Exception as exc:
                print(f"[!] Error downloading match {mid}: {exc}", flush=True)
                failed += 1

            if idx < total:
                time.sleep(random.uniform(delay_min, delay_max))

    if existing_page is not None:
        _run_loop(existing_page)
    else:
        from playwright.sync_api import sync_playwright

        with sync_playwright() as p:
            context, _, page = create_browser_context(
                p, profile_dir=profile_dir, cdp_port=cdp_port, headless=headless
            )
            try:
                _run_loop(page)
            finally:
                try:
                    if not cdp_port and context:
                        context.close()
                except Exception:
                    pass

    return {"downloaded": downloaded, "failed": failed}
