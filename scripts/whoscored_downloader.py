#!/usr/bin/env python3
"""
WhoScored Automated Match Downloader & Replicator.

Recreates the techniques used in open-source scrapers:
  - marinoalfonso/WhoScored_auto_scraper (Playwright persistent profile)
  - Ali-Hasan-Khan/Scrape-Whoscored-Event-Data (Real browser session / HTTP hybrid)
  - soccerdata (Undetected Chrome / CDP extraction of matchCentreData)

Architecture:
  1. Persistent Profile Context: Retains cookies, storage, and clearance across runs.
  2. System Chrome Channel: Uses the real Google Chrome engine to ensure authentic TLS/WebGL fingerprints.
  3. Anti-Detection Flags: Disables AutomationControlled and hides navigator.webdriver.
  4. Human-in-the-Loop Completion: If Cloudflare verification appears, pauses to let the user complete it in the visible UI, then automatically resumes and captures the page.
  5. Dual Artifact Generation: Saves the full raw .html and extracts the structured matchCentreData JSON with Opta event stream.
"""

import argparse
import json
import os
import re
import sys
import time
from pathlib import Path

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8")


def extract_match_centre_data(html_content: str) -> dict | None:
    """Extract matchCentreData JSON object from raw HTML using regex."""
    patterns = [
        r"require\.config\.params\[['\"]args['\"]\]\s*=\s*(\{.+?\});\s*(?:require|var|let|const|<)",
        r"var\s+matchCentreData\s*=\s*(\{.+?\});\s*(?:var|let|const|<)",
        r"matchCentreData\s*:\s*(\{.+?\})\s*,\s*matchId",
    ]
    for pattern in patterns:
        match = re.search(pattern, html_content, re.DOTALL)
        if match:
            raw_json = match.group(1).strip()
            # If the pattern captured require.config.params['args']
            try:
                parsed = json.loads(raw_json)
                if "matchCentreData" in parsed:
                    return parsed["matchCentreData"]
                if "events" in parsed:
                    return parsed
            except json.JSONDecodeError:
                pass

    # Secondary targeted search for the matchCentreData key
    idx = html_content.find("matchCentreData :")
    if idx == -1:
        idx = html_content.find("matchCentreData:")
    if idx != -1:
        # Find opening brace
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
                if c == '"' and not escape:
                    in_str = not in_str
                    continue
                if not in_str:
                    if c == "{":
                        depth += 1
                    elif c == "}":
                        depth -= 1
                        if depth == 0:
                            json_str = html_content[start : i + 1]
                            try:
                                return json.loads(json_str)
                            except Exception:
                                break
    return None


def create_browser_context(
    playwright,
    profile_dir: Path | None = None,
    cdp_port: int | None = None,
    headless: bool = False,
):
    """Create or attach to a Playwright browser context and return (context, browser, page)."""
    browser = None
    if cdp_port:
        print(f"[*] Connecting via Chrome DevTools Protocol (CDP) on port {cdp_port}...")
        browser = playwright.chromium.connect_over_cdp(f"http://localhost:{cdp_port}")
        context = browser.contexts[0] if browser.contexts else browser.new_context()
        page = context.new_page()
        return context, browser, page

    if profile_dir is None:
        profile_dir = Path("mapped_epl/downloaded_matches/.browser_profile")
    profile_dir.mkdir(parents=True, exist_ok=True)
    print(f"[*] Launching persistent browser context with profile: {profile_dir.resolve()}")
    print(f"[*] Mode: Headless = {headless}, Channel = System Chrome")

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
    except Exception as e:
        print(f"[!] System chrome channel failed ({e}), falling back to bundled chromium...")
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


def fetch_match_on_page(
    page,
    match_url: str,
    output_dir: Path,
    timeout_sec: int = 60,
    file_prefix: str | None = None,
) -> dict:
    """Navigate an existing Playwright page to match_url, save HTML and extract matchCentreData."""
    output_dir.mkdir(parents=True, exist_ok=True)

    match_id_search = re.search(r"/matches/(\d+)(?:/|$)", match_url, re.IGNORECASE)
    match_id = match_id_search.group(1) if match_id_search else "unknown_match"
    stem = file_prefix or f"whoscored_{match_id}"

    html_file = output_dir / f"{stem}.html"
    json_file = output_dir / f"{stem}_matchCentreData.json"

    print(f"[*] Navigating to: {match_url}")
    page.goto(match_url, wait_until="commit", timeout=timeout_sec * 1000)

    start_time = time.time()
    match_data = None
    html_content = ""

    while time.time() - start_time < timeout_sec:
        html_content = page.content()
        title = page.title()

        if "Just a moment..." in title or "Attention Required!" in title or "Cloudflare" in title:
            print(f"[!] Cloudflare challenge detected ('{title}'). Waiting for completion in browser UI...")
            time.sleep(3)
            continue

        try:
            js_data = page.evaluate("() => (window.require && window.require.config && window.require.config.params && window.require.config.params['args']) ? window.require.config.params['args'].matchCentreData : null")
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
    print(f"[+] Saved HTML: {html_file.name} ({len(html_content):,} bytes)")

    if not match_data:
        match_data = extract_match_centre_data(html_content)

    if match_data:
        json_file.write_text(json.dumps(match_data, ensure_ascii=False, indent=2), encoding="utf-8")
        events_count = len(match_data.get("events", []))
        home_team = (match_data.get("home") or {}).get("name", "Home")
        away_team = (match_data.get("away") or {}).get("name", "Away")
        score = match_data.get("score", "N/A")
        print(f"[+] Extracted matchCentreData: {home_team} {score} {away_team} | {events_count:,} events -> {json_file.name}")
    else:
        print(f"[!] Warning: matchCentreData not found for {match_id}.")

    return {
        "match_id": match_id,
        "html_path": str(html_file.resolve()),
        "json_path": str(json_file.resolve()) if match_data else None,
        "events_count": len(match_data.get("events", [])) if match_data else 0,
    }


def download_match(
    match_url: str,
    output_dir: Path,
    profile_dir: Path | None = None,
    cdp_port: int | None = None,
    headless: bool = False,
    timeout_sec: int = 60,
) -> dict:
    """
    Downloads match HTML and matchCentreData from WhoScored.
    """
    from playwright.sync_api import sync_playwright

    output_dir.mkdir(parents=True, exist_ok=True)
    match_id_search = re.search(r"/matches/(\d+)(?:/|$)", match_url, re.IGNORECASE)
    match_id = match_id_search.group(1) if match_id_search else "unknown_match"

    print(f"\n========================================================")
    print(f"[*] WhoScored Automated Downloader")
    print(f"[*] Target URL : {match_url}")
    print(f"[*] Match ID   : {match_id}")
    print(f"[*] Output Dir : {output_dir.resolve()}")
    print(f"========================================================\n")

    if profile_dir is None:
        profile_dir = output_dir / ".browser_profile"

    with sync_playwright() as p:
        context, _, page = create_browser_context(
            p, profile_dir=profile_dir, cdp_port=cdp_port, headless=headless
        )
        result = fetch_match_on_page(page, match_url, output_dir, timeout_sec=timeout_sec)

        try:
            if not cdp_port and context:
                context.close()
        except Exception:
            pass

        return result


def main():
    parser = argparse.ArgumentParser(description="WhoScored Automated Downloader & Replicator")
    parser.add_argument(
        "--url",
        type=str,
        default="https://www.whoscored.com/matches/1983821/live/england-premier-league-2026-2027-manchester-city-fulham",
        help="Full WhoScored match URL",
    )
    parser.add_argument(
        "--match-id",
        type=int,
        default=None,
        help="WhoScored match ID (e.g. 1983821)",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="mapped_epl/downloaded_matches",
        help="Directory to save downloaded files",
    )
    parser.add_argument(
        "--cdp",
        type=int,
        default=None,
        help="Connect to existing Chrome via CDP port (e.g. 9222)",
    )
    parser.add_argument(
        "--headless",
        action="store_true",
        help="Run browser in headless mode (not recommended for WhoScored)",
    )
    parser.add_argument(
        "--timeout",
        type=int,
        default=60,
        help="Timeout in seconds for page load",
    )

    args = parser.parse_args()

    url = args.url
    if args.match_id:
        url = f"https://www.whoscored.com/matches/{args.match_id}/live"

    download_match(
        match_url=url,
        output_dir=Path(args.output_dir),
        cdp_port=args.cdp,
        headless=args.headless,
        timeout_sec=args.timeout,
    )


if __name__ == "__main__":
    main()
