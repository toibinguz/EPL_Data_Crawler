"""
crawler_pipeline/whoscored_sync_hf.py
Daily local WhoScored -> Hugging Face incremental sync tool.

Why local Chrome? WhoScored enforces interactive bot checks that block cloud CI runners
(GitHub Actions). This CLI connects Hugging Face Hub directly with a real local Chrome
session to:
  1. Check what matches already exist on Hugging Face (`who_scored_26_27/`).
  2. Discover completed (`FT`) EPL 2026/27 matches on WhoScored using a persistent Chrome profile.
  3. Identify exact missing matches on Hugging Face & download only missing matches locally.
  4. Upload `mapped_epl/whoscored_2627/` cleanly into `who_scored_26_27/` on Hugging Face Hub.
"""

import argparse
import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from crawler_pipeline.config import (  # noqa: E402
    DEFAULT_HF_REPO_ID,
    HF_WHOSCORED_PATH,
    WHOSCORED_2627_DIR,
    WHOSCORED_PROFILE_DIR,
    resolve_hf_token,
)
from crawler_pipeline.extractors.whoscored_ext import (  # noqa: E402
    create_browser_context,
    discover_season_schedule,
    download_missing_matches,
    scan_local_whoscored_dir,
)
from crawler_pipeline.uploader.hf_uploader import (  # noqa: E402
    list_hf_whoscored_match_ids,
    sync_whoscored_to_hf,
)

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description="Daily WhoScored -> Hugging Face WhoScored 26/27 Sync")
    parser.add_argument("--repo-id", default=DEFAULT_HF_REPO_ID, help="Hugging Face dataset repo ID")
    parser.add_argument("--hf-path", default=HF_WHOSCORED_PATH, help="Target folder inside HF repo")
    parser.add_argument("--output-dir", default=str(WHOSCORED_2627_DIR), help="Local WhoScored folder")
    parser.add_argument("--profile-dir", default=str(WHOSCORED_PROFILE_DIR), help="Chrome persistent profile dir")
    parser.add_argument("--cdp", type=int, default=None, help="Optional Chrome DevTools port (e.g. 9222)")
    parser.add_argument("--token", default=None, help="Optional HF token (defaults to HF_TOKEN or access_token_hf.txt)")
    parser.add_argument("--headless", action="store_true", help="Run Chrome headless (not recommended if Cloudflare prompts)")
    parser.add_argument("--timeout", type=int, default=60, help="Page timeout in seconds")
    parser.add_argument("--check-only", action="store_true", help="Report missing matches without downloading or uploading")
    parser.add_argument("--skip-browser", action="store_true", help="Skip Chrome WhoScored check; only compare local vs HF")
    parser.add_argument("--no-sync", action="store_true", help="Download missing matches locally without uploading to HF")
    parser.add_argument("--limit", type=int, default=None, help="Optional max matches to download in one run")
    args = parser.parse_args()

    out_dir = Path(args.output_dir)
    profile_dir = Path(args.profile_dir)
    token = resolve_hf_token(args.token)
    if not token and not args.no_sync:
        print("[!] Error: Missing Hugging Face token (set HF_TOKEN or save it in access_token_hf.txt).", flush=True)
        return 1

    print("=" * 72, flush=True)
    print(f"[*] WhoScored Daily Sync | Repo: {args.repo_id}/{args.hf_path}", flush=True)
    print("=" * 72, flush=True)

    # 1. Inspect Hugging Face remote archive
    hf_state = list_hf_whoscored_match_ids(repo_id=args.repo_id, token=token, path_in_repo=args.hf_path)
    hf_ids = hf_state["complete_ids"]
    hf_files = hf_state["filenames"]
    print(f"[+] Hugging Face ({args.hf_path}): {len(hf_ids)} complete matches ({hf_state['total_files']} files)", flush=True)

    # 2. Inspect local archive
    local_state = scan_local_whoscored_dir(out_dir)
    local_ids = local_state["complete_ids"]
    local_files = local_state["filenames"]
    local_not_on_hf = sorted(local_files - hf_files)
    print(f"[+] Local ({out_dir}): {len(local_ids)} complete matches ({local_state['total_files']} files)", flush=True)
    print(f"[+] Local files not yet on HF: {len(local_not_on_hf)}", flush=True)

    if args.skip_browser:
        if args.check_only:
            print("[*] --check-only + --skip-browser complete.", flush=True)
            return 0
        if local_not_on_hf and not args.no_sync:
            print(f"[*] Uploading {len(local_not_on_hf)} new/missing local file(s) to {args.hf_path}...", flush=True)
            commit_url = sync_whoscored_to_hf(out_dir, args.repo_id, args.hf_path, token)
            print(f"[+] Upload complete: {commit_url}", flush=True)
        else:
            print("[+] Local and HF are already synchronized (skip-browser mode).", flush=True)
        return 0

    # 3. Discover live completed EPL 2026/27 schedule via real Chrome session
    from playwright.sync_api import sync_playwright

    downloaded_cnt = 0
    failed_cnt = 0
    with sync_playwright() as p:
        context, _, page = create_browser_context(
            p, profile_dir=profile_dir, cdp_port=args.cdp, headless=args.headless
        )
        try:
            played = discover_season_schedule(page, output_dir=out_dir, timeout_sec=args.timeout)
            missing_on_hf = [m for m in played if str(m["id"]) not in hf_ids]
            need_download = [m for m in missing_on_hf if str(m["id"]) not in local_ids]
            cached_waiting_upload = [m for m in missing_on_hf if str(m["id"]) in local_ids]

            print("\n--- GAP ANALYSIS (WhoScored vs Local vs Hugging Face) ---", flush=True)
            print(f"  Completed EPL 2026/27 matches on WhoScored : {len(played)}", flush=True)
            print(f"  Already complete on HF ({args.hf_path})        : {len(hf_ids)}", flush=True)
            print(f"  Cached locally but missing on HF            : {len(cached_waiting_upload)}", flush=True)
            print(f"  Missing everywhere (need Chrome download)   : {len(need_download)}", flush=True)

            for m in need_download:
                dt = (m.get("startTimeUtc") or "")[:10]
                print(f"    - [NEED DOWNLOAD] {dt} | ID={m['id']} | {m.get('homeTeamName')} {m.get('homeScore')}-{m.get('awayScore')} {m.get('awayTeamName')}", flush=True)
            for m in cached_waiting_upload:
                dt = (m.get("startTimeUtc") or "")[:10]
                print(f"    - [READY TO UPLOAD] {dt} | ID={m['id']} | {m.get('homeTeamName')} vs {m.get('awayTeamName')}", flush=True)

            if args.check_only:
                print("\n[*] --check-only mode finished (no downloads or uploads performed).", flush=True)
                return 0

            if need_download:
                targets = need_download[: args.limit] if args.limit else need_download
                res = download_missing_matches(
                    targets,
                    output_dir=out_dir,
                    profile_dir=profile_dir,
                    cdp_port=args.cdp,
                    headless=args.headless,
                    timeout_sec=args.timeout,
                    existing_page=page,
                )
                downloaded_cnt = res["downloaded"]
                failed_cnt = res["failed"]
                print(f"\n[+] Browser download summary: downloaded={downloaded_cnt}, failed={failed_cnt}", flush=True)
            else:
                print("\n[+] Zero missing matches to download from WhoScored.", flush=True)
        finally:
            try:
                if not args.cdp and context:
                    context.close()
            except Exception:
                pass

    # 4. Sync delta to Hugging Face (ONLY who_scored_26_27)
    if args.no_sync:
        print("[*] --no-sync requested; skipping Hugging Face upload.", flush=True)
        return 0 if failed_cnt == 0 else 2

    local_after = scan_local_whoscored_dir(out_dir)
    delta_files = sorted(local_after["filenames"] - hf_files)
    if delta_files or downloaded_cnt > 0:
        print(f"\n[*] Uploading {len(delta_files)} new/updated file(s) to {args.repo_id}/{args.hf_path}...", flush=True)
        commit_url = sync_whoscored_to_hf(
            local_dir=out_dir,
            repo_id=args.repo_id,
            path_in_repo=args.hf_path,
            token=token,
            commit_message=f"Incremental WhoScored 26/27 sync (+{downloaded_cnt} matches, +{len(delta_files)} files)",
        )
        hf_after = list_hf_whoscored_match_ids(repo_id=args.repo_id, token=token, path_in_repo=args.hf_path)
        print(f"[+] Commit URL: {commit_url}", flush=True)
        print(f"[+] Verified HF ({args.hf_path}): {len(hf_after['complete_ids'])} complete matches ({hf_after['total_files']} files)", flush=True)
    else:
        print("[+] Hugging Face is already 100% up-to-date. No upload needed.", flush=True)

    return 0 if failed_cnt == 0 else 2


if __name__ == "__main__":
    raise SystemExit(main())
