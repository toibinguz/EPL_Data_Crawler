"""
crawler_pipeline/uploader/hf_uploader.py
Uploads active raw folders (2026_2027_new, who_scored_26_27, and optionally raw_data)
directly to the Hugging Face Dataset repository without touching unrelated folders.
"""

import datetime
import re
from pathlib import Path
from huggingface_hub import HfApi, RepoFile

from crawler_pipeline.config import (
    DEFAULT_HF_REPO_ID,
    HF_WHOSCORED_PATH,
    RAW_DIR,
    SEASON_2627_DIR,
    WHOSCORED_2627_DIR,
    resolve_hf_token,
)

_WS_ID_RE = re.compile(r"(?:^|_)(\d{6,8})_")


def list_hf_whoscored_match_ids(
    repo_id: str = DEFAULT_HF_REPO_ID,
    token: str | None = None,
    path_in_repo: str = HF_WHOSCORED_PATH,
) -> dict:
    """
    Inspect `path_in_repo` on Hugging Face Dataset Hub and return:
      - complete_ids: set of match_ids that have BOTH .html and _matchCentreData.json (>1KB)
      - html_ids: set of match_ids with .html
      - json_ids: set of match_ids with valid _matchCentreData.json
      - total_files: total file count in remote folder
      - filenames: set of remote basenames
    """
    tok = resolve_hf_token(token)
    api = HfApi(token=tok)
    html_ids: set[str] = set()
    json_ids: set[str] = set()
    filenames: set[str] = set()

    try:
        items = list(
            api.list_repo_tree(
                repo_id=repo_id,
                repo_type="dataset",
                path_in_repo=path_in_repo,
                recursive=True,
            )
        )
    except Exception as exc:
        msg = str(exc)
        if "404" in msg or "EntryNotFoundError" in type(exc).__name__:
            return {
                "complete_ids": set(),
                "html_ids": set(),
                "json_ids": set(),
                "total_files": 0,
                "filenames": set(),
            }
        raise

    for item in items:
        if not isinstance(item, RepoFile):
            continue
        name = Path(item.path).name
        filenames.add(name)
        m = _WS_ID_RE.search(name)
        if not m:
            continue
        mid = m.group(1)
        if name.endswith(".html") and (item.size or 0) > 1000:
            html_ids.add(mid)
        elif name.endswith("_matchCentreData.json") and (item.size or 0) > 1000:
            json_ids.add(mid)

    return {
        "complete_ids": html_ids & json_ids,
        "html_ids": html_ids,
        "json_ids": json_ids,
        "total_files": len(filenames),
        "filenames": filenames,
    }


def sync_whoscored_to_hf(
    local_dir: Path = WHOSCORED_2627_DIR,
    repo_id: str = DEFAULT_HF_REPO_ID,
    path_in_repo: str = HF_WHOSCORED_PATH,
    token: str | None = None,
    commit_message: str | None = None,
) -> str:
    """
    Upload `local_dir` ONLY to `path_in_repo` (`who_scored_26_27`) on Hugging Face Dataset Hub.
    Never modifies any other folder on the repository.
    """
    tok = resolve_hf_token(token)
    if not tok:
        raise RuntimeError("Missing HF token (set HF_TOKEN or place token in access_token_hf.txt).")
    if not local_dir.exists():
        raise FileNotFoundError(f"Local WhoScored folder does not exist: {local_dir}")

    api = HfApi(token=tok)
    now_str = datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    msg = commit_message or f"Sync WhoScored 2026/27 raw archive to {path_in_repo} ({now_str})"

    res = api.upload_folder(
        folder_path=str(local_dir),
        path_in_repo=path_in_repo,
        repo_id=repo_id,
        repo_type="dataset",
        ignore_patterns=["*.tmp", ".browser_profile/*"],
        commit_message=msg,
    )
    return str(res.commit_url)


def sync_to_huggingface(repo_id: str = DEFAULT_HF_REPO_ID, token: str | None = None, sync_raw: bool = False) -> None:
    """Sync 2026_2027_new (and optionally raw_data) to Hugging Face Dataset repository."""
    print("=" * 80)
    print(f"SYNCING TO HUGGING FACE DATASET: {repo_id}")
    print("=" * 80)

    tok = resolve_hf_token(token)
    if not tok:
        print("[!] Missing HF_TOKEN; skipping upload.")
        return

    api = HfApi(token=tok)
    now_str = datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%d %H:%M UTC")

    if sync_raw and RAW_DIR.exists():
        print("[*] Uploading raw_data to Hugging Face...")
        api.upload_folder(
            folder_path=str(RAW_DIR),
            path_in_repo="raw_data",
            repo_id=repo_id,
            repo_type="dataset",
            ignore_patterns=["*/cache_matches/*", "*/matches/*"],
            commit_message=f"Sync raw_data - {now_str}",
        )
        print("[+] raw_data synced successfully.")

    if SEASON_2627_DIR.exists():
        print("[*] Uploading 2026_2027_new to Hugging Face...")
        api.upload_folder(
            folder_path=str(SEASON_2627_DIR),
            path_in_repo="2026_2027_new",
            repo_id=repo_id,
            repo_type="dataset",
            commit_message=f"Sync active season 2026_2027_new - {now_str}",
        )
        print("[+] 2026_2027_new synced successfully.")

    print(f"\nSUCCESS: Data synced to https://huggingface.co/datasets/{repo_id}")
