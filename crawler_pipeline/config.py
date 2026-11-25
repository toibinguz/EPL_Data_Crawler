"""
crawler_pipeline/config.py
Central configuration and path resolution for the Raw EPL Crawler & Hugging Face Storage Pipeline.
Scope: Strictly raw data extraction and Hugging Face Dataset archiving (zero cleaning/feature engineering).
"""

import os
from pathlib import Path

PIPELINE_DIR = Path(__file__).resolve().parent
ROOT_DIR = PIPELINE_DIR.parent
RAW_DIR = ROOT_DIR / "raw_data"
SEASON_2627_DIR = ROOT_DIR / "2026_2027_new"
WHOSCORED_2627_DIR = ROOT_DIR / "mapped_epl" / "whoscored_2627"
WHOSCORED_PROFILE_DIR = ROOT_DIR / "mapped_epl" / "downloaded_matches" / ".browser_profile"

TOKEN_FILE = PIPELINE_DIR / "hf_access_key.txt"
TOKEN_FALLBACK = ROOT_DIR / "access_token_hf.txt"

DEFAULT_HF_REPO_ID = os.environ.get("HF_REPO_ID", "Toibinguz/epl-premier-league-dataset")
HF_WHOSCORED_PATH = "who_scored_26_27"

HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/131.0.0.0 Safari/537.36",
    "Accept": "application/json, text/plain, */*",
}


def resolve_hf_token(explicit_token: str | None = None) -> str | None:
    """Resolve Hugging Face token from explicit arg -> HF_TOKEN env -> local token files."""
    if explicit_token and explicit_token.strip():
        return explicit_token.strip()
    env_tok = os.environ.get("HF_TOKEN") or os.environ.get("HUGGINGFACE_HUB_TOKEN")
    if env_tok and env_tok.strip():
        return env_tok.strip()
    for p in (TOKEN_FILE, TOKEN_FALLBACK):
        if p.exists():
            txt = p.read_text(encoding="utf-8", errors="ignore").strip()
            if txt:
                return txt.splitlines()[0].strip()
    return None
