"""
crawler_pipeline.extractors
Individual raw source extractors for Premier League APIs and browser-backed feeds.
"""

from .football_data_ext import fetch_football_data
from .understat_ext import fetch_understat_data
from .fotmob_ext import fetch_fotmob_matches, fetch_fotmob_match_detail
from .fpl_ext import fetch_fpl_data
from .weather_ext import fetch_weather
from .whoscored_ext import (
    create_browser_context,
    discover_season_schedule,
    download_match,
    extract_match_centre_data,
    fetch_match_on_page,
    scan_local_whoscored_dir,
)

__all__ = [
    "fetch_football_data",
    "fetch_understat_data",
    "fetch_fotmob_matches",
    "fetch_fotmob_match_detail",
    "fetch_fpl_data",
    "fetch_weather",
    "create_browser_context",
    "discover_season_schedule",
    "download_match",
    "extract_match_centre_data",
    "fetch_match_on_page",
    "scan_local_whoscored_dir",
]
