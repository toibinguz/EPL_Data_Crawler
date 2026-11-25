"""Locate and load features.csv, quick EDA and the chronological train / dev / test split."""
import glob
import os
import re
from pathlib import Path

import numpy as np
import pandas as pd

from .config import DATASET_DIR, DATA_DIR, DEV_END, DEV_START, HF_REPO, TEST_START, TRAIN_END, TRAIN_START
from .plots import plot_hda_by_season

DATE_CANDIDATES = ["date", "match_date", "kickoff", "utc_date", "datetime"]


# ---------------------------------------------------------------- locating files
def find_features(folder):
    """Best features.csv under folder, preferring .../clean/features.csv and skipping model outputs."""
    hits = [p for p in glob.glob(os.path.join(folder, "**", "features.csv"), recursive=True)
            if "outputs" not in Path(p).parts and "epl_ml_outputs" not in p]
    hits.sort(key=lambda p: ("clean" not in Path(p).parts, len(p)))
    return Path(hits[0]) if hits else None


def locate_features(path=None):
    """Path to features.csv: explicit path -> $EPL_FEATURES_CSV -> $EPL_DATA_DIR -> data/ -> Hugging Face download."""
    path = path or os.environ.get("EPL_FEATURES_CSV")
    if path:
        return Path(path)
    for folder in [os.environ.get("EPL_DATA_DIR"), DATASET_DIR, DATA_DIR]:
        if folder and os.path.isdir(folder):
            hit = find_features(folder)
            if hit:
                return hit
    print("features.csv not found locally -> downloading from Hugging Face:", HF_REPO)
    from huggingface_hub import snapshot_download
    snapshot_download(repo_id=HF_REPO, repo_type="dataset", local_dir=str(DATASET_DIR),
                      allow_patterns=["*features.csv", "*hypotheses.csv", "README.md", "*features_report.json"])
    hit = find_features(DATASET_DIR)
    if hit is None:
        raise FileNotFoundError("features.csv not found - pass --features or set EPL_FEATURES_CSV.")
    return hit


def dataset_root(features_csv):
    """Root of the dataset (the folder that contains clean/)."""
    parent = Path(features_csv).parent
    return parent.parent if parent.name == "clean" else parent


def locate_hypotheses(root):
    hits = glob.glob(os.path.join(root, "**", "hypotheses.csv"), recursive=True)
    return Path(hits[0]) if hits else Path(root) / "EDA/reports/eda_matches_features/hypotheses.csv"


def show_tree(root, max_depth=3, max_files=15):
    """Print the folder structure (up to max_depth levels) to see which files are available."""
    root = os.path.abspath(root)
    if not os.path.exists(root):
        print("NOT FOUND:", root)
        return
    for cur, dirs, files in os.walk(root):
        depth = cur[len(root):].count(os.sep)
        if depth >= max_depth:
            dirs[:] = []
            continue
        dirs[:] = sorted(d for d in dirs if not d.startswith("."))
        print("  " * depth + (os.path.basename(cur) or cur))
        for f in sorted(files)[:max_files]:
            size = os.path.getsize(os.path.join(cur, f)) / 1024
            print("  " * (depth + 1) + f"{f}  ({size:,.0f} KB)")
        if len(files) > max_files:
            print("  " * (depth + 1) + f"... (+{len(files) - max_files} files)")


# ---------------------------------------------------------------- loading
def season_start(s):
    """'2001/02', '2001-02', '2001-2002', 2001 -> 2001"""
    m = re.search(r"(\d{4})", str(s))
    return int(m.group(1)) if m else np.nan


def load_features(features_csv, verbose=True):
    """Load features.csv sorted chronologically. Returns (df, name of the date column or None)."""
    df = pd.read_csv(features_csv)
    df["season_start"] = df["season"].map(season_start)

    date_col = next((c for c in DATE_CANDIDATES if c in df.columns), None)
    if date_col:
        df[date_col] = pd.to_datetime(df[date_col], errors="coerce")
        df = df.sort_values([date_col]).reset_index(drop=True)
    else:
        df = df.sort_values("season_start", kind="stable").reset_index(drop=True)

    if "status" not in df.columns:
        df["status"] = np.where(df["result"].notna(), "played", "upcoming")

    if verbose:
        print("Shape:", df.shape, "| date column:", date_col)
        print("\nstatus:\n", df["status"].value_counts(dropna=False))
        print("\nMatches per season:")
        print(df.groupby(["season", "status"]).size().unstack(fill_value=0).to_string())
    return df, date_col


def played_matches(df):
    return df[df["status"].eq("played")].dropna(subset=["result", "home_goals", "away_goals"]).copy()


def quick_eda(df, played_all, out_dir=None):
    """Label distribution, most-missing columns and home advantage per season (relevant to H5)."""
    print("Label distribution (all played matches):")
    print(played_all["result"].value_counts(normalize=True).round(3))
    print("\nMean goals  home: %.3f | away: %.3f" % (played_all.home_goals.mean(), played_all.away_goals.mean()))

    miss = df.isna().mean().sort_values(ascending=False)
    print("\n20 columns with the most missing values:")
    print(miss.head(20).round(3).to_string())

    ha = played_all.groupby("season_start").agg(home_win=("result", lambda r: (r == "H").mean()),
                                                draw=("result", lambda r: (r == "D").mean()),
                                                away_win=("result", lambda r: (r == "A").mean()))
    plot_hda_by_season(ha, out_dir)


# ---------------------------------------------------------------- splitting
def split_name(y):
    if TRAIN_START <= y <= TRAIN_END:
        return "train"
    if DEV_START <= y <= DEV_END:
        return "dev"
    if y >= TEST_START:
        return "test"
    return "unused"


def split_by_season(df, played_all, date_col=None):
    """Chronological split by season (no shuffling). Returns (train, dev, test, upcoming)."""
    played_all["split"] = played_all["season_start"].map(split_name)
    upcoming = df[df["status"].eq("upcoming")].copy()
    train = played_all[played_all.split == "train"].copy()
    dev = played_all[played_all.split == "dev"].copy()
    test = played_all[played_all.split == "test"].copy()

    summary = played_all.groupby("split").agg(n=("result", "size"),
                                              first_season=("season", "first"), last_season=("season", "last"),
                                              H=("result", lambda r: (r == "H").mean()),
                                              D=("result", lambda r: (r == "D").mean()),
                                              A=("result", lambda r: (r == "A").mean()))
    print(summary.reindex(["train", "dev", "test", "unused"]).dropna(how="all").round(3).to_string())
    print("Upcoming matches:", len(upcoming))

    # Chronological order check: never use the future to predict the past
    if date_col:
        assert train[date_col].max() < dev[date_col].min() <= dev[date_col].max() < test[date_col].min(), \
            "Splits overlap in time!"
        print("OK: train < dev < test in time.")
    return train, dev, test, upcoming

