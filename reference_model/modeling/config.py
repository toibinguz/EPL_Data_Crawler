"""Paths, season splits and switches shared by every modelling module."""
import os
from datetime import datetime
from pathlib import Path

import numpy as np

PROJECT_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_DIR / "data"                      # same data/ folder as the crawling scripts
DATASET_DIR = DATA_DIR / "epl--league-dataset"       # clean/, raw_data/, 2026_2027_new/, README.md
OUTPUTS_DIR = Path(os.environ.get("EPL_OUT_DIR", PROJECT_DIR / "outputs"))

# Hugging Face mirror of the dataset, downloaded when features.csv is not found locally
HF_REPO = "Toibinguz/epl-premier-league-dataset"

SEED = 42
np.random.seed(SEED)

# Splits by season start year (2021 = season 2021/22)
TRAIN_START, TRAIN_END = 2000, 2021   # Train: 2000/01 -> 2021/22 (2000/01: every Elo starts at 1500, form is empty)
DEV_START, DEV_END = 2022, 2023       # Dev:   2022/23 -> 2023/24 (tuning, model and feature-group selection)
TEST_START = 2024                     # Test:  2024/25 ->         (evaluated ONCE at the end)

# Retrain on EVERY played match (test included) before predicting upcoming fixtures?
# Only do this AFTER the test score has been reported. False = as in the PDF (train + dev).
REFIT_WITH_TEST_FOR_PRODUCTION = False

# Use bookmaker odds as FEATURES for Logistic Regression + Poisson?
#   True  -> adds 3 columns: odds_logit_ha = log(pH/pA), odds_log_d = log(pD), odds_missing (1 = no odds)
#   False -> as in the PDF: odds are only a reference baseline (H9), never a model input
# Note: with True, upcoming matches MUST have odds to be predicted well (missing -> median imputed).
USE_ODDS_FEATURE = False

RESULT_ORDER = ["H", "D", "A"]
TARGETS = ["home_goals", "away_goals", "result"]

# Hyper-parameter grids
POISSON_ALPHAS = [0.01, 0.1, 1.0]
LR_C_GRID = [0.01, 0.03, 0.1, 0.3, 1.0]
HGB_GRID = [(lr, depth) for lr in (0.02, 0.05) for depth in (2, 3)]
MAX_GOALS = 10                         # score matrix covers 0..MAX_GOALS goals per side

LEAK_THRESHOLD = 0.55                  # |corr| with the match goal difference above this -> suspected leakage
SUSPICIOUS_LL = 0.90                   # EPL log loss without odds rarely goes below ~0.93
BOOTSTRAP_B = 2000

# Hypothesis tests: rolling-origin seasons (each season learns only from earlier ones)
EVAL_SEASONS = list(range(2015, TEST_START))   # 2015/16 -> 2023/24

# Keep the PDF model (M3) unless the best Dev model beats it significantly (CI95% of delta fully below 0)
PREFER_PDF_IF_NOT_SIGNIFICANT = True

# Train start years compared in the "accuracy by train start" experiment
START_YEARS = [2000, 2009]


def make_run_dir(use_odds=USE_ODDS_FEATURE):
    """Each run writes into its own folder, e.g. outputs/run_20261110_1432_no_odds/, so nothing is overwritten."""
    name = "run_" + datetime.now().strftime("%Y%m%d_%H%M") + ("_odds" if use_odds else "_no_odds")
    out = OUTPUTS_DIR / name
    out.mkdir(parents=True, exist_ok=True)
    return name, out
