"""Feature selection: drop ID / label / odds columns, leakage check, odds features and the PDF feature sets."""
import re
from dataclasses import dataclass

import numpy as np
import pandas as pd

from .config import LEAK_THRESHOLD, TARGETS

ID_LIKE = re.compile(r"(^id$|_id$|^match_?id|team|^date|datetime|kickoff|season|status|^split$|referee|venue|"
                     r"stadium|url|_name$|gameweek|provisional|postponed)", re.I)
# Columns counting how many matches had data: used only for filtering, never as features
META_LIKE = re.compile(r"(_window_n5$|_xg_n5$|_fm_n5$)", re.I)
ODDS_LIKE = re.compile(r"(odd|b365|bet|pinn|^ps[hda]$|^avg[hda]$|^max[hda]$|implied|bookie|bookmaker|market|"
                       r"overround)", re.I)
ODDS_FEATURE_NAMES = ["odds_logit_ha", "odds_log_d", "odds_missing"]


def select_candidates(df):
    """Numeric columns that may be used as features (IDs, labels, odds and meta columns removed)."""
    num_cols = df.select_dtypes(include=[np.number]).columns
    excluded = [c for c in df.columns
                if c in TARGETS or ID_LIKE.search(c) or ODDS_LIKE.search(c) or META_LIKE.search(c)]
    candidates = [c for c in num_cols if c not in excluded]
    print(f"{len(candidates)} numeric columns can be used as features.")
    print("Excluded:", excluded)
    return candidates


def leakage_check(train, candidates, threshold=LEAK_THRESHOLD):
    """Drop columns too correlated with the goal difference of the SAME match (likely post-match stats).

    Returns the list of safe features.
    """
    gd = train["home_goals"] - train["away_goals"]
    corr = train[candidates].corrwith(gd).abs().sort_values(ascending=False)
    leaky = corr[corr > threshold].index.tolist()        # elo_diff is usually only ~0.3-0.4
    print("\nTop 10 |corr| with goal difference:")
    print(corr.head(10).round(3).to_string())
    print("\nSuspected leakage (dropped):", leaky if leaky else "none")
    return [c for c in candidates if c not in leaky]


# ---------------------------------------------------------------- bookmaker odds
@dataclass
class Odds:
    """Bookmaker odds columns (home, draw, away), stored either as probabilities or as decimal odds."""
    cols: tuple
    is_prob: bool

    def probs(self, d):
        """Normalised H/D/A probabilities (overround removed so each row sums to 1)."""
        O = d[list(self.cols)].astype(float).values
        Q = O if self.is_prob else 1.0 / O
        Q = np.where(Q > 0, Q, np.nan)
        return Q / Q.sum(1, keepdims=True)


def find_odds(df):
    """Auto-detect the odds triplet. Returns Odds or None. Fix by hand if detection is wrong,
    e.g. Odds(('B365H', 'B365D', 'B365A'), is_prob=False)."""
    oc = [c for c in df.columns if ODDS_LIKE.search(c) and pd.api.types.is_numeric_dtype(df[c])]

    def pick(pats):
        return next((c for c in oc for p in pats if re.search(p, c, re.I)), None)

    h = pick([r"home", r"_h$", r"^b365h$", r"^psh$", r"^avgh$"])
    d = pick([r"draw", r"_d$", r"^b365d$", r"^psd$", r"^avgd$"])
    a = pick([r"away", r"_a$", r"^b365a$", r"^psa$", r"^avga$"])
    if not (all([h, d, a]) and len({h, d, a}) == 3):
        return None
    cols = (h, d, a)
    return Odds(cols, bool(np.nanmax(df[list(cols)].astype(float).values) <= 1.0))


def add_odds_features(d, odds):
    """Add odds_logit_ha, odds_log_d, odds_missing in place (used as features only when USE_ODDS_FEATURE)."""
    if odds is None:
        return d
    P = odds.probs(d)
    with np.errstate(divide="ignore", invalid="ignore"):
        d["odds_logit_ha"] = np.log(P[:, 0] / P[:, 2])   # > 0: bookmaker favours the home team
        d["odds_log_d"] = np.log(P[:, 1])                 # draw likelihood according to the bookmaker
    ok = np.isfinite(d["odds_logit_ha"]) & np.isfinite(d["odds_log_d"])
    d.loc[~ok, ["odds_logit_ha", "odds_log_d"]] = np.nan  # left for the imputer (median)
    d["odds_missing"] = (~ok).astype(float)
    return d


def odds_feature_list(odds, use_odds, splits):
    """Names of the odds features to feed the models ([] when disabled or unavailable)."""
    feats = ODDS_FEATURE_NAMES if (use_odds and odds is not None) else []
    if use_odds and odds is None:
        print("USE_ODDS_FEATURE = True but no odds columns found -> training WITHOUT odds.")
    if feats:
        print("Odds used as features:", feats)
        print("   Share of matches with odds - train: %.1f%% | dev: %.1f%% | test: %.1f%%"
              % tuple(100 * (1 - d["odds_missing"].mean()) for d in splits))
    return feats


# ---------------------------------------------------------------- feature sets
def have(df, cols, label=""):
    """Keep only the columns present in df, reporting the missing ones."""
    present = [c for c in cols if c in df.columns]
    missing = [c for c in cols if c not in df.columns]
    if missing:
        print(f" [{label}] missing columns: {missing}")
    return present


@dataclass
class FeatureSets:
    """Every feature list the models need, built once from the data."""
    safe: list          # candidate columns that passed the leakage check
    odds_feats: list    # odds features ([] when USE_ODDS_FEATURE is off)
    lr_base: list       # PDF Logistic Regression (6 columns [+ odds])
    poi_feats: list     # PDF Poisson goal model
    elo: list           # ['elo_diff']
    form: list          # points / goals form, ppg, position


def build_feature_sets(df, safe, odds_feats):
    lr_base, poi_feats = pdf_feature_sets(df, odds_feats)
    elo = have(df, ["elo_diff"], "H")
    form = have(df, ["diff_points5", "diff_goals_for5", "diff_goals_against5", "diff_ppg", "position_diff"], "H2")
    return FeatureSets(safe, odds_feats, lr_base, poi_feats, elo, form)


def pdf_feature_sets(df, odds_feats):
    """Feature sets defined in the PDF: 6-column LR and Poisson goal model. Returns (lr_base, poi_feats)."""
    lr_base = have(df, ["elo_diff", "diff_points5", "diff_goals_for5", "diff_goals_against5", "diff_ppg",
                        "position_diff"], "LR 6 cols") + odds_feats
    poi_feats = have(df, ["elo_diff", "home_goals_for5", "home_goals_against5", "away_goals_for5",
                          "away_goals_against5", "home_home_goals_for5", "away_away_goals_against5",
                          "diff_xg5", "diff_xga5"], "Poisson") + odds_feats
    print("LR_BASE  :", lr_base)
    print("POI_FEATS:", poi_feats)
    return lr_base, poi_feats
