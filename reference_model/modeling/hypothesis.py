"""Task 3 - Hypothesis tests H1-H9 (nested models + 2,000-sample paired bootstrap).

Each hypothesis compares a BASE model with BASE + a feature group on the same matches.
Evaluation is rolling-origin: for every season 2015/16 -> 2023/24, train on ALL earlier seasons and predict
that season. Test is never touched.
delta = log loss(with group) - log loss(without): delta < 0 with the CI95% fully below 0 -> the group is useful.

Column names for H4, H7, H8 are matched with regexes. Check the printed table; if detection is wrong,
edit the 'add' lists in build_hypotheses directly.
"""
import os
import re

import numpy as np
import pandas as pd

from .config import EVAL_SEASONS, TEST_START
from .features import have
from .metrics import ll_vec, paired_bootstrap
from .models import fit_predict

FOTMOB_PATTERN = r"possession|pass_accuracy|big_chances|box_shots|box_touches|xgot"
USEFUL, HARMFUL, INCONCLUSIVE = "USEFUL", "HARMFUL", "not enough evidence"


def n5_filter(prefix):
    """Keep only matches where both teams have at least 3 of their last 5 matches with this data source."""
    def f(d):
        h, a = f"home_{prefix}_n5", f"away_{prefix}_n5"
        if {h, a} <= set(d.columns):
            return (d[h] >= 3) & (d[a] >= 3)
        return pd.Series(True, index=d.index)
    return f


def build_hypotheses(df, fs):
    """Feature-group definitions of H1-H8 (+ H10 when odds are model features)."""
    def rx(pattern, exclude=()):
        return [c for c in fs.safe if re.search(pattern, c, re.I) and c not in exclude]

    elo, form = fs.elo, fs.form
    goals_side = have(df, ["home_goals_for5", "home_goals_against5", "away_goals_for5", "away_goals_against5"], "H6")
    fotmob = rx(FOTMOB_PATTERN)
    hyps = {
        "H1": dict(desc="Elo beats the historical label frequency", base="PRIOR", add=elo),
        "H2": dict(desc="Points / goals form adds to Elo", base=elo, add=form),
        "H3": dict(desc="xG / xGA form adds to goals", base=elo + form,
                   add=have(df, ["diff_xg5", "diff_xga5"], "H3"), filter=n5_filter("xg")),
        "H4": dict(desc="Head-to-head adds to Elo and form", base=elo + form, add=rx(r"h2h|head_?to_?head")),
        "H5": dict(desc="Home advantage changes over time", base=elo + form,
                   add=have(df, ["league_home_adv380"], "H5")),
        "H6": dict(desc="Venue-specific form adds to overall form", base=elo + form + goals_side,
                   add=rx(r"^home_home_|^away_away_")),
        "H7": dict(desc="Properly defined rest days help", base=elo + form,
                   add=rx(r"_promoted$|_form_age_days$|_rest_days$")),
        "H8": dict(desc="FotMob style-of-play stats add information", base=elo + form,
                   add=[c for c in fotmob if c.startswith("diff_")] or fotmob, filter=n5_filter("fm")),
    }
    if fs.odds_feats:   # extra test: do odds add information beyond Elo + form?
        hyps["H10"] = dict(desc="Bookmaker odds add to Elo + form", base=elo + form, add=fs.odds_feats)
    for h, s in hyps.items():
        print(f"{h}: +{len(s['add'])} cols -> {s['add'][:12]}{' ...' if len(s['add']) > 12 else ''}")
    return hyps


def rolling_origin(data, feats, C):
    """Predict every EVAL_SEASONS season with a model trained on all earlier seasons."""
    Ps, ys, idx = [], [], []
    for s in EVAL_SEASONS:
        tr = data[data.season_start < s]
        te = data[data.season_start == s]
        if len(te) == 0 or len(tr) < 300:
            continue
        spec = {"kind": "prior"} if feats == "PRIOR" else {"kind": "lr", "feats": feats, "C": C}
        P, _ = fit_predict(spec, tr, te, 0.0)
        Ps.append(P)
        ys.append(te.result.values)
        idx.append(te.index.values)
    return np.vstack(Ps), np.concatenate(ys), np.concatenate(idx)


def run_hypothesis_tests(played_all, hyps, lr_base, C, odds=None, out_dir=None):
    """Run H1-H8 (+H10) and H9 (bookmaker reference). Returns the result table."""
    pre_test = played_all[played_all.season_start < TEST_START].copy()
    rows = []
    for h, s in hyps.items():
        if not s["add"]:
            rows.append({"H": h, "hypothesis": s["desc"], "verdict": "SKIPPED (no columns)"})
            continue
        data = pre_test[s["filter"](pre_test)] if "filter" in s else pre_test
        if data[data.season_start.isin(EVAL_SEASONS)].shape[0] < 300:
            rows.append({"H": h, "hypothesis": s["desc"], "verdict": "SKIPPED (too few matches with data)"})
            continue
        base_feats = s["base"]
        new_feats = s["add"] if base_feats == "PRIOR" else list(dict.fromkeys(base_feats + s["add"]))
        P0, y, _ = rolling_origin(data, base_feats, C)
        P1, _, _ = rolling_origin(data, new_feats, C)
        l0, l1 = ll_vec(P0, y), ll_vec(P1, y)
        d, lo, hi, p = paired_bootstrap(l0, l1)
        verdict = USEFUL if hi < 0 else (HARMFUL if lo > 0 else INCONCLUSIVE)
        first = int(data.loc[data.season_start.isin(EVAL_SEASONS), "season_start"].min())
        rows.append({"H": h, "hypothesis": s["desc"], "n_matches": len(y), "n_added_cols": len(s["add"]),
                     "eval_seasons": f"{first}->{TEST_START - 1}",
                     "LL_base": l0.mean(), "LL_new": l1.mean(), "delta": d, "CI_low": lo, "CI_high": hi,
                     "P(delta>=0)": p, "verdict": verdict})

    # H9 - bookmaker odds as a reference (vs 6-column LR, on the same matches that have odds)
    if odds is not None:
        Pm, y, idx = rolling_origin(pre_test, lr_base, C)
        Pb = odds.probs(pre_test.loc[idx])
        ok = np.isfinite(Pb).all(1)
        lb, lm = ll_vec(Pb[ok], y[ok]), ll_vec(Pm[ok], y[ok])
        d, lo, hi, p = paired_bootstrap(lb, lm)
        rows.append({"H": "H9", "hypothesis": "Bookmaker odds as reference (delta = LR - bookmaker)",
                     "n_matches": int(ok.sum()), "LL_base": lb.mean(), "LL_new": lm.mean(), "delta": d,
                     "CI_low": lo, "CI_high": hi, "P(delta>=0)": p,
                     "verdict": "bookmaker better than model" if lo > 0
                     else ("model >= bookmaker" if hi < 0 else "equivalent")})
    else:
        rows.append({"H": "H9", "hypothesis": "Bookmaker odds as reference", "verdict": "SKIPPED (no odds columns)"})

    res = pd.DataFrame(rows)
    print(res.round(4).to_string(index=False))
    if out_dir is not None:
        res.to_csv(os.path.join(out_dir, "hypothesis_tests.csv"), index=False)
    return res


def accepted_features(hyps, hyp_res):
    """Columns of every hypothesis (except H1) whose group was found USEFUL."""
    verdicts = hyp_res.set_index("H")["verdict"]
    return [c for h, s in hyps.items() if h != "H1" and verdicts.get(h) == USEFUL for c in s["add"]]
