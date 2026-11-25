"""End-to-end train / dev / test pipeline (replaces EPL_train_dev_test_.ipynb).

Usage (from the project root):
    python -m modeling.train_eval                         # find / download features.csv, run everything
    python -m modeling.train_eval --features path/to/features.csv
    python -m modeling.train_eval --use-odds              # bookmaker odds as model features
    python -m modeling.train_eval --skip-experiments      # skip the odds and train-start experiments

Data: epl--league-dataset/clean/features.csv - 9,870 matches (2000/01 -> 2025/26), 113 columns. Looked up in
--features, $EPL_FEATURES_CSV, $EPL_DATA_DIR, data/epl--league-dataset/, data/; otherwise downloaded from
Hugging Face into data/epl--league-dataset/.

Each run writes into outputs/run_<date>_<time>_<odds|no_odds>/ (override the root with $EPL_OUT_DIR).

| Split | Seasons            | Role                                                      |
|-------|--------------------|-----------------------------------------------------------|
| Train | 2000/01 -> 2021/22 | Fitting                                                   |
| Dev   | 2022/23 -> 2023/24 | Hyper-parameters, model choice, feature-group choice      |
| Test  | 2024/25 ->         | Evaluated only ONCE at the end, never used to choose      |

Steps
 0. Configuration and data lookup          6. Task 1 - Logistic Regression H/D/A (6 cols [+ lam_diff])
 1. Load data, quick EDA                   7. Task 3 - Hypothesis tests H1-H9
 2. Chronological split by season          8. Upgraded models (LR+, HistGB, LR + Dixon-Coles blend)
 3. Feature selection + leakage check      9. Final model choice on Dev
 4. Preprocessing and metrics             10. Refit on Train+Dev -> evaluate Test once (+ experiments)
 5. Task 2 - Poisson + Dixon-Coles         11. Task 4 - Predict upcoming matches -> predictions.csv

Improvements over the PDF
- Train lam_diff is computed out-of-fold by season (GroupKFold), so the LR does not over-weight it.
- Besides log loss: RPS (standard for football since H-D-A is ordered), Brier and accuracy.
- H1-H9 use rolling-origin evaluation by season (2015/16 -> 2023/24): many more matches than the 760 Dev
  matches -> narrower confidence intervals, and Test is still untouched.
- Dixon-Coles corrects 0-0, 1-0, 0-1, 1-1 (independent Poisson usually under-predicts draws).
- LR probabilities blended with score-matrix probabilities, weight chosen on Dev.

Interpretation notes
- Models are chosen with Dev only; Test is used exactly once. Changing the model after looking at Test makes
  the Test score no longer unbiased.
- If M5/M6/M7 beat M3 only slightly with a CI containing 0, report M3 (the PDF model) as the main model and
  present the upgrades as additional experiments.
- 2019/20 - 2020/21 (COVID, no crowds) sharply reduced home advantage - why H5 (league_home_adv380) is tested.
- Frequency baseline log loss is usually ~1.06-1.07; bookmakers ~0.95-0.97 on the EPL. A model without odds
  around 0.95-0.97 is reasonable.
"""
import argparse
import json
import os
import shutil
import warnings

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error

from . import config
from .data import (dataset_root, load_features, locate_features, locate_hypotheses, played_matches, quick_eda,
                   show_tree, split_by_season)
from .experiments import accuracy_by_train_start, odds_experiment
from .features import (add_odds_features, build_feature_sets, find_odds, leakage_check, odds_feature_list,
                       select_candidates)
from .hypothesis import accepted_features, build_hypotheses, run_hypothesis_tests
from .metrics import evaluate, ll_vec, rps_vec
from .models import M0, M4, build_dev_leaderboard, choose_final, fit_predict, model_names, tune_lr
from .plots import plot_blend_weight, plot_hypotheses, plot_reliability
from .poisson_dc import prepare_stage, report_score_quality, score_matrix, tune_alpha
from .predict import ID_COLUMNS, predict_upcoming, write_test_predictions

warnings.filterwarnings("ignore")
pd.set_option("display.max_columns", 200)
pd.set_option("display.width", 200)


def section(title):
    print(f"\n{'=' * 100}\n{title}\n{'=' * 100}")


def main(features=None, use_odds=config.USE_ODDS_FEATURE, run_experiments=True,
         refit_with_test=config.REFIT_WITH_TEST_FOR_PRODUCTION):
    # ---------------------------------------------------------------- 0. configuration
    section("0. Configuration and data lookup")
    features_csv = locate_features(features)
    root = dataset_root(features_csv)
    hyp_csv = locate_hypotheses(root)
    run_name, out_dir = config.make_run_dir(use_odds)
    print("DATASET ROOT:", root)
    print("FEATURES_CSV:", features_csv)
    print("Results of this run go to:", out_dir)
    show_tree(root)

    # ---------------------------------------------------------------- 1. load + EDA
    section("1. Load data and quick EDA")
    df, date_col = load_features(features_csv)
    played_all = played_matches(df)
    quick_eda(df, played_all, out_dir)

    # ---------------------------------------------------------------- 2. split
    section("2. Train / Dev / Test split by season")
    train, dev, test, upcoming = split_by_season(df, played_all, date_col)

    # ---------------------------------------------------------------- 3. features + leakage
    section("3. Feature selection and leakage check")
    candidates = select_candidates(df)
    safe = leakage_check(train, candidates)
    odds = find_odds(df)              # used for H9 as reference; as features only with use_odds
    print("\nOdds columns:", odds.cols if odds else None)
    for d in (df, played_all, train, dev, test, upcoming):
        add_odds_features(d, odds)
    odds_feats = odds_feature_list(odds, use_odds, (train, dev, test))
    fs = build_feature_sets(df, safe, odds_feats)
    n_m2, n_m3 = model_names(odds_feats)

    # ---------------------------------------------------------------- 5. Poisson + Dixon-Coles
    section("5. Task 2 - Poisson (home / away lambda) + Dixon-Coles")
    _, best_alpha = tune_alpha(train, dev, fs.poi_feats)
    tr_s, (dev_s,), poi_models_sel, rho_sel = prepare_stage(train, [dev], best_alpha, fs.poi_feats)
    print(f"Dixon-Coles rho (train) = {rho_sel:.4f}")
    report_score_quality(tr_s, dev_s, poi_models_sel, rho_sel, fs.poi_feats)

    # ---------------------------------------------------------------- 6. LR (PDF)
    section("6. Task 1 - Logistic Regression H/D/A (PDF)")
    tab6, C6 = tune_lr(fs.lr_base, "6 cols", tr_s, dev_s, rho_sel)
    tab7, C7 = tune_lr(fs.lr_base + ["lam_diff"], "6 cols + lam_diff", tr_s, dev_s, rho_sel)
    print(pd.concat([tab6, tab7]).round(4).to_string(index=False))
    print(f"\n-> best C: 6 cols = {C6}, 6 cols + lam_diff = {C7}")

    # ---------------------------------------------------------------- 7. hypotheses
    section("7. Task 3 - Hypothesis tests H1-H9")
    if os.path.exists(hyp_csv):
        print("hypotheses.csv:")
        print(pd.read_csv(hyp_csv).to_string())
    else:
        print("Not found:", hyp_csv, "-> using the column groups defined in hypothesis.py.")
    hyps = build_hypotheses(df, fs)
    hyp_res = run_hypothesis_tests(played_all, hyps, fs.lr_base, C6, odds, out_dir)   # C chosen for 6-col LR
    plot_hypotheses(hyp_res, out_dir)

    # ---------------------------------------------------------------- 8. upgraded models
    section("8. Upgraded models and Dev comparison")
    specs, dev_preds, dev_board, (ws, blend_ll, w_best) = build_dev_leaderboard(
        fs, tr_s, dev_s, rho_sel, C6, C7, accepted_features(hyps, hyp_res), odds)

    # ---------------------------------------------------------------- 9. final choice
    section("9. Final model choice on Dev")
    best_name, final_name = choose_final(dev_board, dev_preds, dev_s, ref_name=n_m3)
    final_spec = specs[final_name]
    plot_blend_weight(ws, blend_ll, w_best, out_dir)

    # ---------------------------------------------------------------- 10. test (once)
    section("10. Refit on Train + Dev -> evaluate Test (once)")
    trdev = pd.concat([train, dev])
    trdev_s, (test_s,), _, rho_eval = prepare_stage(trdev, [test], best_alpha, fs.poi_feats)
    print(f"Dixon-Coles rho (train+dev) = {rho_eval:.4f}  | test: {len(test_s)} matches")

    test_preds = {n: fit_predict(specs[n], trdev_s, test_s, rho_eval)[0]
                  for n in dict.fromkeys([M0, n_m2, n_m3, M4, best_name, final_name])}
    test_board = pd.DataFrame([evaluate(P, test_s.result, n) for n, P in test_preds.items()]).sort_values("log_loss")
    if odds is not None:
        Pb = odds.probs(test_s)
        ok = np.isfinite(Pb).all(1)
        if ok.any():
            ref = evaluate(Pb[ok], test_s.result.values[ok], "(reference) Bookmaker")
            test_board = pd.concat([test_board, pd.DataFrame([ref])])
    print(f"=== TEST RESULTS ({config.TEST_START}/{str(config.TEST_START + 1)[2:]} ->) ===")
    print(test_board.round(4).to_string(index=False))

    P_final_test = test_preds[final_name]
    by_season = (test_s.assign(ll=ll_vec(P_final_test, test_s.result), rps=rps_vec(P_final_test, test_s.result))
                 .groupby("season").agg(n=("ll", "size"), log_loss=("ll", "mean"), rps=("rps", "mean")))
    print("\nFinal model per season:")
    print(by_season.round(4).to_string())

    M_test = score_matrix(test_s.lam_h, test_s.lam_a, rho_eval)
    print("\nMAE home goals: %.3f | away goals: %.3f" % (mean_absolute_error(test_s.home_goals, test_s.lam_h),
                                                       mean_absolute_error(test_s.away_goals, test_s.lam_a)))
    test_board.to_csv(os.path.join(out_dir, "test_metrics.csv"), index=False)

    plot_reliability(P_final_test, test_s.result, final_name, out_dir)
    # Standardised LR coefficients - easy to explain to examiners
    _, lr_model = fit_predict(specs[n_m3], trdev_s, test_s.head(1), rho_eval)
    coef = pd.DataFrame(lr_model.named_steps["lr"].coef_.T, index=specs[n_m3]["feats"],
                        columns=[f"coef_{c}" for c in lr_model.classes_])
    print("\nLogistic Regression coefficients (per standard deviation of each feature):")
    print(coef.round(3).to_string())

    if run_experiments:
        section("10b. Experiment: adding bookmaker odds")
        odds_experiment(odds, fs.lr_base, tr_s, dev_s, trdev_s, test_s, dev_preds[final_name], P_final_test,
                        final_name, rho_sel, rho_eval, out_dir)
        section("10c. Accuracy by train start year")
        accuracy_by_train_start(train, dev, test, final_spec, final_name, best_alpha, fs.poi_feats,
                                config.START_YEARS, out_dir)

    # ---------------------------------------------------------------- 11. predictions
    section("11. Task 4 - Predictions")
    id_cols = [c for c in ID_COLUMNS if c in df.columns]
    write_test_predictions(test_s, M_test, P_final_test, id_cols, out_dir)
    prod_train = played_all if refit_with_test else trdev
    predict_upcoming(upcoming, prod_train, final_name, final_spec, best_alpha, fs.poi_feats, odds_feats, id_cols,
                     out_dir, refit_with_test)

    # ---------------------------------------------------------------- 12. summary
    section("12. Output files")
    # Save the configuration and choices of this run, to know later which setup produced which files
    run_info = {
        "run_name": run_name, "features_csv": str(features_csv),
        "train_seasons": f"{config.TRAIN_START}/{config.TRAIN_START + 1} -> {config.TRAIN_END}/{config.TRAIN_END + 1}",
        "dev_seasons": f"{config.DEV_START}/{config.DEV_START + 1} -> {config.DEV_END}/{config.DEV_END + 1}",
        "test_from": f"{config.TEST_START}/{config.TEST_START + 1}",
        "use_odds_feature": bool(odds_feats), "odds_feats": odds_feats,
        "poisson_alpha": best_alpha, "rho_dixon_coles_eval": float(rho_eval),
        "C_6cols": C6, "C_6cols_lam": C7, "final_model": final_name,
        "test_log_loss_final": float(test_board.set_index("model").loc[final_name, "log_loss"]),
    }
    with open(os.path.join(out_dir, "run_info.json"), "w", encoding="utf-8") as f:
        json.dump(run_info, f, ensure_ascii=False, indent=2)

    print("Results folder:", out_dir)
    for name in sorted(os.listdir(out_dir)):
        print(f"   {name:40s} {os.path.getsize(os.path.join(out_dir, name)) / 1024:8.1f} KB")
    zip_path = shutil.make_archive(str(out_dir), "zip", out_dir)   # one archive to download / share
    print("\nArchive:", zip_path)
    return run_info


def parse_args(argv=None):
    p = argparse.ArgumentParser(description="EPL train / dev / test modelling pipeline")
    p.add_argument("--features", help="path to features.csv (default: auto-detect or download)")
    p.add_argument("--use-odds", action="store_true", default=config.USE_ODDS_FEATURE,
                   help="use bookmaker odds as model features")
    p.add_argument("--skip-experiments", action="store_true", help="skip the odds and train-start experiments")
    p.add_argument("--refit-with-test", action="store_true", default=config.REFIT_WITH_TEST_FOR_PRODUCTION,
                   help="fit the production model on every played match (only after reporting the test score)")
    return p.parse_args(argv)


if __name__ == "__main__":
    a = parse_args()
    main(a.features, a.use_odds, not a.skip_experiments, a.refit_with_test)
