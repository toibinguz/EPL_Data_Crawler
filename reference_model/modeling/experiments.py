"""Side experiments that never change the chosen model or the prediction files.

1. Odds experiment - compare on the SAME test matches that have odds:
   final model (no odds) | bookmaker (normalised odds probabilities) |
   LR + odds (LR 6 cols + lam_diff + log(pH/pA), log(pD); C chosen on Dev) |
   blend w * model + (1 - w) * bookmaker (w chosen on Dev).
   delta = log loss(model) - log loss(bookmaker): negative means the model beats the bookmaker.

2. Accuracy by train start year - retrain the SAME final model with the SAME hyper-parameters, only changing
   the first training season (season_start >= year). Dev / Test unchanged: Dev model learns on train,
   Test model learns on train + dev. Log loss is reported too because accuracy ignores confidence.
"""
import os

import numpy as np
import pandas as pd

from .config import LR_C_GRID, RESULT_ORDER, SUSPICIOUS_LL
from .metrics import evaluate, ll_vec, paired_bootstrap
from .models import fit_predict
from .plots import plot_accuracy_by_train_start
from .poisson_dc import prepare_stage


# ---------------------------------------------------------------- 1. odds experiment
def _with_bookmaker(d, odds):
    d = d.copy()
    P = odds.probs(d)
    d["bk_H"], d["bk_D"], d["bk_A"] = P[:, 0], P[:, 1], P[:, 2]
    with np.errstate(divide="ignore", invalid="ignore"):
        d["odds_logit_ha"] = np.log(d["bk_H"] / d["bk_A"])
        d["odds_log_d"] = np.log(d["bk_D"])
    d["has_odds"] = np.isfinite(P).all(1) & (P > 0).all(1)
    return d


def odds_experiment(odds, lr_base, tr_s, dev_s, trdev_s, test_s, dev_pred_final, test_pred_final, final_name,
                    rho_sel, rho_eval, out_dir):
    if odds is None:
        print("No odds columns -> skipping this experiment.")
        return None
    tr_o, dev_o, trdev_o, test_o = (_with_bookmaker(d, odds) for d in (tr_s, dev_s, trdev_s, test_s))
    for name, d in [("train", tr_o), ("dev", dev_o), ("test", test_o)]:
        print(f"{name:5s}: {d.has_odds.mean():.1%} matches with odds ({d.has_odds.sum()}/{len(d)})")

    lr_odds = list(dict.fromkeys(lr_base + ["lam_diff"] + ["odds_logit_ha", "odds_log_d"]))
    tr_ok, dev_ok = tr_o[tr_o.has_odds], dev_o[dev_o.has_odds]
    dev_mask = dev_o.has_odds.values

    # Choose C for LR + odds on Dev
    c_rows = []
    for C in LR_C_GRID:
        P, _ = fit_predict({"kind": "lr", "feats": lr_odds, "C": C}, tr_ok, dev_ok, rho_sel)
        c_rows.append((C, float(ll_vec(P, dev_ok.result).mean())))
    C_odds = min(c_rows, key=lambda r: r[1])[0]
    print("\nLR + odds - Dev log loss by C:", {c: round(v, 4) for c, v in c_rows}, "-> C =", C_odds)

    # Choose the blend weight (final model + bookmaker) on Dev
    P_dev_model = dev_pred_final[dev_mask]
    P_dev_bk = dev_ok[["bk_H", "bk_D", "bk_A"]].values
    ws = np.round(np.arange(0, 1.01, 0.05), 2)
    ll_w = [ll_vec(w * P_dev_model + (1 - w) * P_dev_bk, dev_ok.result).mean() for w in ws]
    w_bk = float(ws[int(np.argmin(ll_w))])
    print(f"Blend - best w on Dev = {w_bk} (w = model weight, 1 - w = bookmaker weight)")

    # Evaluate on Test (same matches with odds)
    te_mask = test_o.has_odds.values
    te_ok = test_o[te_mask]
    y_te = te_ok.result.values
    P_te_model = test_pred_final[te_mask]
    P_te_bk = te_ok[["bk_H", "bk_D", "bk_A"]].values
    P_te_lro, lr_odds_model = fit_predict({"kind": "lr", "feats": lr_odds, "C": C_odds},
                                          trdev_o[trdev_o.has_odds], te_ok, rho_eval)
    P_te_blend = w_bk * P_te_model + (1 - w_bk) * P_te_bk

    preds = {f"Final model ({final_name})": P_te_model,
             "Bookmaker": P_te_bk,
             "LR + odds": P_te_lro,
             f"Blend model x {w_bk} + bookmaker x {1 - w_bk:.2f}": P_te_blend}
    board = pd.DataFrame([evaluate(P, y_te, n) for n, P in preds.items()]).sort_values("log_loss")
    print(f"\n=== TEST ({len(te_ok)} matches with odds) ===")
    print(board.round(4).to_string(index=False))

    # vs bookmaker: bootstrap CI95%
    ll_bk = ll_vec(P_te_bk, y_te)
    rows = []
    for n, P in preds.items():
        if n == "Bookmaker":
            continue
        d, lo, hi, _ = paired_bootstrap(ll_bk, ll_vec(P, y_te))
        rows.append({"model": n, "delta_vs_bookmaker": d, "CI_low": lo, "CI_high": hi,
                     "verdict": "better than bookmaker" if hi < 0
                     else ("worse than bookmaker" if lo > 0 else "equivalent to bookmaker")})
    print("\nvs bookmaker (negative delta = better than bookmaker):")
    print(pd.DataFrame(rows).round(4).to_string(index=False))

    # LR + odds coefficients: how much do odds dominate the other columns?
    coef = pd.DataFrame(lr_odds_model.named_steps["lr"].coef_.T, index=lr_odds,
                        columns=[f"coef_{c}" for c in lr_odds_model.classes_])
    print("\nLR + odds coefficients (standardised):")
    print(coef.round(3).to_string())

    # Final model vs bookmaker per season
    per_season = te_ok.assign(model=ll_vec(P_te_model, y_te), bookie=ll_bk).groupby("season")[["model", "bookie"]].mean()
    per_season["gap"] = per_season.model - per_season.bookie
    print("\nLog loss per season (Test):")
    print(per_season.round(4).to_string())

    board.to_csv(os.path.join(out_dir, "odds_experiment_test.csv"), index=False)
    if board.log_loss.min() < SUSPICIOUS_LL:
        print(" A model has log loss < %.2f - check the odds columns (closing odds published after kick-off?)."
              % SUSPICIOUS_LL)
    return board


# ---------------------------------------------------------------- 2. accuracy by train start year
def accuracy_by_train_start(train, dev, test, final_spec, final_name, alpha, poi_feats, start_years, out_dir):
    acc_rows, season_rows, test_hits = [], [], {}
    for sy in start_years:
        tr_y = train[train.season_start >= sy]
        # Dev: learn on train (from sy)
        tr_y_s, (dev_y_s,), _, rho_y = prepare_stage(tr_y, [dev], alpha, poi_feats)
        P_dev_y, _ = fit_predict(final_spec, tr_y_s, dev_y_s, rho_y)
        # Test: learn on train (from sy) + dev
        trdev_y_s, (test_y_s,), _, rho_y2 = prepare_stage(pd.concat([tr_y, dev]), [test], alpha, poi_feats)
        P_test_y, _ = fit_predict(final_spec, trdev_y_s, test_y_s, rho_y2)

        ev_dev, ev_test = evaluate(P_dev_y, dev_y_s.result), evaluate(P_test_y, test_y_s.result)
        acc_rows.append({"train_from": f"{sy}/{str(sy + 1)[2:]}", "start_year": sy, "n_train": len(tr_y),
                         "acc_dev": ev_dev["accuracy"], "acc_test": ev_test["accuracy"],
                         "logloss_dev": ev_dev["log_loss"], "logloss_test": ev_test["log_loss"],
                         "rps_test": ev_test["rps"]})
        hit = np.array(RESULT_ORDER)[P_test_y.argmax(1)] == test_y_s.result.values
        test_hits[sy] = hit.astype(float)
        for season, g in test_y_s.assign(hit=hit).groupby("season"):
            season_rows.append({"start_year": sy, "season": season, "accuracy": g.hit.mean(), "n": len(g)})

    acc_tab = pd.DataFrame(acc_rows)
    season_tab = pd.DataFrame(season_rows)
    always_h_dev, always_h_test = (dev.result == "H").mean(), (test.result == "H").mean()
    print(f"Model: {final_name}")
    print(acc_tab.round(4).to_string(index=False))
    print(f'\nBaseline "always predict home win": Dev {always_h_dev:.3f} | Test {always_h_test:.3f}')

    plot_accuracy_by_train_start(acc_tab, season_tab, start_years, always_h_dev, always_h_test, out_dir)
    acc_tab.to_csv(os.path.join(out_dir, "accuracy_by_train_start.csv"), index=False)
    season_tab.to_csv(os.path.join(out_dir, "accuracy_by_train_start_per_season.csv"), index=False)

    # Is the difference significant? (bootstrap on Test, first two years of the list)
    if len(start_years) >= 2:
        a, b = start_years[:2]
        d, lo, hi, _ = paired_bootstrap(1 - test_hits[a], 1 - test_hits[b])   # error delta: negative = b better
        print(f"Test: accuracy(train from {b}) - accuracy(train from {a}) = {-d:+.3%}  CI95% [{-hi:+.3%}, {-lo:+.3%}]")
        print("-> CI contains 0: difference not statistically significant." if lo < 0 < hi
              else "-> Difference is statistically significant.")
    return acc_tab, season_tab
