"""Task 1 and upgraded models: model specs, fitting, LR / HGB tuning, Dev leaderboard and final model choice.

Models compared on Dev:
  M0 label frequency | M1 LR Elo only | M2 LR 6 cols | M3 LR 6 cols + lam_diff (the PDF model)
  M4 Dixon-Coles (H/D/A probabilities from the score matrix)
  M5 LR+ = M3 + feature groups accepted as USEFUL by the hypothesis tests
  M6 HistGradientBoosting (small learning rate, shallow trees, early stopping) on every safe column
  M7 Blend: w * (best LR) + (1 - w) * Dixon-Coles, w chosen on Dev
"""
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier

from .config import (HGB_GRID, LR_C_GRID, PREFER_PDF_IF_NOT_SIGNIFICANT, RESULT_ORDER, SEED, SUSPICIOUS_LL)
from .metrics import evaluate, ll_vec, paired_bootstrap
from .poisson_dc import matrix_to_hda, score_matrix
from .preprocessing import lr_pipe

M0, M1, M4, M5, M6, M7 = "M0 Frequency", "M1 LR Elo", "M4 Dixon-Coles", "M5 LR+", "M6 HistGB", "M7 Blend"


def model_names(odds_feats):
    """Display names of M2 / M3 (tagged when odds are used as features)."""
    tag = " + odds" if odds_feats else ""
    return f"M2 LR 6 cols{tag}", f"M3 LR 6 + lam_diff{tag}"


def proba_HDA(model, X):
    """predict_proba follows model.classes_ = ['A','D','H'] -> reorder to H, D, A."""
    P = model.predict_proba(X)
    cls = list(model.classes_)
    return P[:, [cls.index(k) for k in RESULT_ORDER]]


def fit_predict(spec, tr, te, rho):
    """Train the model described by spec on tr. Returns (H/D/A probabilities on te, fitted model or None)."""
    k = spec["kind"]
    if k == "prior":
        p = tr["result"].value_counts(normalize=True).reindex(RESULT_ORDER).fillna(0).values
        return np.tile(p, (len(te), 1)), None
    if k == "lr":
        m = lr_pipe(spec["C"]).fit(tr[spec["feats"]], tr["result"])
        return proba_HDA(m, te[spec["feats"]]), m
    if k == "dc":
        return matrix_to_hda(score_matrix(te.lam_h, te.lam_a, rho)), None
    if k == "hgb":
        m = HistGradientBoostingClassifier(learning_rate=spec["lr"], max_depth=spec["depth"], max_iter=1000,
                                           min_samples_leaf=50, l2_regularization=1.0, early_stopping=True,
                                           validation_fraction=0.15, n_iter_no_change=30, random_state=SEED)
        feats = [c for c in spec["feats"] if tr[c].nunique(dropna=True) >= 2]   # drop constant / all-NaN columns
        m.fit(tr[feats], tr["result"])
        return proba_HDA(m, te[feats]), m
    if k == "blend":
        P1, m1 = fit_predict(spec["a"], tr, te, rho)
        P2, _ = fit_predict(spec["b"], tr, te, rho)
        return spec["w"] * P1 + (1 - spec["w"]) * P2, m1
    raise ValueError(k)


def tune_lr(feats, label, tr_s, dev_s, rho):
    """Tune C by Dev log loss. Returns (table, best C)."""
    res = []
    for C in LR_C_GRID:
        P, _ = fit_predict({"kind": "lr", "feats": feats, "C": C}, tr_s, dev_s, rho)
        scores = {k: v for k, v in evaluate(P, dev_s.result).items() if k not in ("model", "n")}
        res.append({"feature_set": label, "C": C, **scores})
    res = pd.DataFrame(res)
    return res, float(res.loc[res.log_loss.idxmin()].C)


def tune_hgb(feats, tr_s, dev_s, rho):
    rows = []
    for lr_, depth in HGB_GRID:
        P, _ = fit_predict({"kind": "hgb", "feats": feats, "lr": lr_, "depth": depth}, tr_s, dev_s, rho)
        rows.append({"lr": lr_, "depth": depth, **evaluate(P, dev_s.result)})
    tab = pd.DataFrame(rows)
    print("\nHGB:")
    print(tab.drop(columns=["model", "n"]).round(4).to_string(index=False))
    return tab, tab.loc[tab.log_loss.idxmin()]


def build_dev_leaderboard(fs, tr_s, dev_s, rho, C6, C7, accepted, odds=None):
    """Fit M0-M7 on train, score on Dev.

    fs: FeatureSets. Returns (specs, dev_preds, dev_board, blend_info) where blend_info = (ws, blend_ll, w_best).
    """
    n_m2, n_m3 = model_names(fs.odds_feats)
    lr_plus = list(dict.fromkeys(fs.lr_base + ["lam_diff"] + accepted))
    print("Extra columns for LR+ :", accepted if accepted else "(no group passed the tests -> LR+ = M3)")
    tab_plus, C_plus = tune_lr(lr_plus, "LR+", tr_s, dev_s, rho)
    print(tab_plus.round(4).to_string(index=False))

    hgb_feats = list(dict.fromkeys(fs.safe + ["lam_diff"] + fs.odds_feats))
    _, best_hgb = tune_hgb(hgb_feats, tr_s, dev_s, rho)

    specs = {
        M0: {"kind": "prior"},
        M1: {"kind": "lr", "feats": fs.elo, "C": C6},
        n_m2: {"kind": "lr", "feats": fs.lr_base, "C": C6},
        n_m3: {"kind": "lr", "feats": fs.lr_base + ["lam_diff"], "C": C7},
        M4: {"kind": "dc"},
        M5: {"kind": "lr", "feats": lr_plus, "C": C_plus},
        M6: {"kind": "hgb", "feats": hgb_feats, "lr": float(best_hgb.lr), "depth": int(best_hgb.depth)},
    }
    dev_preds = {name: fit_predict(spec, tr_s, dev_s, rho)[0] for name, spec in specs.items()}

    # Blend: best LR (M3 or M5) with Dixon-Coles
    lr_best = min([n_m3, M5], key=lambda n: ll_vec(dev_preds[n], dev_s.result).mean())
    ws = np.round(np.arange(0, 1.01, 0.05), 2)
    blend_ll = [ll_vec(w * dev_preds[lr_best] + (1 - w) * dev_preds[M4], dev_s.result).mean() for w in ws]
    w_best = float(ws[int(np.argmin(blend_ll))])
    specs[M7] = {"kind": "blend", "a": specs[lr_best], "b": specs[M4], "w": w_best}
    dev_preds[M7] = w_best * dev_preds[lr_best] + (1 - w_best) * dev_preds[M4]
    print(f"\nBlend: {lr_best} x w={w_best} + Dixon-Coles x {1 - w_best:.2f}")

    dev_board = pd.DataFrame([evaluate(P, dev_s.result, n) for n, P in dev_preds.items()]).sort_values("log_loss")

    # "Too good to be true" check: EPL prediction without odds rarely goes below ~0.93 log loss.
    # A much lower score is almost certainly leakage (post-match columns, duplicated matches across splits...).
    sus = dev_board[dev_board.log_loss < SUSPICIOUS_LL]["model"].tolist()
    if sus:
        print(f" SUSPECTED LEAKAGE: {sus} have log loss < {SUSPICIOUS_LL} -> excluded from selection. "
              "Check the feature columns!")
    dev_board["suspected_leak"] = dev_board["model"].isin(sus)
    if odds is not None:
        Pb = odds.probs(dev_s)
        ok = np.isfinite(Pb).all(1)
        print(f"(Reference) Bookmaker on Dev: log loss = {ll_vec(Pb[ok], dev_s.result.values[ok]).mean():.4f}")
    print("\n=== DEV LEADERBOARD ===")
    print(dev_board.round(4).to_string(index=False))
    return specs, dev_preds, dev_board, (ws, blend_ll, w_best)


def choose_final(dev_board, dev_preds, dev_s, ref_name, prefer_pdf=PREFER_PDF_IF_NOT_SIGNIFICANT):
    """Best non-leaky Dev model; fall back to the PDF model when the gain is not significant.

    Returns (best_name, final_name).
    """
    best_name = dev_board[~dev_board.suspected_leak].iloc[0]["model"]
    final_name = best_name
    print("Best on Dev:", best_name)
    if best_name != ref_name:
        d, lo, hi, _ = paired_bootstrap(ll_vec(dev_preds[ref_name], dev_s.result),
                                        ll_vec(dev_preds[best_name], dev_s.result))
        print(f"vs PDF model ({ref_name}): delta log loss = {d:+.4f}  CI95% [{lo:+.4f}, {hi:+.4f}]")
        if hi >= 0 and prefer_pdf:
            final_name = ref_name
            print("CI contains 0 -> improvement not certain -> keep the PDF model.")
    print("==> Final model:", final_name)
    return best_name, final_name
