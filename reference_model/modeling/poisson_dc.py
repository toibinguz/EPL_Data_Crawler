"""Task 2 - Poisson goal model (home lambda, away lambda) with the Dixon-Coles low-score correction.

- alpha in {0.01, 0.1, 1} is tuned by MAE on Dev (as in the PDF).
- lam_diff for train is computed OUT-OF-FOLD by season (each season is predicted by a model that never saw it).
  Fitting Poisson and predicting on the same train rows would let lam_diff "know" the label, and the
  Logistic Regression would over-weight it.
- Dixon-Coles: one parameter rho, estimated on train, adjusts the probabilities of 0-0, 1-0, 0-1, 1-1.
"""
import numpy as np
import pandas as pd
from scipy.optimize import minimize_scalar
from scipy.stats import poisson
from sklearn.linear_model import PoissonRegressor
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import Pipeline

from .config import MAX_GOALS, POISSON_ALPHAS, RESULT_ORDER
from .metrics import ll_vec
from .preprocessing import make_prep

_I, _J = np.meshgrid(np.arange(MAX_GOALS + 1), np.arange(MAX_GOALS + 1), indexing="ij")
REGION = {"H": _I > _J, "D": _I == _J, "A": _I < _J}   # cells of the score matrix for each result


def poisson_pipe(alpha):
    return Pipeline(make_prep() + [("poi", PoissonRegressor(alpha=alpha, max_iter=3000))])


def fit_poisson(data, alpha, feats):
    mh = poisson_pipe(alpha).fit(data[feats], data["home_goals"])
    ma = poisson_pipe(alpha).fit(data[feats], data["away_goals"])
    return mh, ma


def predict_lams(models, data, feats):
    mh, ma = models
    return mh.predict(data[feats]), ma.predict(data[feats])


def oof_lams(data, alpha, feats, n_splits=5):
    """Out-of-fold lambdas, grouped by season."""
    lh = np.full(len(data), np.nan)
    la = np.full(len(data), np.nan)
    groups = data["season_start"].values
    gkf = GroupKFold(n_splits=min(n_splits, len(np.unique(groups))))
    for tr_idx, va_idx in gkf.split(data, groups=groups):
        m = fit_poisson(data.iloc[tr_idx], alpha, feats)
        lh[va_idx], la[va_idx] = predict_lams(m, data.iloc[va_idx], feats)
    return lh, la


def score_matrix(lh, la, rho=0.0):
    """Score probability matrix M[match, home goals, away goals], optionally Dixon-Coles corrected."""
    lh = np.asarray(lh, float)
    la = np.asarray(la, float)
    g = np.arange(MAX_GOALS + 1)
    M = poisson.pmf(g[None, :, None], lh[:, None, None]) * poisson.pmf(g[None, None, :], la[:, None, None])
    if rho != 0.0:   # Dixon-Coles tau factor
        M[:, 0, 0] *= 1 - lh * la * rho
        M[:, 0, 1] *= 1 + lh * rho
        M[:, 1, 0] *= 1 + la * rho
        M[:, 1, 1] *= 1 - rho
    M = np.clip(M, 1e-12, None)
    return M / M.sum(axis=(1, 2), keepdims=True)


def matrix_to_hda(M):
    return np.stack([M[:, REGION[k]].sum(1) for k in RESULT_ORDER], axis=1)


def fit_rho(lh, la, hg, ag):
    """Maximum-likelihood Dixon-Coles rho."""
    hg = np.clip(np.asarray(hg, int), 0, MAX_GOALS)
    ag = np.clip(np.asarray(ag, int), 0, MAX_GOALS)

    def nll(rho):
        M = score_matrix(lh, la, rho)
        return -np.log(M[np.arange(len(hg)), hg, ag]).sum()

    return minimize_scalar(nll, bounds=(-0.25, 0.25), method="bounded").x


def tune_alpha(train, dev, feats, alphas=POISSON_ALPHAS):
    """Pick alpha by mean home/away MAE on Dev. Returns (table, best alpha)."""
    rows = []
    for a in alphas:
        lh, la = predict_lams(fit_poisson(train, a, feats), dev, feats)
        rows.append({"alpha": a,
                     "MAE_home": mean_absolute_error(dev.home_goals, lh),
                     "MAE_away": mean_absolute_error(dev.away_goals, la)})
    tab = pd.DataFrame(rows)
    tab["MAE_avg"] = tab[["MAE_home", "MAE_away"]].mean(1)
    print(tab.round(4).to_string(index=False))
    best = float(tab.loc[tab.MAE_avg.idxmin(), "alpha"])
    print("-> best alpha:", best)
    return tab, best


def prepare_stage(tr, te_list, alpha, feats):
    """Add lam_h, lam_a, lam_diff: tr gets out-of-fold values, every te uses a model fit on the whole tr.

    Returns (tr, [te...], poisson models, rho).
    """
    tr = tr.copy()
    tr["lam_h"], tr["lam_a"] = oof_lams(tr, alpha, feats)
    models = fit_poisson(tr, alpha, feats)
    outs = []
    for te in te_list:
        te = te.copy()
        if len(te):
            te["lam_h"], te["lam_a"] = predict_lams(models, te, feats)
        else:
            te["lam_h"] = te["lam_a"] = np.array([], float)
        outs.append(te)
    for d in [tr] + outs:
        d["lam_diff"] = d["lam_h"] - d["lam_a"]
    rho = fit_rho(tr.lam_h, tr.lam_a, tr.home_goals, tr.away_goals)
    return tr, outs, models, rho


def report_score_quality(tr_s, dev_s, models, rho, feats):
    """Compare in-sample vs out-of-fold lam_diff and score quality of independent Poisson vs Dixon-Coles on Dev."""
    lh_in, la_in = predict_lams(models, tr_s, feats)
    gd = tr_s.home_goals - tr_s.away_goals
    print("corr(lam_diff, goal diff) on train - in-sample: %.3f | out-of-fold: %.3f"
          % (np.corrcoef(lh_in - la_in, gd)[0, 1], np.corrcoef(tr_s.lam_diff, gd)[0, 1]))

    for rho_name, r in [("Independent Poisson", 0.0), ("Dixon-Coles", rho)]:
        M = score_matrix(dev_s.lam_h, dev_s.lam_a, r)
        hg = np.clip(dev_s.home_goals.astype(int), 0, MAX_GOALS).values
        ag = np.clip(dev_s.away_goals.astype(int), 0, MAX_GOALS).values
        p_true = M[np.arange(len(hg)), hg, ag]
        flat = M.reshape(len(M), -1).argmax(1)
        exact = ((flat // (MAX_GOALS + 1) == hg) & (flat % (MAX_GOALS + 1) == ag)).mean()
        P = matrix_to_hda(M)
        print(f"{rho_name:19s} | score log-lik/match: {np.log(p_true).mean():.4f} | exact score: {exact:.3f} "
              f"| mean P(D): {P[:, 1].mean():.3f} (actual {np.mean(dev_s.result == 'D'):.3f}) "
              f"| H/D/A log loss: {ll_vec(P, dev_s.result).mean():.4f}")
