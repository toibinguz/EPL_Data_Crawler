"""Task 4 - Prediction files: predictions_test.csv (test set) and predictions.csv (upcoming fixtures, if any).

pred_home_goals, pred_away_goals = most likely score INSIDE the region matching pred_result
(H: home > away, D: equal, A: home < away).
likeliest_score = most likely score over the whole matrix (may disagree with the predicted result, often 1-1).
"""
import os

import joblib
import numpy as np

from .config import RESULT_ORDER
from .models import fit_predict
from .poisson_dc import REGION, prepare_stage, score_matrix

ID_COLUMNS = ["match_id", "season", "date", "gameweek", "home_team", "away_team"]


def score_table(M, P):
    pred_result = np.array(RESULT_ORDER)[P.argmax(1)]
    phg, pag, likeliest = [], [], []
    for n in range(len(M)):
        masked = np.where(REGION[pred_result[n]], M[n], -1)
        i, j = np.unravel_index(masked.argmax(), masked.shape)
        phg.append(i)
        pag.append(j)
        gi, gj = np.unravel_index(M[n].argmax(), M[n].shape)
        likeliest.append(f"{gi}-{gj}")
    return pred_result, phg, pag, likeliest


def _prediction_frame(data, M, P, id_cols):
    out = data[id_cols].copy()
    out["p_home"], out["p_draw"], out["p_away"] = P[:, 0].round(4), P[:, 1].round(4), P[:, 2].round(4)
    out["pred_result"], out["pred_home_goals"], out["pred_away_goals"], out["likeliest_score"] = score_table(M, P)
    out["exp_home_goals"], out["exp_away_goals"] = data.lam_h.round(2).values, data.lam_a.round(2).values
    return out


def write_test_predictions(test_s, M_test, P_test, id_cols, out_dir):
    """Predictions on the TEST set (model trained on train + dev) - for reporting against real results."""
    pt = _prediction_frame(test_s, M_test, P_test, id_cols)
    pt["true_result"] = test_s.result.values
    pt["true_score"] = (test_s.home_goals.astype(int).astype(str).values + "-"
                        + test_s.away_goals.astype(int).astype(str).values)
    pt.to_csv(os.path.join(out_dir, "predictions_test.csv"), index=False)
    pred_score = pt.pred_home_goals.astype(str) + "-" + pt.pred_away_goals.astype(str)
    print("Wrote predictions_test.csv |", f"H/D/A correct: {(pt.pred_result == pt.true_result).mean():.3f}",
          f"| exact score (pred): {(pred_score == pt.true_score).mean():.3f}")
    print(pt.head(10).to_string(index=False))
    return pt


def predict_upcoming(upcoming, prod_train, final_name, final_spec, alpha, poi_feats, odds_feats, id_cols,
                     out_dir, refit_with_test=False):
    """Fit the final model on prod_train and predict upcoming fixtures -> predictions.csv + final_models.joblib."""
    if len(upcoming) == 0:
        print("\nfeatures.csv currently has no upcoming matches (every match has been played).")
        print("Once the pipeline builds features for 2026/27 (folder 2026_2027_new/), rerun to write predictions.csv.")
        return None
    if odds_feats and upcoming["odds_missing"].mean() > 0:
        print(f"{int(upcoming['odds_missing'].sum())} upcoming matches have no odds -> odds median-imputed, "
              "these predictions are less reliable.")
    prod_s, (up_s,), poi_models, rho = prepare_stage(prod_train, [upcoming], alpha, poi_feats)
    P_up, final_model = fit_predict(final_spec, prod_s, up_s, rho)
    M_up = score_matrix(up_s.lam_h, up_s.lam_a, rho)

    pred = _prediction_frame(up_s, M_up, P_up, id_cols)
    pred["model"] = final_name
    out_path = os.path.join(out_dir, "predictions.csv")
    pred.to_csv(out_path, index=False)
    print("Wrote", out_path, "| training data:", "all played matches" if refit_with_test else "train+dev")
    print(pred.head(20).to_string(index=False))

    joblib.dump({"final_spec": final_spec, "final_model": final_model, "poisson_models": poi_models,
                 "rho": rho, "alpha": alpha, "poi_feats": poi_feats, "result_order": RESULT_ORDER},
                os.path.join(out_dir, "final_models.joblib"))
    return pred
