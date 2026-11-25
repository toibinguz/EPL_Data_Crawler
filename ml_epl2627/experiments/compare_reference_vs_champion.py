"""
ml_epl2627/experiments/compare_reference_vs_champion.py
Strict 1-to-1 aligned benchmark & diagnostic audit comparing:
  - Reference models from reference_model/modeling (M0, M2, M3 LR, M4 Dixon-Coles, M6 HGB, M7 Blend)
    evaluated under BOTH:
      (a) Original Reference cutoff (Train<=21/22, Dev=22/23..23/24, refit <=23/24 -> predict 25/26 & 26/27)
      (b) Fair causal cutoff (refit <=24/25 -> predict 25/26; refit <=25/26 -> predict 26/27)
  - Champion_v1_Legacy, Champion_v2_BayesOnly, Champion_v2_New
  - Market_Odds (de-vigged market average from epl_clean.db & features.csv)

Matches are strictly 1-to-1 aligned by (date, home_code, away_code) across:
  - Season 2025/26 (N = 380)
  - Season 2026/27 (N = 80 overlapping fixtures in clean/features.csv through 2026-10-25)
"""

import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent.parent
ML_DIR = ROOT / "ml_epl2627"
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(ML_DIR) not in sys.path:
    sys.path.insert(0, str(ML_DIR))

from reference_model.modeling.features import (  # noqa: E402
    build_feature_sets,
    find_odds,
    leakage_check,
    select_candidates,
)
from reference_model.modeling.models import fit_predict, tune_hgb, tune_lr  # noqa: E402
from reference_model.modeling.poisson_dc import prepare_stage, tune_alpha  # noqa: E402

from src.bayesian_models import estimate_historical_negbin_and_rho  # noqa: E402
from src.metrics import bootstrap_rps_diff, clip_probs, evaluate_all  # noqa: E402
from src.strict_models import (  # noqa: E402
    evaluate_fixed_elo,
    fit_bayesian_poisson_glm_laplace,
    fit_bayesian_ridge_logclip,
    predict_elo_on_test,
    prepare_next_season_ratings,
    run_elo_walkforward,
    walkforward_bayesian_season,
)

FEATURES_CSV = ROOT / "clean" / "features.csv"
HIST_CSV = ML_DIR / "data" / "history_1426.csv"
TEST_CSV = ML_DIR / "data" / "matches_2627.csv"
OUT_CSV = ML_DIR / "results" / "reference_vs_champion_comparison.csv"

FULL_TO_CODE = {
    "Arsenal": "ARS",
    "Aston Villa": "AVL",
    "Bournemouth": "BOU",
    "Brentford": "BRE",
    "Brighton": "BHA",
    "Burnley": "BUR",
    "Chelsea": "CHE",
    "Coventry": "COV",
    "Crystal Palace": "CRY",
    "Everton": "EVE",
    "Fulham": "FUL",
    "Hull": "HUL",
    "Ipswich": "IPS",
    "Leeds": "LEE",
    "Liverpool": "LIV",
    "Manchester City": "MCI",
    "Manchester United": "MUN",
    "Newcastle": "NEW",
    "Nottingham Forest": "NFO",
    "Sunderland": "SUN",
    "Tottenham": "TOT",
    "West Ham": "WHU",
    "Wolves": "WOL",
}


def mean_p_true(y_true: np.ndarray, probs: np.ndarray) -> float:
    p = clip_probs(probs)
    return float(np.mean(p[np.arange(len(y_true)), y_true]))


def match_played_counts(df: pd.DataFrame) -> np.ndarray:
    counts: dict[str, int] = {}
    out = np.empty(len(df), dtype=float)
    for idx, r in enumerate(df.itertuples(index=False)):
        h, a = r.home_team, r.away_team
        out[idx] = float(min(counts.get(h, 0), counts.get(a, 0)))
        counts[h] = counts.get(h, 0) + 1
        counts[a] = counts.get(a, 0) + 1
    return out


def align_by_match_key(df_ref_sub: pd.DataFrame, df_ours: pd.DataFrame) -> tuple[pd.DataFrame, np.ndarray]:
    """Aligns df_ref_sub rows 1-to-1 with df_ours rows by (home_code, away_code, goals)."""
    ref_map = {}
    for idx, r in df_ref_sub.iterrows():
        key = (FULL_TO_CODE[r["home_team"]], FULL_TO_CODE[r["away_team"]], int(r["home_goals"]), int(r["away_goals"]))
        assert key not in ref_map, f"Duplicate key in reference: {key}"
        ref_map[key] = idx

    ours_idx = []
    ref_idx = []
    for o_i, r in df_ours.iterrows():
        key = (r["home_team"], r["away_team"], int(r["home_goals"]), int(r["away_goals"]))
        if key in ref_map:
            ours_idx.append(o_i)
            ref_idx.append(ref_map[key])

    assert len(ref_idx) == len(df_ref_sub), f"Alignment mismatch: matched {len(ref_idx)} vs ref {len(df_ref_sub)}"
    return df_ref_sub.loc[ref_idx].reset_index(drop=True), np.array(ours_idx, dtype=int)


def run_reference_suite(df_feat: pd.DataFrame) -> dict:
    """
    Runs actual reference_model.modeling code under:
      1) Orig protocol: Train=2000-2021, Dev=2022-2023, Refit=<=2023/24 -> Test=2025/26 & 2026/27
      2) Fair cutoff: Refit<=2024/25 -> 2025/26; Refit<=2025/26 -> 2026/27
    Returns predictions dict plus diagnostic audit facts.
    """
    df = df_feat.copy()
    df["season_start"] = df["season"].str.slice(0, 4).astype(int)
    df["date_dt"] = pd.to_datetime(df["date"], errors="coerce")
    df = df.sort_values(["date_dt", "season_start"]).reset_index(drop=True)
    played = df[df["status"].eq("played")].dropna(subset=["result", "home_goals", "away_goals"]).copy()

    train = played[played["season_start"].between(2000, 2021)].copy()
    dev = played[played["season_start"].between(2022, 2023)].copy()
    trdev_orig = pd.concat([train, dev], ignore_index=True)

    tr_fair_25 = played[played["season_start"] <= 2024].copy()
    tr_fair_26 = played[played["season_start"] <= 2025].copy()
    eval_25 = played[played["season"] == "2025/26"].reset_index(drop=True)
    eval_26 = played[played["season"] == "2026/27"].reset_index(drop=True)

    candidates = select_candidates(df)
    safe = leakage_check(train, candidates)
    fs = build_feature_sets(df, safe, [])
    odds_ref = find_odds(df)

    # Audit diagnostics on Reference pipeline
    xg_nan_train = float(train[["diff_xg5", "diff_xga5"]].isna().mean().mean())
    xg_nan_dev = float(dev[["diff_xg5", "diff_xga5"]].isna().mean().mean())

    # Tune Poisson alpha, LR C, HGB, and Blend w strictly on Train (2000-21) -> Dev (2022-23)
    _, best_alpha = tune_alpha(train, dev, fs.poi_feats)
    tr_s, (dev_s,), _, rho_dev = prepare_stage(train, [dev], best_alpha, fs.poi_feats)
    _, C6 = tune_lr(fs.lr_base, "6 cols", tr_s, dev_s, rho_dev)
    _, C7 = tune_lr(fs.lr_base + ["lam_diff"], "6 cols + lam_diff", tr_s, dev_s, rho_dev)
    hgb_feats = list(dict.fromkeys(fs.safe + ["lam_diff"]))
    _, best_hgb = tune_hgb(hgb_feats, tr_s, dev_s, rho_dev)

    # Tune M7 Blend weight on Dev (M3 vs M4 Dixon-Coles)
    p_m3_dev = fit_predict({"kind": "lr", "feats": fs.lr_base + ["lam_diff"], "C": C7}, tr_s, dev_s, rho_dev)[0]
    p_m4_dev = fit_predict({"kind": "dc"}, tr_s, dev_s, rho_dev)[0]
    ws = np.round(np.arange(0, 1.01, 0.05), 2)
    from reference_model.modeling.metrics import ll_vec
    w_best = float(ws[int(np.argmin([ll_vec(w * p_m3_dev + (1 - w) * p_m4_dev, dev_s.result).mean() for w in ws]))])

    specs = {
        "Ref_M0_Prior": {"kind": "prior"},
        "Ref_M3_LR": {"kind": "lr", "feats": fs.lr_base + ["lam_diff"], "C": C7},
        "Ref_M4_DC": {"kind": "dc"},
        "Ref_M6_HGB": {"kind": "hgb", "feats": hgb_feats, "lr": float(best_hgb.lr), "depth": int(best_hgb.depth)},
        "Ref_M7_Blend": {"kind": "blend", "a": {"kind": "lr", "feats": fs.lr_base + ["lam_diff"], "C": C7}, "b": {"kind": "dc"}, "w": w_best},
    }

    # Verify M3 LR coefficients including lam_diff weight and diff_xg5 inside Poisson
    trdev_s_orig, (ev25_orig, ev26_orig), poi_orig, rho_orig = prepare_stage(trdev_orig, [eval_25, eval_26], best_alpha, fs.poi_feats)
    preds_25, preds_26 = {}, {}
    lr_orig_model = None
    for name, sp in specs.items():
        p25, m_fit = fit_predict(sp, trdev_s_orig, ev25_orig, rho_orig)
        p26, _ = fit_predict(sp, trdev_s_orig, ev26_orig, rho_orig)
        preds_25[f"{name}_Orig23"] = clip_probs(p25)
        preds_26[f"{name}_Orig23"] = clip_probs(p26)
        if name == "Ref_M3_LR":
            lr_orig_model = m_fit

    # Fair cutoff: Refit <=2024/25 for 2025/26, Refit <=2025/26 for 2026/27
    tr25_s, (ev25_fair,), _, rho_25 = prepare_stage(tr_fair_25, [eval_25], best_alpha, fs.poi_feats)
    tr26_s, (ev26_fair,), _, rho_26 = prepare_stage(tr_fair_26, [eval_26], best_alpha, fs.poi_feats)
    for name in ("Ref_M3_LR", "Ref_M4_DC", "Ref_M6_HGB", "Ref_M7_Blend"):
        sp = specs[name]
        preds_25[f"{name}_Fair"] = clip_probs(fit_predict(sp, tr25_s, ev25_fair, rho_25)[0])
        preds_26[f"{name}_Fair"] = clip_probs(fit_predict(sp, tr26_s, ev26_fair, rho_26)[0])

    # Inspect Poisson coefficients on trdev_orig
    poi_h_coef = dict(zip(fs.poi_feats, np.round(poi_orig[0].named_steps["poi"].coef_, 4)))
    lr_coefs = pd.DataFrame(lr_orig_model.named_steps["lr"].coef_.T, index=specs["Ref_M3_LR"]["feats"], columns=list(lr_orig_model.classes_))

    # Check Bookmaker Odds column agreement in features.csv
    p_mkt_ref_25 = clip_probs(odds_ref.probs(eval_25)) if odds_ref else None
    p_mkt_ref_26 = clip_probs(odds_ref.probs(eval_26)) if odds_ref else None

    return {
        "eval_25": eval_25,
        "eval_26": eval_26,
        "preds_25": preds_25,
        "preds_26": preds_26,
        "p_mkt_ref_25": p_mkt_ref_25,
        "p_mkt_ref_26": p_mkt_ref_26,
        "diag": {
            "best_alpha": best_alpha,
            "C6": C6,
            "C7": C7,
            "w_blend": w_best,
            "rho_dev": round(float(rho_dev), 4),
            "rho_orig23": round(float(rho_orig), 4),
            "rho_fair25": round(float(rho_26), 4),
            "xg_nan_train_2000_21": round(xg_nan_train, 4),
            "xg_nan_dev_2022_23": round(xg_nan_dev, 4),
            "n_safe_feats": len(fs.safe),
            "poi_h_coef": poi_h_coef,
            "lr_coefs": lr_coefs,
        },
    }


def eval_row(window: str, model: str, y_true: np.ndarray, p_mod: np.ndarray, p_v2: np.ndarray, p_mkt: np.ndarray) -> dict:
    m = evaluate_all(y_true, p_mod)
    b_v2 = bootstrap_rps_diff(y_true, p_mod, p_v2)
    b_mk = bootstrap_rps_diff(y_true, p_mod, p_mkt)
    return {
        "window": window,
        "model": model,
        "n": m["n"],
        "rps": round(m["rps"], 5),
        "log_loss": round(m["log_loss"], 5),
        "brier": round(m["brier"], 5),
        "top1_acc": round(m["accuracy"], 4),
        "mean_p_true": round(mean_p_true(y_true, p_mod), 4),
        "ece": round(m["ece"], 4),
        "diff_vs_v2": round(b_v2["mean_diff"], 5),
        "p_better_v2": round(b_v2["p_better"], 4),
        "diff_vs_mkt": round(b_mk["mean_diff"], 5),
        "p_better_mkt": round(b_mk["p_better"], 4),
    }


def main() -> None:
    t0 = time.time()
    df_feat = pd.read_csv(FEATURES_CSV)
    df_hist = pd.read_csv(HIST_CSV)
    df_2627 = pd.read_csv(TEST_CSV)

    print("=== RUNNING REFERENCE PIPELINE & DIAGNOSTICS ===")
    ref = run_reference_suite(df_feat)

    # Align ours cleanly with reference 25/26 (380) and 26/27 (80)
    df_val = df_hist[df_hist["season"] == "2025/26"].reset_index(drop=True)
    eval_25_aligned, idx_ours_25 = align_by_match_key(ref["eval_25"], df_val)
    eval_26_aligned, idx_ours_26 = align_by_match_key(ref["eval_26"], df_2627)

    # Re-order ref prediction dicts to exact order of eval_25_aligned / eval_26_aligned (already matching ref["eval_25"].index if identical)
    order_25 = ref["eval_25"].merge(eval_25_aligned[["date", "home_team", "away_team"]].reset_index(), on=["date", "home_team", "away_team"], how="right")["index"].to_numpy() if False else None

    # Wait: align_by_match_key preserves df_ours order in idx_ours_*, wait: ref_idx preserves ref rows matching df_ours order!
    # Let's index ref prediction matrices directly via key lookup so order matches idx_ours_25 & idx_ours_26 100% verbatim!
    def reorder_preds(df_ref_orig: pd.DataFrame, df_aligned: pd.DataFrame, preds_dict: dict) -> dict:
        key_to_i = {
            (FULL_TO_CODE[r["home_team"]], FULL_TO_CODE[r["away_team"]], int(r["home_goals"]), int(r["away_goals"])): i
            for i, (_, r) in enumerate(df_ref_orig.iterrows())
        }
        perm = np.array([
            key_to_i[(FULL_TO_CODE[r["home_team"]], FULL_TO_CODE[r["away_team"]], int(r["home_goals"]), int(r["away_goals"]))]
            for _, r in df_aligned.iterrows()
        ], dtype=int)
        return {k: v[perm] for k, v in preds_dict.items()}, perm

    preds_ref_25, perm_25 = reorder_preds(ref["eval_25"], eval_25_aligned, ref["preds_25"])
    preds_ref_26, perm_26 = reorder_preds(ref["eval_26"], eval_26_aligned, ref["preds_26"])

    # Compute Champion v1, Champion v2, Champion v2 BayesOnly walk-forward
    df_train_all = df_hist[df_hist["season"] <= "2024/25"].reset_index(drop=True)
    df_train_ex = df_train_all[df_train_all["season"] != "2020/21"].reset_index(drop=True)
    df_train_r4 = df_train_all[df_train_all["season"] >= "2021/22"].reset_index(drop=True)
    alpha_val, rho_val = estimate_historical_negbin_and_rho(df_train_r4)
    alpha_test, rho_test = estimate_historical_negbin_and_rho(df_hist[df_hist["season"] >= "2021/22"])

    # Champion v2
    elo_v2 = evaluate_fixed_elo(df_train_ex, df_val, k=41.369, hfa=140.0, shrink=0.143, promo_init=1460.0, w_xg=0.45)
    p_elo_v2_25 = elo_v2["val_probs"]
    p_elo_v2_26_all, _ = predict_elo_on_test(df_2627, elo_v2)

    glm_kw = {"prior_scale": 0.35, "half_life_days": 360.0, "w_goals": 0.35, "hfa_prior": 0.22, "hfa_prior_std": 0.14, "elo_priors": None, "promo_offset": 0.0}
    p_glm_25 = walkforward_bayesian_season(df_train_r4, df_val, fit_bayesian_poisson_glm_laplace, glm_kw, True, alpha_val, rho_val, 400, 42)
    p_glm_26_all = walkforward_bayesian_season(df_hist[df_hist["season"] >= "2021/22"], df_2627, fit_bayesian_poisson_glm_laplace, glm_kw, True, alpha_test, rho_test, 400, 42)

    p_v2_25 = clip_probs(0.60 * p_elo_v2_25 + 0.40 * p_glm_25)
    p_v2_26_all = clip_probs(0.60 * p_elo_v2_26_all + 0.40 * p_glm_26_all)

    # Champion v1 Legacy
    elo_v1 = evaluate_fixed_elo(df_train_r4, df_val, k=20.0, hfa=65.0, shrink=0.35, promo_init=1420.0, w_xg=0.0)
    _, end_v1_tr, last_v1_tr = run_elo_walkforward(df_train_r4, k=20.0, hfa=65.0, shrink=0.35, promo_init=1420.0, w_xg=0.0)
    teams_val = sorted(set(df_val["home_team"]).union(df_val["away_team"]))
    promo_val = set(teams_val) - set(df_train_r4[df_train_r4["season"] == "2024/25"]["home_team"]).union(df_train_r4[df_train_r4["season"] == "2024/25"]["away_team"])
    priors_v1_25 = prepare_next_season_ratings(end_v1_tr, last_v1_tr, teams_val, 0.35, 1420.0)
    p_v1_bayes_25 = walkforward_bayesian_season(
        pd.DataFrame(), df_val, fit_bayesian_ridge_logclip,
        {"ridge_alpha": 18.0, "half_life_days": 90.0, "w_goals": 0.35, "hfa_prior": 0.22, "elo_priors": priors_v1_25, "promo_offset": 0.10},
        False, alpha_val, rho_val, 400, 42, promo_val,
    )
    w_v1_25 = np.exp(-match_played_counts(df_val) / 25.0)[:, None]
    p_v1_25 = clip_probs(w_v1_25 * elo_v1["val_probs"] + (1.0 - w_v1_25) * p_v1_bayes_25)

    teams_test = sorted(set(df_2627["home_team"]).union(df_2627["away_team"]))
    promo_test = set(teams_test) - set(df_val["home_team"]).union(df_val["away_team"])
    priors_v1_26 = prepare_next_season_ratings(elo_v1["end_ratings_tv"], elo_v1["last_teams_tv"], teams_test, 0.35, 1420.0)
    p_v1_elo_26_all, _ = predict_elo_on_test(df_2627, elo_v1)
    p_v1_bayes_26_all = walkforward_bayesian_season(
        pd.DataFrame(), df_2627, fit_bayesian_ridge_logclip,
        {"ridge_alpha": 18.0, "half_life_days": 90.0, "w_goals": 0.35, "hfa_prior": 0.22, "elo_priors": priors_v1_26, "promo_offset": 0.10},
        False, alpha_test, rho_test, 400, 42, promo_test,
    )
    w_v1_26_all = np.exp(-match_played_counts(df_2627) / 25.0)[:, None]
    p_v1_26_all = clip_probs(w_v1_26_all * p_v1_elo_26_all + (1.0 - w_v1_26_all) * p_v1_bayes_26_all)

    # Slice ours to exact aligned indices (idx_ours_25: 380, idx_ours_26: 80)
    y_25 = df_val["target"].to_numpy(dtype=int)[idx_ours_25]
    y_26 = df_2627["target"].to_numpy(dtype=int)[idx_ours_26]
    p_mkt_25 = clip_probs(df_val[["odds_prob_home", "odds_prob_draw", "odds_prob_away"]].to_numpy(dtype=float)[idx_ours_25])
    p_mkt_26 = clip_probs(df_2627[["odds_prob_home", "odds_prob_draw", "odds_prob_away"]].to_numpy(dtype=float)[idx_ours_26])

    # Verify Market Odds parity between features.csv and our DB
    mkt_diff_25 = float(np.nanmean(np.abs(p_mkt_25 - ref["p_mkt_ref_25"][perm_25])))
    mkt_diff_26 = float(np.nanmean(np.abs(p_mkt_26 - ref["p_mkt_ref_26"][perm_26])))
    print(f"\n[+] Audit Market Odds MAE (features.csv vs epl_clean.db): 25/26={mkt_diff_25:.5f}, 26/27={mkt_diff_26:.5f}")
    print(f"[+] Reference Diagnostics: { {k: v for k, v in ref['diag'].items() if k not in ('poi_h_coef', 'lr_coefs')} }")
    print(f"[+] Reference Poisson Home Coefs: {ref['diag']['poi_h_coef']}")
    print(f"[+] Reference M3 LR Coefs:\n{ref['diag']['lr_coefs'].round(4)}")

    models_25 = [
        ("Ref_M0_Prior_Orig23", preds_ref_25["Ref_M0_Prior_Orig23"]),
        ("Ref_M3_LR_Orig23", preds_ref_25["Ref_M3_LR_Orig23"]),
        ("Ref_M4_DC_Orig23", preds_ref_25["Ref_M4_DC_Orig23"]),
        ("Ref_M6_HGB_Orig23", preds_ref_25["Ref_M6_HGB_Orig23"]),
        ("Ref_M7_Blend_Orig23", preds_ref_25["Ref_M7_Blend_Orig23"]),
        ("Ref_M3_LR_Fair24", preds_ref_25["Ref_M3_LR_Fair"]),
        ("Ref_M4_DC_Fair24", preds_ref_25["Ref_M4_DC_Fair"]),
        ("Ref_M6_HGB_Fair24", preds_ref_25["Ref_M6_HGB_Fair"]),
        ("Ref_M7_Blend_Fair24", preds_ref_25["Ref_M7_Blend_Fair"]),
        ("Champion_v1_Legacy", p_v1_25[idx_ours_25]),
        ("Champion_v2_BayesOnly", p_glm_25[idx_ours_25]),
        ("Champion_v2_New", p_v2_25[idx_ours_25]),
        ("Market_Odds", p_mkt_25),
    ]

    models_26 = [
        ("Ref_M0_Prior_Orig23", preds_ref_26["Ref_M0_Prior_Orig23"]),
        ("Ref_M3_LR_Orig23", preds_ref_26["Ref_M3_LR_Orig23"]),
        ("Ref_M4_DC_Orig23", preds_ref_26["Ref_M4_DC_Orig23"]),
        ("Ref_M6_HGB_Orig23", preds_ref_26["Ref_M6_HGB_Orig23"]),
        ("Ref_M7_Blend_Orig23", preds_ref_26["Ref_M7_Blend_Orig23"]),
        ("Ref_M3_LR_Fair25", preds_ref_26["Ref_M3_LR_Fair"]),
        ("Ref_M4_DC_Fair25", preds_ref_26["Ref_M4_DC_Fair"]),
        ("Ref_M6_HGB_Fair25", preds_ref_26["Ref_M6_HGB_Fair"]),
        ("Ref_M7_Blend_Fair25", preds_ref_26["Ref_M7_Blend_Fair"]),
        ("Champion_v1_Legacy", p_v1_26_all[idx_ours_26]),
        ("Champion_v2_BayesOnly", p_glm_26_all[idx_ours_26]),
        ("Champion_v2_New", p_v2_26_all[idx_ours_26]),
        ("Market_Odds", p_mkt_26),
    ]

    rows = []
    p_v2_al_25 = p_v2_25[idx_ours_25]
    p_v2_al_26 = p_v2_26_all[idx_ours_26]
    for name, p in models_25:
        rows.append(eval_row("Aligned_2025_26 (N=380)", name, y_25, p, p_v2_al_25, p_mkt_25))
    for name, p in models_26:
        rows.append(eval_row("Aligned_2026_27 (N=80)", name, y_26, p, p_v2_al_26, p_mkt_26))

    df_out = pd.DataFrame(rows)
    df_out.to_csv(OUT_CSV, index=False)

    for win in ("Aligned_2025_26 (N=380)", "Aligned_2026_27 (N=80)"):
        print(f"\n=== {win} ===")
        sub = df_out[df_out["window"] == win][["model", "rps", "log_loss", "brier", "top1_acc", "mean_p_true", "ece", "diff_vs_v2", "p_better_v2", "diff_vs_mkt", "p_better_mkt"]]
        print(sub.to_string(index=False))

    print(f"\n[+] Saved {OUT_CSV} in {time.time() - t0:.1f}s")


if __name__ == "__main__":
    main()
