"""
ml_epl2627/experiments/run_strict_validation.py
Strict 3-way temporal validation addressing Elo hardcoding, Bayes distribution/noise model,
multi-season history depth + distribution shift, and independent vs double-counted fusion:
  - Train Pool       : <= 2024/25 (Recent4 2021-25 vs ExCOVID 2014-25 vs InclCOVID 2014-25)
  - Validation Holdout: 2025/26 (N=380) — ALL hyperparameter & fusion decisions made strictly here
  - Locked Test      : 2026/27 (N=110 all, N=80 post-MD3) — ZERO re-tuning allowed
"""

import json
import math
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.append(str(ROOT))

from src.metrics import evaluate_all, bootstrap_rps_diff, rps_per_match, clip_probs  # noqa: E402
from src.bayesian_models import (  # noqa: E402
    estimate_historical_negbin_and_rho,
    predict_bayesian_posterior,
)
from src.strict_models import (  # noqa: E402
    evaluate_fixed_elo,
    fit_bayesian_poisson_glm_laplace,
    fit_bayesian_ridge_logclip,
    predict_elo_on_test,
    prepare_next_season_ratings,
    tune_elo_strict,
    walkforward_bayesian_season,
)

HIST_CSV = ROOT / "data" / "history_1426.csv"
TEST_CSV = ROOT / "data" / "matches_2627.csv"
OUT_DIR = ROOT / "results"


def match_played_counts(df: pd.DataFrame) -> np.ndarray:
    """Returns min(played_before_match(home), played_before_match(away)) causally."""
    counts: dict[str, int] = {}
    out = np.empty(len(df), dtype=float)
    for idx, r in enumerate(df.itertuples(index=False)):
        h, a = r.home_team, r.away_team
        out[idx] = float(min(counts.get(h, 0), counts.get(a, 0)))
        counts[h] = counts.get(h, 0) + 1
        counts[a] = counts.get(a, 0) + 1
    return out


def get_promoted_teams(df_prev_season: pd.DataFrame, df_target: pd.DataFrame) -> set[str]:
    prev_teams = set(df_prev_season["home_team"]).union(df_prev_season["away_team"])
    cur_teams = set(df_target["home_team"]).union(df_target["away_team"])
    return cur_teams - prev_teams


def walkforward_champion_legacy(
    df_prior_source: pd.DataFrame,
    df_target: pd.DataFrame,
    elo_fit: dict,
    use_val_ratings: bool,
    alpha_nb: float,
    rho: float,
) -> np.ndarray:
    """
    Reproduces Legacy Champion architecture on any target season:
      1) Handcrafted Elo_init -> prior for Bayesian Ridge (log-clipped, current season only)
      2) Hardcoded exp(-n_played / 25.0) blend between Elo and BayesRidge(EloPrior)
    """
    teams_target = sorted(set(df_target["home_team"]).union(df_target["away_team"]))
    promoted = get_promoted_teams(df_prior_source, df_target)
    if use_val_ratings:
        elo_priors = prepare_next_season_ratings(
            elo_fit["end_ratings_tv"],
            elo_fit["last_teams_tv"],
            teams_target,
            elo_fit["shrink"],
            elo_fit["promo_init"],
        )
        p_elo, _ = predict_elo_on_test(df_target, elo_fit)
    else:
        # Evaluate on Validation 2025/26 using Train-only end ratings
        from src.strict_models import run_elo_walkforward, ordered_logit_probs
        init_val = prepare_next_season_ratings(
            elo_fit["end_ratings_train"],
            elo_fit["last_teams_train"],
            teams_target,
            elo_fit["shrink"],
            elo_fit["promo_init"],
        )
        elo_priors = init_val
        diffs_val, _, _ = run_elo_walkforward(
            df_target,
            k=elo_fit["k"],
            hfa=elo_fit["hfa"],
            shrink=elo_fit["shrink"],
            promo_init=elo_fit["promo_init"],
            w_xg=0.0,
            initial_ratings=init_val,
            prev_season_teams=elo_fit["last_teams_train"],
        )
        c1, c2, beta = elo_fit["head_train_only"]
        p_elo = clip_probs(ordered_logit_probs(diffs_val, c1, c2, beta))

    p_bayes = walkforward_bayesian_season(
        df_hist_before=pd.DataFrame(),
        df_target=df_target,
        fit_fn=fit_bayesian_ridge_logclip,
        fit_kwargs={
            "ridge_alpha": 18.0,
            "half_life_days": 90.0,
            "w_goals": 0.35,
            "hfa_prior": 0.22,
            "elo_priors": elo_priors,
            "promo_offset": 0.10,
        },
        use_multi_season_history=False,
        alpha_nb=alpha_nb,
        rho=rho,
        n_draws=250,
        promoted_teams=promoted,
    )
    n_min = match_played_counts(df_target)
    w_elo = np.exp(-n_min / 25.0)[:, None]
    return clip_probs(w_elo * p_elo + (1.0 - w_elo) * p_bayes)


def main() -> None:
    t0 = time.time()
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    df_hist = pd.read_csv(HIST_CSV)
    df_2627 = pd.read_csv(TEST_CSV)

    df_val = df_hist[df_hist["season"] == "2025/26"].reset_index(drop=True)
    df_train_all = df_hist[df_hist["season"] <= "2024/25"].reset_index(drop=True)

    train_splits = {
        "Recent4_2125": df_train_all[df_train_all["season"] >= "2021/22"].reset_index(drop=True),
        "Full10_ExCOVID": df_train_all[df_train_all["season"] != "2020/21"].reset_index(drop=True),
        "Full11_InclCOVID": df_train_all.copy(),
    }

    y_val = df_val["target"].to_numpy(dtype=int)
    y_test = df_2627["target"].to_numpy(dtype=int)
    post_md3_mask = np.arange(len(df_2627)) >= 30
    y_test_md3 = y_test[post_md3_mask]

    n_val = match_played_counts(df_val)
    n_test = match_played_counts(df_2627)

    promo_val = get_promoted_teams(df_train_all[df_train_all["season"] == "2024/25"], df_val)
    promo_test = get_promoted_teams(df_val, df_2627)

    # Leak-free NegBin dispersion & Dixon-Coles rho:
    # For Val (2025/26), fit only on <= 2024/25; for Test (2026/27), fit on <= 2025/26.
    alpha_val, rho_val = estimate_historical_negbin_and_rho(train_splits["Recent4_2125"])
    alpha_test, rho_test = estimate_historical_negbin_and_rho(df_hist[df_hist["season"] >= "2021/22"])
    print(f"[+] Dispersion (Train<=24/25): alpha_nb={alpha_val:.4f}, rho={rho_val:.4f}")
    print(f"[+] Dispersion (Train+Val<=25/26): alpha_nb={alpha_test:.4f}, rho={rho_test:.4f}")

    # =====================================================================
    # STEP 1: ELO VALIDATION EXPERIMENT (Handcrafted vs Tuned, Depth & COVID)
    # =====================================================================
    print("\n=== STEP 1: ELO EXPERIMENTS (Train<=24/25 -> Val 25/26) ===")
    elo_configs = {}
    elo_val_rows = []

    for split_name, df_tr in train_splits.items():
        # Handcrafted Elo
        eh = evaluate_fixed_elo(df_tr, df_val, k=20.0, hfa=65.0, shrink=0.35, promo_init=1420.0, w_xg=0.0)
        # Also save train-only end ratings for legacy Champion Val run
        from src.strict_models import run_elo_walkforward
        _, end_r_tr, last_tr = run_elo_walkforward(df_tr, k=20.0, hfa=65.0, shrink=0.35, promo_init=1420.0, w_xg=0.0)
        eh["end_ratings_train"] = end_r_tr
        eh["last_teams_train"] = last_tr

        key_h = f"Elo_Hand_{split_name}"
        elo_configs[key_h] = eh
        m_h = evaluate_all(y_val, eh["val_probs"])
        elo_val_rows.append({"model": key_h, "split": split_name, "params": "K=20,HFA=65,sh=0.35,pr=1420,wxg=0", **m_h})

        # Validated Goals-only Elo (for Recent4 and ExCOVID)
        if split_name != "Full11_InclCOVID":
            et_g = tune_elo_strict(df_tr, df_val, allow_xg=False)
            key_g = f"Elo_ValTuned_Goals_{split_name}"
            elo_configs[key_g] = et_g
            m_g = evaluate_all(y_val, et_g["val_probs"])
            elo_val_rows.append({"model": key_g, "split": split_name, "params": f"K={et_g['k']},HFA={et_g['hfa']},sh={et_g['shrink']},pr={et_g['promo_init']},wxg=0", **m_g})

            et_x = tune_elo_strict(df_tr, df_val, allow_xg=True)
            key_x = f"Elo_ValTuned_xG_{split_name}"
            elo_configs[key_x] = et_x
            m_x = evaluate_all(y_val, et_x["val_probs"])
            elo_val_rows.append({"model": key_x, "split": split_name, "params": f"K={et_x['k']},HFA={et_x['hfa']},sh={et_x['shrink']},pr={et_x['promo_init']},wxg={et_x['w_xg']}", **m_x})

    # Historical (<=2023/24 train -> 2024/25 early-stop 1Y) Elo evaluated out-of-sample on Val 2025/26
    df_ex = train_splits["Full10_ExCOVID"]
    df_ex_sub_tr = df_ex[df_ex["season"] <= "2023/24"].reset_index(drop=True)
    df_ex_sub_es = df_ex[df_ex["season"] == "2024/25"].reset_index(drop=True)
    hist_tuned = tune_elo_strict(df_ex_sub_tr, df_ex_sub_es, allow_xg=True)
    eh_hist = evaluate_fixed_elo(
        df_ex, df_val,
        k=hist_tuned["k"], hfa=hist_tuned["hfa"], shrink=hist_tuned["shrink"],
        promo_init=hist_tuned["promo_init"], w_xg=hist_tuned["w_xg"],
    )
    elo_configs["Elo_HistN1_ExCOVID"] = eh_hist
    m_eh = evaluate_all(y_val, eh_hist["val_probs"])
    elo_val_rows.append({
        "model": "Elo_HistN1_ExCOVID",
        "split": "Full10_ExCOVID",
        "params": f"K={hist_tuned['k']},HFA={hist_tuned['hfa']},sh={hist_tuned['shrink']},pr={hist_tuned['promo_init']},wxg={hist_tuned['w_xg']}",
        **m_eh,
    })

    # Multi-year historical Elo tuning (<=2021/22 train -> 2022/23..2024/25 = 3 seasons / 9 promoted clubs)
    df_ex_3y_tr = df_ex[df_ex["season"] <= "2021/22"].reset_index(drop=True)
    df_ex_3y_es = df_ex[df_ex["season"] >= "2022/23"].reset_index(drop=True)
    hist3_tuned = tune_elo_strict(df_ex_3y_tr, df_ex_3y_es, allow_xg=True)
    eh_hist3 = evaluate_fixed_elo(
        df_ex, df_val,
        k=hist3_tuned["k"], hfa=hist3_tuned["hfa"], shrink=hist3_tuned["shrink"],
        promo_init=hist3_tuned["promo_init"], w_xg=hist3_tuned["w_xg"],
    )
    elo_configs["Elo_Hist3Y_ExCOVID"] = eh_hist3
    m_eh3 = evaluate_all(y_val, eh_hist3["val_probs"])
    elo_val_rows.append({
        "model": "Elo_Hist3Y_ExCOVID",
        "split": "Full10_ExCOVID",
        "params": f"K={hist3_tuned['k']},HFA={hist3_tuned['hfa']},sh={hist3_tuned['shrink']},pr={hist3_tuned['promo_init']},wxg={hist3_tuned['w_xg']}",
        **m_eh3,
    })

    df_elo_val = pd.DataFrame(elo_val_rows).sort_values("rps").reset_index(drop=True)
    print(df_elo_val[["model", "params", "rps", "log_loss", "brier", "accuracy", "ece"]].to_string(index=False))
    best_elo_key = str(df_elo_val.iloc[0]["model"])
    best_elo_fit = elo_configs[best_elo_key]
    print(f"[*] Best Elo selected strictly on Val 2025/26: {best_elo_key} (RPS={df_elo_val.iloc[0]['rps']:.5f})")

    # =====================================================================
    # STEP 2: BAYESIAN ABLATION (Ridge LogClip vs Poisson GLM Laplace vs Multi-Season)
    # =====================================================================
    print("\n=== STEP 2: BAYESIAN ABLATIONS (Zero-Elo Pure Branches on Val 25/26) ===")
    bayes_specs = []

    # 2A: BayesRidge_SeasonFlat (tune ridge_alpha, half_life, w_goals)
    best_ridge = {"val_rps": 999.0}
    for ra in (12.0, 18.0, 28.0):
        for hl in (90.0, 150.0):
            for wg in (0.25, 0.35):
                kw = {"ridge_alpha": ra, "half_life_days": hl, "w_goals": wg, "hfa_prior": 0.22, "elo_priors": None, "promo_offset": 0.12}
                vp = walkforward_bayesian_season(pd.DataFrame(), df_val, fit_bayesian_ridge_logclip, kw, False, alpha_val, rho_val, 120, 42, promo_val)
                vrps = float(np.mean(rps_per_match(y_val, vp)))
                if vrps < best_ridge["val_rps"]:
                    best_ridge = {"val_rps": vrps, "kw": kw, "val_probs": vp}
    best_ridge["val_probs"] = walkforward_bayesian_season(pd.DataFrame(), df_val, fit_bayesian_ridge_logclip, best_ridge["kw"], False, alpha_val, rho_val, 400, 42, promo_val)
    bayes_specs.append(("BayesRidge_SeasonFlat", fit_bayesian_ridge_logclip, False, None, best_ridge))

    # 2B: BayesGLM_SeasonFlat (true Poisson GLM Laplace, zero log clipping, zero Elo)
    best_glm_s = {"val_rps": 999.0}
    for ps in (0.22, 0.30, 0.42):
        for hl in (90.0, 180.0):
            for wg in (0.25, 0.35):
                kw = {"prior_scale": ps, "half_life_days": hl, "w_goals": wg, "hfa_prior": 0.22, "hfa_prior_std": 0.14, "elo_priors": None, "promo_offset": 0.12}
                vp = walkforward_bayesian_season(pd.DataFrame(), df_val, fit_bayesian_poisson_glm_laplace, kw, False, alpha_val, rho_val, 120, 42, promo_val)
                vrps = float(np.mean(rps_per_match(y_val, vp)))
                if vrps < best_glm_s["val_rps"]:
                    best_glm_s = {"val_rps": vrps, "kw": kw, "val_probs": vp}
    best_glm_s["val_probs"] = walkforward_bayesian_season(pd.DataFrame(), df_val, fit_bayesian_poisson_glm_laplace, best_glm_s["kw"], False, alpha_val, rho_val, 400, 42, promo_val)
    bayes_specs.append(("BayesGLM_SeasonFlat", fit_bayesian_poisson_glm_laplace, False, None, best_glm_s))

    # 2C: BayesGLM_MultiSeason_Recent4 (2021-25 direct time-decayed history, zero Elo)
    best_glm_r4 = {"val_rps": 999.0}
    for ps in (0.25, 0.35):
        for hl in (150.0, 240.0, 360.0):
            for wg in (0.25, 0.35):
                for po in (0.0, 0.12):
                    kw = {"prior_scale": ps, "half_life_days": hl, "w_goals": wg, "hfa_prior": 0.22, "hfa_prior_std": 0.14, "elo_priors": None, "promo_offset": po}
                    vp = walkforward_bayesian_season(train_splits["Recent4_2125"], df_val, fit_bayesian_poisson_glm_laplace, kw, True, alpha_val, rho_val, 120, 42, promo_val)
                    vrps = float(np.mean(rps_per_match(y_val, vp)))
                    if vrps < best_glm_r4["val_rps"]:
                        best_glm_r4 = {"val_rps": vrps, "kw": kw, "val_probs": vp}
    best_glm_r4["val_probs"] = walkforward_bayesian_season(train_splits["Recent4_2125"], df_val, fit_bayesian_poisson_glm_laplace, best_glm_r4["kw"], True, alpha_val, rho_val, 400, 42, promo_val)
    bayes_specs.append(("BayesGLM_MultiSeason_Recent4", fit_bayesian_poisson_glm_laplace, True, "Recent4", best_glm_r4))

    # 2D: BayesGLM_MultiSeason_ExCOVID (2014-25 ex-COVID direct time-decayed history, zero Elo)
    best_glm_ex = {"val_rps": 999.0}
    for ps in (0.25, 0.35):
        for hl in (180.0, 300.0, 480.0):
            for wg in (0.25, 0.35):
                for po in (0.0, 0.12):
                    kw = {"prior_scale": ps, "half_life_days": hl, "w_goals": wg, "hfa_prior": 0.22, "hfa_prior_std": 0.14, "elo_priors": None, "promo_offset": po}
                    vp = walkforward_bayesian_season(train_splits["Full10_ExCOVID"], df_val, fit_bayesian_poisson_glm_laplace, kw, True, alpha_val, rho_val, 120, 42, promo_val)
                    vrps = float(np.mean(rps_per_match(y_val, vp)))
                    if vrps < best_glm_ex["val_rps"]:
                        best_glm_ex = {"val_rps": vrps, "kw": kw, "val_probs": vp}
    best_glm_ex["val_probs"] = walkforward_bayesian_season(train_splits["Full10_ExCOVID"], df_val, fit_bayesian_poisson_glm_laplace, best_glm_ex["kw"], True, alpha_val, rho_val, 400, 42, promo_val)
    bayes_specs.append(("BayesGLM_MultiSeason_ExCOVID", fit_bayesian_poisson_glm_laplace, True, "ExCOVID", best_glm_ex))

    bayes_val_rows = []
    for name, _, _, _, info in bayes_specs:
        m_b = evaluate_all(y_val, info["val_probs"])
        bayes_val_rows.append({"model": name, "kw": json.dumps(info["kw"]), **m_b})
    df_bayes_val = pd.DataFrame(bayes_val_rows).sort_values("rps").reset_index(drop=True)
    print(df_bayes_val[["model", "rps", "log_loss", "brier", "accuracy", "ece", "kw"]].to_string(index=False))

    best_bayes_name = str(df_bayes_val.iloc[0]["model"])
    best_bayes_tuple = [t for t in bayes_specs if t[0] == best_bayes_name][0]
    print(f"[*] Best Pure Bayesian branch selected strictly on Val 2025/26: {best_bayes_name} (RPS={df_bayes_val.iloc[0]['rps']:.5f})")

    # =====================================================================
    # STEP 3: FUSION & FINAL LOCKED TEST EVALUATION (2026/27 N=110 & N=80)
    # =====================================================================
    print("\n=== STEP 3: STACKING / FUSION TUNING ON VAL 2025/26 ===")
    # A) Clean Independent Stack: Best_Elo + BayesGLM_SeasonFlat with validated exp(-n / tau)
    best_tau, best_stack_rps, best_stack_val_p = 25.0, 999.0, None
    for tau in (8.0, 15.0, 25.0, 40.0, 65.0, 100.0, 1e9):
        w = np.exp(-n_val / tau)[:, None]
        p_st = clip_probs(w * best_elo_fit["val_probs"] + (1.0 - w) * best_glm_s["val_probs"])
        r = float(np.mean(rps_per_match(y_val, p_st)))
        if r < best_stack_rps:
            best_stack_rps, best_tau, best_stack_val_p = r, tau, p_st

    # B) Static Stack weight between Best_Elo and Best_MultiSeason_BayesGLM
    best_w_static, best_st2_rps, best_st2_val_p = 0.0, 999.0, None
    best_multi_val_p = best_bayes_tuple[4]["val_probs"]
    for w_s in np.linspace(0.0, 1.0, 21):
        p_st = clip_probs(w_s * best_elo_fit["val_probs"] + (1.0 - w_s) * best_multi_val_p)
        r = float(np.mean(rps_per_match(y_val, p_st)))
        if r < best_st2_rps:
            best_st2_rps, best_w_static, best_st2_val_p = r, float(w_s), p_st

    print(f"[*] Independent Dynamic Stack (BestElo + SeasonFlat GLM): best tau={best_tau:.1f} -> Val RPS={best_stack_rps:.5f}")
    print(f"[*] Static Pool (w*BestElo + (1-w)*{best_bayes_name}): best w_elo={best_w_static:.2f} -> Val RPS={best_st2_rps:.5f}")

    # Now freeze everything and evaluate ALL models on Locked Test 2026/27 (N=110 & Post-MD3 N=80)
    p_elo_test, _ = predict_elo_on_test(df_2627, best_elo_fit)
    p_elo_histn1_test, _ = predict_elo_on_test(df_2627, elo_configs["Elo_HistN1_ExCOVID"])
    p_elo_hist3y_test, _ = predict_elo_on_test(df_2627, elo_configs["Elo_Hist3Y_ExCOVID"])
    p_elo_hand_test, _ = predict_elo_on_test(df_2627, elo_configs["Elo_Hand_Recent4_2125"])

    # Legacy Champion (double-counted prior + hardcoded exp(-n/25)) on both Val and Test
    p_champ_val = walkforward_champion_legacy(df_train_all[df_train_all["season"] == "2024/25"], df_val, elo_configs["Elo_Hand_Recent4_2125"], False, alpha_val, rho_val)
    p_champ_test = walkforward_champion_legacy(df_val, df_2627, elo_configs["Elo_Hand_Recent4_2125"], True, alpha_test, rho_test)

    # Pure Bayes candidates on Test 2026/27 (higher MC draws=400 for clean evaluation)
    p_ridge_s_test = walkforward_bayesian_season(pd.DataFrame(), df_2627, fit_bayesian_ridge_logclip, best_ridge["kw"], False, alpha_test, rho_test, 400, 42, promo_test)
    p_glm_s_test = walkforward_bayesian_season(pd.DataFrame(), df_2627, fit_bayesian_poisson_glm_laplace, best_glm_s["kw"], False, alpha_test, rho_test, 400, 42, promo_test)
    p_glm_r4_test = walkforward_bayesian_season(df_hist[df_hist["season"] >= "2021/22"], df_2627, fit_bayesian_poisson_glm_laplace, best_glm_r4["kw"], True, alpha_test, rho_test, 400, 42, promo_test)
    p_glm_ex_test = walkforward_bayesian_season(df_hist[df_hist["season"] != "2020/21"], df_2627, fit_bayesian_poisson_glm_laplace, best_glm_ex["kw"], True, alpha_test, rho_test, 400, 42, promo_test)

    best_multi_test_p = {"BayesGLM_MultiSeason_Recent4": p_glm_r4_test, "BayesGLM_MultiSeason_ExCOVID": p_glm_ex_test}.get(best_bayes_name, p_glm_s_test)

    w_dyn_test = np.exp(-n_test / best_tau)[:, None]
    p_stack_dyn_test = clip_probs(w_dyn_test * p_elo_test + (1.0 - w_dyn_test) * p_glm_s_test)
    p_stack_static_test = clip_probs(best_w_static * p_elo_test + (1.0 - best_w_static) * best_multi_test_p)

    p_mkt_val = clip_probs(df_val[["odds_prob_home", "odds_prob_draw", "odds_prob_away"]].to_numpy(dtype=float))
    p_mkt_test = clip_probs(df_2627[["odds_prob_home", "odds_prob_draw", "odds_prob_away"]].to_numpy(dtype=float))

    final_models = [
        ("Champion_Legacy", p_champ_val, p_champ_test),
        ("Elo_Hand_Recent4", elo_configs["Elo_Hand_Recent4_2125"]["val_probs"], p_elo_hand_test),
        ("Elo_HistN1_ExCOVID", elo_configs["Elo_HistN1_ExCOVID"]["val_probs"], p_elo_histn1_test),
        ("Elo_Hist3Y_ExCOVID", elo_configs["Elo_Hist3Y_ExCOVID"]["val_probs"], p_elo_hist3y_test),
        (f"Best_Elo ({best_elo_key})", best_elo_fit["val_probs"], p_elo_test),
        ("BayesRidge_SeasonFlat", best_ridge["val_probs"], p_ridge_s_test),
        ("BayesGLM_SeasonFlat", best_glm_s["val_probs"], p_glm_s_test),
        ("BayesGLM_MultiRecent4", best_glm_r4["val_probs"], p_glm_r4_test),
        ("BayesGLM_MultiExCOVID", best_glm_ex["val_probs"], p_glm_ex_test),
        (f"Stack_IndepDyn (tau={best_tau:.0f})", best_stack_val_p, p_stack_dyn_test),
        (f"Stack_Multi (w_elo={best_w_static:.2f})", best_st2_val_p, p_stack_static_test),
        ("Market_Odds", p_mkt_val, p_mkt_test),
    ]

    summary_rows = []
    for name, pv, pt in final_models:
        mv = evaluate_all(y_val, pv)
        mt = evaluate_all(y_test, pt)
        mt3 = evaluate_all(y_test_md3, pt[post_md3_mask])
        boot_vs_champ_t = bootstrap_rps_diff(y_test, pt, p_champ_test)
        boot_vs_mkt_t = bootstrap_rps_diff(y_test, pt, p_mkt_test)
        summary_rows.append(
            {
                "model": name,
                "val2526_rps": mv["rps"],
                "val2526_ll": mv["log_loss"],
                "val2526_acc": mv["accuracy"],
                "test2627_rps": mt["rps"],
                "test2627_ll": mt["log_loss"],
                "test2627_acc": mt["accuracy"],
                "test2627_ece": mt["ece"],
                "test_md3_rps": mt3["rps"],
                "test_md3_ll": mt3["log_loss"],
                "test_md3_acc": mt3["accuracy"],
                "diff_vs_champ": boot_vs_champ_t["mean_diff"],
                "p_better_vs_champ": boot_vs_champ_t["p_better"],
                "diff_vs_mkt": boot_vs_mkt_t["mean_diff"],
                "p_better_vs_mkt": boot_vs_mkt_t["p_better"],
            }
        )

    df_summary = pd.DataFrame(summary_rows)
    df_elo_val.to_csv(OUT_DIR / "strict_elo_validation.csv", index=False)
    df_bayes_val.to_csv(OUT_DIR / "strict_bayes_validation.csv", index=False)
    df_summary.to_csv(OUT_DIR / "strict_validation_summary.csv", index=False)

    meta = {
        "best_elo_key": best_elo_key,
        "best_elo_params": {k: best_elo_fit[k] for k in ("k", "hfa", "shrink", "promo_init", "w_xg", "val_rps")},
        "best_ridge_kw": best_ridge["kw"],
        "best_glm_season_kw": best_glm_s["kw"],
        "best_glm_recent4_kw": best_glm_r4["kw"],
        "best_glm_excovid_kw": best_glm_ex["kw"],
        "best_bayes_name": best_bayes_name,
        "stack_dyn_tau": best_tau,
        "stack_static_w_elo": best_w_static,
    }
    (OUT_DIR / "strict_validation_meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")

    print("\n=== FINAL 3-WAY BENCHMARK SUMMARY (Val 2025/26 -> Locked Test 2026/27) ===")
    cols = ["model", "val2526_rps", "val2526_ll", "test2627_rps", "test2627_ll", "test_md3_rps", "diff_vs_champ", "p_better_vs_champ"]
    print(df_summary[cols].to_string(index=False))
    print(f"\n[+] Completed in {time.time() - t0:.1f}s. Saved to {OUT_DIR}")


if __name__ == "__main__":
    main()
