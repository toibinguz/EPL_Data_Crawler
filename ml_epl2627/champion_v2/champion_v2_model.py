"""
ml_epl2627/champion_v2/champion_v2_model.py
Production Champion v2 Model (`EPLChampionV2`) for EPL 2026/27.

Architecture (Synthesized & Independently Validated on 2025/26 Holdout):
  1. Data Hygiene:
     - Elo runs on 2014/15-2025/26 excluding the 2020/21 COVID closed-doors anomaly
       (home_adv=+0.011 vs normal +0.21..+0.42).
     - Bayesian Poisson GLM trains directly on recent multi-season matches (2021/22+)
       with exponential time decay (half-life = 360 days).
  2. Branch 1 — Stabilized xG-Elo:
     - Updates ratings with continuous match performance (55% goal outcome + 45% xG sigmoid).
     - Hyperparameters (K, HFA, shrink, promo_init bounded in [1400, 1460]) tuned strictly
       on Validation 2025/26 to prevent single-season overfitting on 3 promoted clubs.
     - Ordered Logistic calibration head fit strictly on historical seasons (excluding warmup).
  3. Branch 2 — Pure Multi-Season Bayesian Poisson GLM (Exact Laplace Approximation):
     - Completely independent of Elo (flat zero-centered prior, zero double-counting).
     - Damped Newton-Raphson MAP + exact Hessian inverse Sigma = H(theta_MAP)^(-1)
       (eliminates ad-hoc log(max(0.25, perf)) clipping).
     - 400-draw Monte Carlo integration over 2D attack/defense epistemic uncertainty
       through a Dixon-Coles adjusted Poisson score matrix.
  4. Independent Fusion:
     - Static probability pool: w_elo * P_Elo + (1 - w_elo) * P_BayesGLM,
       with w_elo frozen strictly from Validation 2025/26.
"""

import math
import sys
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.stats import poisson

ML_DIR = Path(__file__).resolve().parent.parent
if str(ML_DIR) not in sys.path:
    sys.path.insert(0, str(ML_DIR))

from src.bayesian_models import (  # noqa: E402
    estimate_historical_negbin_and_rho,
    fit_ordered_logit_head,
    ordered_logit_probs,
    predict_bayesian_posterior,
)
from src.strict_models import (  # noqa: E402
    fit_bayesian_poisson_glm_laplace,
    prepare_next_season_ratings,
    run_elo_walkforward,
)


class EPLChampionV2:
    """Locked standalone Champion v2 predictor (`Stabilized xG-Elo + Multi-Season Laplace Bayes GLM`)."""

    def __init__(
        self,
        k: float = 41.369,
        hfa: float = 140.0,
        shrink: float = 0.143,
        promo_init: float = 1460.0,
        w_xg: float = 0.45,
        prior_scale: float = 0.35,
        half_life_days: float = 360.0,
        w_goals: float = 0.35,
        w_elo: float = 0.60,
    ):
        self.k = k
        self.hfa = hfa
        self.shrink = shrink
        self.promo_init = promo_init
        self.w_xg = w_xg
        self.prior_scale = prior_scale
        self.half_life_days = half_life_days
        self.w_goals = w_goals
        self.w_elo = w_elo

        self.teams_2627: list[str] = []
        self.elo_current: dict[str, float] = {}
        self.ordered_head: tuple[float, float, float] = (-0.4, 0.7, 0.004)
        self.glm_model: dict = {}
        self.alpha_nb: float = 0.0
        self.rho: float = -0.0382
        self.n_matches_2627: int = 0
        self.n_hist_matches: int = 0

    def fit(self, df_hist_full: pd.DataFrame, df_2627: pd.DataFrame) -> "EPLChampionV2":
        """Fit Elo trajectory + Multi-Season Bayesian Poisson GLM causally across history + 2026/27."""
        df_hist_elo = (
            df_hist_full[(df_hist_full["season"] <= "2025/26") & (df_hist_full["season"] != "2020/21")]
            .sort_values(["date", "match_id"])
            .reset_index(drop=True)
        )
        df_2627 = df_2627.sort_values(["date", "ws_match_id"]).reset_index(drop=True)
        self.teams_2627 = sorted(set(df_2627["home_team"]).union(df_2627["away_team"]))
        self.n_matches_2627 = len(df_2627)

        # Estimate historical NegBin overdispersion & Dixon-Coles rho cleanly before 2026/27
        df_recent = df_hist_full[(df_hist_full["season"] >= "2021/22") & (df_hist_full["season"] <= "2025/26")].copy()
        self.alpha_nb, self.rho = estimate_historical_negbin_and_rho(df_recent)

        # 1. Fit Stabilized xG-Elo over clean 2014/15-2025/26 (ExCOVID) -> walk forward through 2026/27
        diffs_hist, end_r_hist, last_teams_hist = run_elo_walkforward(
            df_hist_elo,
            k=self.k,
            hfa=self.hfa,
            shrink=self.shrink,
            promo_init=self.promo_init,
            w_xg=self.w_xg,
        )
        warmup_season = str(df_hist_elo["season"].iloc[0])
        fit_mask = (df_hist_elo["season"] != warmup_season).to_numpy()
        self.ordered_head = fit_ordered_logit_head(
            diffs_hist[fit_mask],
            df_hist_elo["target"].to_numpy(dtype=int)[fit_mask],
        )

        init_2627 = prepare_next_season_ratings(
            end_r_hist,
            last_teams_hist,
            self.teams_2627,
            self.shrink,
            self.promo_init,
        )
        _, self.elo_current, _ = run_elo_walkforward(
            df_2627,
            k=self.k,
            hfa=self.hfa,
            shrink=self.shrink,
            promo_init=self.promo_init,
            w_xg=self.w_xg,
            initial_ratings=init_2627,
            prev_season_teams=last_teams_hist,
        )

        # 2. Fit Pure Multi-Season Bayesian Poisson GLM (Laplace, flat zero-Elo prior)
        df_glm_all = pd.concat([df_recent, df_2627], ignore_index=True)
        self.n_hist_matches = len(df_recent)
        max_date = pd.to_datetime(df_2627["date"]).max()
        self.glm_model = fit_bayesian_poisson_glm_laplace(
            df_glm_all,
            self.teams_2627,
            prior_scale=self.prior_scale,
            half_life_days=self.half_life_days,
            w_goals=self.w_goals,
            hfa_prior=0.22,
            hfa_prior_std=0.14,
            elo_priors=None,
            promo_offset=0.0,
            ref_date=max_date,
        )
        return self

    def predict_match(self, home_team: str, away_team: str, n_draws: int = 400, seed: int = 42) -> dict:
        if home_team not in self.elo_current or away_team not in self.elo_current:
            raise ValueError(f"Unknown team code. Valid 2026/27 teams: {self.teams_2627}")

        p_elo = ordered_logit_probs(
            np.array([(self.elo_current[home_team] + self.hfa) - self.elo_current[away_team]]),
            *self.ordered_head,
        )[0]
        post = predict_bayesian_posterior(
            self.glm_model,
            home_team,
            away_team,
            alpha_nb=self.alpha_nb,
            rho=self.rho,
            n_draws=n_draws,
            seed=seed,
        )
        p_glm = post["mean_probs"]
        std_glm = post["std_probs"]

        p_final = self.w_elo * p_elo + (1.0 - self.w_elo) * p_glm
        p_final = p_final / p_final.sum()

        t2i, n_t = self.glm_model["t2i"], self.glm_model["n_t"]
        hi, ai = t2i[home_team], t2i[away_team]
        th = self.glm_model["theta"]
        xg_h = self.glm_model["mean_rate"] * math.exp(np.clip(th[2 * n_t] + th[hi] + th[n_t + ai], -1.5, 1.5))
        xg_a = self.glm_model["mean_rate"] * math.exp(np.clip(th[ai] + th[n_t + hi], -1.5, 1.5))

        k_grid = np.arange(7)
        mat = np.outer(poisson.pmf(k_grid, xg_h), poisson.pmf(k_grid, xg_a))
        mat /= mat.sum()
        flat_idx = np.argsort(mat.ravel())[::-1][:3]
        top_scores = [(int(i // 7), int(i % 7), round(float(mat.ravel()[i]) * 100.0, 1)) for i in flat_idx]

        return {
            "home_team": home_team,
            "away_team": away_team,
            "pH": round(float(p_final[0]), 4),
            "pD": round(float(p_final[1]), 4),
            "pA": round(float(p_final[2]), 4),
            "pH_elo": round(float(p_elo[0]), 4),
            "pD_elo": round(float(p_elo[1]), 4),
            "pA_elo": round(float(p_elo[2]), 4),
            "pH_glm": round(float(p_glm[0]), 4),
            "pD_glm": round(float(p_glm[1]), 4),
            "pA_glm": round(float(p_glm[2]), 4),
            "std_pH": round(float(std_glm[0]), 4),
            "std_pD": round(float(std_glm[1]), 4),
            "std_pA": round(float(std_glm[2]), 4),
            "xg_home": round(float(xg_h), 2),
            "xg_away": round(float(xg_a), 2),
            "w_elo": round(float(self.w_elo), 4),
            "top_scores": top_scores,
        }

    def get_rankings_df(self) -> pd.DataFrame:
        t2i, n_t = self.glm_model["t2i"], self.glm_model["n_t"]
        th = self.glm_model["theta"]
        rows = []
        for t in self.teams_2627:
            ti = t2i[t]
            att = float(th[ti])
            dfn = float(th[n_t + ti])
            rows.append(
                {
                    "team": t,
                    "net_glm": round(att - dfn, 4),
                    "attack": round(att, 4),
                    "defense": round(dfn, 4),
                    "elo_xg": round(self.elo_current[t], 1),
                    "exp_gf_home": round(self.glm_model["mean_rate"] * math.exp(th[2 * n_t] + att), 2),
                    "exp_ga_home": round(self.glm_model["mean_rate"] * math.exp(dfn), 2),
                }
            )
        df = pd.DataFrame(rows).sort_values("net_glm", ascending=False).reset_index(drop=True)
        df.insert(0, "rank", np.arange(1, len(df) + 1))
        return df
