"""
ml_epl2627/champion/champion_model.py
Locked Production Champion Model (`Champion_Unified`) for EPL 2026/27.

Architecture:
  1. Historical Base: Elo updated via continuous Expected Performance (50% Goals + 50% Shot_xG/xT)
     with conservative hyperparameters (K=20, HFA=65, off-season shrink=0.35, promoted=1420)
     and Ordered Logistic probability head fit strictly on 2021/22-2025/26.
  2. Current-Season Likelihood: Bayesian Ridge Event State-Space on denoised match performance
     (Goals + self-contained Shot-xG + Opta xT), time-decayed (half-life 90 days) and shrunk
     toward the Elo_Formula_xG prior, with discrete 3x3x3 Grid Search re-selected strictly on
     past 2026/27 matches.
  3. Distributional Output: 300-draw Monte Carlo integration over the 2D posterior covariance matrix
     passed through a Dixon-Coles adjusted Poisson score matrix (rho=-0.04), blended smoothly with
     Elo_Formula_xG (w_elo = exp(-n_matches / 25.0)).
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
    fit_ordered_logit_head,
    ordered_logit_probs,
    predict_bayesian_posterior,
    vectorized_score_hda,
)
from src.metrics import rps_per_match  # noqa: E402


class EPLChampionModel:
    """Locked standalone Champion_Unified predictor for EPL 2026/27."""

    def __init__(self, k: float = 20.0, hfa: float = 65.0, shrink: float = 0.35, promo_init: float = 1420.0, rho: float = -0.04):
        self.k = k
        self.hfa = hfa
        self.shrink = shrink
        self.promo_init = promo_init
        self.rho = rho

        self.teams_2627: list[str] = []
        self.promoted: set[str] = set()
        self.elo_init_xg: dict[str, float] = {}
        self.elo_current: dict[str, float] = {}
        self.ordered_head: tuple[float, float, float] = (-0.4, 0.7, 0.004)
        self.latest_cfg: dict = {"ridge_alpha": 18.0, "weights": (0.35, 0.35, 0.30), "prior_scale": 0.45}
        self.ss_model: dict = {}
        self.n_matches_2627: int = 0

    def _fit_historical_elo_xg(self, df_hist: pd.DataFrame, teams_2627: list[str]):
        assert df_hist["season"].max() == "2025/26", "Historical Elo must strictly end at 2025/26!"
        ratings = {}
        prev_sec = None
        diffs, tgts, is_eval = [], [], []

        for _, r in df_hist.iterrows():
            sec = r["season"]
            if prev_sec is not None and sec != prev_sec:
                for t in list(ratings.keys()):
                    ratings[t] = 1500.0 + (1.0 - self.shrink) * (ratings[t] - 1500.0)
            prev_sec = sec

            ht, at = r["home_team"], r["away_team"]
            hr = ratings.get(ht, self.promo_init if sec != "2021/22" else 1500.0)
            ar = ratings.get(at, self.promo_init if sec != "2021/22" else 1500.0)
            diff = (hr + self.hfa) - ar
            diffs.append(diff)
            tgts.append(int(r["target"]))
            is_eval.append(sec != "2021/22")

            exp_h = 1.0 / (1.0 + 10.0 ** (-diff / 400.0))
            hg, ag = float(r["home_goals"]), float(r["away_goals"])
            h_xg = float(r["home_xg"]) if pd.notna(r.get("home_xg")) else hg
            a_xg = float(r["away_xg"]) if pd.notna(r.get("away_xg")) else ag
            perf_diff = (0.5 * hg + 0.5 * h_xg) - (0.5 * ag + 0.5 * a_xg)
            actual_h = 1.0 / (1.0 + math.exp(-1.1 * np.clip(perf_diff, -4.0, 4.0)))
            delta = self.k * 1.15 * (actual_h - exp_h)
            ratings[ht] = hr + delta
            ratings[at] = ar - delta

        teams_2526 = set(df_hist[df_hist["season"] == "2025/26"]["home_team"]).union(
            df_hist[df_hist["season"] == "2025/26"]["away_team"]
        )
        self.promoted = set(teams_2627) - teams_2526
        self.elo_init_xg = {
            t: (1500.0 + (1.0 - self.shrink) * (ratings[t] - 1500.0) if t in teams_2526 else self.promo_init)
            for t in teams_2627
        }
        mask = np.array(is_eval, dtype=bool)
        self.ordered_head = fit_ordered_logit_head(np.array(diffs)[mask], np.array(tgts)[mask])

    def _update_elo_2627(self, ratings: dict, row: pd.Series):
        ht, at = row["home_team"], row["away_team"]
        diff = (ratings[ht] + self.hfa) - ratings[at]
        exp_h = 1.0 / (1.0 + 10.0 ** (-diff / 400.0))
        hg, ag = float(row["home_goals"]), float(row["away_goals"])
        h_perf = 0.5 * hg + 0.5 * (0.65 * float(row["home_opta_npxg"]) + 0.35 * float(row["home_xt"]) * 0.90)
        a_perf = 0.5 * ag + 0.5 * (0.65 * float(row["away_opta_npxg"]) + 0.35 * float(row["away_xt"]) * 0.90)
        actual_h = 1.0 / (1.0 + math.exp(-1.1 * np.clip(h_perf - a_perf, -4.0, 4.0)))
        delta = self.k * 1.15 * (actual_h - exp_h)
        ratings[ht] += delta
        ratings[at] -= delta

    def _fit_bayesian_ss(
        self,
        df_past: pd.DataFrame,
        ridge_alpha: float,
        weights: tuple[float, float, float],
        prior_scale: float,
        half_life: float = 90.0,
    ) -> dict:
        t2i = {t: i for i, t in enumerate(self.teams_2627)}
        n_t = len(self.teams_2627)
        n_cols = 2 * n_t + 1
        w_g, w_xg, w_xt = weights

        prior = np.zeros(n_cols, dtype=float)
        for t, idx in t2i.items():
            elo_z = (self.elo_init_xg.get(t, 1500.0) - 1500.0) / 400.0
            prior[idx] = prior_scale * elo_z + (-0.10 if t in self.promoted else 0.0)
            prior[n_t + idx] = -prior_scale * elo_z + (0.10 if t in self.promoted else 0.0)
        prior[2 * n_t] = 0.22

        prior_sigma2 = 0.16
        if len(df_past) == 0:
            return {"t2i": t2i, "n_t": n_t, "theta": prior, "sigma": prior_sigma2 * (1.0 / ridge_alpha) * np.eye(n_cols), "mean_rate": 1.35}

        rows_X, rows_y, rows_w = [], [], []
        max_date = pd.to_datetime(df_past["date"]).max()
        for _, r in df_past.iterrows():
            hi, ai = t2i[r["home_team"]], t2i[r["away_team"]]
            dt_days = max(0.0, float((max_date - pd.to_datetime(r["date"])).days))
            wt = math.exp(-math.log(2.0) * dt_days / half_life)

            perf_h = w_g * float(r["home_goals"]) + w_xg * float(r["home_opta_npxg"]) + w_xt * (float(r["home_xt"]) * 0.90)
            perf_a = w_g * float(r["away_goals"]) + w_xg * float(r["away_opta_npxg"]) + w_xt * (float(r["away_xt"]) * 0.90)

            xh = np.zeros(n_cols)
            xh[hi], xh[n_t + ai], xh[2 * n_t] = 1.0, 1.0, 1.0
            rows_X.append(xh)
            rows_y.append(math.log(max(0.25, perf_h) / 1.35))
            rows_w.append(wt)

            xa = np.zeros(n_cols)
            xa[ai], xa[n_t + hi] = 1.0, 1.0
            rows_X.append(xa)
            rows_y.append(math.log(max(0.25, perf_a) / 1.35))
            rows_w.append(wt)

        X = np.vstack(rows_X)
        y = np.array(rows_y)
        W = np.diag(rows_w)
        A = X.T @ W @ X + ridge_alpha * np.eye(n_cols)
        A_inv = np.linalg.inv(A)
        theta = prior + A_inv @ (X.T @ W @ (y - X @ prior))

        resid = y - X @ theta
        eff_df = max(1.0, float(np.sum(rows_w)) - float(np.trace(X.T @ W @ X @ A_inv)))
        emp_sigma2 = float(np.sum(np.array(rows_w) * (resid ** 2)) / eff_df)
        sigma2 = 0.5 * emp_sigma2 + 0.5 * prior_sigma2

        theta[:n_t] -= np.mean(theta[:n_t])
        theta[n_t : 2 * n_t] -= np.mean(theta[n_t : 2 * n_t])
        return {"t2i": t2i, "n_t": n_t, "theta": theta, "sigma": sigma2 * A_inv, "mean_rate": 1.35}

    def _select_greedy_cfg(self, df_2627: pd.DataFrame) -> dict:
        if len(df_2627) < 20:
            return {"ridge_alpha": 18.0, "weights": (0.35, 0.35, 0.30), "prior_scale": 0.45}

        val_start = max(10, len(df_2627) - 30)
        unique_dates = sorted(df_2627["date"].iloc[val_start:].unique())
        best_score = 999.0
        best_cfg = {"ridge_alpha": 18.0, "weights": (0.35, 0.35, 0.30), "prior_scale": 0.45}

        for alpha in (10.0, 18.0, 30.0):
            for w in ((0.35, 0.35, 0.30), (0.20, 0.40, 0.40), (0.50, 0.25, 0.25)):
                for ps in (0.35, 0.45, 0.60):
                    probs_val, y_val = [], []
                    for d_val in unique_dates:
                        tr_sub = df_2627[df_2627["date"] < d_val]
                        va_sub = df_2627[df_2627["date"] == d_val]
                        m = self._fit_bayesian_ss(tr_sub, alpha, w, ps)
                        t2i, n_t = m["t2i"], m["n_t"]
                        for _, r in va_sub.iterrows():
                            hi, ai = t2i[r["home_team"]], t2i[r["away_team"]]
                            lh = m["mean_rate"] * math.exp(np.clip(m["theta"][2 * n_t] + m["theta"][hi] + m["theta"][n_t + ai], -1.5, 1.5))
                            la = m["mean_rate"] * math.exp(np.clip(m["theta"][ai] + m["theta"][n_t + hi], -1.5, 1.5))
                            probs_val.append(vectorized_score_hda(np.array([lh]), np.array([la]), 0.0, self.rho)[0])
                            y_val.append(int(r["target"]))
                    score = float(np.mean(rps_per_match(np.array(y_val), np.array(probs_val))))
                    if score < best_score:
                        best_score = score
                        best_cfg = {"ridge_alpha": alpha, "weights": w, "prior_scale": ps, "inner_rps": round(score, 5)}
        return best_cfg

    def fit(self, df_hist: pd.DataFrame, df_2627: pd.DataFrame):
        df_2627 = df_2627.sort_values(["date", "ws_match_id"]).reset_index(drop=True)
        self.teams_2627 = sorted(set(df_2627["home_team"]).union(df_2627["away_team"]))
        self._fit_historical_elo_xg(df_hist, self.teams_2627)

        self.elo_current = dict(self.elo_init_xg)
        for _, row in df_2627.iterrows():
            self._update_elo_2627(self.elo_current, row)

        self.n_matches_2627 = len(df_2627)
        self.latest_cfg = self._select_greedy_cfg(df_2627)
        self.ss_model = self._fit_bayesian_ss(
            df_2627,
            self.latest_cfg["ridge_alpha"],
            self.latest_cfg["weights"],
            self.latest_cfg["prior_scale"],
        )
        return self

    def predict_match(self, home_team: str, away_team: str, n_draws: int = 300, seed: int = 42) -> dict:
        if home_team not in self.elo_current or away_team not in self.elo_current:
            raise ValueError(f"Unknown team code. Valid: {self.teams_2627}")

        p_elo = ordered_logit_probs(
            np.array([(self.elo_current[home_team] + self.hfa) - self.elo_current[away_team]]),
            *self.ordered_head,
        )[0]
        post = predict_bayesian_posterior(self.ss_model, home_team, away_team, alpha_nb=0.0, rho=self.rho, n_draws=n_draws, seed=seed)
        p_ss = post["mean_probs"]
        std_ss = post["std_probs"]

        w_elo = math.exp(-self.n_matches_2627 / 25.0)
        p_final = w_elo * p_elo + (1.0 - w_elo) * p_ss
        p_final = p_final / p_final.sum()

        t2i, n_t = self.ss_model["t2i"], self.ss_model["n_t"]
        hi, ai = t2i[home_team], t2i[away_team]
        xg_h = self.ss_model["mean_rate"] * math.exp(self.ss_model["theta"][2 * n_t] + self.ss_model["theta"][hi] + self.ss_model["theta"][n_t + ai])
        xg_a = self.ss_model["mean_rate"] * math.exp(self.ss_model["theta"][ai] + self.ss_model["theta"][n_t + hi])

        # Compute top 3 most likely scorelines under Poisson(xg_h, xg_a)
        k = np.arange(7)
        mat = np.outer(poisson.pmf(k, xg_h), poisson.pmf(k, xg_a))
        mat /= mat.sum()
        flat_idx = np.argsort(mat.ravel())[::-1][:3]
        top_scores = [(int(i // 7), int(i % 7), round(float(mat.ravel()[i]) * 100.0, 1)) for i in flat_idx]

        return {
            "home_team": home_team,
            "away_team": away_team,
            "pH": round(float(p_final[0]), 4),
            "pD": round(float(p_final[1]), 4),
            "pA": round(float(p_final[2]), 4),
            "std_pH": round(float(std_ss[0]), 4),
            "std_pD": round(float(std_ss[1]), 4),
            "std_pA": round(float(std_ss[2]), 4),
            "xg_home": round(float(xg_h), 2),
            "xg_away": round(float(xg_a), 2),
            "w_elo_blend": round(float(w_elo), 4),
            "top_scores": top_scores,
        }

    def get_rankings_df(self) -> pd.DataFrame:
        t2i, n_t = self.ss_model["t2i"], self.ss_model["n_t"]
        rows = []
        for t in self.teams_2627:
            ti = t2i[t]
            att = float(self.ss_model["theta"][ti])
            dfn = float(self.ss_model["theta"][n_t + ti])
            rows.append(
                {
                    "team": t,
                    "net_rating": round(att - dfn, 4),
                    "attack": round(att, 4),
                    "defense": round(dfn, 4),
                    "elo_xg": round(self.elo_current[t], 1),
                    "exp_gf_home": round(self.ss_model["mean_rate"] * math.exp(self.ss_model["theta"][2 * n_t] + att), 2),
                    "exp_ga_home": round(self.ss_model["mean_rate"] * math.exp(dfn), 2),
                }
            )
        df = pd.DataFrame(rows).sort_values("net_rating", ascending=False).reset_index(drop=True)
        df.insert(0, "rank", np.arange(1, len(df) + 1))
        return df
