# EPL 2026/27 ML Experiment Plan

## 1. Objective & Philosophy
With `N = 107` completed EPL 2026/27 matches, Signal-to-Noise Ratio (SNR) is low and outcome variance (`1X2` goals) is high. The goal is **NOT** to maximize raw training/CV accuracy via high-dimensional tree models (which overfit on `N=107`), but to achieve **well-calibrated outcome probabilities** evaluated under strict **Zero Data Leakage Walk-Forward Validation**.

- **Primary Metric:** Ranked Probability Score (**RPS**) — respects ordered outcomes (`H > D > A`).
- **Secondary Metrics:** Multiclass **LogLoss**, **Brier Score**, **ECE** (Expected Calibration Error, 5 bins), **Accuracy**, plus paired **Bootstrap 95% CI** (`n=2000`) against trivial baselines.

## 2. Benchmark Ladder
Every model predicts match $t$ using strictly matches with `date < date(t)`:
- **B0_NaivePrior**: Empirical `H/D/A` proportions from prior seasons (`2021/22–2025/26`) + expanding `2026/27`.
- **B1_TablePPG**: Regularized Multinomial Logistic Regression (`L2`) on 3 trivial table features (`delta_ppg`, `delta_gd_pg`, `home=1`).
- **B2_Elo**: Time-continuous Elo (`K=20`, `HFA=65`, `0.35` off-season shrink toward `1500`; newly promoted clubs initialized at `1420`) mapped via calibrated Multinomial Logistic on `delta_elo`.
- **B3_DixonColes_Goals**: Time-decayed Dixon-Coles Bivariate Poisson with L2 Ridge shrinkage on team attack/defense parameters, using `2025/26` (down-weighted) + expanding `2026/27` scores.
- **B4_EventStateSpace_xT**: Noise-reduced **Opponent-Adjusted Ridge + Poisson/Logit** model using WhoScored `2026/27` event metrics (`goals`, `box_shots`, `xT`, `team_rating`) shrunk toward pre-season Elo priors.
- **B4_XGB_OverfitDiag**: `XGBClassifier` (`max_depth=4`, `n_estimators=80`) trained ONLY on expanding `2026/27` raw rolling features (`12` features) to empirically test small-sample overfitting vs trivial baselines.
- **B5_MarketOdds**: De-vigged market closing probabilities from `epl_clean.db` on the overlapping `2026/27` subset.

## 3. Distribution Shift & Multi-Season Usage
- Quantify distribution shift between `2021/22–2025/26` (`N = 1900`) and `2026/27` (`N = 107`) on `H/D/A` rates, goals/match, home goal advantage, and shots/match (`KS-test`, `Chi-square`).
- Use historical seasons **only** for structural league parameters (home advantage, draw propensity, score conversion slope) and decayed off-season priors — never mixing raw unscaled team stats across seasons.

## 4. Opta Power Ranking Usage Rule
`mapped_epl/opta_power_ranking_261124.json` (snapshot `2026-11-23T03:05:49Z`, post-MD11/12) has **ZERO training/prediction value** due to point-in-time unavailability. It is used **strictly post-hoc** to measure Spearman/Pearson correlation with our learned post-MD11 team strength ratings.
