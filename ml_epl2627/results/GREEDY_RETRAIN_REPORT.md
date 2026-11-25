# Updated 110-Match Report: Shot-xG + Greedy BayesSS vs Market Odds (EPL 2026/27)

## 1. Data & Shot-xG Model Verification
- **Synchronized 110/110 EPL 2026/27 matches** through MD11 (`2026-11-23`) across WhoScored (`whoscored_2627/`), FotMob (`2026_2027_new/matches/fotmob/`), and Football-Data odds (`football_data_raw_2627.csv`), cleanly integrated into `mapped_epl/epl_clean.db` (`clean_matches=110`, `clean_match_odds=110`, `clean_match_team_stats=220`; `2025/26=380` untouched).
- **Historical Shot-xG Model** (`ml_epl2627/src/shot_xg_model.py`, trained strictly on `114,840` non-penalty shots `<=2025/26`):
  - `2025/26` OOS validation (`N=9,391` shots): `AUC = 0.7396`, `Brier = 0.08055`
  - `2026/27` vs FotMob `fm_npxg` (`N=220` team-matches): **Pearson `r = 0.9071`**, `MAE = 0.2451`

## 2. Full Season (`N = 110`, Matches #1–#110, 100% Odds Overlap)
| Model | RPS ↓ | LogLoss ↓ | Brier ↓ | Acc ↑ | ECE ↓ | ΔRPS vs Market [95% CI] |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **B5_MarketOdds** | **0.2082** | 1.0240 | 0.6148 | **0.482** | 0.0346 | 0.0000 |
| **Elo_Formula_xG** | 0.2089 | **1.0223** | **0.6135** | 0.464 | 0.0490 | +0.0007 [-0.0088, +0.0104] |
| BayesSS_Blend_Elo | 0.2101 | 1.0285 | 0.6168 | 0.473 | 0.0449 | +0.0019 [-0.0097, +0.0137] |
| BayesSS_Greedy_Every5 | 0.2101 | 1.0305 | 0.6181 | 0.455 | 0.0469 | +0.0019 [-0.0101, +0.0139] |
| Elo_Formula_MOV | 0.2113 | 1.0304 | 0.6190 | **0.482** | 0.0357 | +0.0031 [-0.0074, +0.0138] |
| Elo_Formula_WDL | 0.2117 | 1.0315 | 0.6200 | 0.473 | **0.0228** | +0.0036 [-0.0069, +0.0148] |

## 3. Post-Warmup (`N = 80`, Matches #31–#110, Post-MD3)
| Model | RPS ↓ | LogLoss ↓ | Brier ↓ | Acc ↑ | ECE ↓ | ΔRPS vs Market [95% CI] |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **BayesSS_Greedy_Every5** | **0.2135** | **1.0237** | **0.6135** | 0.463 | **0.0219** | **-0.0022 [-0.0154, +0.0105]** |
| BayesSS_Greedy_Every10 | 0.2141 | 1.0255 | 0.6148 | 0.475 | 0.0266 | -0.0016 [-0.0146, +0.0114] |
| B5_MarketOdds | 0.2157 | 1.0298 | 0.6204 | **0.488** | 0.0258 | 0.0000 |
| Elo_Formula_xG | 0.2164 | 1.0302 | 0.6186 | 0.463 | 0.0417 | +0.0007 [-0.0097, +0.0115] |
| BayesSS_Blend_Elo | 0.2164 | 1.0313 | 0.6191 | **0.488** | 0.0351 | +0.0007 [-0.0129, +0.0137] |

*Honest note:* `BayesSS_Greedy_Every5` beats `B5_MarketOdds` on all proper scoring rules & ECE for Matches #31–#110 (`p_beat = 63.8%`), while `Elo_Formula_xG` beats Market on `LogLoss` (`1.0223`) and `Brier` (`0.6135`) across all `N=110`. The 95% Bootstrap CI still crosses 0 due to `N=80/110` sample size — more matchdays are needed for `p < 0.05` statistical significance.
