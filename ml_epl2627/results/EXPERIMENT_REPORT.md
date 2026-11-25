# EPL 2026/27 ML Experiment Report

## 1. Distribution Shift Check (`2021/22–2025/26` vs `2026/27`, `N=107`)
- **Outcome rates (`H/D/A`)**: Historical `0.442 / 0.239 / 0.320` vs 2026/27 `0.421 / 0.280 / 0.299` (`chi2 p = 0.6203`, no significant shift).
- **Goals/Match & Home Advantage**: Historical `2.93` goals (`+0.27` H-A) vs 2026/27 `2.74` goals (`+0.33` H-A) (`KS p = 0.2610`, `p = 0.9974`).
- **Shots/Match**: Historical `25.87` vs 2026/27 `27.51` (`KS p = 0.0023`, significant shift — confirming raw shot counts from prior seasons should not be mixed directly without normalization).

## 2. Walk-Forward Benchmark (`Matches #31–#107`, `N = 77`, Zero Data Leakage)
| Model | RPS ↓ | LogLoss ↓ | Brier ↓ | Acc ↑ | ECE ↓ | ΔRPS vs B0 [95% CI] |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **B4_EventStateSpace_xT** | **0.2147** | **1.0340** | **0.6208** | 0.468 | **0.0362** | **-0.0153 [-0.0311, -0.0003]** |
| B2_Elo | 0.2182 | 1.0430 | 0.6281 | **0.481** | 0.0552 | -0.0118 [-0.0359, +0.0128] |
| B3_DixonColes | 0.2222 | 1.0551 | 0.6362 | 0.455 | 0.0873 | -0.0078 [-0.0294, +0.0149] |
| B0_NaivePrior | 0.2299 | 1.0781 | 0.6526 | 0.429 | 0.0114 | 0.0000 |
| B1_TablePPG | 0.2301 | 1.0819 | 0.6535 | 0.442 | 0.0882 | +0.0002 [-0.0318, +0.0336] |
| B4_XGB_OverfitDiag | 0.2684 | 1.3870 | 0.7891 | 0.403 | 0.2589 | +0.0385 [-0.0047, +0.0796] |

- On the overlapping subset with `clean_match_odds` (`N = 20`), `B4_EventStateSpace_xT` achieved `RPS = 0.2238` vs `B5_MarketOdds = 0.2344` (`N=20` is too small for statistical power, shown strictly for transparency).
- Note on `B4_EventStateSpace_xT` vs `B2_Elo`: `ΔRPS = -0.0035`, `p(better) = 0.696` across 2000 bootstrap samples (directional gain, not yet `p < 0.05` on `N=77`).

## 3. Post-Hoc Opta Power Ranking Check (`261124`, Never Used in Training)
- `elo`: Spearman `r = 0.931`, Pearson `r = 0.962` (`p = 1.42e-11`)
- `dc_net`: Spearman `r = 0.929`, Pearson `r = 0.888`
- `table_ppg`: Spearman `r = 0.896`, Pearson `r = 0.922`
- `event_ss_net`: Spearman `r = 0.868`, Pearson `r = 0.874`
