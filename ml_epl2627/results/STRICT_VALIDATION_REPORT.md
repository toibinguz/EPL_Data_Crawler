# Strict 3-Way Validation Report (EPL 2014/15 – 2026/27)

## 1. Strict Zero-Leakage Protocol & Historical Distribution Shift

All experiments follow a strict 3-way temporal split:
- **Train Pool (`<= 2024/25`)**: exported from `mapped_epl/epl_clean.db` via [export_full_history.py](file:///z:/Desktop/1x2/ml_epl2627/data/export_full_history.py) (`4,560` played matches across 12 seasons `2014/15–2025/26`, zero missing values in `goals`, `xG`, or `odds`):
  - `Recent4_2125`: `2021/22 – 2024/25` (`1,520` matches)
  - `Full10_ExCOVID`: `2014/15 – 2024/25` excluding `2020/21` (`3,800` matches)
  - `Full11_InclCOVID`: `2014/15 – 2024/25` including `2020/21` (`4,180` matches)
- **Validation Holdout (`2025/26`, `N=380`)**: used exclusively for hyperparameter selection (`Elo`, `Bayesian GLM`, `Ridge`, and stacking weights/decay) and model ranking.
- **Locked Out-of-Sample Test (`2026/27`, `N=110` all matches & `N=80` post-MD3)**: strictly frozen hyperparameters from Validation `2025/26`.

Key empirical distribution shifts observed in `history_1426.csv`:
- `2020/21` (COVID closed-doors): `home_adv = +0.011`, `home_win_rate = 0.379`, `avg_xg = 2.361` (structural anomaly vs typical EPL `+0.21..+0.42` goal advantage).
- `2023/24`: scoring surge (`avg_goals = 3.279`, `avg_xg = 2.927`) vs normal `2.57–2.85`.
- `2024/25`: suppressed home advantage (`home_adv = +0.092`, `h_rate = 0.408`) and collapse of all 3 promoted clubs.
- `2025/26` (Val): `avg_goals = 2.750`, `home_adv = +0.303`, `h_rate = 0.426`.

---

## 2. Step 1 — Elo Validation & Promoted-Team Parameter Fragility (`Val 2025/26, N=380`)

| Model | Params | Val 25/26 RPS | Val LogLoss | Val Acc | Val ECE |
|---|---:|---:|---:|---:|---:|
| `Elo_ValTuned_xG_Full10_ExCOVID` | `K=35.72, HFA=117.78, sh=0.123, pr=1550.0, wxg=0.45` | **0.207005** | **1.021417** | **0.5026** | 0.0311 |
| `Elo_ValTuned_xG_Recent4_2125` | `K=38.68, HFA=128.05, sh=0.053, pr=1550.0, wxg=0.45` | 0.207038 | 1.022191 | 0.4974 | **0.0267** |
| `Elo_ValTuned_Goals_Recent4_2125` | `K=21.79, HFA=0.0, sh=0.016, pr=1550.0, wxg=0.0` | 0.208368 | 1.026100 | 0.4868 | 0.0436 |
| `Elo_Hist3Y_ExCOVID` (tuned on `22–25`) | `K=46.99, HFA=122.66, sh=0.109, pr=1250.0, wxg=0.7` | 0.209214 | 1.027898 | 0.4921 | 0.0376 |
| `Elo_Hand_Recent4_2125` | `K=20, HFA=65, sh=0.35, pr=1420, wxg=0.0` | 0.209995 | 1.031538 | 0.4842 | 0.0631 |
| `Elo_Hand_Full11_InclCOVID` | `K=20, HFA=65, sh=0.35, pr=1420, wxg=0.0` | 0.210010 | 1.031174 | 0.4895 | 0.0743 |
| `Elo_Hand_Full10_ExCOVID` | `K=20, HFA=65, sh=0.35, pr=1420, wxg=0.0` | 0.210075 | 1.031134 | 0.4868 | 0.0507 |
| `Elo_HistN1_ExCOVID` (tuned on `24/25`) | `K=28.97, HFA=71.07, sh=0.317, pr=1250.0, wxg=0.7` | 0.213897 | 1.043136 | 0.4895 | 0.0618 |

**Conclusions for Elo:**
1. Blending `xG` into Elo updates (`w_xg = 0.45`) materially improves out-of-sample RPS (`0.2084 -> 0.2070` on Val `2025/26`, `0.2114 -> 0.2080` on Test `2026/27`).
2. Excluding COVID `2020/21` improves calibration (`ECE: 0.0743 -> 0.0507` under identical fixed parameters).
3. Optimizing `promo_init` on a single validation season is unstable: tuning on `2024/25` collapsed `promo_init` to `1250` (`Val RPS = 0.21390`, `Test RPS = 0.21738`), while tuning on `2025/26` pushed `promo_init` to the upper bound `1550`. Multi-year `2022–25` historical tuning (`Elo_Hist3Y_ExCOVID`, `0.20921`) beat handcrafted Elo (`0.21000`) on `2025/26`, though structural stability favors constraining `promo_init` around historical baseline (`~1420`) rather than fitting it to 3 teams per season.

---

## 3. Step 2 — Pure Zero-Elo Bayesian Branch (`Val 2025/26, N=380`)

| Model | Hyperparameters | Val RPS | Val LogLoss | Val Acc | Val ECE |
|---|---|---:|---:|---:|---:|
| `BayesGLM_MultiRecent4` | `prior_scale=0.35, half_life=360d, w_goals=0.35, promo_offset=0.0` | **0.208333** | 1.029685 | 0.4895 | 0.0314 |
| `BayesGLM_MultiExCOVID` | `prior_scale=0.35, half_life=300d, w_goals=0.35, promo_offset=0.0` | 0.208399 | **1.029518** | **0.4974** | **0.0179** |
| `BayesGLM_SeasonFlat` | `prior_scale=0.42, half_life=180d, w_goals=0.35, promo_offset=0.12` | 0.211339 | 1.041198 | 0.4947 | 0.0264 |
| `BayesRidge_SeasonFlat` | `ridge_alpha=12.0, half_life=150d, w_goals=0.35, promo_offset=0.12` | 0.214066 | 1.046048 | 0.4658 | 0.0132 |

**Conclusions for Bayesian Branch:**
1. Replacing `log(max(0.25, perf))` Ridge with exact **Bayesian Poisson GLM (Laplace)** removes bias on low-scoring matches and improves `SeasonFlat` Val RPS by **`-0.00273`** (`0.21407 -> 0.21134`).
2. Directly passing multi-season time-decayed match history into Bayesian Poisson GLM (`half_life = 300–360d`) cuts Val RPS further to **`0.20833`** (`-0.00301`) with zero Elo needed.
3. Because optimal half-life is `~1` year (`300–360d`), `Recent4 (2021–25)` and `Full10_ExCOVID (2014–25 ex-COVID)` perform nearly identically (`0.20833` vs `0.20840` Val; `0.20672` vs `0.20680` Test).

---

## 4. Step 3 — Full 3-Way Benchmark Summary (`Val 2025/26` -> `Locked Test 2026/27`)

| Model | Val 25/26 RPS | Val LL | Test 26/27 RPS (`N=110`) | Test LL | Test Acc | Post-MD3 RPS (`N=80`) | Diff vs Legacy Champ | `P(Better)` vs Champ |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| `Champion_Legacy` | 0.20866 | 1.02883 | 0.21039 | 1.02790 | 48.18% | 0.21820 | `0.00000` | — |
| `Elo_Hand_Recent4` | 0.21000 | 1.03154 | 0.21137 | 1.03065 | 48.18% | 0.21976 | `+0.00098` | 16.00% |
| `Elo_HistN1_ExCOVID` | 0.21390 | 1.04314 | 0.21738 | 1.04641 | 44.55% | 0.22287 | `+0.00699` | 12.15% |
| `Elo_Hist3Y_ExCOVID` | 0.20921 | 1.02790 | 0.21131 | 1.02816 | 45.45% | 0.21786 | `+0.00093` | 41.55% |
| `Best_Elo` (`ValTuned_xG_ExCOVID`) | 0.20701 | 1.02142 | 0.20798 | 1.02096 | **50.91%** | 0.21725 | `-0.00241` | 77.60% |
| `BayesRidge_SeasonFlat` | 0.21407 | 1.04605 | 0.21675 | 1.05015 | 42.73% | 0.21861 | `+0.00636` | 12.40% |
| `BayesGLM_SeasonFlat` | 0.21134 | 1.04120 | 0.21621 | 1.05709 | 43.64% | 0.21744 | `+0.00582` | 13.05% |
| **`BayesGLM_MultiRecent4` (Pure, Zero Elo)** | 0.20833 | 1.02969 | **0.20672** | 1.02211 | 50.00% | 0.21615 | `-0.00367` | 80.25% |
| `BayesGLM_MultiExCOVID` (Pure, Zero Elo) | 0.20840 | 1.02952 | 0.20680 | 1.02200 | 50.00% | 0.21605 | `-0.00358` | 81.90% |
| `Stack_IndepDyn (tau=40)` | 0.20661 | 1.02276 | 0.20730 | 1.01909 | 50.00% | 0.21608 | `-0.00309` | 86.85% |
| **`Stack_Multi (w_elo=0.65)`** | **0.20649** | **1.02087** | **0.20661** | **1.01837** | 49.09% | **0.21594** | **`-0.00378`** | **92.35%** |
| `Market_Odds` (Bookmaker Benchmark) | **0.20528** | **1.01525** | 0.20816 | 1.02395 | 48.18% | **0.21569** | `-0.00223` | 67.45% |
