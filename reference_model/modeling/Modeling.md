# Mô hình hóa – Train / Dev / Test

Tài liệu này mô tả chi tiết package `modeling/`. Package được tách từ notebook `EPL_train_dev_test_.ipynb` (vốn chạy trên Colab) thành các module Python chạy được bằng một lệnh.

Bài toán gồm hai phần:
- **Phân loại H/D/A**: dự đoán xác suất đội nhà thắng, hòa hoặc đội khách thắng.
- **Dự đoán tỉ số**: dự đoán số bàn kỳ vọng của mỗi đội (λ), từ đó suy ra ma trận xác suất các tỉ số.

---

## Mục lục

1. [Cách chạy](#1-cách-chạy)
2. [Dữ liệu đầu vào](#2-dữ-liệu-đầu-vào)
3. [Luồng xử lý](#3-luồng-xử-lý)
4. [Chia Train / Dev / Test](#4-chia-train--dev--test)
5. [Chọn đặc trưng và kiểm tra rò rỉ](#5-chọn-đặc-trưng-và-kiểm-tra-rò-rỉ-dữ-liệu)
6. [Tiền xử lý](#6-tiền-xử-lý)
7. [Thước đo đánh giá](#7-thước-đo-đánh-giá)
8. [Nhiệm vụ 2 – Poisson + Dixon–Coles](#8-nhiệm-vụ-2--poisson--dixoncoles-dự-đoán-tỉ-số)
9. [Nhiệm vụ 1 – Logistic Regression](#9-nhiệm-vụ-1--logistic-regression-hda)
10. [Nhiệm vụ 3 – Kiểm định giả thuyết H1–H9](#10-nhiệm-vụ-3--kiểm-định-giả-thuyết-h1h9)
11. [Các mô hình M0–M7 và chọn mô hình](#11-các-mô-hình-m0m7-và-chọn-mô-hình-cuối)
12. [Đánh giá trên Test](#12-đánh-giá-trên-test)
13. [Thử nghiệm phụ](#13-thử-nghiệm-phụ)
14. [Nhiệm vụ 4 – Xuất dự đoán](#14-nhiệm-vụ-4--xuất-dự-đoán)
15. [File đầu ra](#15-file-đầu-ra)
16. [Mô tả từng module](#16-mô-tả-từng-module)
17. [Tùy chỉnh](#17-tùy-chỉnh)
18. [Lưu ý và hạn chế](#18-lưu-ý-và-hạn-chế)

---

## 1. Cách chạy

Chạy từ **thư mục gốc** project, không chạy từ trong `modeling/`, vì package dùng import tương đối:

```bash
python -m modeling.train_eval [--features PATH] [--use-odds] [--skip-experiments] [--refit-with-test]
```

| Tham số | Ý nghĩa | Mặc định |
|---|---|---|
| `--features PATH` | Đường dẫn tới `features.csv` | Tự tìm (xem mục 2) |
| `--use-odds` | Dùng kèo nhà cái làm **đặc trưng** cho LR và Poisson | Tắt |
| `--skip-experiments` | Bỏ qua thử nghiệm odds (10b) và thử năm bắt đầu train (10c) | Chạy cả hai |
| `--refit-with-test` | Mô hình dự đoán trận sắp đá được học trên **mọi** trận đã đá, kể cả test | Tắt (học trên train + dev) |

Biến môi trường (tùy chọn):

| Biến | Ý nghĩa |
|---|---|
| `EPL_FEATURES_CSV` | Đường dẫn tới `features.csv` |
| `EPL_DATA_DIR` | Thư mục dataset, script tự tìm `features.csv` bên trong |
| `EPL_OUT_DIR` | Thư mục gốc chứa kết quả (mặc định `outputs/`) |

Log được in ra màn hình theo từng mục, đánh số 0 → 12 giống notebook. Muốn lưu log:

```bash
python -m modeling.train_eval > run.log 2>&1
```

Trên Windows, nếu gặp lỗi encoding khi ghi log, đặt `PYTHONIOENCODING=utf-8` trước khi chạy.

---

## 2. Dữ liệu đầu vào

**File:** `epl--league-dataset/clean/features.csv`, gồm 9.870 trận (2000/01 → 2026/27, mùa 2026/27 mới có 80 trận) và 113 cột.

**Thứ tự tìm file** (`data.locate_features`):
1. Tham số `--features`
2. Biến `EPL_FEATURES_CSV`
3. Biến `EPL_DATA_DIR`
4. `data/epl--league-dataset/`
5. `data/` (tìm đệ quy, ưu tiên `.../clean/features.csv`)
6. Không thấy ở đâu cả thì tải từ Hugging Face (`Toibinguz/epl-premier-league-dataset`) về `data/epl--league-dataset/`. Chỉ tải các file cần: `features.csv`, `hypotheses.csv`, `README.md`, `features_report.json`.

**Các nhóm cột chính:**

| Nhóm | Ví dụ |
|---|---|
| Định danh | `match_id`, `season`, `date`, `home_team`, `away_team`, `status` |
| Nhãn | `home_goals`, `away_goals`, `result` (H/D/A) |
| Elo | `home_elo`, `away_elo`, `elo_diff`, `elo_home_win_prob` |
| Phong độ 5 trận | `home_goals_for5`, `diff_points5`, `diff_xg5`, `diff_xga5`, … |
| Phong độ theo sân | `home_home_goals_for5`, `away_away_goals_against5`, … |
| Bảng xếp hạng | `diff_ppg`, `position_diff` |
| Đối đầu | `h2h_matches`, `h2h_home_points`, `h2h_home_goal_diff` |
| Nghỉ / lên hạng | `home_rest_days`, `home_form_age_days`, `home_promoted`, … |
| FotMob (lối chơi) | `diff_possession5`, `diff_big_chances5`, `diff_box_touches5`, `diff_xgot5`, … |
| Lợi thế sân nhà của giải | `league_home_adv380` |
| Kèo nhà cái | `odds_home_prob`, `odds_draw_prob`, `odds_away_prob` (đã là xác suất) |
| Số trận có dữ liệu | `home_xg_n5`, `home_fm_n5`, … (chỉ dùng để lọc) |

Nếu file không có cột `status`, trận có `result` được coi là `played`, còn lại là `upcoming`.

**Tỉ lệ thiếu dữ liệu:** các cột FotMob thiếu 60–78% (chỉ có từ những mùa gần đây). Cột xG có từ mùa 2014/15. Các ô trống được điền bằng trung vị trong bước tiền xử lý.

---

## 3. Luồng xử lý

```
features.csv
   │
   ▼  data.py           load, sắp xếp theo ngày, EDA, chia train / dev / test theo mùa
   ▼  features.py       loại cột ID/nhãn/odds → kiểm tra rò rỉ → bộ cột an toàn
   ▼  poisson_dc.py     dò alpha trên Dev → λ nhà, λ khách (out-of-fold) → lam_diff, ρ Dixon–Coles
   ▼  models.py         dò C cho LR (6 cột, 6 cột + lam_diff)
   ▼  hypothesis.py     kiểm định H1–H9 bằng rolling-origin 2015/16 → 2023/24
   ▼  models.py         M0–M7 trên Dev → bảng xếp hạng → chọn mô hình cuối
   ▼  train_eval.py     học lại trên Train + Dev → đánh giá Test MỘT LẦN
   ▼  experiments.py    (phụ) thử odds, thử năm bắt đầu train
   ▼  predict.py        predictions_test.csv, predictions.csv, final_models.joblib
   ▼
outputs/run_<ngày>_<giờ>_<odds|no_odds>/
```

---

## 4. Chia Train / Dev / Test

Dữ liệu được chia **theo mùa, không xáo trộn**, để không dùng tương lai đoán quá khứ. Mốc chia đặt trong `config.py`, tính theo năm bắt đầu mùa (2021 = mùa 2021/22).

| Tập | Mùa | Số trận | H / D / A | Vai trò |
|---|---|---|---|---|
| **Train** | 2000/01 → 2021/22 | 8.270 | 45.9% / 24.9% / 29.2% | Huấn luyện |
| **Dev** | 2022/23 → 2023/24 | 760 | 47.2% / 22.2% / 30.5% | Dò siêu tham số, chọn mô hình, chọn nhóm đặc trưng |
| **Test** | 2024/25 → nay | 840 | 41.4% / 26.2% / 32.4% | Chỉ đánh giá **một lần** ở cuối |

Script kiểm tra bằng `assert` rằng ngày cuối của train < ngày đầu của dev, và ngày cuối của dev < ngày đầu của test.

Tập test có tỉ lệ đội nhà thắng thấp hơn hẳn (41.4% so với khoảng 46% ở train), nên test "khó" hơn dev. Đây là lý do log loss trên test cao hơn dev.

---

## 5. Chọn đặc trưng và kiểm tra rò rỉ dữ liệu

**Bước 1: loại cột** (`features.select_candidates`), dùng regex:
- `ID_LIKE`: cột id, tên đội, ngày, mùa, status, trọng tài, sân, gameweek, …
- `ODDS_LIKE`: các cột kèo nhà cái. Odds không được làm đặc trưng mặc định vì trận sắp đá có thể chưa có odds.
- `META_LIKE`: các cột `*_n5` (số trận có dữ liệu), chỉ dùng để lọc.
- Nhãn: `home_goals`, `away_goals`, `result`.

Kết quả: còn **87 cột số** làm ứng viên.

**Bước 2: kiểm tra rò rỉ** (`features.leakage_check`). Tính |tương quan| giữa mỗi cột với hiệu số bàn thắng **của chính trận đó** trên tập train. Cột nào > 0.55 thì gần như chắc là thống kê sau trận (ví dụ xG của chính trận) và bị loại. Để so sánh, `elo_diff`, một đặc trưng mạnh và hợp lệ, chỉ đạt khoảng 0.44.

Top tương quan: `elo_diff` 0.441, `elo_home_win_prob` 0.439, `diff_xg5` 0.359, `position_diff` 0.358. **Không có cột nào bị loại.**

**Bước 3: odds** (`features.find_odds`). Script tự dò bộ 3 cột odds (nhà / hòa / khách), xác định odds đang lưu dạng xác suất hay odds thập phân, rồi chuẩn hóa để bỏ overround (tổng = 1). Odds dùng cho:
- **H9**: mốc tham chiếu, luôn chạy nếu có cột odds.
- **Đặc trưng** khi bật `--use-odds`: `odds_logit_ha = log(pH/pA)`, `odds_log_d = log(pD)`, `odds_missing`.

**Các bộ đặc trưng theo PDF** (`features.pdf_feature_sets`):

| Bộ | Cột |
|---|---|
| `LR_BASE` (6 cột) | `elo_diff`, `diff_points5`, `diff_goals_for5`, `diff_goals_against5`, `diff_ppg`, `position_diff` |
| `POI_FEATS` (Poisson) | `elo_diff`, `home_goals_for5`, `home_goals_against5`, `away_goals_for5`, `away_goals_against5`, `home_home_goals_for5`, `away_away_goals_against5`, `diff_xg5`, `diff_xga5` |

---

## 6. Tiền xử lý

`preprocessing.make_prep()` cho mọi mô hình tuyến tính (LR, Poisson):

```
SimpleImputer(median) → ClipPercentile(1, 99) → StandardScaler
```

- **Imputer**: điền ô trống bằng trung vị học từ tập fit.
- **ClipPercentile** (class tự viết): cắt mỗi cột về khoảng [phân vị 1, phân vị 99] của tập fit để giảm ảnh hưởng ngoại lai.
- **StandardScaler**: chuẩn hóa về trung bình 0, độ lệch chuẩn 1. Nhờ vậy hệ số LR so sánh được với nhau.

Mọi bước đều fit **chỉ trên tập train** (nằm trong `Pipeline`), nên không rò rỉ thông tin từ dev/test. HistGradientBoosting không cần bước này vì tự xử lý được ô trống.

---

## 7. Thước đo đánh giá

`metrics.py`, mọi thước đo đều **càng thấp càng tốt**, trừ accuracy:

| Thước đo | Công thức / ý nghĩa |
|---|---|
| **Log loss** | `−log P(kết quả thật)`. Thước đo chính, phạt nặng khi tự tin mà sai |
| **RPS** | Ranked Probability Score. Tính đến thứ tự H < D < A: đoán H mà kết quả D thì bị phạt nhẹ hơn kết quả A. Chuẩn cho dự đoán bóng đá |
| **Brier** | Tổng bình phương sai số giữa xác suất và one-hot của nhãn |
| **Accuracy** | Tỉ lệ đoán đúng kết quả có xác suất cao nhất. Bỏ qua độ tự tin nên chỉ để tham khảo |

**Paired bootstrap** (`metrics.paired_bootstrap`): so sánh hai mô hình trên **cùng tập trận**.
- Δ = loss(mô hình mới) − loss(mô hình gốc). Δ âm nghĩa là mô hình mới tốt hơn.
- Lấy mẫu lại 2.000 lần để có khoảng tin cậy 95%.
- CI nằm hẳn dưới 0 thì cải thiện **có ý nghĩa thống kê**. CI chứa 0 thì chưa đủ bằng chứng.

Mốc tham chiếu trên EPL: đoán theo tần suất khoảng 1.06–1.07, nhà cái khoảng 0.95–0.97. Mô hình không dùng odds đạt khoảng 0.95–0.97 trên dev là hợp lý. **Thấp hơn 0.90** thì gần như chắc chắn có rò rỉ, nên mô hình đó bị loại khỏi lựa chọn (`SUSPICIOUS_LL`).

---

## 8. Nhiệm vụ 2 – Poisson + Dixon–Coles (dự đoán tỉ số)

`poisson_dc.py`

**Mô hình:** hai `PoissonRegressor` độc lập, một cho số bàn đội nhà (λ_nhà) và một cho số bàn đội khách (λ_khách), cùng dùng `POI_FEATS`.

**Dò `alpha`** (hệ số regularization) trong {0.01, 0.1, 1} theo MAE trung bình trên dev:

| alpha | MAE nhà | MAE khách | MAE TB |
|---|---|---|---|
| **0.01** | 1.002 | 0.916 | **0.959** |
| 0.1 | 1.004 | 0.914 | 0.959 |
| 1.0 | 1.017 | 0.912 | 0.965 |

**`lam_diff` = λ_nhà − λ_khách** được đưa vào LR làm đặc trưng. Điểm quan trọng: trên tập train, `lam_diff` được tính **out-of-fold theo mùa** (`GroupKFold` 5 fold, nhóm theo mùa). Mỗi trận được dự đoán bởi mô hình chưa từng thấy mùa của trận đó. Nếu fit rồi dự đoán ngay trên chính tập train, `lam_diff` sẽ "biết trước" nhãn và LR sẽ đánh giá quá cao đặc trưng này. Dev/test dùng mô hình fit trên toàn bộ train. Hàm thực hiện: `prepare_stage`.

**Ma trận tỉ số** (`score_matrix`): `M[trận, bàn nhà, bàn khách] = Poisson(i; λ_nhà) × Poisson(j; λ_khách)` với i, j từ 0 đến 10, sau đó chuẩn hóa tổng = 1. Từ ma trận suy ra:
- P(H) = tổng các ô i > j, P(D) = tổng các ô i = j, P(A) = tổng các ô i < j (`matrix_to_hda`)
- Tỉ số có khả năng cao nhất

**Hiệu chỉnh Dixon–Coles:** Poisson độc lập thường đoán sai các tỉ số thấp. Dixon–Coles nhân thêm hệ số τ cho 4 ô:

| Tỉ số | Hệ số |
|---|---|
| 0-0 | 1 − λ_nhà·λ_khách·ρ |
| 0-1 | 1 + λ_nhà·ρ |
| 1-0 | 1 + λ_khách·ρ |
| 1-1 | 1 − ρ |

ρ được ước lượng bằng maximum likelihood trên train, trong khoảng [−0.25, 0.25] (`fit_rho`). Kết quả: **ρ = −0.0389** (train) và −0.0351 (train + dev).

**Chất lượng trên dev:**

| | Log-lik tỉ số / trận | Đúng tỉ số | P(D) TB (thực tế 0.222) | Log loss H/D/A |
|---|---|---|---|---|
| Poisson độc lập | −3.0550 | 11.8% | 0.227 | 0.9508 |
| Dixon–Coles | −3.0556 | 11.2% | 0.235 | 0.9512 |

Trên dữ liệu này, Dixon–Coles gần như không khác Poisson độc lập (ρ nhỏ).

---

## 9. Nhiệm vụ 1 – Logistic Regression H/D/A

`models.py`, `LogisticRegression` đa lớp, kèm tiền xử lý ở mục 6.

Dò `C` (nghịch đảo regularization) trong {0.01, 0.03, 0.1, 0.3, 1} theo log loss trên dev:

| Bộ đặc trưng | C tốt nhất | Log loss | RPS | Accuracy |
|---|---|---|---|---|
| 6 cột (M2) | 0.01 | 0.9534 | 0.1964 | 55.4% |
| 6 cột + `lam_diff` (M3) | 0.01 | 0.9526 | 0.1961 | 55.5% |

C nhỏ nhất thắng, nghĩa là regularization mạnh tốt hơn. Các đặc trưng tương quan cao với nhau (Elo, điểm, hạng đều đo "sức mạnh đội").

`proba_HDA` sắp lại cột xác suất từ thứ tự `classes_` của sklearn (A, D, H) về thứ tự H, D, A.

**Hệ số LR** (mô hình cuối, tính theo mỗi độ lệch chuẩn của đặc trưng):

| Đặc trưng | coef_H | coef_D | coef_A |
|---|---|---|---|
| `elo_diff` | +0.293 | +0.042 | −0.335 |
| `lam_diff` | +0.295 | −0.023 | −0.272 |
| `position_diff` | +0.097 | −0.019 | −0.078 |
| `diff_points5` | −0.099 | +0.022 | +0.077 |
| `diff_goals_against5` | −0.073 | +0.031 | +0.042 |
| `diff_goals_for5` | +0.051 | −0.021 | −0.030 |
| `diff_ppg` | +0.048 | −0.029 | −0.020 |

`elo_diff` và `lam_diff` chiếm phần lớn sức dự đoán. Các cột phong độ có hệ số nhỏ, một số còn ngược dấu do đa cộng tuyến với Elo.

---

## 10. Nhiệm vụ 3 – Kiểm định giả thuyết H1–H9

`hypothesis.py`

**Phương pháp:** mỗi giả thuyết so sánh mô hình lồng nhau, **mô hình gốc** với **mô hình gốc + nhóm đặc trưng**, trên cùng một tập trận.

**Đánh giá rolling-origin:** với mỗi mùa 2015/16 → 2023/24, huấn luyện LR trên **tất cả các mùa trước đó** rồi dự đoán mùa đó. Cách này cho khoảng 3.400 trận đánh giá (nhiều hơn hẳn 760 trận dev), nên khoảng tin cậy hẹp hơn, và vẫn **không đụng tới test**.

Với H3 (xG) và H8 (FotMob), chỉ giữ các trận mà cả hai đội có ít nhất 3/5 trận gần nhất có dữ liệu (`n5_filter`).

**Kết quả:**

| H | Giả thuyết | Gốc | Thêm | Δ log loss | CI 95% | Kết luận |
|---|---|---|---|---|---|---|
| H1 | Elo tốt hơn đoán theo tần suất | tần suất | `elo_diff` | −0.0947 | [−0.108, −0.082] | **Có ích** |
| H2 | Phong độ bổ sung ngoài Elo | Elo | 5 cột phong độ | −0.0015 | [−0.0028, −0.0001] | **Có ích** |
| H3 | xG / xGA bổ sung ngoài bàn thắng | Elo + phong độ | `diff_xg5`, `diff_xga5` | −0.0032 | [−0.0059, −0.0003] | **Có ích** |
| H4 | Đối đầu | Elo + phong độ | 3 cột h2h | +0.0002 | [−0.0010, +0.0015] | Chưa đủ bằng chứng |
| H5 | Lợi thế sân nhà thay đổi theo thời kỳ | Elo + phong độ | `league_home_adv380` | −0.0007 | [−0.0022, +0.0009] | Chưa đủ bằng chứng |
| H6 | Phong độ theo sân | Elo + phong độ + bàn thắng 2 đội | 6 cột home_home / away_away | +0.0007 | [−0.0005, +0.0018] | Chưa đủ bằng chứng |
| H7 | Số ngày nghỉ, lên hạng | Elo + phong độ | 6 cột | +0.0004 | [−0.0011, +0.0019] | Chưa đủ bằng chứng |
| H8 | Chỉ số lối chơi FotMob | Elo + phong độ | 6 cột diff_* | +0.0002 | [−0.0036, +0.0044] | Chưa đủ bằng chứng |
| H9 | Nhà cái làm mốc tham chiếu | nhà cái | LR 6 cột | +0.0179 | [+0.012, +0.024] | Nhà cái tốt hơn mô hình |

H10 (odds bổ sung ngoài Elo + phong độ) chỉ chạy khi bật `--use-odds`.

**Ý nghĩa:**
- Elo là đặc trưng quan trọng nhất, giảm log loss khoảng 0.095 so với đoán theo tần suất.
- Phong độ và xG chỉ cải thiện rất ít (khoảng 0.002–0.003) nhưng có ý nghĩa thống kê.
- Các nhóm còn lại không bổ sung được gì đáng kể sau khi đã có Elo + phong độ.
- Nhà cái vẫn tốt hơn mô hình khoảng 0.018 log loss.

Tên cột của H4, H7, H8 được dò bằng regex. Nếu dữ liệu đổi tên cột, kiểm tra bảng in ra (`H4: +3 cols -> [...]`) và sửa danh sách `add` trong `build_hypotheses`.

---

## 11. Các mô hình M0–M7 và chọn mô hình cuối

| Mô hình | Mô tả |
|---|---|
| **M0** Frequency | Luôn dự đoán tỉ lệ H/D/A của tập train |
| **M1** LR Elo | LR chỉ với `elo_diff` |
| **M2** LR 6 cols | LR với 6 cột của PDF |
| **M3** LR 6 + lam_diff | **Mô hình của PDF**: 6 cột + `lam_diff` |
| **M4** Dixon–Coles | Xác suất H/D/A suy từ ma trận tỉ số |
| **M5** LR+ | M3 + các nhóm đặc trưng **có ích** ở H2–H8 (ở đây: phong độ + xG) |
| **M6** HistGB | `HistGradientBoostingClassifier` trên toàn bộ cột an toàn + `lam_diff`. Learning rate nhỏ, cây nông (depth 2–3), `min_samples_leaf=50`, L2 = 1, early stopping. Dò `lr` ∈ {0.02, 0.05} × `depth` ∈ {2, 3} |
| **M7** Blend | w · (LR tốt hơn giữa M3 và M5) + (1 − w) · Dixon–Coles. Dò w từ 0 đến 1, bước 0.05, trên dev. Kết quả: w = 0.25 |

**Bảng xếp hạng trên dev:**

| Mô hình | Log loss | RPS | Brier | Accuracy |
|---|---|---|---|---|
| M7 Blend | **0.9511** | 0.1956 | 0.5632 | 55.4% |
| M4 Dixon–Coles | 0.9512 | 0.1956 | 0.5633 | 55.7% |
| M3 LR 6 + lam_diff | 0.9526 | 0.1961 | 0.5641 | 55.5% |
| M5 LR+ | 0.9526 | 0.1960 | 0.5640 | 55.5% |
| M1 LR Elo | 0.9529 | 0.1960 | 0.5644 | 55.9% |
| M2 LR 6 cols | 0.9534 | 0.1964 | 0.5648 | 55.4% |
| M6 HistGB | 0.9597 | 0.1969 | 0.5679 | 56.1% |
| M0 Frequency | 1.0527 | 0.2308 | 0.6353 | 47.2% |
| *(tham chiếu) Nhà cái* | *0.9375* | | | |

**Quy tắc chọn** (`models.choose_final`):
1. Lấy mô hình có log loss dev thấp nhất, bỏ các mô hình nghi rò rỉ (log loss < 0.90).
2. Nếu mô hình đó không phải M3, so với M3 bằng paired bootstrap. Khi `PREFER_PDF_IF_NOT_SIGNIFICANT = True` mà CI chứa 0 thì **giữ M3**, vì cải thiện chưa chắc chắn và M3 dễ giải thích hơn.

Kết quả: M7 tốt nhất trên dev, nhưng Δ = −0.0015 với CI [−0.0054, +0.0023] chứa 0. Vì vậy **mô hình cuối là M3 – LR 6 cột + `lam_diff`**.

Nhận xét: các mô hình chênh nhau rất ít (0.951–0.953). Thêm đặc trưng hay dùng mô hình phức tạp hơn (HistGB) không giúp được gì, HistGB còn kém hơn vì dễ overfit.

---

## 12. Đánh giá trên Test

Học lại Poisson (tính lại `lam_diff` out-of-fold và ρ) và các mô hình trên **Train + Dev**, rồi đánh giá trên test **đúng một lần**:

| Mô hình | Log loss | RPS | Brier | Accuracy |
|---|---|---|---|---|
| M4 Dixon–Coles | 1.0091 | 0.2061 | 0.6053 | 50.2% |
| M7 Blend | 1.0093 | 0.2061 | 0.6055 | 49.9% |
| **M3 LR 6 + lam_diff (mô hình cuối)** | **1.0118** | **0.2069** | **0.6071** | **49.6%** |
| M2 LR 6 cols | 1.0132 | 0.2074 | 0.6081 | 49.4% |
| M0 Frequency | 1.0854 | 0.2323 | 0.6582 | 41.4% |
| *(tham chiếu) Nhà cái* | *0.9955* | *0.2016* | *0.5961* | *51.4%* |

**Theo từng mùa (mô hình cuối):**

| Mùa | Số trận | Log loss | RPS |
|---|---|---|---|
| 2024/25 | 380 | 0.9926 | 0.2045 |
| 2025/26 | 380 | 1.0249 | 0.2081 |
| 2026/27 | 80 | 1.0402 | 0.2121 |

**Dự đoán số bàn:** MAE bàn nhà 0.969, bàn khách 0.850. Đoán đúng chính xác tỉ số: 8.5%.

**Nhận xét:**
- Mô hình hơn hẳn đoán theo tần suất (log loss −0.074, accuracy +8.2 điểm phần trăm) và hơn mốc "luôn đoán đội nhà thắng" (41.4%).
- Log loss trên test (1.01) cao hơn dev (0.95) vì test có tỉ lệ hòa và đội khách thắng cao hơn bình thường, khó đoán hơn. Nhà cái cũng tăng từ 0.94 lên 1.00.
- Mô hình kém nhà cái khoảng 0.016 log loss, đúng như kỳ vọng với một mô hình chỉ dùng dữ liệu công khai.

**Biểu đồ hiệu chỉnh** (`reliability_test.png`): so sánh xác suất dự đoán với tần suất thực tế cho từng kết quả H, D, A. Đường càng gần đường chéo thì xác suất càng đáng tin.

---

## 13. Thử nghiệm phụ

Hai thử nghiệm này **không làm thay đổi** mô hình đã chọn hay file dự đoán. Bỏ qua bằng `--skip-experiments`.

### 13.1. Thêm kèo nhà cái (10b)

So sánh trên cùng 840 trận test có odds:

| Mô hình | Log loss | Accuracy | Δ so với nhà cái (CI 95%) |
|---|---|---|---|
| Nhà cái | 0.9955 | 51.4% | – |
| LR + odds (6 cột + `lam_diff` + `log(pH/pA)` + `log(pD)`, C = 1.0) | 1.0017 | 50.6% | +0.0062 [+0.001, +0.011] (kém hơn) |
| Mô hình cuối (không odds) | 1.0118 | 49.6% | +0.0163 [+0.005, +0.028] (kém hơn) |
| Blend mô hình + nhà cái | 0.9955 | 51.4% | w tốt nhất trên dev = 0, tức chỉ dùng nhà cái |

Khi có odds, hệ số của `odds_logit_ha` (±0.5–0.7) lấn át mọi đặc trưng khác. Kết luận: odds chứa gần như toàn bộ thông tin mà các đặc trưng của mình có, và nhiều hơn thế. Mô hình không thêm được gì cho nhà cái.

### 13.2. Accuracy theo năm bắt đầu train (10c)

Học lại cùng mô hình cuối với cùng siêu tham số, chỉ đổi mùa bắt đầu của tập train (sửa `START_YEARS` trong `config.py`):

| Train từ | Số trận train | Acc dev | Acc test | Log loss dev | Log loss test |
|---|---|---|---|---|---|
| 2000/01 | 8.270 | 55.5% | 49.6% | 0.9526 | 1.0118 |
| 2009/10 | 4.940 | 55.1% | 49.6% | 0.9507 | 1.0097 |

Chênh lệch accuracy trên test = 0.0%, CI [−0.6%, +0.6%], **không có ý nghĩa thống kê**. Bỏ bớt dữ liệu cũ làm log loss tốt lên rất nhẹ, nhưng không đáng kể.

---

## 14. Nhiệm vụ 4 – Xuất dự đoán

`predict.py`

Với mỗi trận, file dự đoán có các cột:

| Cột | Ý nghĩa |
|---|---|
| `match_id`, `season`, `date`, `gameweek`, `home_team`, `away_team` | Định danh (cột nào có trong dữ liệu thì giữ) |
| `p_home`, `p_draw`, `p_away` | Xác suất H/D/A của mô hình cuối |
| `pred_result` | Kết quả có xác suất cao nhất |
| `pred_home_goals`, `pred_away_goals` | Tỉ số có xác suất cao nhất **trong vùng khớp với `pred_result`** (H: nhà > khách, D: bằng nhau, A: nhà < khách) |
| `likeliest_score` | Tỉ số có xác suất cao nhất trên **toàn ma trận**. Có thể khác `pred_result`, thường là 1-1 |
| `exp_home_goals`, `exp_away_goals` | Số bàn kỳ vọng λ_nhà, λ_khách |
| `true_result`, `true_score` | Kết quả thật (chỉ có trong `predictions_test.csv`) |
| `model` | Tên mô hình (chỉ có trong `predictions.csv`) |

- **`predictions_test.csv`**: dự đoán trên tập test, mô hình học trên train + dev. Dùng để báo cáo và so với kết quả thật.
- **`predictions.csv`**: dự đoán các trận `upcoming`. Mô hình học trên train + dev, hoặc trên mọi trận đã đá nếu dùng `--refit-with-test`. **Lần chạy hiện tại không có trận upcoming** vì `features.csv` chỉ chứa trận đã đá, nên file này chưa được tạo. Khi dataset có thêm trận sắp đá (`status = upcoming`), chạy lại là có.
- **`final_models.joblib`**: mô hình cuối, 2 mô hình Poisson, ρ, alpha và danh sách cột. Chỉ được lưu khi có trận upcoming.

Tải lại mô hình đã lưu:

```python
import joblib
bundle = joblib.load("outputs/run_.../final_models.joblib")
bundle["final_model"], bundle["poisson_models"], bundle["rho"], bundle["poi_feats"]
```

---

## 15. File đầu ra

Mỗi lần chạy tạo `outputs/run_<YYYYMMDD>_<HHMM>_<odds|no_odds>/` và file `.zip` cùng tên, nên không bao giờ ghi đè lần chạy trước.

| File | Nội dung |
|---|---|
| `run_info.json` | Cấu hình và lựa chọn của lần chạy: mốc chia, alpha, ρ, C, mô hình cuối, log loss test |
| `hda_rate_by_season.png` | Tỉ lệ H/D/A theo mùa (thấy rõ COVID 2020/21 làm mất lợi thế sân nhà) |
| `hypothesis_tests.csv` / `.png` | Bảng và biểu đồ kiểm định H1–H9 |
| `blend_weight.png` | Log loss dev theo trọng số blend w |
| `test_metrics.csv` | Bảng kết quả trên test |
| `reliability_test.png` | Biểu đồ hiệu chỉnh xác suất của mô hình cuối trên test |
| `odds_experiment_test.csv` | Kết quả thử nghiệm odds |
| `accuracy_by_train_start.csv` / `.png` / `_per_season.csv` | Thử nghiệm năm bắt đầu train |
| `predictions_test.csv` | Dự đoán trên tập test |
| `predictions.csv`, `final_models.joblib` | Dự đoán trận sắp đá và mô hình (chỉ khi có trận upcoming) |

---

## 16. Mô tả từng module

| Module | Hàm / lớp chính | Mô tả |
|---|---|---|
| `config.py` | `make_run_dir()` | Hằng số: đường dẫn, mốc chia, `USE_ODDS_FEATURE`, `REFIT_WITH_TEST_FOR_PRODUCTION`, lưới siêu tham số, ngưỡng rò rỉ, `EVAL_SEASONS`, `START_YEARS` |
| `data.py` | `locate_features`, `load_features`, `quick_eda`, `split_by_season`, `show_tree` | Tìm/tải dữ liệu, sắp theo ngày, EDA, chia tập và kiểm tra thứ tự thời gian |
| `features.py` | `select_candidates`, `leakage_check`, `Odds`, `find_odds`, `add_odds_features`, `FeatureSets`, `build_feature_sets` | Chọn cột, kiểm tra rò rỉ, xử lý odds, gom các bộ đặc trưng vào `FeatureSets` |
| `preprocessing.py` | `ClipPercentile`, `make_prep`, `lr_pipe` | Pipeline tiền xử lý |
| `metrics.py` | `ll_vec`, `rps_vec`, `evaluate`, `paired_bootstrap` | Thước đo và bootstrap |
| `poisson_dc.py` | `tune_alpha`, `prepare_stage`, `oof_lams`, `score_matrix`, `matrix_to_hda`, `fit_rho`, `REGION` | Mô hình tỉ số |
| `models.py` | `fit_predict`, `tune_lr`, `tune_hgb`, `build_dev_leaderboard`, `choose_final` | Mô tả mô hình bằng dict `spec` (`kind` = `prior` / `lr` / `dc` / `hgb` / `blend`), huấn luyện, xếp hạng, chọn mô hình |
| `hypothesis.py` | `build_hypotheses`, `rolling_origin`, `run_hypothesis_tests`, `accepted_features` | Kiểm định H1–H9 |
| `experiments.py` | `odds_experiment`, `accuracy_by_train_start` | Hai thử nghiệm phụ |
| `predict.py` | `score_table`, `write_test_predictions`, `predict_upcoming` | Xuất dự đoán |
| `plots.py` | `plot_*` | Vẽ và lưu PNG (backend `Agg`, không mở cửa sổ) |
| `train_eval.py` | `main`, `parse_args` | Ghép mọi bước theo thứ tự 0 → 12 |

**Dùng lại từng phần trong notebook hoặc script khác:**

```python
from modeling.data import locate_features, load_features, played_matches, split_by_season
from modeling.poisson_dc import prepare_stage, score_matrix
from modeling.models import fit_predict
from modeling.metrics import evaluate

df, date_col = load_features(locate_features(), verbose=False)
train, dev, test, upcoming = split_by_season(df, played_matches(df), date_col)
feats = ["elo_diff", "home_goals_for5", "home_goals_against5", "away_goals_for5", "away_goals_against5"]
tr_s, (dev_s,), _, rho = prepare_stage(train, [dev], alpha=0.01, feats=feats)
P, model = fit_predict({"kind": "lr", "feats": ["elo_diff", "lam_diff"], "C": 0.01}, tr_s, dev_s, rho)
print(evaluate(P, dev_s.result, "LR elo + lam_diff"))
```

---

## 17. Tùy chỉnh

| Muốn… | Sửa ở đâu |
|---|---|
| Đổi mốc chia train/dev/test | `TRAIN_START`, `TRAIN_END`, `DEV_START`, `DEV_END`, `TEST_START` trong `config.py` |
| Dùng odds làm đặc trưng | `--use-odds` hoặc `USE_ODDS_FEATURE = True` |
| Luôn chọn mô hình tốt nhất trên dev, kể cả khi không có ý nghĩa thống kê | `PREFER_PDF_IF_NOT_SIGNIFICANT = False` |
| Đổi lưới siêu tham số | `POISSON_ALPHAS`, `LR_C_GRID`, `HGB_GRID` |
| Đổi ngưỡng rò rỉ | `LEAK_THRESHOLD`, `SUSPICIOUS_LL` |
| Đổi các mùa dùng để kiểm định | `EVAL_SEASONS` |
| Thử năm bắt đầu train khác | `START_YEARS` |
| Đổi bộ cột của LR / Poisson | `pdf_feature_sets` trong `features.py` |
| Sửa nhóm cột của giả thuyết | `build_hypotheses` trong `hypothesis.py` |
| Thêm loại mô hình mới | Thêm nhánh `kind` mới trong `models.fit_predict`, rồi thêm vào `specs` trong `build_dev_leaderboard` |
| Chỉ định tay cột odds | Thay `find_odds(df)` trong `train_eval.py` bằng `Odds(("B365H", "B365D", "B365A"), is_prob=False)` |

---

## 18. Lưu ý và hạn chế

- **Test chỉ dùng một lần.** Nếu xem kết quả test rồi quay lại sửa mô hình, điểm test không còn khách quan nữa.
- **Tập test có 80 trận của mùa 2026/27** (mùa đang đá), vì `TEST_START = 2024` không có mốc kết thúc. Muốn test đúng 2024/25 → 2025/26 thì cần thêm điều kiện lọc trong `data.split_name`.
- **Chưa có trận upcoming** trong `features.csv` hiện tại, nên `predictions.csv` chưa được tạo.
- **Đặc trưng FotMob thiếu nhiều** (60–78%), chỉ có ở những mùa gần đây. Vì vậy H8 chưa kết luận được.
- **Mùa COVID 2019/20 – 2020/21** (không khán giả) làm lợi thế sân nhà giảm mạnh. `league_home_adv380` nhằm nắm bắt điều này nhưng chưa cho cải thiện có ý nghĩa (H5).
- **Khi bật `--use-odds`**, trận sắp đá **phải** có odds. Thiếu odds thì giá trị bị điền trung vị và dự đoán kém tin cậy.
- **Các mô hình chênh nhau rất nhỏ** (khoảng 0.001–0.002 log loss). Nên báo cáo M3 làm mô hình chính và trình bày M4–M7 như thí nghiệm bổ sung.
- **Kết quả tái lập được:** `SEED = 42`. LR và Poisson là thuật toán tất định. HistGB có `random_state` cố định.
- **So với notebook gốc:** con số giống hệt nhau. Khác biệt chỉ là biểu đồ được lưu PNG thay vì hiển thị, kết quả ghi vào `outputs/` thay vì Google Drive, và tên cột / nhãn đổi sang tiếng Anh (ví dụ `verdict = USEFUL` thay cho `kết luận = CÓ ÍCH`).
