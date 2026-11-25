# Dự đoán kết quả Premier League – Team 5

Project dự đoán **kết quả Thắng/Hòa/Thua (H/D/A)** và **tỉ số** các trận Premier League. Project gồm hai phần:

1. **Thu thập và tiền xử lý dữ liệu**: crawl kết quả và thống kê từng trận, rồi tính thuộc tính cho mỗi trận.
2. **Mô hình hóa** (package `modeling/`): chia train/dev/test theo mùa, huấn luyện Poisson + Dixon–Coles và Logistic Regression, kiểm định giả thuyết, đánh giá trên tập test và xuất file dự đoán.

## Cấu trúc thư mục

```
Project-Datascience-Team-5/
├── README.md                           # file này
├── Data_crawling_and_Preprocessing.md  # chi tiết phần thu thập và tiền xử lý
├── Modeling.md                         # chi tiết phần mô hình hóa
│
├── crawl_epl_season.py                 # crawl thống kê theo mùa (Understat, ESPN, Transfermarkt)
├── fetch_fixtures.py                   # lịch thi đấu và vòng tiếp theo
├── update_epl_final.py                 # cập nhật kết quả từng trận vào epl_final.csv
├── build_features.py                   # tính 51 thuộc tính cho mỗi trận
├── build_dataset.py                    # chạy cả quy trình, xuất data/epl_dataset.csv
│
├── modeling/                           # pipeline mô hình hóa (tách từ EPL_train_dev_test_.ipynb)
│   ├── config.py                       # đường dẫn, mốc chia tập, cờ bật/tắt, lưới siêu tham số
│   ├── data.py                         # tìm/tải features.csv, EDA, chia train/dev/test
│   ├── features.py                     # chọn cột, kiểm tra rò rỉ, đặc trưng odds
│   ├── preprocessing.py                # Imputer → ClipPercentile → StandardScaler
│   ├── metrics.py                      # log loss, RPS, Brier, accuracy, bootstrap
│   ├── poisson_dc.py                   # Poisson + Dixon–Coles, lam_diff out-of-fold
│   ├── models.py                       # các mô hình M0–M7, dò siêu tham số, chọn mô hình
│   ├── hypothesis.py                   # kiểm định H1–H9 (rolling-origin)
│   ├── experiments.py                  # thử nghiệm odds, thử năm bắt đầu train
│   ├── predict.py                      # xuất predictions_test.csv, predictions.csv
│   ├── plots.py                        # biểu đồ (lưu PNG)
│   └── train_eval.py                   # script chính, chạy từ đầu đến cuối
│
├── data/                               # dữ liệu (không đưa lên git, tạo lại bằng script)
└── outputs/                            # kết quả mỗi lần chạy mô hình (không đưa lên git)
```

## Cài đặt

Cần Python 3.9 trở lên.

```bash
pip install requests pandas numpy lxml scipy scikit-learn matplotlib joblib huggingface_hub
```

## Chạy nhanh

**Phần 1: cập nhật dữ liệu** (sau mỗi vòng đấu)

```bash
python build_dataset.py              # cập nhật kết quả mới, tính thuộc tính → data/epl_dataset.csv
python crawl_epl_season.py 2026      # (tùy chọn) thống kê tổng hợp của mùa đang đá
```

**Phần 2: huấn luyện, đánh giá và dự đoán** (chạy từ thư mục gốc)

```bash
python -m modeling.train_eval                     # tự tìm features.csv, không có thì tải từ Hugging Face
python -m modeling.train_eval --features path/to/features.csv
python -m modeling.train_eval --use-odds          # dùng kèo nhà cái làm đặc trưng
python -m modeling.train_eval --skip-experiments  # bỏ qua 2 thử nghiệm phụ (chạy nhanh hơn)
```

Mỗi lần chạy ghi kết quả vào `outputs/run_<ngày>_<giờ>_<odds|no_odds>/` kèm file `.zip`. Chạy trọn bộ mất khoảng vài phút.

## Kết quả chính (lần chạy không dùng odds)

| | Log loss | RPS | Accuracy |
|---|---|---|---|
| Đoán theo tần suất (M0) | 1.085 | 0.232 | 41.4% |
| **Mô hình cuối: LR 6 cột + `lam_diff` (M3)** | **1.012** | **0.207** | **49.6%** |
| Dixon–Coles (M4) | 1.009 | 0.206 | 50.2% |
| Nhà cái (chỉ để tham chiếu) | 0.996 | 0.202 | 51.4% |

Các số trên tính trên tập test: 840 trận từ mùa 2024/25 đến nay. Giải thích chi tiết ở [Modeling.md](Modeling.md).

## Tài liệu

- [Data_crawling_and_Preprocessing.md](Data_crawling_and_Preprocessing.md): nguồn dữ liệu, từng script crawl, cách tính thuộc tính.
- [Modeling.md](Modeling.md): cách chia dữ liệu, các mô hình, kiểm định giả thuyết, kết quả, file đầu ra, cách tùy chỉnh.
