# Quá trình Tiến hóa Mô hình: Từ Baseline đến Champion Hợp nhất (EPL 2026/27)

## Giai đoạn 0: Vấn đề cốt lõi
Dữ liệu mùa 2026/27 chỉ có `N = 107 → 110` trận, tỷ lệ tín hiệu/nhiễu (SNR) cực thấp vì kết quả thắng/thua 1 trận chịu nhiều may rủi (bóng đập cột, penalty phút bù giờ).

## Bước 1: Dựng Thang Đo Chuẩn & Chứng minh bẫy Overfit (`run_benchmark.py`, `N=77`)
- **Hướng suy luận:** Phải có những mô hình cực kỳ đơn giản ("tầm thường") để làm chuẩn so sánh; nếu mô hình phức tạp thua baseline thì vứt bỏ.
- **Thí nghiệm & Kết quả:**
  - Đoán theo tỷ lệ lịch sử (`B0_NaivePrior`): RPS = `0.2299`
  - Chỉ dùng Bảng xếp hạng (`B1_TablePPG`, điểm + hiệu số): RPS = `0.2301` (không thắng nổi đoán mù vì bảng điểm đầu mùa nhiễu lịch thi đấu).
  - Cây `XGBoost` với 12 feature lăn (`B4_XGB_OverfitDiag`): RPS = **`0.2684`**, LogLoss = `1.3870`, ECE = `0.2589` → **Thất bại nặng nề vì học vẹt trên mẫu nhỏ**.
- **Quyết định:** Loại bỏ hoàn toàn mô hình cây nhiều feature; chỉ giữ mô hình thống kê ít tham số có điều chuẩn.

## Bước 2: Dùng Sức mạnh Lịch sử & Kiểm tra Lệch Phân phối (`shift_analysis.py`, `run_ablation_history.py`)
- **Hướng suy luận:** Với `N = 110`, nếu bắt đầu mùa mới từ con số 0 (`1500` cho mọi đội) thì 30 trận đầu dự đoán rất tệ. Nhưng trộn thẳng dữ liệu mùa cũ có bị lệch phân phối không?
- **Thí nghiệm & Kết quả:**
  - Kiểm định thống kê `2021/22–2025/26` (`N=1,900`) vs `2026/27` (`N=107`): Tỷ lệ Thắng/Hòa/Thua (`chi2 p=0.620`) và Bàn thắng/Lợi thế sân nhà (`KS p=0.261 / 0.997`) **không lệch**, nhưng Số cú sút/trận (`p=0.0023`) **có lệch** → Không được trộn feature đếm thô từ mùa cũ, chỉ lấy **quy luật giải** và **điểm Elo co về trung bình hè (`shrink=0.35`)**.
  - Ablation: Elo không lịch sử đạt `0.2405` → thêm quy luật giải giảm còn `0.2224` → thêm Elo đầu mùa giảm mạnh còn **`0.2099`** (`p_beat = 99.6%`).
  - Thử tối ưu quá sát 4 tham số Elo trên 4 mùa cũ (`K=29.9, HFA=110, shrink=0.10`): chỉ nhỉnh ở 30 trận đầu (`0.1883` vs `0.1891`) nhưng tệ hơn về sau (`0.2192` vs `0.2180`, ECE kém gấp đôi) → **Giữ bộ tham số bảo thủ (`K=20, HFA=65, shrink=0.35, promo=1420`)**.

## Bước 3: Khử nhiễu bằng Không gian Trạng thái Bayes + Học Phân phối (`bayesian_models.py`)
- **Hướng suy luận:** Thay vì đoán nhãn Thắng/Hòa/Thua rời rạc, hãy dự đoán hiệu suất tấn công/phòng ngự đã khử nhiễu (`Goals + Shot_xG + Opta xT`), hiệu chỉnh theo độ khó đối thủ bằng Bayesian Ridge, rồi lấy mẫu Monte Carlo qua phân phối hậu nghiệm để ra xác suất kèm độ bất định tự nhiên.
- **Thí nghiệm & Kết quả:**
  - Chuyển sang Bayesian Posterior đạt `RPS = 0.2137` (`N=77`) vs Elo `0.2180`.
  - Kiểm chứng độ bất định: Nhóm trận bất định thấp đạt `RPS = 0.2032` (`Acc 48.1%`) vs nhóm bất định cao `RPS = 0.2156` (`Acc 39.6%`). Thử Negative Binomial thấy `alpha_nb = 1e-05` → Poisson + Dixon-Coles đủ chuẩn.

## Bước 4: Đổi Công thức Elo (`Elo_Formula_xG`) & Tái huấn luyện Tham lam (`run_greedy_retrain.py`)
- **Hướng suy luận:**
  1. Giữ nguyên `K=20, HFA=65, shrink=0.35`, nhưng cập nhật Elo bằng hiệu suất liên tục (`50% Goals + 50% xG/xT`) thay vì Thắng/Thua thô để giảm nhiễu ngay trong Elo.
  2. Mỗi 5 trận, chạy Grid Search rời rạc `3×3×3` trên quá khứ 26/27 để chọn độ co và trọng số thích nghi với mùa giải.
  3. Đồng bộ đủ `110/110` trận + `110/110` Kèo Nhà cái và tự train `Shot-xG` từ `114,840` cú sút lịch sử (`r = 0.9071` với FotMob `fm_npxg`).
- **Kết quả (`N = 110` & `N = 80`):**
  - Toàn mùa (`N = 110`): `Elo_Formula_xG` đạt `RPS = 0.2089`, `LogLoss = 1.0223` (sát Kèo Nhà cái `0.2082`).
  - Vòng 4–11 (`N = 80`): `BayesSS_Greedy_Every5` đạt **`RPS = 0.2135`** (vượt Kèo Nhà cái `0.2157`), tự chuyển trọng số sang `(0.20 Goals, 0.40 Shot_xG, 0.40 xT)`.

## Bước 5: Hợp nhất thành Champion Chốt (`Champion_Unified` trong `champion/`)
- **Hướng suy luận:** Lấy thế mạnh đầu mùa của `Elo_Formula_xG` làm prior và blend mượt (`e^{-n/25}`) với `BayesSS_Greedy5_xGPrior` về sau — không thêm tham số tự do mới.
- **Kết quả kiểm chứng cuối (`110/110` trận):**
  - Toàn mùa (`N = 110`): `RPS = 0.2094`, `Acc = 47.27%`, **`ECE = 0.0323`** (hiệu chuẩn tốt hơn Nhà cái `0.0346`).
  - Vòng 4–11 (`N = 80`): **`RPS = 0.2147`** (vượt Nhà cái `0.2157`), `LogLoss = 1.0268`, `Acc = 48.75%` (ngang Nhà cái), `ECE = 0.0251`.
