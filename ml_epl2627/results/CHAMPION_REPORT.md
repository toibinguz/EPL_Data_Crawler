# Báo cáo Kiểm chứng Mô hình Champion Hợp nhất (EPL 2026/27 — 110 trận)

## 1. Cấu trúc Mô hình Champion Hợp nhất (`Champion_Unified`)
Mô hình kết hợp mượt hai thành phần theo số trận đã đấu ở mùa 2026/27 (`w_elo = exp(-số_trận_đã_đấu / 25)`):
1. **Thành phần Elo cập nhật bằng Bàn thắng Kỳ vọng (`Elo_Formula_xG`, chiếm ưu thế ở 3 vòng đầu):**
   - Khởi tạo từ lịch sử Ngoại hạng Anh `2021/22–2025/26` (học hệ số co về trung bình `0.35`, đội mới lên hạng `1420`) + hàm Ordered Logistic.
   - Sau mỗi trận, cập nhật Elo bằng hiệu số hiệu suất kỳ vọng liên tục (`50% bàn thắng thật + 50% hiệu suất tạo cơ hội từ xG và xT`).
2. **Thành phần Không gian Trạng thái Bayes tái huấn luyện mỗi 5 trận (`BayesSS_Greedy5_xGPrior`, chiếm ưu thế từ vòng 4 trở đi):**
   - Biến mục tiêu khử nhiễu: `Hiệu_suất = w_g * Bàn_thắng + w_xg * Bàn_thắng_kỳ_vọng_Shot_xG + w_xt * (0.90 * Kỳ_vọng_đe_dọa_xT)`.
   - Tối ưu lại trọng số mỗi 5 trận chỉ trên các trận quá khứ trong mùa 2026/27.
   - Sinh ra phân phối xác suất `(Thắng_nhà, Hòa, Thắng_khách)` kèm độ lệch chuẩn bất định tự nhiên qua 300 lần lấy mẫu Monte Carlo từ ma trận hiệp phương sai hậu nghiệm + mô hình tỷ số Poisson có hiệu chỉnh Dixon-Coles.

## 2. Kết quả kiểm chứng Walk-Forward trên 110 trận EPL 2026/27
- **Toàn mùa (`N = 110`, đối chiếu 100% với kèo nhà cái):**
  - Kèo nhà cái (`B5_MarketOdds`): RPS `0.2082`, LogLoss `1.0240`, Brier `0.6148`, Độ chính xác `48.18%`, Sai số hiệu chuẩn `0.0346`.
  - **`Elo_Formula_xG`**: RPS `0.2089`, LogLoss **`1.0223`**, Brier **`0.6135`**, Độ chính xác `46.36%`, Sai số hiệu chuẩn `0.0490`.
  - **`Champion_Unified`**: RPS **`0.2094`**, LogLoss `1.0274`, Brier `0.6160`, Độ chính xác **`47.27%`**, Sai số hiệu chuẩn **`0.0323`** (hiệu chuẩn tốt hơn cả nhà cái).
- **Sau giai đoạn khởi động (`N = 80`, Trận `#31–#110`):**
  - `BayesSS_Greedy_Every5`: RPS **`0.2135`**, LogLoss **`1.0237`**, Brier **`0.6135`**, Sai số hiệu chuẩn `0.0219`.
  - **`Champion_Unified`**: RPS **`0.2147`**, LogLoss `1.0268`, Brier `0.6156`, Độ chính xác **`48.75%`** (ngang nhà cái), Sai số hiệu chuẩn `0.0251`.
  - Kèo nhà cái (`B5_MarketOdds`): RPS `0.2157`, LogLoss `1.0298`, Brier `0.6204`, Độ chính xác `48.75%`, Sai số hiệu chuẩn `0.0258`.
