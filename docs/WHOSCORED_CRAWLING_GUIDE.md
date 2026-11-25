# Hướng Dẫn Crawl Dữ Liệu Sự Kiện Từ WhoScored (Opta Event Stream)

Tài liệu này mô tả chi tiết cơ chế hoạt động, cách vượt qua lớp bảo vệ Cloudflare hợp lệ, cấu trúc dữ liệu `matchCentreData`, và hướng dẫn vận hành bộ công cụ tự động trong thư mục `scripts/`.

---

## 1. Tại sao HTTP thông thường (`requests`, `curl`, `tls_requests`) bị chặn?

* Các trang lịch thi đấu (`/Regions/.../Tournaments/...`) thường trả về `HTTP 200`, nhưng các trang chi tiết trận đấu (`/matches/{id}/live`) được bảo vệ bởi **Cloudflare Bot Management (Turnstile)**.
* Khi gửi request bằng `requests`, `curl_cffi`, `tls_requests`, hoặc trình duyệt headless mặc định (`navigator.webdriver = true`, không có lịch sử cookie/WebGL thật), Cloudflare sẽ trả về `HTTP 403 Forbidden` kèm trang chờ `"Just a moment..."`.

### Cơ chế xử lý thành công (Kế thừa từ `WhoScored_auto_scraper` & `soccerdata`)
1. **Dùng trình duyệt Chrome thật (`channel="chrome"`):** Khởi chạy chính file thực thi Google Chrome cài đặt trên máy (`headless=False`), giữ nguyên chứng chỉ TLS, WebGL, fonts và audio fingerprint thật.
2. **Ẩn dấu vết tự động hóa:**
   * Thêm cờ `--disable-blink-features=AutomationControlled`.
   * Tiêm JS xóa `navigator.webdriver` (`Object.defineProperty(navigator, 'webdriver', { get: () => undefined })`).
3. **Persistent Browser Profile (`launch_persistent_context`):** Lưu toàn bộ cookie (`cf_clearance`, `__cf_bm`), localStorage và cache vào thư mục cố định (`mapped_epl/downloaded_matches/.browser_profile`). Nếu Cloudflare yêu cầu xác thực lần đầu, người dùng chỉ cần bấm xác nhận 1 lần trên giao diện Chrome đang mở; toàn bộ các lượt tải sau sẽ tự động tái sử dụng phiên hợp lệ.
4. **Cầu nối CDP (`--cdp 9222`):** Cho phép script gắn trực tiếp vào cửa sổ Chrome người dùng đang mở sẵn (qua `--remote-debugging-port=9222`) để điều hướng và trích xuất DOM tức thì.

---

## 2. Cấu trúc dữ liệu bên trong WhoScored

### A. Khám phá lịch thi đấu theo mùa (`discover_season_schedule`)
* **Giải Ngoại Hạng Anh (EPL):** `Region = 252`, `Tournament = 2` -> `https://www.whoscored.com/Regions/252/Tournaments/2/England-Premier-League`.
* Từ trang giải đấu, lấy `stage_id` (ví dụ mùa `2026/27` có `stage_id = 25544`).
* Endpoint JSON lịch thi đấu theo từng tháng (gọi bằng `fetch(..., { credentials: 'include' })` ngay bên trong phiên trình duyệt):
  ```text
  https://www.whoscored.com/tournaments/{stage_id}/data/?d=YYYYMM
  ```
  Ví dụ: `d=202608` (tháng 8/2026), `d=202610` (tháng 10/2026).

### B. Dữ liệu sự kiện trận đấu (`matchCentreData`)
Trên mỗi trang `/matches/{match_id}/live`, toàn bộ dữ liệu Opta được nhúng trực tiếp trong biến JS:
```javascript
window.require.config.params['args'].matchCentreData
```
Cấu trúc JSON gồm:
* `home` / `away`: Tên đội, `teamId`, đội hình (`formations`), danh sách cầu thủ (`players`).
* `playerIdNameDictionary`: Bảng ánh xạ `{ "playerId": "Player Name" }`.
* `events`: Mảng ~1.400–1.800 sự kiện trên bóng mỗi trận, mỗi sự kiện có:
  * `minute`, `second`, `expandedMinute`, `period` (`FirstHalf`, `SecondHalf`)
  * `type.displayName` (`Pass`, `Tackle`, `Interception`, `Clearance`, `BallRecovery`, `TakeOn`, `Aerial`, `Miss`, `Post`, `AttemptSaved`, `Goal`,...)
  * `outcomeType.displayName` (`Successful` / `Unsuccessful`)
  * `x`, `y`, `endX`, `endY` (tọa độ chuẩn Opta `0..100` theo chiều dài và chiều rộng sân)
  * `qualifiers` (góc chuyền, độ dài, `PassEndX`, `PassEndY`, `Zone`,...)

---

## 3. Hướng dẫn sử dụng các script

### 1) Tải 1 trận đấu đơn lẻ ([scripts/whoscored_downloader.py](file:///z:/Desktop/1x2/scripts/whoscored_downloader.py))
* **Tải bằng Persistent Profile (tự động mở Chrome):**
  ```bash
  python scripts/whoscored_downloader.py --url "https://www.whoscored.com/matches/1983821/live/england-premier-league-2026-2027-manchester-city-fulham" --output-dir "mapped_epl/downloaded_matches"
  ```
  Hoặc chỉ cần truyền `--match-id`:
  ```bash
  python scripts/whoscored_downloader.py --match-id 1983821
  ```
* **Tải qua CDP kết nối Chrome đang mở sẵn:**
  ```bash
  chrome.exe --remote-debugging-port=9222
  python scripts/whoscored_downloader.py --cdp 9222 --match-id 1983821
  ```

### 2) Quét hàng loạt toàn bộ mùa giải 2026/27 ([scripts/whoscored_season_scraper.py](file:///z:/Desktop/1x2/scripts/whoscored_season_scraper.py))
* **Chạy lần đầu / tiếp tục tải phần còn thiếu** (lưu vào `mapped_epl/whoscored_2627/`, tự động bỏ qua trận đã tải):
  ```bash
  python scripts/whoscored_season_scraper.py
  ```
* **Cập nhật khi có vòng đấu mới diễn ra** (tải lại `schedule_2627.json` và chỉ tải thêm các trận mới đá xong):
  ```bash
  python scripts/whoscored_season_scraper.py --refresh-schedule
  ```
* **Chạy thử nghiệm giới hạn số trận:**
  ```bash
  python scripts/whoscored_season_scraper.py --limit 3
  ```

### 3) Nạp sự kiện Opta vào SQLite ([mapped_epl/ingest_whoscored_events.py](file:///z:/Desktop/1x2/mapped_epl/ingest_whoscored_events.py))
Nạp file `.html` hoặc `_matchCentreData.json` vào bảng `clean_match_events_opta` trong `mapped_epl/epl_clean.db`, tự động khớp `club_id` và `player_key` (`dim_player`):
```bash
python mapped_epl/ingest_whoscored_events.py "mapped_epl/downloaded_matches/whoscored_1983821.html"
```

---

## 4. Lưu ý vận hành & An toàn
* **Rate Limiting:** Script mặc định nghỉ ngẫu nhiên `2.0–3.5 giây` giữa mỗi trận để tránh kích hoạt giới hạn tốc độ từ máy chủ WhoScored.
* **Dữ liệu đã lưu:** Toàn bộ 107 trận đã hoàn thành của EPL 2026/27 (tính đến `2026-11-22`) đã được tải và lưu trữ tại `mapped_epl/whoscored_2627/` (gồm cả `.html` gốc và `_matchCentreData.json`).
