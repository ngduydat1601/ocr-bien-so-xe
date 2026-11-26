# Hệ thống Nhận diện Biển số Xe Việt Nam (AI LPR — YOLO + SVTR26 Fine-tuned)

Dự án nhận diện biển số **Ô tô & Xe máy Việt Nam** thời gian thực (Ảnh đơn, Thư mục hàng loạt tốc độ cao và Video Tracking HUD), đạt **97.43%** trên tập Ô tô độc lập và **98.65%** trên tập Xe máy thực tế (~8–14 ms/ảnh trên GPU).

---

## 1. Cài đặt & Chạy nhanh (cho thành viên mới clone repo)

```powershell
# 1. Tạo môi trường ảo Python 3.10 / 3.11
python -m venv .venv
.\.venv\Scripts\Activate.ps1

# 2. Cài đặt thư viện
pip install -r yeu_cau_thu_vien.txt

# 3. Chạy Web Demo đơn giản (cổng 8503)
streamlit run web_don_gian.py --server.port 8503
```

---

## 2. Cấu trúc thư mục chính

```text
ocr_bien_so_xe/
├── web_don_gian.py                 # Giao diện Web Demo (Streamlit)
├── luu_csdl.py                     # 👈 ĐIỂM CẮM CHO NGƯỜI PHỤ TRÁCH CSDL (chỉ cần code trong file này)
├── nguon/                          # Mã nguồn lõi đã tách theo từng chức năng
│   ├── cau_hinh.py                 # Đường dẫn, tên model mặc định, từ điển OCR & tham số xử lý
│   ├── tien_xu_ly.py               # Đọc ảnh Unicode, chuẩn hóa & kiểm tra định dạng biển số VN (Ô tô + Xe máy)
│   ├── mo_hinh_ai.py               # Nạp model GPU & chạy suy luận (YOLO, Gộp 2 Detector NMS, SVTR26 ONNX)
│   ├── xu_ly_khung_hinh.py         # Xử lý khung hình đơn lẻ & xử lý hàng loạt đa luồng + Batch GPU
│   └── xu_ly_video_hud.py          # Theo dõi phương tiện ByteTrack + Khử lóa & Bảng HUD thời gian thực cho Video
├── mo_hinh/
│   ├── svtr26_hon_hop.onnx         # 🌟 Model SVTR26-Tiny OCR mặc định (Đa năng Ô tô + Xe máy)
│   ├── litealpr_yolov8n_eff.pt     # Model YOLO phát hiện biển số mặc định (GPU FP16, siêu nhẹ)
│   ├── yolo_bien_so_vn.pt          # Model YOLO phát hiện biển số (hỗ trợ bắt cả ô biển góc màn hình)
│   ├── yolo26n.pt                  # Model YOLO phát hiện & theo dõi phương tiện cho chế độ Video HUD
│   ├── best.onnx                   # Model SVTR26-Tiny OCR chuyên Xe máy (dự phòng)
│   └── best.onnx.data              # Trọng số đi kèm best.onnx
├── huan_luyen_litealpr/            # Các Notebook Google Colab hướng dẫn Fine-tune OCR & YOLO Detect
└── yeu_cau_thu_vien.txt            # Danh sách gói Python cần cài
```

---

## 3. Luồng dữ liệu & Hợp đồng bàn giao cho phần CSDL (Data Contract)

```text
Ảnh / Video Frame
   └─▶ xu_ly_mot_khung_hinh() [web_don_gian.py]
         ├─ kq["anh_goc"]             (numpy BGR — ảnh toàn cảnh sạch chưa vẽ khung)
         └─ item trong kq["cac_ket_qua"]:
              ├─ item["crop_goc"]     (numpy BGR — ảnh crop sát biển số)
              └─ item["du_lieu_db"]   (dict 10 trường chuẩn tên cột DB)
                     └─▶ luu_ket_qua_bien_so(item["du_lieu_db"], kq["anh_goc"], item["crop_goc"])
                           ├─ Tự lưu ảnh vào du_lieu/anh_luu/YYYY/MM/DD/full & crop
                           └─ INSERT 15 trường vào Database (SQLite / PostgreSQL)
```

### Bảng đặc tả 15 trường Database

| # | Tên cột | Kiểu dữ liệu | Nguồn tạo | Mô tả |
|---:|---|---|---|---|
| 1 | `id` | `INTEGER / BIGINT` | **DB tự sinh** | Khóa chính tự tăng (`AUTOINCREMENT` / `BIGSERIAL`) |
| 2 | `captured_at` | `TIMESTAMP` | **Backend/DB** | Thời điểm chụp (`YYYY-MM-DD HH:MM:SS`), đánh index |
| 3 | `plate_raw` | `VARCHAR(20)` | **AI (`du_lieu_db`)** | Chuỗi OCR nguyên bản trước chuẩn hóa (`29G216139`) |
| 4 | `plate_normalized` | `VARCHAR(20)` | **AI (`du_lieu_db`)** | Biển số chuẩn hóa (`29G2-161.39`), đánh index |
| 5 | `format_status` | `VARCHAR(15)` | **AI (`du_lieu_db`)** | Trạng thái định dạng: `VALID` / `UNCERTAIN` / `INVALID` |
| 6 | `det_confidence` | `REAL / FLOAT` | **AI (`du_lieu_db`)** | Độ tin cậy YOLO phát hiện biển số (`0.0 – 1.0`) |
| 7 | `ocr_confidence` | `REAL / FLOAT` | **AI (`du_lieu_db`)** | Độ tin cậy OCR đọc ký tự (`0.0 – 1.0`) |
| 8 | `det_latency_ms` | `REAL / FLOAT` | **AI (`du_lieu_db`)** | Thời gian YOLO detect (ms) |
| 9 | `ocr_latency_ms` | `REAL / FLOAT` | **AI (`du_lieu_db`)** | Thời gian OCR đọc ký tự cho biển số này (ms) |
| 10 | `total_latency_ms` | `REAL / FLOAT` | **AI (`du_lieu_db`)** | Tổng độ trễ xử lý biển số (`det + ocr`, ms) |
| 11 | `model_version` | `VARCHAR(100)` | **AI (`du_lieu_db`)** | Phiên bản model (`yolo_bien_so_vn.pt + svtr26_vn`) |
| 12 | `plate_bbox` | `TEXT / JSONB` | **AI (`du_lieu_db`)** | Tọa độ hộp biển số `[x1, y1, x2, y2]` |
| 13 | `image_full_path` | `TEXT` | **`luu_csdl.py`** | Đường dẫn tương đối ảnh toàn cảnh (`YYYY/MM/DD/full/...jpg`) |
| 14 | `image_crop_path` | `TEXT` | **`luu_csdl.py`** | Đường dẫn tương đối ảnh crop (`YYYY/MM/DD/crop/...jpg`) |
| 15 | `is_correct` | `BOOLEAN (NULL)` | **Đối soát sau** | Mặc định `NULL`; cập nhật `True/False` khi kiểm tra đúng/sai |

### Schema SQL mẫu (dùng cho cả SQLite và PostgreSQL)

```sql
CREATE TABLE lich_su_bien_so (
    id                INTEGER PRIMARY KEY AUTOINCREMENT, -- Đổi thành BIGSERIAL nếu dùng PostgreSQL
    captured_at       TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    plate_raw         VARCHAR(20),
    plate_normalized  VARCHAR(20),
    format_status     VARCHAR(15),
    det_confidence    REAL,
    ocr_confidence    REAL,
    det_latency_ms    REAL,
    ocr_latency_ms    REAL,
    total_latency_ms  REAL,
    model_version     VARCHAR(100),
    plate_bbox        TEXT,
    image_full_path   TEXT,
    image_crop_path   TEXT,
    is_correct        BOOLEAN
);

CREATE INDEX idx_plate_time ON lich_su_bien_so (plate_normalized, captured_at);
```

---

## 4. Hướng dẫn dành riêng cho người phụ trách CSDL

1. Tạo nhánh riêng: `git checkout -b feat/database`.
2. Mở file **`luu_csdl.py`** — hàm `luu_anh_ra_o_dia(...)` và `chuan_bi_ban_ghi_day_du(...)` đã được viết sẵn, bạn chỉ cần viết code kết nối DB và câu lệnh `INSERT` / `UPDATE` vào đúng 2 hàm:
   - `luu_ket_qua_bien_so(du_lieu_db, anh_goc, anh_crop)`
   - `cap_nhat_doi_soat(record_id, is_correct)`
3. Mở giao diện web (`http://localhost:8503`), bấm chọn 1 ảnh và mở mục **📦 Output chuẩn bàn giao CSDL (`du_lieu_db`)** để xem trực tiếp dữ liệu JSON thật mà AI trả về.
4. Commit & tạo Pull Request gộp vào `main`.
