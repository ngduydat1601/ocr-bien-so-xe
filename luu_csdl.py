"""Điểm cắm (Interface) cho tầng Lưu trữ Ảnh & Cơ sở dữ liệu (Database).

Người phụ trách CSDL chỉ cần triển khai logic SQLite / PostgreSQL trong file này,
không cần sửa code AI trong `web_don_gian.py`.
"""

import json
import re
import sqlite3
from contextlib import closing
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np


THU_MUC_GOC = Path(__file__).resolve().parent
THU_MUC_LUU_ANH = THU_MUC_GOC / "du_lieu" / "anh_luu"
DUONG_DAN_CSDL = THU_MUC_GOC / "du_lieu" / "lich_su_bien_so.sqlite3"


def _ket_noi_csdl() -> sqlite3.Connection:
    """Tạo CSDL, bảng và index nếu chưa có; bên gọi chịu trách nhiệm đóng kết nối."""
    DUONG_DAN_CSDL.parent.mkdir(parents=True, exist_ok=True)
    ket_noi = sqlite3.connect(DUONG_DAN_CSDL, timeout=30)
    try:
        with ket_noi:
            ket_noi.execute("""
                CREATE TABLE IF NOT EXISTS lich_su_bien_so (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    captured_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    plate_raw VARCHAR(20),
                    plate_normalized VARCHAR(20),
                    format_status VARCHAR(15),
                    det_confidence REAL,
                    ocr_confidence REAL,
                    det_latency_ms REAL,
                    ocr_latency_ms REAL,
                    total_latency_ms REAL,
                    model_version VARCHAR(100),
                    plate_bbox TEXT,
                    image_full_path TEXT,
                    image_crop_path TEXT,
                    is_correct BOOLEAN
                )
            """)
            ket_noi.execute("""
                CREATE INDEX IF NOT EXISTS idx_plate_time
                ON lich_su_bien_so (plate_normalized, captured_at)
            """)
        return ket_noi
    except Exception:
        ket_noi.close()
        raise


def luu_anh_ra_o_dia(
    anh_goc: np.ndarray,
    anh_crop: np.ndarray,
    plate_normalized: str,
    thoi_diem: datetime | None = None,
) -> tuple[str, str]:
    """Lưu ảnh toàn cảnh (full) và ảnh biển số (crop) theo cấu trúc YYYY/MM/DD/.

    Trả về cặp ĐƯỜNG DẪN TƯƠNG ĐỐI (dùng dấu '/') để lưu vào 2 cột:
    - image_full_path: VD '2026/11/22/full/20261122_141530_123456_29G216139.jpg'
    - image_crop_path: VD '2026/11/22/crop/20261122_141530_123456_29G216139_crop.jpg'
    """
    now = thoi_diem or datetime.now()
    ngay_rel = now.strftime("%Y/%m/%d")
    dau_thoi_gian = now.strftime("%Y%m%d_%H%M%S_%f")
    bien_an_toan = re.sub(r"[^A-Z0-9]", "", (plate_normalized or "UNKNOWN").upper()) or "UNKNOWN"

    thu_muc_full = THU_MUC_LUU_ANH / ngay_rel / "full"
    thu_muc_crop = THU_MUC_LUU_ANH / ngay_rel / "crop"
    thu_muc_full.mkdir(parents=True, exist_ok=True)
    thu_muc_crop.mkdir(parents=True, exist_ok=True)

    ten_full = f"{dau_thoi_gian}_{bien_an_toan}.jpg"
    ten_crop = f"{dau_thoi_gian}_{bien_an_toan}_crop.jpg"

    # Nén JPEG 85% cho ảnh toàn cảnh (nhẹ ổ đĩa), 95% cho ảnh crop biển số (giữ nét)
    ok1, buf1 = cv2.imencode(".jpg", anh_goc, [int(cv2.IMWRITE_JPEG_QUALITY), 85])
    if ok1:
        (thu_muc_full / ten_full).write_bytes(buf1.tobytes())

    ok2, buf2 = cv2.imencode(".jpg", anh_crop, [int(cv2.IMWRITE_JPEG_QUALITY), 95])
    if ok2:
        (thu_muc_crop / ten_crop).write_bytes(buf2.tobytes())

    image_full_path = f"{ngay_rel}/full/{ten_full}"
    image_crop_path = f"{ngay_rel}/crop/{ten_crop}"
    return image_full_path, image_crop_path


def chuan_bi_ban_ghi_day_du(
    du_lieu_db: dict,
    anh_goc: np.ndarray,
    anh_crop: np.ndarray,
) -> dict:
    """Bổ sung đủ 15 trường theo schema Database (tự lưu ảnh ra ổ đĩa lấy đường dẫn tương đối)."""
    thoi_diem = datetime.now()
    image_full_path, image_crop_path = luu_anh_ra_o_dia(
        anh_goc, anh_crop, du_lieu_db.get("plate_normalized", ""), thoi_diem=thoi_diem
    )

    return {
        "id": None,  # Database tự sinh (AUTOINCREMENT / UUID) khi INSERT
        "captured_at": thoi_diem.strftime("%Y-%m-%d %H:%M:%S"),
        **du_lieu_db,
        "image_full_path": image_full_path,
        "image_crop_path": image_crop_path,
        "is_correct": None,  # Mặc định NULL; cập nhật True/False khi có đối soát
    }


def luu_ket_qua_bien_so(
    du_lieu_db: dict,
    anh_goc: np.ndarray,
    anh_crop: np.ndarray,
) -> int | None:
    """Điểm cắm chính cho người phụ trách CSDL.

    Tham số đầu vào từ `web_don_gian.py`:
    - du_lieu_db: dict gồm 10 trường phía AI (plate_raw, plate_normalized, format_status,
      det_confidence, ocr_confidence, det_latency_ms, ocr_latency_ms, total_latency_ms,
      model_version, plate_bbox).
    - anh_goc: ảnh toàn cảnh gốc chưa vẽ khung (numpy BGR).
    - anh_crop: ảnh vùng biển số đã cắt (numpy BGR).

    Trả về: id bản ghi vừa INSERT vào CSDL.
    """
    ban_ghi = chuan_bi_ban_ghi_day_du(du_lieu_db, anh_goc, anh_crop)

    ban_ghi["plate_bbox"] = json.dumps(ban_ghi["plate_bbox"], ensure_ascii=False)
    with closing(_ket_noi_csdl()) as ket_noi, ket_noi:
        con_tro = ket_noi.execute("""
            INSERT INTO lich_su_bien_so (
                captured_at, plate_raw, plate_normalized, format_status,
                det_confidence, ocr_confidence, det_latency_ms, ocr_latency_ms,
                total_latency_ms, model_version, plate_bbox,
                image_full_path, image_crop_path, is_correct
            ) VALUES (
                :captured_at, :plate_raw, :plate_normalized, :format_status,
                :det_confidence, :ocr_confidence, :det_latency_ms, :ocr_latency_ms,
                :total_latency_ms, :model_version, :plate_bbox,
                :image_full_path, :image_crop_path, :is_correct
            )
        """, ban_ghi)
        return con_tro.lastrowid


def luu_nhieu_ket_qua_bien_so(
    danh_sach_muc: list[tuple[dict, np.ndarray, np.ndarray]],
) -> list[int]:
    """Lưu hàng loạt bản ghi biển số trong cùng 1 giao dịch SQLite (dùng cho chế độ Folder & Video).

    Mỗi phần tử trong `danh_sach_muc` là bộ 3: `(du_lieu_db, anh_goc, anh_crop)`.
    Trả về danh sách các `id` vừa được tạo.
    """
    if not danh_sach_muc:
        return []

    cac_ban_ghi = []
    for du_lieu_db, anh_goc, anh_crop in danh_sach_muc:
        ban_ghi = chuan_bi_ban_ghi_day_du(du_lieu_db, anh_goc, anh_crop)
        ban_ghi["plate_bbox"] = json.dumps(ban_ghi["plate_bbox"], ensure_ascii=False)
        cac_ban_ghi.append(ban_ghi)

    ds_id: list[int] = []
    cau_lenh = """
        INSERT INTO lich_su_bien_so (
            captured_at, plate_raw, plate_normalized, format_status,
            det_confidence, ocr_confidence, det_latency_ms, ocr_latency_ms,
            total_latency_ms, model_version, plate_bbox,
            image_full_path, image_crop_path, is_correct
        ) VALUES (
            :captured_at, :plate_raw, :plate_normalized, :format_status,
            :det_confidence, :ocr_confidence, :det_latency_ms, :ocr_latency_ms,
            :total_latency_ms, :model_version, :plate_bbox,
            :image_full_path, :image_crop_path, :is_correct
        )
    """
    with closing(_ket_noi_csdl()) as ket_noi, ket_noi:
        for ban_ghi in cac_ban_ghi:
            con_tro = ket_noi.execute(cau_lenh, ban_ghi)
            if con_tro.lastrowid is not None:
                ds_id.append(int(con_tro.lastrowid))
    return ds_id


def cap_nhat_doi_soat(record_id: int, is_correct: bool) -> bool:
    """Cập nhật kết quả đối soát đúng/sai (`is_correct`) cho một bản ghi đã lưu."""
    if not isinstance(is_correct, bool):
        raise TypeError("is_correct phải là bool (True hoặc False).")
    with closing(_ket_noi_csdl()) as ket_noi, ket_noi:
        con_tro = ket_noi.execute(
            "UPDATE lich_su_bien_so SET is_correct = ? WHERE id = ?",
            (is_correct, record_id),
        )
        return con_tro.rowcount > 0
