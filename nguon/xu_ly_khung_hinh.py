"""Luồng xử lý trọn gói 1 khung hình: YOLO Detect → Lọc góc PiP → Crop → SVTR26 ONNX → Vẽ khung → Output DB."""

import time

import cv2
import numpy as np
from ultralytics import YOLO

from nguon.cau_hinh import (
    PHIEN_BAN_MO_HINH,
    TY_LE_PADDING_CROP,
    VUNG_PIP_X_MIN,
    VUNG_PIP_Y_MIN,
)
from nguon.mo_hinh_ai import doc_ky_tu_bang_svtr26
from nguon.tien_xu_ly import chuan_hoa_bien_so_vn, danh_gia_trang_thai_bien_so


def xu_ly_mot_khung_hinh(
    anh_bgr: np.ndarray,
    mo_hinh_yolo: YOLO,
    bo_svtr_onnx: tuple,
    thiet_bi: str,
    nguong_conf: float,
    imgsz: int,
    bo_qua_goc_pip: bool,
    bat_chuan_hoa: bool,
    model_version: str = PHIEN_BAN_MO_HINH,
) -> dict:
    """Chạy litealpr_yolov8n_eff.pt + best.onnx và trả về ảnh kết quả kèm dữ liệu chuẩn bàn giao CSDL."""
    h, w = anh_bgr.shape[:2]

    # 1. Phát hiện biển số bằng YOLO
    t0 = time.perf_counter()
    kq_yolo = mo_hinh_yolo.predict(
        source=anh_bgr, imgsz=imgsz, conf=nguong_conf, iou=0.5, device=thiet_bi, verbose=False
    )[0]
    t_yolo_ms = (time.perf_counter() - t0) * 1000.0

    anh_ve = anh_bgr.copy()
    cac_ket_qua = []
    t_ocr_ms = 0.0

    if kq_yolo.boxes is not None and len(kq_yolo.boxes) > 0:
        boxes = kq_yolo.boxes.xyxy.cpu().numpy()
        confs = kq_yolo.boxes.conf.cpu().numpy()

        for box, det_conf in zip(boxes, confs):
            x1, y1, x2, y2 = [int(v) for v in box]
            # Bỏ qua vùng ảnh chèn phụ (PiP) ở góc dưới phải của camera bãi xe
            if bo_qua_goc_pip and x1 > VUNG_PIP_X_MIN * w and y1 > VUNG_PIP_Y_MIN * h:
                continue

            px = int((x2 - x1) * TY_LE_PADDING_CROP)
            py = int((y2 - y1) * TY_LE_PADDING_CROP)
            x1p, y1p = max(0, x1 - px), max(0, y1 - py)
            x2p, y2p = min(w, x2 + px), min(h, y2 + py)
            anh_crop = anh_bgr[y1p:y2p, x1p:x2p].copy()
            if anh_crop.size == 0:
                continue

            # 2. Đọc ký tự trực tiếp từ crop gốc bằng SVTR26 ONNX (best.onnx)
            t1 = time.perf_counter()
            raw_text, ocr_conf = doc_ky_tu_bang_svtr26(anh_crop, bo_svtr_onnx[0], bo_svtr_onnx[1])
            t_ocr_bien_ms = (time.perf_counter() - t1) * 1000.0
            t_ocr_ms += t_ocr_bien_ms

            ty_le_khung = (x2 - x1) / max(1, y2 - y1)
            text_cuoi = chuan_hoa_bien_so_vn(raw_text, ty_le_khung) if bat_chuan_hoa else raw_text
            trang_thai_format = danh_gia_trang_thai_bien_so(text_cuoi, ocr_conf)

            # 3. Vẽ khung & nhãn lên ảnh hiển thị
            cv2.rectangle(anh_ve, (x1, y1), (x2, y2), (0, 230, 80), 3)
            nhan = f"{text_cuoi or '?'} ({ocr_conf*100:.0f}%)"
            (tw, th), _ = cv2.getTextSize(nhan, cv2.FONT_HERSHEY_SIMPLEX, 0.8, 2)
            y_nen = max(0, y1 - th - 10)
            cv2.rectangle(anh_ve, (x1, y_nen), (x1 + tw + 10, y_nen + th + 10), (0, 230, 80), -1)
            cv2.putText(anh_ve, nhan, (x1 + 5, y_nen + th + 4), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 2)

            # 4. Đóng gói dữ liệu chuẩn tên cột Database
            du_lieu_db = {
                "plate_raw": raw_text,
                "plate_normalized": text_cuoi,
                "format_status": trang_thai_format,
                "det_confidence": round(float(det_conf), 4),
                "ocr_confidence": round(float(ocr_conf), 4),
                "det_latency_ms": round(t_yolo_ms, 2),
                "ocr_latency_ms": round(t_ocr_bien_ms, 2),
                "total_latency_ms": round(t_yolo_ms + t_ocr_bien_ms, 2),
                "model_version": model_version,
                "plate_bbox": [x1, y1, x2, y2],
            }

            cac_ket_qua.append({
                "crop_goc": anh_crop,
                "raw_text": raw_text,
                "text_chuan": text_cuoi,
                "format_status": trang_thai_format,
                "det_conf": round(float(det_conf) * 100, 1),
                "ocr_conf": round(ocr_conf * 100, 1),
                "du_lieu_db": du_lieu_db,
            })

    tong_ms = t_yolo_ms + t_ocr_ms
    return {
        "anh_goc": anh_bgr,
        "anh_ve": anh_ve,
        "cac_ket_qua": cac_ket_qua,
        "yolo_ms": round(t_yolo_ms, 1),
        "ocr_ms": round(t_ocr_ms, 1),
        "tong_ms": round(tong_ms, 1),
    }
