"""Luồng xử lý khung hình đơn lẻ và xử lý hàng loạt tốc độ cao (Đa luồng đọc ảnh + Batch YOLO)."""

from collections.abc import Iterator
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
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
from nguon.mo_hinh_ai import doc_ky_tu_bang_svtr26, doc_lo_ky_tu_svtr26
from nguon.tien_xu_ly import chuan_hoa_bien_so_vn, danh_gia_trang_thai_bien_so, doc_anh_unicode


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

    # 1. Phát hiện biển số bằng YOLO (bật FP16 Tensor Cores khi chạy trên GPU)
    dung_fp16 = thiet_bi != "cpu"
    t0 = time.perf_counter()
    kq_yolo = mo_hinh_yolo.predict(
        source=anh_bgr, imgsz=imgsz, conf=nguong_conf, iou=0.5, half=dung_fp16, device=thiet_bi, verbose=False
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
            # Giữ nhãn trong khung ảnh (biển sát mép trên/phải thì đặt nhãn xuống dưới & lùi sang trái)
            x_nhan = max(0, min(x1, w - tw - 10))
            y_nen = y1 - th - 10 if y1 - th - 10 >= 0 else min(h - th - 10, y2 + 2)
            cv2.rectangle(anh_ve, (x_nhan, y_nen), (x_nhan + tw + 10, y_nen + th + 10), (0, 230, 80), -1)
            cv2.putText(anh_ve, nhan, (x_nhan + 5, y_nen + th + 4), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 2)

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

        # Ưu tiên đưa biển hợp lệ chuẩn VN (VALID) lên đầu danh sách kết quả
        cac_ket_qua.sort(key=lambda x: (1 if x["format_status"] == "VALID" else 0, x["det_conf"]), reverse=True)

    tong_ms = t_yolo_ms + t_ocr_ms
    return {
        "anh_goc": anh_bgr,
        "anh_ve": anh_ve,
        "cac_ket_qua": cac_ket_qua,
        "yolo_ms": round(t_yolo_ms, 1),
        "ocr_ms": round(t_ocr_ms, 1),
        "tong_ms": round(tong_ms, 1),
    }


def xu_ly_danh_sach_anh_nhanh(
    ds_duong_dan: list[Path],
    mo_hinh_yolo: YOLO,
    bo_svtr_onnx: tuple,
    thiet_bi: str,
    nguong_conf: float,
    imgsz: int,
    bo_qua_goc_pip: bool,
    bat_chuan_hoa: bool,
    kich_thuoc_lo: int = 8,
    so_luong_doc: int = 4,
    model_version: str = PHIEN_BAN_MO_HINH,
) -> Iterator[list[tuple[Path, dict]]]:
    """Xử lý hàng loạt ảnh tốc độ cao theo Phương án A:
    - 4 luồng đọc file & giải mã JPEG gối đầu (prefetch lô kế tiếp trong lúc GPU bận).
    - 1 luồng AI chạy gom lô Batch YOLO trên GPU + Batch SVTR26 ONNX trên CPU.
    """
    if not ds_duong_dan:
        return

    cac_lo_duong_dan = [
        ds_duong_dan[i : i + kich_thuoc_lo] for i in range(0, len(ds_duong_dan), kich_thuoc_lo)
    ]

    with ThreadPoolExecutor(max_workers=so_luong_doc) as pool:
        # Giao việc đọc trước lô đầu tiên trực tiếp cho 4 worker
        ds_future = [(p, pool.submit(doc_anh_unicode, p)) for p in cac_lo_duong_dan[0]]

        for idx_lo in range(len(cac_lo_duong_dan)):
            lo_hien_tai = [(p, fut.result()) for p, fut in ds_future]
            lo_hien_tai = [(p, anh) for p, anh in lo_hien_tai if anh is not None]

            # Gối đầu: giao ngay việc đọc & giải mã lô kế tiếp trong lúc GPU xử lý lô hiện tại
            if idx_lo + 1 < len(cac_lo_duong_dan):
                ds_future = [(p, pool.submit(doc_anh_unicode, p)) for p in cac_lo_duong_dan[idx_lo + 1]]

            if not lo_hien_tai:
                continue

            ds_anh_bgr = [cap[1] for cap in lo_hien_tai]

            # 1. Batch YOLO Detect trên GPU (FP16 Tensor Cores) cho cả lô ảnh cùng lúc
            dung_fp16 = thiet_bi != "cpu"
            t0 = time.perf_counter()
            ds_kq_yolo = mo_hinh_yolo.predict(
                source=ds_anh_bgr,
                imgsz=imgsz,
                conf=nguong_conf,
                iou=0.5,
                half=dung_fp16,
                batch=len(ds_anh_bgr),
                device=thiet_bi,
                verbose=False,
            )
            yolo_moi_anh_ms = ((time.perf_counter() - t0) * 1000.0) / len(ds_anh_bgr)

            # 2. Thu thập toàn bộ vùng crop trong lô để chạy Batch SVTR26 ONNX 1 lần
            ds_crop: list[np.ndarray] = []
            ds_meta_crop: list[tuple[int, float, float, list[int]]] = []

            for idx_anh, ((_, anh_bgr), kq_yolo) in enumerate(zip(lo_hien_tai, ds_kq_yolo)):
                if kq_yolo.boxes is None or len(kq_yolo.boxes) == 0:
                    continue
                h, w = anh_bgr.shape[:2]
                boxes = kq_yolo.boxes.xyxy.cpu().numpy()
                confs = kq_yolo.boxes.conf.cpu().numpy()

                for box, det_conf in zip(boxes, confs):
                    x1, y1, x2, y2 = [int(v) for v in box]
                    if bo_qua_goc_pip and x1 > VUNG_PIP_X_MIN * w and y1 > VUNG_PIP_Y_MIN * h:
                        continue
                    px = int((x2 - x1) * TY_LE_PADDING_CROP)
                    py = int((y2 - y1) * TY_LE_PADDING_CROP)
                    anh_crop = anh_bgr[max(0, y1 - py) : min(h, y2 + py), max(0, x1 - px) : min(w, x2 + px)]
                    if anh_crop.size == 0:
                        continue
                    ds_crop.append(anh_crop)
                    ds_meta_crop.append((idx_anh, float(det_conf), (x2 - x1) / max(1, y2 - y1), [x1, y1, x2, y2]))

            t1 = time.perf_counter()
            ds_kq_ocr = doc_lo_ky_tu_svtr26(ds_crop, bo_svtr_onnx[0], bo_svtr_onnx[1])
            ocr_moi_crop_ms = ((time.perf_counter() - t1) * 1000.0) / max(1, len(ds_crop))

            ds_cac_ket_qua: list[list[dict]] = [[] for _ in range(len(lo_hien_tai))]
            for (idx_anh, det_conf, ty_le_khung, bbox_bien), (raw_text, ocr_conf) in zip(ds_meta_crop, ds_kq_ocr):
                text_cuoi = chuan_hoa_bien_so_vn(raw_text, ty_le_khung) if bat_chuan_hoa else raw_text
                trang_thai_format = danh_gia_trang_thai_bien_so(text_cuoi, ocr_conf)
                du_lieu_db = {
                    "plate_raw": raw_text,
                    "plate_normalized": text_cuoi,
                    "format_status": trang_thai_format,
                    "det_confidence": round(float(det_conf), 4),
                    "ocr_confidence": round(float(ocr_conf), 4),
                    "det_latency_ms": round(yolo_moi_anh_ms, 2),
                    "ocr_latency_ms": round(ocr_moi_crop_ms, 2),
                    "total_latency_ms": round(yolo_moi_anh_ms + ocr_moi_crop_ms, 2),
                    "model_version": model_version,
                    "plate_bbox": bbox_bien,
                }
                ds_cac_ket_qua[idx_anh].append({
                    "raw_text": raw_text,
                    "text_chuan": text_cuoi,
                    "format_status": trang_thai_format,
                    "det_conf": round(det_conf * 100, 1),
                    "ocr_conf": round(ocr_conf * 100, 1),
                    "du_lieu_db": du_lieu_db,
                })

            ket_qua_lo: list[tuple[Path, dict]] = []
            for idx_anh, (duong_dan, _) in enumerate(lo_hien_tai):
                cac_ket_qua = ds_cac_ket_qua[idx_anh]
                cac_ket_qua.sort(key=lambda x: (1 if x["format_status"] == "VALID" else 0, x["det_conf"]), reverse=True)
                t_ocr_ms = ocr_moi_crop_ms * len(cac_ket_qua)
                ket_qua_lo.append((
                    duong_dan,
                    {
                        "cac_ket_qua": cac_ket_qua,
                        "yolo_ms": round(yolo_moi_anh_ms, 1),
                        "ocr_ms": round(t_ocr_ms, 1),
                        "tong_ms": round(yolo_moi_anh_ms + t_ocr_ms, 1),
                    },
                ))

            yield ket_qua_lo
