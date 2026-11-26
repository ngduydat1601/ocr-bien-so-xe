"""Luồng xử lý Video 2 tầng (Theo dõi phương tiện ByteTrack + Bắt biển từ crop gốc 4K + Vẽ bảng HUD)."""

from collections import deque
from pathlib import Path
import time

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from ultralytics import YOLO

from nguon.cau_hinh import LOP_PHUONG_TIEN_COCO, TY_LE_PADDING_CROP
from nguon.mo_hinh_ai import doc_ky_tu_bang_svtr26, doc_lo_ky_tu_svtr26
from nguon.tien_xu_ly import chuan_hoa_bien_so_vn, danh_gia_trang_thai_bien_so

# Bảng màu chuẩn theo phong cách HUD Vàng - Đen thời gian thực
MAU_VANG_BGR = (25, 208, 255)
MAU_VANG_RGB = (255, 208, 25)
MAU_DIEM_TAM_BGR = (130, 120, 255)
MAU_NEN_HUD_BGR = (22, 24, 30)
MAU_VIEN_HUD_BGR = (48, 125, 165)


def _tai_bo_font() -> tuple[ImageFont.FreeTypeFont | ImageFont.ImageFont, ...]:
    """Tải phông chữ đậm hỗ trợ đầy đủ dấu Tiếng Việt trên Windows."""
    cac_ung_vien = [
        Path("C:/Windows/Fonts/arialbd.ttf"),
        Path("C:/Windows/Fonts/segoeuib.ttf"),
        Path("C:/Windows/Fonts/tahomabd.ttf"),
    ]
    for p in cac_ung_vien:
        if p.exists():
            return (
                ImageFont.truetype(str(p), 15),  # Nhãn trên đầu xe & banner biển số HUD
                ImageFont.truetype(str(p), 14),  # Tiêu đề chính bảng HUD
                ImageFont.truetype(str(p), 12),  # Chữ phụ thẻ xe HUD
            )
    mac_dinh = ImageFont.load_default()
    return mac_dinh, mac_dinh, mac_dinh


FONT_NHAN_XE, FONT_TIEU_DE_HUD, FONT_THE_HUD = _tai_bo_font()
_CLAHE_KHU_LOA = cv2.createCLAHE(clipLimit=1.8, tileGridSize=(4, 4))


def khu_loa_bien_so(crop_bgr: np.ndarray) -> tuple[np.ndarray, bool]:
    """Khử lóa nắng / chói đèn trên vùng crop biển số (chỉ kích hoạt khi vùng lõi thực sự bị dư sáng)."""
    if crop_bgr is None or crop_bgr.size == 0:
        return crop_bgr, False
    h, w = crop_bgr.shape[:2]
    yh1, yh2 = int(h * 0.12), max(int(h * 0.12) + 1, int(h * 0.88))
    xw1, xw2 = int(w * 0.10), max(int(w * 0.10) + 1, int(w * 0.90))
    loi = crop_bgr[yh1:yh2, xw1:xw2]
    if loi.size == 0:
        return crop_bgr, False

    xam = cv2.cvtColor(loi, cv2.COLOR_BGR2GRAY)
    tb_sang = float(np.mean(xam))
    ty_le_choi = float(np.mean(xam >= 235))
    do_tuong_phan = float(np.std(xam))

    if tb_sang < 178.0 and ty_le_choi < 0.28:
        return crop_bgr, False

    lab = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2LAB)
    l_kenh, a_kenh, b_kenh = cv2.split(lab)
    l_loi = l_kenh[yh1:yh2, xw1:xw2]
    p_thap = float(np.percentile(l_loi, 6))
    p_cao = max(p_thap + 25.0, float(np.percentile(l_loi, 96)))
    l_norm = np.clip((l_kenh.astype(np.float32) - p_thap) / (p_cao - p_thap), 0.0, 1.0)

    gamma = 1.65 if (tb_sang > 205.0 or do_tuong_phan < 32.0) else 1.35
    l_gamma = (np.power(l_norm, gamma) * 232.0 + 10.0).astype(np.uint8)
    l_cuoi = _CLAHE_KHU_LOA.apply(l_gamma)
    return cv2.cvtColor(cv2.merge([l_cuoi, a_kenh, b_kenh]), cv2.COLOR_LAB2BGR), True


def _cham_diem_quang_hoc_crop(crop_bgr: np.ndarray, ocr_conf: float) -> tuple[float, bool]:
    """Chấm điểm chất lượng hiển thị thực tế: ưu tiên nét căng, tương phản tốt, không bị cháy nắng."""
    h, w = crop_bgr.shape[:2]
    yh1, yh2 = int(h * 0.12), max(int(h * 0.12) + 1, int(h * 0.88))
    xw1, xw2 = int(w * 0.10), max(int(w * 0.10) + 1, int(w * 0.90))
    loi = crop_bgr[yh1:yh2, xw1:xw2]
    xam = cv2.cvtColor(loi if loi.size > 0 else crop_bgr, cv2.COLOR_BGR2GRAY)

    ty_le_chay = float(np.mean(xam >= 238))
    he_so_chong_loa = max(0.22, 1.0 - 0.85 * ty_le_chay)
    he_so_tuong_phan = min(1.30, max(0.35, float(np.std(xam)) / 46.0))
    do_net_lap = float(cv2.Laplacian(xam, cv2.CV_64F).var())
    he_so_sac_net = min(1.30, max(0.30, float(np.log1p(do_net_lap)) / 5.5))
    he_so_co = min(1.35, (h * w) / 4000.0)

    diem = float(ocr_conf) * he_so_chong_loa * he_so_tuong_phan * he_so_sac_net * he_so_co
    dang_bi_choi = ty_le_chay >= 0.28 or float(np.mean(xam)) >= 182.0
    return diem, dang_bi_choi


class BoTheoDoiVideoHUD:
    """Quản lý trạng thái theo dõi phương tiện (#ID), bầu chọn biển số ổn định và dựng giao diện HUD."""

    def __init__(self, chieu_rong_hien: int = 1280, chieu_rong_hud: int = 300, so_the_hud: int = 4):
        self.chieu_rong_hien = chieu_rong_hien
        self.chieu_rong_hud = chieu_rong_hud
        self.so_the_hud = so_the_hud
        self.quy_dao: dict[int, deque[tuple[int, int]]] = {}
        self.ho_so_xe: dict[int, dict] = {}
        self.buoc_hien_tai = 0

    def dat_lai(self, mo_hinh_xe: YOLO | None = None) -> None:
        """Xóa lịch sử quỹ đạo & bộ đếm #ID khi chuyển sang video mới."""
        self.quy_dao.clear()
        self.ho_so_xe.clear()
        self.buoc_hien_tai = 0
        if mo_hinh_xe is not None and getattr(mo_hinh_xe, "predictor", None) is not None:
            mo_hinh_xe.predictor = None

    def xu_ly_khung_video(
        self,
        khung_goc_bgr: np.ndarray,
        mo_hinh_xe: YOLO,
        mo_hinh_bien: YOLO,
        bo_svtr_onnx: tuple,
        thiet_bi: str,
        ds_ma_lop_xe: list[int],
        nguong_conf_xe: float = 0.35,
        nguong_conf_bien: float = 0.25,
        bat_chuan_hoa: bool = True,
    ) -> dict:
        """Chạy theo dõi xe + đọc biển từ crop độ phân giải gốc + kết xuất khung hình kèm HUD bên phải."""
        self.buoc_hien_tai += 1
        h_goc, w_goc = khung_goc_bgr.shape[:2]
        ty_le_hien = self.chieu_rong_hien / float(max(1, w_goc))
        h_hien = int(h_goc * ty_le_hien)
        khung_hien = (
            cv2.resize(khung_goc_bgr, (self.chieu_rong_hien, h_hien), interpolation=cv2.INTER_LINEAR)
            if w_goc != self.chieu_rong_hien
            else khung_goc_bgr.copy()
        )
        ty_le_nguoc = 1.0 / ty_le_hien
        dung_fp16 = thiet_bi != "cpu"

        # 1. Phát hiện & theo dõi phương tiện trên khung 1280px (nhanh gấp 4 lần so với nạp thẳng mảng 4K)
        t0 = time.perf_counter()
        kq_xe = mo_hinh_xe.track(
            source=khung_hien,
            imgsz=640,
            conf=nguong_conf_xe,
            iou=0.5,
            classes=ds_ma_lop_xe,
            half=dung_fp16,
            device=thiet_bi,
            persist=True,
            tracker="bytetrack.yaml",
            verbose=False,
        )[0]
        t_xe_ms = (time.perf_counter() - t0) * 1000.0

        ds_xe_frame: list[dict] = []
        if kq_xe.boxes is not None and kq_xe.boxes.id is not None and len(kq_xe.boxes) > 0:
            xyxy_all = kq_xe.boxes.xyxy.cpu().numpy()
            confs_all = kq_xe.boxes.conf.cpu().numpy()
            clss_all = kq_xe.boxes.cls.cpu().numpy().astype(int)
            ids_all = kq_xe.boxes.id.cpu().numpy().astype(int)

            for box, conf_xe, ma_lop, ma_id in zip(xyxy_all, confs_all, clss_all, ids_all):
                xh1, yh1, xh2, yh2 = [float(v) for v in box]
                if (xh2 - xh1) < 22 or (yh2 - yh1) < 22:
                    continue
                x1, y1 = int(xh1 * ty_le_nguoc), int(yh1 * ty_le_nguoc)
                x2, y2 = int(xh2 * ty_le_nguoc), int(yh2 * ty_le_nguoc)
                ten_loai = LOP_PHUONG_TIEN_COCO.get(int(ma_lop), "Xe")
                hs = self.ho_so_xe.setdefault(
                    int(ma_id),
                    {
                        "id": int(ma_id),
                        "loai_xe": ten_loai,
                        "diem_bau_chon": {},
                        "anh_theo_bien": {},
                        "bien_chot": "",
                        "ocr_tot_nhat": 0.0,
                        "lan_quet_cuoi": -99,
                        "buoc_thay_cuoi": 0,
                    },
                )
                hs["loai_xe"] = ten_loai
                hs["buoc_thay_cuoi"] = self.buoc_hien_tai
                ds_xe_frame.append({
                    "id": int(ma_id),
                    "loai_xe": ten_loai,
                    "conf_xe": float(conf_xe),
                    "box_hien": (int(xh1), int(yh1), int(xh2), int(yh2)),
                    "box_goc": (max(0, x1), max(0, y1), min(w_goc, x2), min(h_goc, y2)),
                })

        # 2. Cắt crop xe ở độ phân giải gốc 4K (thu gọn tối đa 768px nếu xe quá sát camera)
        ds_crop_xe: list[np.ndarray] = []
        ds_meta_xe: list[tuple[dict, float]] = []
        for xe in ds_xe_frame:
            hs = self.ho_so_xe[xe["id"]]
            tong_diem_chot = hs["diem_bau_chon"].get(hs["bien_chot"], 0.0)
            anh_hien_tai_con_choi = (
                hs["anh_theo_bien"].get(hs["bien_chot"], (0.0, 0.0, None, True))[3]
                if hs["bien_chot"] in hs["anh_theo_bien"]
                else True
            )
            # Nếu ảnh đại diện hiện tại đã rõ & không chói thì giảm tần suất quét lại; còn nếu đang chói thì quét thường xuyên hơn để bắt khoảnh khắc râm mát!
            chu_ky_nghi = 2 if anh_hien_tai_con_choi else 6
            if tong_diem_chot >= 2.7 and (self.buoc_hien_tai - hs["lan_quet_cuoi"]) < chu_ky_nghi:
                continue
            x1, y1, x2, y2 = xe["box_goc"]
            crop_xe = khung_goc_bgr[y1:y2, x1:x2]
            if crop_xe.size == 0:
                continue
            hc, wc = crop_xe.shape[:2]
            ty_le_c = 1.0
            if max(hc, wc) > 768:
                ty_le_c = 768.0 / float(max(hc, wc))
                crop_xe = cv2.resize(crop_xe, (int(wc * ty_le_c), int(hc * ty_le_c)), interpolation=cv2.INTER_LINEAR)
            hs["lan_quet_cuoi"] = self.buoc_hien_tai
            ds_crop_xe.append(crop_xe)
            ds_meta_xe.append((xe, ty_le_c))

        # 3. Chạy Batch YOLO tìm biển số trên các vùng crop xe + Batch SVTR26 đọc chữ
        t1 = time.perf_counter()
        t_ocr_ms = 0.0
        if ds_crop_xe:
            kq_bien_lo = mo_hinh_bien.predict(
                source=ds_crop_xe,
                imgsz=640,
                conf=nguong_conf_bien,
                iou=0.5,
                half=dung_fp16,
                batch=min(16, len(ds_crop_xe)),
                device=thiet_bi,
                verbose=False,
            )
            t_bien_ms = (time.perf_counter() - t1) * 1000.0

            ds_crop_bien: list[np.ndarray] = []
            ds_lien_ket_id: list[tuple[int, np.ndarray, float]] = []

            for (xe, ty_le_c), crop_xe, kq_b in zip(ds_meta_xe, ds_crop_xe, kq_bien_lo):
                if kq_b.boxes is None or len(kq_b.boxes) == 0:
                    continue
                hx, wx = crop_xe.shape[:2]
                xg1, yg1, xg2, yg2 = xe["box_goc"]
                dt_xe_nay = max(1, (xg2 - xg1) * (yg2 - yg1))

                boxes_b = kq_b.boxes.xyxy.cpu().numpy()
                confs_b = kq_b.boxes.conf.cpu().numpy()
                for bb, dc in zip(boxes_b, confs_b):
                    if float(dc) < 0.45:
                        continue
                    bx1, by1, bx2, by2 = [int(v) for v in bb]
                    # Bỏ qua biển số đang bị cắt cụt sát rìa khung hình gốc 4K (tránh đoán mò nửa biển số)
                    gx1 = xg1 + int(bx1 / ty_le_c)
                    gy1 = yg1 + int(by1 / ty_le_c)
                    gx2 = xg1 + int(bx2 / ty_le_c)
                    gy2 = yg1 + int(by2 / ty_le_c)
                    if gx1 <= 6 or gy1 <= 6 or gx2 >= w_goc - 6 or gy2 >= h_goc - 6:
                        continue
                    if (gx2 - gx1) < 38 or (gy2 - gy1) < 16:
                        continue

                    tam_bx_goc = (gx1 + gx2) * 0.5
                    tam_by_goc = (gy1 + gy2) * 0.5
                    # Trên xe lớn (Ô tô/Xe tải/Xe buýt), không lấy biển số lơ lửng sát mép ngoài 8% (thường của xe máy đi cạnh)
                    if xe["loai_xe"] != "Xe máy":
                        le_ngang = (xg2 - xg1) * 0.08
                        if tam_bx_goc < xg1 + le_ngang or tam_bx_goc > xg2 - le_ngang:
                            continue

                    bi_xe_nho_che = False
                    for xe_khac in ds_xe_frame:
                        if xe_khac["id"] == xe["id"]:
                            continue
                        kx1, ky1, kx2, ky2 = xe_khac["box_goc"]
                        dt_khac = max(1, (kx2 - kx1) * (ky2 - ky1))
                        if dt_khac < dt_xe_nay * 0.85 and kx1 <= tam_bx_goc <= kx2 and ky1 <= tam_by_goc <= ky2:
                            bi_xe_nho_che = True
                            break
                    if bi_xe_nho_che:
                        continue

                    px = int((gx2 - gx1) * TY_LE_PADDING_CROP)
                    py = int((gy2 - gy1) * TY_LE_PADDING_CROP)
                    crop_bien = khung_goc_bgr[max(0, gy1 - py) : min(h_goc, gy2 + py), max(0, gx1 - px) : min(w_goc, gx2 + px)].copy()
                    if crop_bien.size == 0:
                        continue
                    ty_le_b = (gx2 - gx1) / max(1, gy2 - gy1)
                    ds_crop_bien.append(crop_bien)
                    ds_lien_ket_id.append((xe["id"], crop_bien, ty_le_b))
                    break

            if ds_crop_bien:
                t2 = time.perf_counter()
                kq_ocr_lo = doc_lo_ky_tu_svtr26(ds_crop_bien, bo_svtr_onnx[0], bo_svtr_onnx[1])
                t_ocr_ms = (time.perf_counter() - t2) * 1000.0

                for (ma_id, crop_bien, ty_le_b), (raw_text, ocr_conf) in zip(ds_lien_ket_id, kq_ocr_lo):
                    crop_khu_loa, da_khu_loa = khu_loa_bien_so(crop_bien)
                    text_chuan = chuan_hoa_bien_so_vn(raw_text, ty_le_b) if bat_chuan_hoa else raw_text
                    trang_thai = danh_gia_trang_thai_bien_so(text_chuan, ocr_conf)

                    # Nếu biển bị chói nắng làm lần đọc đầu kém tự tin hoặc sai chuẩn -> đọc lại trên bản đã khử lóa
                    if da_khu_loa and (trang_thai != "VALID" or ocr_conf < 0.82):
                        raw_kl, conf_kl = doc_ky_tu_bang_svtr26(crop_khu_loa, bo_svtr_onnx[0], bo_svtr_onnx[1])
                        text_kl = chuan_hoa_bien_so_vn(raw_kl, ty_le_b) if bat_chuan_hoa else raw_kl
                        tt_kl = danh_gia_trang_thai_bien_so(text_kl, conf_kl)
                        if tt_kl == "VALID" and (trang_thai != "VALID" or conf_kl > ocr_conf):
                            text_chuan, ocr_conf, trang_thai = text_kl, conf_kl, tt_kl

                    if ocr_conf < 0.72 or trang_thai != "VALID":
                        continue
                    hs = self.ho_so_xe[ma_id]
                    hs["diem_bau_chon"][text_chuan] = hs["diem_bau_chon"].get(text_chuan, 0.0) + float(ocr_conf)

                    # Chấm điểm trên ảnh gốc để ưu tiên giữ frame tự nhiên không chói; nếu vẫn chói thì hiển thị bản đã khử lóa sắc nét!
                    diem_quang_hoc, con_choi = _cham_diem_quang_hoc_crop(crop_bien, ocr_conf)
                    anh_luu_hud = crop_khu_loa if da_khu_loa else crop_bien
                    cu = hs["anh_theo_bien"].get(text_chuan)
                    if cu is None or diem_quang_hoc >= cu[0]:
                        hs["anh_theo_bien"][text_chuan] = (diem_quang_hoc, float(ocr_conf), anh_luu_hud, con_choi)

                    # Chỉ chính thức chốt biển số lên HUD khi đã xác nhận ổn định >= 2 lần đọc (tổng điểm >= 1.55)
                    ung_vien_tot = max(hs["diem_bau_chon"], key=hs["diem_bau_chon"].get)
                    if hs["diem_bau_chon"][ung_vien_tot] >= 1.55:
                        hs["bien_chot"] = ung_vien_tot
                        hs["ocr_tot_nhat"] = hs["anh_theo_bien"][ung_vien_tot][1]
        else:
            t_bien_ms = 0.0

        # 4. Dựng khung hình hiển thị (1280px) kèm quỹ đạo, khung xe và bảng HUD bên phải
        anh_hien = self._ve_giao_dien_hud(khung_hien, ds_xe_frame)
        tong_ms = t_xe_ms + t_bien_ms + t_ocr_ms
        so_xe_co_bien = sum(1 for v in self.ho_so_xe.values() if v["bien_chot"])

        return {
            "anh_ve": anh_hien,
            "xe_ms": round(t_xe_ms, 1),
            "bien_ocr_ms": round(t_bien_ms + t_ocr_ms, 1),
            "tong_ms": round(tong_ms, 1),
            "so_xe_khung_hinh": len(ds_xe_frame),
            "tong_xe_da_chot_bien": so_xe_co_bien,
        }

    def _ve_giao_dien_hud(self, khung: np.ndarray, ds_xe_frame: list[dict]) -> np.ndarray:
        h_hien, w_hien = khung.shape[:2]

        # Bảng HUD đặt ở cột riêng bên phải video (không che khung hình)
        x_hud_trai = w_hien + 12
        ds_nhan_pil: list[tuple[int, int, str]] = []

        # 4.1. Vẽ quỹ đạo vàng, chấm tâm đáy & khung viền vàng quanh từng xe
        for xe in ds_xe_frame:
            ma_id = xe["id"]
            x1, y1, x2, y2 = xe["box_hien"]

            tam_day = ((x1 + x2) // 2, y2)
            lich_su = self.quy_dao.setdefault(ma_id, deque(maxlen=36))
            lich_su.append(tam_day)

            # Vẽ vệt quỹ đạo di chuyển màu vàng
            if len(lich_su) >= 2:
                pts = list(lich_su)
                for idx_p in range(1, len(pts)):
                    do_day = 1 if idx_p < len(pts) // 2 else 2
                    cv2.line(khung, pts[idx_p - 1], pts[idx_p], MAU_VANG_BGR, do_day, cv2.LINE_AA)

            cv2.rectangle(khung, (x1, y1), (x2, y2), MAU_VANG_BGR, 2)
            cv2.circle(khung, tam_day, 5, MAU_DIEM_TAM_BGR, -1, cv2.LINE_AA)

            hs = self.ho_so_xe[ma_id]
            phan_tram = int(round(xe["conf_xe"] * 100))
            if hs["bien_chot"]:
                chuoi_nhan = f"#{ma_id} {xe['loai_xe']} {phan_tram}% | {hs['bien_chot']}"
            else:
                chuoi_nhan = f"#{ma_id} {xe['loai_xe']} {phan_tram}%"
            ds_nhan_pil.append((x1, y1, chuoi_nhan))

        # 4.2. Ghép video bên trái + cột Bảng HUD bên phải trên cùng 1 khung nền tối
        khung_video = khung
        khung = np.full((h_hien, w_hien + self.chieu_rong_hud + 24, 3), (14, 15, 18), dtype=np.uint8)
        khung[:, :w_hien] = khung_video
        y_hud_tren, y_hud_duoi = 12, h_hien - 12
        x_hud_phai = x_hud_trai + self.chieu_rong_hud
        cv2.rectangle(khung, (x_hud_trai, y_hud_tren), (x_hud_phai, y_hud_duoi), MAU_NEN_HUD_BGR, -1)
        cv2.rectangle(khung, (x_hud_trai, y_hud_tren), (x_hud_phai, y_hud_duoi), MAU_VIEN_HUD_BGR, 1)
        cv2.line(khung, (x_hud_trai + 10, y_hud_tren + 48), (x_hud_phai - 10, y_hud_tren + 48), MAU_VANG_BGR, 1)

        # Chọn tối đa N biển số duy nhất gần đây nhất (không lặp trùng cùng 1 biển số trên bảng HUD)
        id_dang_thay = {xe["id"] for xe in ds_xe_frame}
        ung_vien_hud = [v for v in self.ho_so_xe.values() if v["bien_chot"] and v["bien_chot"] in v["anh_theo_bien"]]
        ung_vien_hud.sort(
            key=lambda v: (1 if v["id"] in id_dang_thay else 0, v["buoc_thay_cuoi"], v["diem_bau_chon"][v["bien_chot"]]),
            reverse=True,
        )
        ds_the = []
        bien_da_hien = set()
        for v in ung_vien_hud:
            if v["bien_chot"] in bien_da_hien:
                continue
            bien_da_hien.add(v["bien_chot"])
            ds_the.append(v)
            if len(ds_the) >= self.so_the_hud:
                break

        # Tính chiều cao mỗi thẻ vừa khít chiều cao khung hình
        y_bat_dau_the = y_hud_tren + 58
        khoang_trong = y_hud_duoi - y_bat_dau_the - 8
        cao_moi_the = min(148, max(110, khoang_trong // max(1, self.so_the_hud) - 8))

        thong_tin_chu_hud: list[tuple[int, int, int, str, str]] = []
        for idx_t, hs in enumerate(ds_the):
            yt1 = y_bat_dau_the + idx_t * (cao_moi_the + 8)
            yt2 = yt1 + cao_moi_the
            if yt2 > y_hud_duoi - 4:
                break
            xt1, xt2 = x_hud_trai + 10, x_hud_phai - 10

            # Viền thẻ mờ
            cv2.rectangle(khung, (xt1, yt1), (xt2, yt2), (55, 65, 80), 1)

            # Vùng đặt ảnh crop biển số ở giữa thẻ
            y_crop_top = yt1 + 24
            y_banner_top = yt2 - 28
            cao_vung_crop = max(24, y_banner_top - y_crop_top - 6)
            rong_vung_crop = (xt2 - xt1) - 24

            crop_b = hs["anh_theo_bien"][hs["bien_chot"]][2]
            ch, cw = crop_b.shape[:2]
            sc = min(rong_vung_crop / float(max(1, cw)), cao_vung_crop / float(max(1, ch)))
            nw, nh = max(1, int(cw * sc)), max(1, int(ch * sc))
            crop_rs = cv2.resize(crop_b, (nw, nh), interpolation=cv2.INTER_CUBIC)

            xc_dat = xt1 + ((xt2 - xt1) - nw) // 2
            yc_dat = y_crop_top + (cao_vung_crop - nh) // 2
            khung[yc_dat : yc_dat + nh, xc_dat : xc_dat + nw] = crop_rs
            cv2.rectangle(khung, (xc_dat - 1, yc_dat - 1), (xc_dat + nw, yc_dat + nh), (180, 180, 180), 1)

            # Thanh banner nền vàng bên dưới thẻ
            cv2.rectangle(khung, (xt1 + 6, y_banner_top), (xt2 - 6, yt2 - 5), MAU_VANG_BGR, -1)

            tieu_de_the = f"#{hs['id']} {hs['loai_xe']} ({int(round(hs['ocr_tot_nhat'] * 100))}%)"
            thong_tin_chu_hud.append((xt1, yt1, y_banner_top, tieu_de_the, hs["bien_chot"]))

        # 4.3. Kết xuất toàn bộ chữ Tiếng Việt sắc nét bằng PIL trong 1 lần chuyển đổi duy nhất
        anh_pil = Image.fromarray(cv2.cvtColor(khung, cv2.COLOR_BGR2RGB))
        ve = ImageDraw.Draw(anh_pil)

        # Nhãn vàng trên đầu xe (giữ trong phạm vi video, không tràn sang cột HUD)
        for x1, y1, chuoi in ds_nhan_pil:
            hop_chu = ve.textbbox((0, 0), chuoi, font=FONT_NHAN_XE)
            tw, th = hop_chu[2] - hop_chu[0], hop_chu[3] - hop_chu[1]
            x_hop = min(max(2, x1), max(2, w_hien - tw - 14))
            y_hop = max(2, y1 - th - 10)
            ve.rectangle([x_hop, y_hop, x_hop + tw + 12, y_hop + th + 8], fill=MAU_VANG_RGB)
            ve.text((x_hop + 6, y_hop + 3), chuoi, font=FONT_NHAN_XE, fill=(10, 10, 10))

        # Tiêu đề bảng HUD bên phải
        ve.text((x_hud_trai + 12, y_hud_tren + 8), "BẢNG PHÂN TÍCH PHƯƠNG TIỆN", font=FONT_TIEU_DE_HUD, fill=(245, 245, 245))
        ve.text((x_hud_trai + 12, y_hud_tren + 28), "THEO DÕI THỜI GIAN THỰC", font=FONT_THE_HUD, fill=MAU_VANG_RGB)

        if not thong_tin_chu_hud:
            ve.text(
                (x_hud_trai + 18, y_bat_dau_the + 20),
                "Đang chờ phương tiện rõ biển số...",
                font=FONT_THE_HUD,
                fill=(170, 175, 185),
            )

        for xt1, yt1, y_banner_top, tieu_de_the, bien_chot in thong_tin_chu_hud:
            xt2 = x_hud_phai - 10
            ve.text((xt1 + 8, yt1 + 5), tieu_de_the, font=FONT_THE_HUD, fill=MAU_VANG_RGB)
            hop_b = ve.textbbox((0, 0), bien_chot, font=FONT_NHAN_XE)
            bw_t = hop_b[2] - hop_b[0]
            xb_giua = xt1 + ((xt2 - xt1) - bw_t) // 2
            ve.text((xb_giua, y_banner_top + 3), bien_chot, font=FONT_NHAN_XE, fill=(10, 10, 10))

        return cv2.cvtColor(np.asarray(anh_pil), cv2.COLOR_RGB2BGR)
