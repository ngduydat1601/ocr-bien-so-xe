"""Giao diện Web Demo Nhận diện Biển số Xe Việt Nam (Streamlit).

Sử dụng mặc định cặp mô hình tối ưu nhất cho cả Ô tô & Xe máy:
- Phát hiện biển số: litealpr_yolov8n_eff.pt (GPU FP16)
- Đọc ký tự OCR: svtr26_hon_hop.onnx (SVTR26-Tiny Đa năng Ô tô + Xe máy)
"""

import hashlib
import re
import sqlite3
import sys
import tempfile
import time
from pathlib import Path

import cv2
import pandas as pd
import streamlit as st
import torch

THU_MUC_HIEN_TAI = Path(__file__).resolve().parent
if str(THU_MUC_HIEN_TAI) not in sys.path:
    sys.path.insert(0, str(THU_MUC_HIEN_TAI))

cau_hinh_stdout = getattr(sys.stdout, "reconfigure", None)
if callable(cau_hinh_stdout):
    cau_hinh_stdout(encoding="utf-8")

from luu_csdl import luu_ket_qua_bien_so, luu_nhieu_ket_qua_bien_so
from nguon.cau_hinh import (
    KICH_THUOC_SVTR_THEO_TY_LE,
    LOP_PHUONG_TIEN_COCO,
    MO_HINH_DETECT,
    MO_HINH_OCR,
    MO_HINH_PHUONG_TIEN,
    NHAN_HIEN_THI_OCR,
    THU_MUC_GOC,
    THU_MUC_MO_HINH,
    TY_LE_PADDING_CROP,
)
from nguon.mo_hinh_ai import nap_mo_hinh_yolo, nap_svtr26_onnx
from nguon.tien_xu_ly import doc_anh_tu_bytes, doc_anh_unicode, trich_nhan_tu_ten_file
from nguon.xu_ly_khung_hinh import xu_ly_danh_sach_anh_nhanh, xu_ly_mot_khung_hinh
from nguon.xu_ly_video_hud import BoTheoDoiVideoHUD
from nguon.xuat_bao_cao import DUONG_DAN_EXCEL_TONG_HOP, luu_ket_qua_muc_kiem_tra


DUOI_ANH_HOP_LE = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}
DUOI_VIDEO_HOP_LE = {".mp4", ".avi", ".mov", ".mkv", ".webm"}


def _doc_thong_so_video(duong_dan: str) -> dict:
    """Đọc nhanh độ phân giải, FPS và thời lượng video mà không tốn RAM nạp frame."""
    cap_tam = cv2.VideoCapture(duong_dan)
    if not cap_tam.isOpened():
        return {"hop_le": False, "rong": 0, "cao": 0, "fps": 0.0, "so_frame": 0, "thoi_luong_s": 0.0, "nhan_res": "Không rõ"}
    rong = int(cap_tam.get(cv2.CAP_PROP_FRAME_WIDTH))
    cao = int(cap_tam.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = float(cap_tam.get(cv2.CAP_PROP_FPS) or 0.0)
    so_frame = int(cap_tam.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    cap_tam.release()
    thoi_luong_s = (so_frame / fps) if (fps > 0 and so_frame > 0) else 0.0
    return {
        "hop_le": True,
        "rong": rong,
        "cao": cao,
        "fps": fps,
        "so_frame": so_frame,
        "thoi_luong_s": thoi_luong_s,
        "nhan_res": f"{cao}p ({rong}x{cao})",
    }

st.set_page_config(page_title="Nhận diện Biển số Xe Việt Nam", page_icon="🚘", layout="wide")
st.title("🚘 Nhận diện Biển số Xe Việt Nam (YOLO + SVTR26)")
st.session_state.setdefault("cac_ban_ghi_da_luu", {})


@st.cache_resource
def tai_yolo_cache(duong_dan: str, thiet_bi_chay: str):
    import numpy as np
    mo_hinh = nap_mo_hinh_yolo(duong_dan)
    # Warmup trước kernel CUDA FP16 để lần bấm đầu tiên không bị trễ khởi tạo
    nhap = [np.zeros((640, 640, 3), dtype=np.uint8) for _ in range(8)]
    mo_hinh.predict(
        source=nhap, imgsz=640, conf=0.25, half=(thiet_bi_chay != "cpu"), batch=8, device=thiet_bi_chay, verbose=False
    )
    mo_hinh.predict(
        source=nhap[0], imgsz=640, conf=0.25, half=(thiet_bi_chay != "cpu"), device=thiet_bi_chay, verbose=False
    )
    return mo_hinh


@st.cache_resource
def tai_svtr_cache(duong_dan: str):
    sess, ten_vao = nap_svtr26_onnx(duong_dan)
    import numpy as np
    # Warmup đủ 4 tỷ lệ kích thước SVTR26 để ngay cả ảnh đầu tiên cũng đạt tốc độ thực tế
    for img_w, img_h in KICH_THUOC_SVTR_THEO_TY_LE:
        for co_n in (1, 8):
            sess.run(None, {ten_vao: np.zeros((co_n, 3, img_h, img_w), dtype=np.float32)})
    return sess, ten_vao


@st.cache_resource
def tai_yolo_phuong_tien_cache(duong_dan: str, thiet_bi_chay: str):
    import numpy as np
    mo_hinh = nap_mo_hinh_yolo(duong_dan)
    mo_hinh.predict(
        source=np.zeros((640, 640, 3), dtype=np.uint8),
        imgsz=640,
        conf=0.35,
        half=(thiet_bi_chay != "cpu"),
        device=thiet_bi_chay,
        verbose=False,
    )
    return mo_hinh


# ----------------- THANH CẤU HÌNH BÊN TRÁI -----------------
co_cuda = torch.cuda.is_available()
thiet_bi = "0" if co_cuda else "cpu"

with st.sidebar:
    st.header("⚙️ Cấu hình")
    if co_cuda:
        st.success(f"GPU CUDA: **{torch.cuda.get_device_name(0)}**")
    else:
        st.warning("Đang chạy CPU")

    # Quét danh sách các model OCR SVTR26 ONNX trong thư mục mo_hinh/ (bỏ qua detector & bản svtr26 gốc)
    ds_file_ocr = sorted(
        p.name for p in THU_MUC_MO_HINH.glob("*.onnx")
        if "yolo" not in p.name.lower() and "416" not in p.name.lower() and p.name != "svtr26_goc.onnx"
    )
    if not ds_file_ocr:
        ds_file_ocr = [MO_HINH_OCR.name]
    idx_mac_dinh = ds_file_ocr.index(MO_HINH_OCR.name) if MO_HINH_OCR.name in ds_file_ocr else 0

    ten_ocr_chon = st.selectbox(
        "Model OCR biển số (GPU):",
        ds_file_ocr,
        index=idx_mac_dinh,
        format_func=lambda ten: NHAN_HIEN_THI_OCR.get(ten, ten),
    )
    duong_dan_ocr_chon = THU_MUC_MO_HINH / ten_ocr_chon

    st.caption(f"Model phát hiện biển số (GPU): LiteALPR YOLOv8n-Eff ({MO_HINH_DETECT.name})")

    nguong_conf = st.slider("Ngưỡng tin cậy YOLO:", 0.10, 0.90, 0.25, 0.05)
    imgsz = st.select_slider("Kích thước ảnh YOLO (imgsz):", [416, 640, 960], value=640)

    st.markdown("---")
    st.subheader("🔧 Tối ưu nhận diện")
    bo_qua_goc_pip = st.checkbox("Bỏ qua góc dưới phải (ảnh PiP phụ)", value=True)
    bat_chuan_hoa = st.checkbox("Chuẩn hóa biển số VN (Ô tô + Xe máy)", value=True)

    st.markdown("---")
    st.subheader("🚗 Lọc loại xe (Chế độ Video HUD)")
    ds_ten_loai_xe = list(LOP_PHUONG_TIEN_COCO.values())
    cac_loai_xe_chon = st.multiselect(
        "Theo dõi & đọc biển các nhóm xe:",
        options=ds_ten_loai_xe,
        default=ds_ten_loai_xe,
    )
    nguong_conf_xe = st.slider("Ngưỡng tin cậy phát hiện xe:", 0.20, 0.85, 0.35, 0.05)

# Kiểm tra và nạp model lên GPU
if not MO_HINH_DETECT.exists() or not duong_dan_ocr_chon.exists():
    st.error(f"Thiếu file model bắt buộc: `{MO_HINH_DETECT}` hoặc `{duong_dan_ocr_chon}`.")
    st.stop()

mo_hinh_yolo = tai_yolo_cache(str(MO_HINH_DETECT), thiet_bi)
bo_svtr_onnx = tai_svtr_cache(str(duong_dan_ocr_chon))
dang_chay_ocr_gpu = "CUDAExecutionProvider" in bo_svtr_onnx[0].get_providers()
st.sidebar.caption(
    f"Detect: {MO_HINH_DETECT.name} | OCR: {ten_ocr_chon} ({'GPU CUDA' if dang_chay_ocr_gpu else 'CPU'})"
)
PHIEN_BAN_MO_HINH = f"{MO_HINH_DETECT.name} + {ten_ocr_chon}"

# ----------------- CHỌN ĐẦU VÀO: ẢNH HOẶC VIDEO -----------------
loai_vao = st.radio("Chọn loại đầu vào:", ("🖼️ Ảnh", "🎬 Video"), horizontal=True)

if loai_vao == "🖼️ Ảnh":
    nguon_anh = st.radio(
        "Chế độ đầu vào ảnh:",
        ("Tải 1 ảnh từ máy", "Thư mục chứa nhiều ảnh (Folder)"),
        horizontal=True,
    )

    if nguon_anh == "Tải 1 ảnh từ máy":
        f_anh = st.file_uploader("Tải lên ảnh (.jpg, .jpeg, .png, .webp):", type=["jpg", "jpeg", "png", "webp"])
        if f_anh is not None:
            noi_dung_anh = f_anh.getvalue()
            khoa_anh = (
                hashlib.sha256(noi_dung_anh).hexdigest(), nguong_conf, imgsz,
                bo_qua_goc_pip, bat_chuan_hoa, PHIEN_BAN_MO_HINH, thiet_bi,
            )
            if st.session_state.get("khoa_anh_da_xu_ly") != khoa_anh:
                anh_bgr = doc_anh_tu_bytes(noi_dung_anh)
                if anh_bgr is None:
                    st.error("Không đọc được ảnh. Hãy chọn một file ảnh hợp lệ.")
                    st.stop()
                st.session_state["ket_qua_anh"] = xu_ly_mot_khung_hinh(
                    anh_bgr, mo_hinh_yolo, bo_svtr_onnx, thiet_bi,
                    nguong_conf, imgsz, bo_qua_goc_pip, bat_chuan_hoa,
                    model_version=PHIEN_BAN_MO_HINH,
                )
                st.session_state["khoa_anh_da_xu_ly"] = khoa_anh
            kq = st.session_state["ket_qua_anh"]

            m1, m2, m3, m4 = st.columns(4)
            m1.metric("⚡ YOLO Detect", f"{kq['yolo_ms']} ms")
            m2.metric("🔤 SVTR26 ONNX", f"{kq['ocr_ms']} ms")
            m3.metric("⏱️ Tổng thời gian", f"{kq['tong_ms']} ms")
            m4.metric("🎯 Số biển số", len(kq["cac_ket_qua"]))

            c_trai, c_phai = st.columns([1.45, 1.0])
            with c_trai:
                st.subheader(f"Ảnh kết quả: {f_anh.name}")
                st.image(cv2.cvtColor(kq["anh_ve"], cv2.COLOR_BGR2RGB), width="stretch")

            with c_phai:
                st.subheader("✂️ Vùng Crop & Kết quả Text")
                if not kq["cac_ket_qua"]:
                    st.warning("Không phát hiện được biển số nào. Thử giảm ngưỡng tin cậy YOLO.")
                for idx, item in enumerate(kq["cac_ket_qua"], 1):
                    with st.container(border=True):
                        st.markdown(f"### Biển #{idx}: `{item['text_chuan'] or '(Không đọc được)'}` — `{item['format_status']}`")
                        st.caption(f"Raw: `{item['raw_text']}` | Det: **{item['det_conf']}%** | OCR: **{item['ocr_conf']}%**")
                        st.image(cv2.cvtColor(item["crop_goc"], cv2.COLOR_BGR2RGB), caption="Ảnh crop biển số", width=260)
                        with st.expander("📦 Output chuẩn bàn giao CSDL (du_lieu_db)"):
                            st.json(item["du_lieu_db"])
                        khoa_luu = (*khoa_anh, idx)
                        da_luu = st.session_state["cac_ban_ghi_da_luu"]
                        if st.button(
                            "Lưu vào CSDL", key=f"luu_bien_so_{idx}",
                            disabled=khoa_luu in da_luu,
                        ):
                            try:
                                record_id = luu_ket_qua_bien_so(
                                    item["du_lieu_db"], kq["anh_goc"], item["crop_goc"],
                                )
                            except (sqlite3.Error, OSError, ValueError, TypeError) as exc:
                                st.error(f"Chưa lưu được vào CSDL: {exc}")
                            else:
                                da_luu[khoa_luu] = record_id
                                st.rerun()
                        if khoa_luu in da_luu:
                            st.success(f"Đã lưu vào CSDL — ID: {da_luu[khoa_luu]}")
        else:
            st.info("💡 Hãy tải 1 ảnh lên để chạy nhận diện.")

    else:
        duong_dan_mac_dinh = str(THU_MUC_GOC / "du_lieu" / "anh_mau")
        duong_dan_nhap = st.text_input(
            "Nhập đường dẫn thư mục chứa ảnh (có thể dán trực tiếp từ Windows Explorer):",
            value=duong_dan_mac_dinh,
        ).strip().strip('"').strip("'")

        thu_muc = Path(duong_dan_nhap)
        if not duong_dan_nhap or not thu_muc.exists() or not thu_muc.is_dir():
            st.warning(f"Không tìm thấy thư mục hợp lệ: `{duong_dan_nhap}`")
        else:
            ds_anh = sorted(p for p in thu_muc.rglob("*") if p.is_file() and p.suffix.lower() in DUOI_ANH_HOP_LE)
            st.caption(f"Tìm thấy **{len(ds_anh)}** ảnh (quét cả thư mục con) trong `{thu_muc}`.")

            if ds_anh and st.button(f"🚀 Xử lý toàn bộ {len(ds_anh)} ảnh", type="primary"):
                thanh_tien_do = st.progress(0.0, text="Đang bắt đầu xử lý...")
                bang_ket_qua = []
                tong_thoi_gian_ai_ms = 0.0
                so_phat_hien = 0
                so_hop_le = 0
                so_co_nhan = 0
                so_dung_nhan = 0
                ds_ocr_conf = []

                t_bat_dau = time.perf_counter()
                da_xu_ly = 0
                for ket_qua_lo in xu_ly_danh_sach_anh_nhanh(
                    ds_anh, mo_hinh_yolo, bo_svtr_onnx, thiet_bi,
                    nguong_conf, imgsz, bo_qua_goc_pip, bat_chuan_hoa,
                    kich_thuoc_lo=8, so_luong_doc=4, model_version=PHIEN_BAN_MO_HINH,
                ):
                    for duong_dan_anh, kq in ket_qua_lo:
                        da_xu_ly += 1
                        tong_thoi_gian_ai_ms += kq["tong_ms"]

                        nhan_that = trich_nhan_tu_ten_file(duong_dan_anh.name)
                        if kq["cac_ket_qua"]:
                            so_phat_hien += 1
                            b0 = kq["cac_ket_qua"][0]
                            bien_doc = b0["text_chuan"]
                            bien_doc_sach = re.sub(r"[^A-Z0-9]", "", bien_doc.upper())
                            det_conf = b0["det_conf"]
                            ocr_conf = b0["ocr_conf"]
                            trang_thai = b0["format_status"]
                            if trang_thai == "VALID":
                                so_hop_le += 1
                            du_lieu_db_anh = b0["du_lieu_db"]
                            ds_ocr_conf.append(ocr_conf)
                        else:
                            bien_doc, bien_doc_sach = "", ""
                            det_conf, ocr_conf, trang_thai = 0.0, 0.0, "KHONG_THAY"
                            du_lieu_db_anh = None

                        if nhan_that is not None:
                            so_co_nhan += 1
                            khop = bien_doc_sach == nhan_that
                            if khop:
                                so_dung_nhan += 1
                            ket_luan_khop = "✅ Đúng" if khop else "❌ Sai"
                        else:
                            ket_luan_khop = "—"

                        bang_ket_qua.append({
                            "Tên file": duong_dan_anh.name,
                            "Nhãn tên file": nhan_that or "—",
                            "Biển nhận diện": bien_doc or "(Không thấy)",
                            "Biển viết liền": bien_doc_sach,
                            "Đối chiếu": ket_luan_khop,
                            "Trạng thái": trang_thai,
                            "Det (%)": det_conf,
                            "OCR (%)": ocr_conf,
                            "Thời gian (ms)": kq["tong_ms"],
                            "_duong_dan": str(duong_dan_anh),
                            "_du_lieu_db": du_lieu_db_anh,
                        })

                    thanh_tien_do.progress(
                        min(1.0, da_xu_ly / len(ds_anh)),
                        text=f"Đang xử lý {da_xu_ly}/{len(ds_anh)} ảnh (Đa luồng đọc + Batch YOLO)...",
                    )

                tong_thoi_gian_thuc_s = time.perf_counter() - t_bat_dau
                thanh_tien_do.empty()

                st.session_state["kq_thu_muc"] = {
                    "ten_thu_muc": thu_muc.name,
                    "bang": bang_ket_qua,
                    "tong_anh": len(bang_ket_qua),
                    "so_phat_hien": so_phat_hien,
                    "so_hop_le": so_hop_le,
                    "so_co_nhan": so_co_nhan,
                    "so_dung_nhan": so_dung_nhan,
                    "tong_thuc_s": tong_thoi_gian_thuc_s,
                    "tong_ai_ms": tong_thoi_gian_ai_ms,
                    "ocr_tb": (sum(ds_ocr_conf) / len(ds_ocr_conf)) if ds_ocr_conf else 0.0,
                }
                st.session_state["da_luu_thu_muc"] = {}
                st.session_state.pop("thong_bao_xuat_anh", None)

            if "kq_thu_muc" in st.session_state:
                tk = st.session_state["kq_thu_muc"]
                tong_anh = max(1, tk["tong_anh"])
                tb_ms = tk["tong_ai_ms"] / tong_anh
                ty_le_det = (tk["so_phat_hien"] / tong_anh) * 100.0

                c1, c2, c3, c4 = st.columns(4)
                c1.metric("⏱️ Tổng thời gian xử lý", f"{tk['tong_thuc_s']:.2f} s", f"AI thuần: {tb_ms:.1f} ms/ảnh")
                if tk["so_co_nhan"] > 0:
                    do_chinh_xac = (tk["so_dung_nhan"] / tk["so_co_nhan"]) * 100.0
                    c2.metric("🎯 Độ chính xác (Exact Match)", f"{do_chinh_xac:.2f}%", f"{tk['so_dung_nhan']}/{tk['so_co_nhan']} ảnh đúng")
                else:
                    c2.metric("✅ Biển hợp lệ (VALID)", f"{tk.get('so_hop_le', 0)}/{tk['tong_anh']}", "Tên file không chứa nhãn")
                c3.metric("🔍 Tỷ lệ phát hiện biển", f"{ty_le_det:.1f}%", f"{tk['so_phat_hien']}/{tk['tong_anh']} ảnh")
                c4.metric("🔤 Độ tin cậy OCR TB", f"{tk['ocr_tb']:.1f}%")

                # Khung xuất Báo cáo Kết quả Kiểm tra (Excel phân Sheet theo từng mục + CSV riêng)
                with st.container(border=True):
                    st.markdown("**📊 Xuất file Báo cáo Kết quả Kiểm tra (Phân mục rõ ràng: 50 / 500 / 5000 / 10000 ảnh)**")
                    bx1, bx2, bx3 = st.columns([1.6, 1.1, 1.1])
                    ten_goi_y_anh = f"Anh_{tk.get('ten_thu_muc', 'Muc')}_{tk['tong_anh']}"
                    with bx1:
                        ten_muc_anh = st.text_input(
                            "Tên mục kiểm tra (tên Sheet trong file Báo cáo):",
                            value=ten_goi_y_anh,
                            key="ten_muc_bao_cao_anh",
                        )
                    with bx2:
                        st.write("")
                        if st.button("📤 Ghi vào Báo cáo (.xlsx & .csv)", type="primary", use_container_width=True, key="nut_xuat_anh"):
                            df_xuat_anh = pd.DataFrame([
                                {
                                    "STT": idx_d,
                                    "Tên file": d["Tên file"],
                                    "Biển nhận diện": "" if d["Biển nhận diện"] == "(Không thấy)" else d["Biển nhận diện"],
                                    "Biển viết liền": d.get("Biển viết liền", ""),
                                    "Trạng thái": d["Trạng thái"],
                                    "Det (%)": d["Det (%)"],
                                    "OCR (%)": d["OCR (%)"],
                                    "Thời gian AI (ms)": d["Thời gian (ms)"],
                                }
                                for idx_d, d in enumerate(tk["bang"], 1)
                            ])
                            dd_xlsx, dd_csv = luu_ket_qua_muc_kiem_tra(
                                nhom_kiem_tra="1. Kiểm tra Ảnh",
                                ten_muc=ten_muc_anh.strip() or ten_goi_y_anh,
                                quy_mo_do_phan_giai=f"{tk['tong_anh']} ảnh",
                                so_phat_hien=tk["so_phat_hien"],
                                so_hop_le=tk.get("so_hop_le", 0),
                                tong_thoi_gian_s=tk["tong_thuc_s"],
                                ai_trung_binh_ms=tb_ms,
                                thong_luong_hien_thi=f"{tk['tong_anh'] / max(1e-6, tk['tong_thuc_s']):.1f} ảnh/giây",
                                phien_ban_model=PHIEN_BAN_MO_HINH,
                                df_chi_tiet=df_xuat_anh,
                            )
                            st.session_state["thong_bao_xuat_anh"] = (str(dd_xlsx), str(dd_csv), ten_muc_anh.strip() or ten_goi_y_anh)
                    with bx3:
                        st.write("")
                        if DUONG_DAN_EXCEL_TONG_HOP.exists():
                            st.download_button(
                                "⬇️ Tải Báo cáo Tổng hợp (.xlsx)",
                                data=DUONG_DAN_EXCEL_TONG_HOP.read_bytes(),
                                file_name=DUONG_DAN_EXCEL_TONG_HOP.name,
                                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                                use_container_width=True,
                                key="tai_xlsx_anh",
                            )
                    if "thong_bao_xuat_anh" in st.session_state:
                        p_x, p_c, m_ten = st.session_state["thong_bao_xuat_anh"]
                        st.success(f"✅ Đã lưu mục **{m_ten}** vào file tổng hợp `{p_x}` và file riêng `{p_c}`!")

                da_luu_tm: dict[str, int] = st.session_state.setdefault("da_luu_thu_muc", {})

                # Khung lưu hàng loạt kết quả Folder vào CSDL
                with st.container(border=True):
                    cl1, cl2 = st.columns([1.8, 1.2])
                    with cl1:
                        pham_vi_luu = st.radio(
                            "💾 Tùy chọn lưu kết quả thư mục vào CSDL:",
                            ("Chỉ lưu biển hợp lệ (VALID)", "Lưu tất cả ảnh tìm thấy biển"),
                            horizontal=True,
                        )
                    ds_ung_vien_luu = [
                        dong for dong in tk["bang"]
                        if dong.get("_du_lieu_db") is not None
                        and dong["_duong_dan"] not in da_luu_tm
                        and (pham_vi_luu != "Chỉ lưu biển hợp lệ (VALID)" or dong["Trạng thái"] == "VALID")
                    ]
                    with cl2:
                        if st.button(
                            f"💾 Lưu {len(ds_ung_vien_luu)} ảnh vào CSDL",
                            disabled=(len(ds_ung_vien_luu) == 0),
                            use_container_width=True,
                        ):
                            thanh_luu = st.progress(0.0, text="Đang lưu ảnh & ghi vào SQLite...")
                            tong_can_luu = len(ds_ung_vien_luu)
                            da_ghi = 0
                            try:
                                for vi_tri in range(0, tong_can_luu, 40):
                                    nhom = ds_ung_vien_luu[vi_tri : vi_tri + 40]
                                    lo_ghi = []
                                    ds_duong_nhom = []
                                    for dong in nhom:
                                        anh_g = doc_anh_unicode(Path(dong["_duong_dan"]))
                                        if anh_g is None:
                                            continue
                                        hg, wg = anh_g.shape[:2]
                                        x1, y1, x2, y2 = dong["_du_lieu_db"]["plate_bbox"]
                                        px, py = int((x2 - x1) * TY_LE_PADDING_CROP), int((y2 - y1) * TY_LE_PADDING_CROP)
                                        anh_c = anh_g[max(0, y1 - py) : min(hg, y2 + py), max(0, x1 - px) : min(wg, x2 + px)]
                                        if anh_c.size == 0:
                                            continue
                                        lo_ghi.append((dong["_du_lieu_db"], anh_g, anh_c))
                                        ds_duong_nhom.append(dong["_duong_dan"])
                                    cac_id_moi = luu_nhieu_ket_qua_bien_so(lo_ghi)
                                    for dd, rid in zip(ds_duong_nhom, cac_id_moi):
                                        da_luu_tm[dd] = rid
                                    da_ghi += len(cac_id_moi)
                                    thanh_luu.progress(min(1.0, (vi_tri + len(nhom)) / tong_can_luu), text=f"Đã lưu {da_ghi}/{tong_can_luu} ảnh...")
                            except (sqlite3.Error, OSError, ValueError, TypeError) as exc:
                                thanh_luu.empty()
                                st.error(f"Lỗi khi lưu thư mục vào CSDL: {exc}")
                            else:
                                thanh_luu.empty()
                                st.rerun()
                    if da_luu_tm:
                        st.success(f"✅ Đã lưu tổng cộng **{len(da_luu_tm)}** ảnh từ thư mục này vào CSDL.")

                df_kq = pd.DataFrame(tk["bang"])
                df_kq["Trạng thái CSDL"] = df_kq["_duong_dan"].map(
                    lambda d: f"✅ ID: {da_luu_tm[d]}" if d in da_luu_tm else "—"
                )
                chi_xem_sai = st.checkbox("Chỉ lọc các ảnh nhận sai / không thấy biển", value=False)
                df_hien = df_kq[df_kq["Đối chiếu"] == "❌ Sai"] if (chi_xem_sai and tk["so_co_nhan"] > 0) else (
                    df_kq[df_kq["Trạng thái"] == "KHONG_THAY"] if chi_xem_sai else df_kq
                )
                df_bang_web = df_hien.head(1000) if len(df_hien) > 1000 else df_hien
                if len(df_hien) > 1000:
                    st.caption(
                        f"ℹ️ Đang hiển thị **1.000/{len(df_hien)}** dòng đầu trên trình duyệt để mượt trang web "
                        f"(file Báo cáo `.xlsx / .csv` vẫn xuất đầy đủ 100% cả **{len(tk['bang'])}** ảnh)."
                    )
                st.dataframe(df_bang_web.drop(columns=["_duong_dan", "_du_lieu_db"], errors="ignore"), use_container_width=True, hide_index=True)

                if not df_hien.empty:
                    ds_chon_xem = df_hien["Tên file"].head(500).tolist()
                    ten_xem = st.selectbox("Xem chi tiết ảnh kết quả (tối đa 500 mục đầu danh sách lọc):", ds_chon_xem)
                    dong_xem = df_hien.loc[df_hien["Tên file"] == ten_xem].iloc[0]
                    duong_dan_xem = dong_xem["_duong_dan"]
                    anh_xem = doc_anh_unicode(Path(duong_dan_xem))
                    if anh_xem is not None:
                        kq_xem = xu_ly_mot_khung_hinh(
                            anh_xem, mo_hinh_yolo, bo_svtr_onnx, thiet_bi,
                            nguong_conf, imgsz, bo_qua_goc_pip, bat_chuan_hoa,
                            model_version=PHIEN_BAN_MO_HINH,
                        )
                        st.image(cv2.cvtColor(kq_xem["anh_ve"], cv2.COLOR_BGR2RGB), caption=ten_xem, width=650)
                        if kq_xem["cac_ket_qua"]:
                            da_co_le = duong_dan_xem in da_luu_tm
                            if st.button(
                                f"💾 Lưu ảnh `{ten_xem}` vào CSDL",
                                key=f"luu_le_tm_{ten_xem}",
                                disabled=da_co_le,
                            ):
                                b_le = kq_xem["cac_ket_qua"][0]
                                rid_le = luu_ket_qua_bien_so(b_le["du_lieu_db"], kq_xem["anh_goc"], b_le["crop_goc"])
                                if rid_le is not None:
                                    da_luu_tm[duong_dan_xem] = rid_le
                                    st.rerun()
                            if da_co_le:
                                st.caption(f"✅ Ảnh này đã được lưu vào CSDL (ID: {da_luu_tm[duong_dan_xem]}).")

else:
    thu_muc_video_test = THU_MUC_GOC / "video test"
    ds_video_san = (
        sorted(p for p in thu_muc_video_test.iterdir() if p.is_file() and p.suffix.lower() in DUOI_VIDEO_HOP_LE)
        if thu_muc_video_test.exists()
        else []
    )

    cac_nguon_vid = ["📎 Nhập đường dẫn file / thư mục Video (Không giới hạn dung lượng — cho Video 10p & 30p)"]
    if ds_video_san:
        cac_nguon_vid.append("📂 Chọn nhanh từ thư mục video test")
    cac_nguon_vid.append("⬆️ Tải file video từ máy (tối đa 5 GB)")

    lua_chon_nguon_vid = st.radio("Nguồn video:", tuple(cac_nguon_vid), horizontal=True)

    duong_dan_video_chay: str | None = None
    ten_file_video_hien: str = "Video"
    f_vid_tai = None

    if lua_chon_nguon_vid.startswith("📎"):
        mac_dinh_vid = str(ds_video_san[0]) if ds_video_san else ""
        duong_vid_nhap = st.text_input(
            "Dán đường dẫn file Video (.mp4, .avi, .mov, .mkv) hoặc thư mục chứa các mức 144p → 1080p:",
            value=mac_dinh_vid,
        ).strip().strip('"').strip("'")
        if duong_vid_nhap:
            p_vid = Path(duong_vid_nhap)
            if p_vid.exists() and p_vid.is_file() and p_vid.suffix.lower() in DUOI_VIDEO_HOP_LE:
                duong_dan_video_chay = str(p_vid)
                ten_file_video_hien = p_vid.stem
            elif p_vid.exists() and p_vid.is_dir():
                ds_vid_trong_tm = sorted(
                    q for q in p_vid.rglob("*") if q.is_file() and q.suffix.lower() in DUOI_VIDEO_HOP_LE
                )
                if ds_vid_trong_tm:
                    chon_rel = st.selectbox(
                        f"Tìm thấy {len(ds_vid_trong_tm)} video trong thư mục — chọn file muốn kiểm tra:",
                        ds_vid_trong_tm,
                        format_func=lambda q: str(q.relative_to(p_vid)),
                    )
                    duong_dan_video_chay = str(chon_rel)
                    ten_file_video_hien = chon_rel.stem
                else:
                    st.warning(f"Không tìm thấy file video hợp lệ nào trong thư mục `{p_vid}`.")
            else:
                st.warning(f"Đường dẫn không tồn tại hoặc không phải file video hợp lệ: `{duong_vid_nhap}`")

    elif lua_chon_nguon_vid.startswith("📂"):
        ten_clip = st.selectbox(
            f"Chọn video ({len(ds_video_san)} clip trong thư mục `video test`):",
            [p.name for p in ds_video_san],
        )
        duong_dan_video_chay = str(thu_muc_video_test / ten_clip)
        ten_file_video_hien = Path(ten_clip).stem

    else:
        f_vid_tai = st.file_uploader("Tải lên video (.mp4, .avi, .mov, .mkv — tối đa 5 GB):", type=["mp4", "avi", "mov", "mkv"])
        if f_vid_tai is not None:
            ten_file_video_hien = Path(f_vid_tai.name).stem

    cv_opt1, cv_opt2 = st.columns([1.1, 1.7])
    with cv_opt1:
        buoc_frame = st.slider("Xử lý mỗi N frame (30 FPS chọn 2–3, 60 FPS chọn 3–5):", 1, 10, 3)
    with cv_opt2:
        st.write("")
        quet_nhanh_vid = st.checkbox(
            "⚡ Quét nhanh Video dài (10–30 phút): Giữ 100% độ chính xác AI, chỉ cập nhật hình minh họa mỗi 30 bước",
            value=False,
        )

    if duong_dan_video_chay is not None:
        ts_xem_truoc = _doc_thong_so_video(duong_dan_video_chay)
        if ts_xem_truoc["hop_le"]:
            st.caption(
                f"🎬 Thông số video: **{ts_xem_truoc['nhan_res']}** | **{ts_xem_truoc['fps']:.1f} FPS** | "
                f"**{ts_xem_truoc['so_frame']:,} frame** (~**{ts_xem_truoc['thoi_luong_s'] / 60.0:.2f} phút**)"
            )

    ds_ma_lop_xe = [ma for ma, ten in LOP_PHUONG_TIEN_COCO.items() if ten in cac_loai_xe_chon]

    if "vid_dang_chay" not in st.session_state:
        st.session_state.vid_dang_chay = False

    b1, b2, _ = st.columns([1, 1, 4])
    if b1.button("▶️ Bắt đầu nhận diện HUD", type="primary", use_container_width=True):
        st.session_state.vid_dang_chay = True
        st.session_state["ds_xe_video_cho_luu"] = []
        st.session_state["bien_video_da_luu"] = {}
        st.session_state.pop("thong_ke_video", None)
        st.session_state.pop("thong_bao_xuat_vid", None)
    if b2.button("⏹️ Dừng", use_container_width=True):
        st.session_state.vid_dang_chay = False

    co_nguon_hop_le = (duong_dan_video_chay is not None) or (f_vid_tai is not None)
    if co_nguon_hop_le and st.session_state.vid_dang_chay:
        if not MO_HINH_PHUONG_TIEN.exists():
            st.error(f"Thiếu model phát hiện phương tiện: `{MO_HINH_PHUONG_TIEN}`")
            st.stop()
        if not ds_ma_lop_xe:
            st.warning("Hãy chọn ít nhất 1 loại phương tiện ở thanh bên trái.")
            st.stop()

        mo_hinh_xe = tai_yolo_phuong_tien_cache(str(MO_HINH_PHUONG_TIEN), thiet_bi)
        bo_hud = BoTheoDoiVideoHUD(chieu_rong_hien=1280, chieu_rong_hud=300, so_the_hud=4)
        bo_hud.dat_lai(mo_hinh_xe)

        if duong_dan_video_chay is not None:
            duong_mo_cap = duong_dan_video_chay
        else:
            tam = tempfile.NamedTemporaryFile(delete=False, suffix=Path(f_vid_tai.name).suffix)
            while True:
                manh = f_vid_tai.read(16 * 1024 * 1024)
                if not manh:
                    break
                tam.write(manh)
            tam.close()
            duong_mo_cap = tam.name

        ts_vid = _doc_thong_so_video(duong_mo_cap)
        cap = cv2.VideoCapture(duong_mo_cap)
        tong_frame_vid = ts_vid["so_frame"]

        t1, t2, t3, t4 = st.columns(4)
        the_xe, the_bien, the_fps, the_chot = t1.empty(), t2.empty(), t3.empty(), t4.empty()
        thanh_vid = st.progress(0.0, text="Đang khởi chạy luồng Video HUD...")
        khung_vid = st.empty()

        chi_so = 0
        so_chot_truoc = -1
        tong_ai_vid_ms = 0.0
        t_bat_dau_vid = time.perf_counter()

        while cap.isOpened() and st.session_state.vid_dang_chay:
            ok, frame = cap.read()
            if not ok:
                st.session_state.vid_dang_chay = False
                break
            chi_so += 1
            if chi_so % buoc_frame != 0:
                continue

            buoc_tiep = bo_hud.buoc_hien_tai + 1
            can_ve_hud = (not quet_nhanh_vid) or (buoc_tiep == 1) or (buoc_tiep % 30 == 0)

            kq_v = bo_hud.xu_ly_khung_video(
                frame,
                mo_hinh_xe=mo_hinh_xe,
                mo_hinh_bien=mo_hinh_yolo,
                bo_svtr_onnx=bo_svtr_onnx,
                thiet_bi=thiet_bi,
                ds_ma_lop_xe=ds_ma_lop_xe,
                nguong_conf_xe=nguong_conf_xe,
                nguong_conf_bien=nguong_conf,
                bat_chuan_hoa=bat_chuan_hoa,
                model_version=PHIEN_BAN_MO_HINH,
                ve_hud=can_ve_hud,
            )
            tong_ai_vid_ms += kq_v["tong_ms"]

            # Đồng bộ vào session_state định kỳ để dù bấm Dừng giữa chừng vẫn giữ nguyên kết quả & thống kê
            if kq_v["tong_xe_da_chot_bien"] != so_chot_truoc or bo_hud.buoc_hien_tai % 15 == 0:
                st.session_state["ds_xe_video_cho_luu"] = bo_hud.lay_danh_sach_chot_duy_nhat()
                so_chot_truoc = kq_v["tong_xe_da_chot_bien"]
                da_chay_s = time.perf_counter() - t_bat_dau_vid
                st.session_state["thong_ke_video"] = {
                    "ten_video": ten_file_video_hien,
                    "cao_p": ts_vid["cao"],
                    "nhan_res": ts_vid["nhan_res"],
                    "thoi_luong_vid_s": ts_vid["thoi_luong_s"],
                    "frame_da_doc": chi_so,
                    "tong_frame_vid": tong_frame_vid,
                    "so_buoc_ai": bo_hud.buoc_hien_tai,
                    "buoc_frame": buoc_frame,
                    "tong_thuc_s": da_chay_s,
                    "tb_ai_ms": tong_ai_vid_ms / max(1, bo_hud.buoc_hien_tai),
                    "quet_nhanh": quet_nhanh_vid,
                }

            cap_nhat_chi_so = (not quet_nhanh_vid) or can_ve_hud or (bo_hud.buoc_hien_tai % 10 == 0)
            if cap_nhat_chi_so:
                the_xe.metric("🚗 Track Xe (ByteTrack)", f"{kq_v['xe_ms']} ms", f"{kq_v['so_xe_khung_hinh']} xe trong hình")
                the_bien.metric("🔤 Detect + SVTR26", f"{kq_v['bien_ocr_ms']} ms")
                the_fps.metric("🚀 Tốc độ AI", f"{1000.0 / max(1.0, kq_v['tong_ms']):.1f} FPS", f"Tổng {kq_v['tong_ms']} ms")
                the_chot.metric("🎯 Đã chốt biển số", f"{kq_v['tong_xe_da_chot_bien']} xe")
                if tong_frame_vid > 0:
                    da_chay_s = time.perf_counter() - t_bat_dau_vid
                    fps_thuc_vid = chi_so / max(1e-6, da_chay_s)
                    con_lai_s = max(0.0, (tong_frame_vid - chi_so) / max(1e-6, fps_thuc_vid))
                    thanh_vid.progress(
                        min(1.0, chi_so / tong_frame_vid),
                        text=f"Frame {chi_so:,}/{tong_frame_vid:,} | Tốc độ luồng: {fps_thuc_vid:.1f} frame/s | Ước còn: {con_lai_s:.0f}s",
                    )

            if kq_v["anh_ve"] is not None:
                khung_vid.image(cv2.cvtColor(kq_v["anh_ve"], cv2.COLOR_BGR2RGB), use_container_width=True)

        cap.release()
        thanh_vid.empty()
        tong_thuc_vid_s = time.perf_counter() - t_bat_dau_vid
        st.session_state["ds_xe_video_cho_luu"] = bo_hud.lay_danh_sach_chot_duy_nhat()
        st.session_state["thong_ke_video"] = {
            "ten_video": ten_file_video_hien,
            "cao_p": ts_vid["cao"],
            "nhan_res": ts_vid["nhan_res"],
            "thoi_luong_vid_s": ts_vid["thoi_luong_s"],
            "frame_da_doc": chi_so,
            "tong_frame_vid": tong_frame_vid,
            "so_buoc_ai": bo_hud.buoc_hien_tai,
            "buoc_frame": buoc_frame,
            "tong_thuc_s": tong_thuc_vid_s,
            "tb_ai_ms": tong_ai_vid_ms / max(1, bo_hud.buoc_hien_tai),
            "quet_nhanh": quet_nhanh_vid,
        }

    elif not st.session_state.get("ds_xe_video_cho_luu") and "thong_ke_video" not in st.session_state:
        st.info("💡 Dán đường dẫn file/thư mục video (hoặc chọn clip mẫu) rồi bấm **▶️ Bắt đầu nhận diện HUD**.")

    ds_cho_luu: list[dict] = st.session_state.get("ds_xe_video_cho_luu", [])
    tk_vid: dict | None = st.session_state.get("thong_ke_video")

    # Khung xuất Báo cáo Kết quả Kiểm tra cho Mục Video (hỗ trợ cả 6 mức 144p -> 1080p & 10p/30p)
    if tk_vid is not None:
        with st.container(border=True):
            st.markdown(
                f"**📊 Xuất file Báo cáo Kết quả Kiểm tra (Mục Video: `{tk_vid['ten_video']}` — {tk_vid['nhan_res']})**  \n"
                f"Đã quét **{tk_vid['frame_da_doc']:,}** frame ({tk_vid['so_buoc_ai']:,} bước AI, mỗi {tk_vid['buoc_frame']} frame) "
                f"trong **{tk_vid['tong_thuc_s']:.2f} giây** | AI TB: **{tk_vid['tb_ai_ms']:.1f} ms/bước** | "
                f"Chốt được: **{len(ds_cho_luu)} biển số hợp lệ**"
            )
            vx1, vx2, vx3 = st.columns([1.6, 1.1, 1.1])
            ten_goi_y_vid = f"Video_{tk_vid['ten_video']}_{tk_vid['cao_p']}p" if tk_vid["cao_p"] else f"Video_{tk_vid['ten_video']}"
            with vx1:
                ten_muc_vid = st.text_input(
                    "Tên mục kiểm tra Video (tên Sheet trong file Báo cáo):",
                    value=ten_goi_y_vid,
                    key="ten_muc_bao_cao_vid",
                )
            with vx2:
                st.write("")
                if st.button("📤 Ghi vào Báo cáo (.xlsx & .csv)", type="primary", use_container_width=True, key="nut_xuat_vid"):
                    df_xuat_vid = pd.DataFrame([
                        {
                            "STT": idx_v,
                            "Mã theo dõi (#ID)": m["nhan_id"],
                            "Loại phương tiện": m["loai_xe"],
                            "Biển số chốt": m["bien_so"],
                            "Biển viết liền": re.sub(r"[^A-Z0-9]", "", m["bien_so"].upper()),
                            "Trạng thái": m["du_lieu_db"]["format_status"],
                            "Det (%)": round(m["du_lieu_db"]["det_confidence"] * 100.0, 1),
                            "OCR (%)": m["ocr_phan_tram"],
                            "Số lần xác nhận": m["so_lan_xac_nhan"],
                            "Thời gian AI (ms)": m["du_lieu_db"]["total_latency_ms"],
                        }
                        for idx_v, m in enumerate(ds_cho_luu, 1)
                    ])
                    if df_xuat_vid.empty:
                        df_xuat_vid = pd.DataFrame(columns=[
                            "STT", "Mã theo dõi (#ID)", "Loại phương tiện", "Biển số chốt",
                            "Biển viết liền", "Trạng thái", "Det (%)", "OCR (%)", "Số lần xác nhận", "Thời gian AI (ms)",
                        ])
                    fps_luong = tk_vid["frame_da_doc"] / max(1e-6, tk_vid["tong_thuc_s"])
                    dd_xlsx_v, dd_csv_v = luu_ket_qua_muc_kiem_tra(
                        nhom_kiem_tra="2. Kiểm tra Video",
                        ten_muc=ten_muc_vid.strip() or ten_goi_y_vid,
                        quy_mo_do_phan_giai=f"{tk_vid['nhan_res']} | {tk_vid['frame_da_doc']} frame (bước {tk_vid['buoc_frame']})",
                        so_phat_hien=len(ds_cho_luu),
                        so_hop_le=len(ds_cho_luu),
                        tong_thoi_gian_s=tk_vid["tong_thuc_s"],
                        ai_trung_binh_ms=tk_vid["tb_ai_ms"],
                        thong_luong_hien_thi=f"{fps_luong:.1f} frame/giây",
                        phien_ban_model=PHIEN_BAN_MO_HINH,
                        df_chi_tiet=df_xuat_vid,
                    )
                    st.session_state["thong_bao_xuat_vid"] = (str(dd_xlsx_v), str(dd_csv_v), ten_muc_vid.strip() or ten_goi_y_vid)
            with vx3:
                st.write("")
                if DUONG_DAN_EXCEL_TONG_HOP.exists():
                    st.download_button(
                        "⬇️ Tải Báo cáo Tổng hợp (.xlsx)",
                        data=DUONG_DAN_EXCEL_TONG_HOP.read_bytes(),
                        file_name=DUONG_DAN_EXCEL_TONG_HOP.name,
                        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                        use_container_width=True,
                        key="tai_xlsx_vid",
                    )
            if "thong_bao_xuat_vid" in st.session_state:
                p_xv, p_cv, m_ten_v = st.session_state["thong_bao_xuat_vid"]
                st.success(f"✅ Đã lưu mục **{m_ten_v}** vào file tổng hợp `{p_xv}` và file riêng `{p_cv}`!")

    if ds_cho_luu:
        st.subheader(f"📋 Danh sách {len(ds_cho_luu)} phương tiện nhận diện từ video (Xem trước & Chọn lưu CSDL)")
        bien_da_luu_vid: dict[str, int] = st.session_state.setdefault("bien_video_da_luu", {})

        df_vid = pd.DataFrame([
            {
                "Chọn lưu": (muc["bien_so"] not in bien_da_luu_vid),
                "Mã xe (#ID)": muc["nhan_id"],
                "Loại xe": muc["loai_xe"],
                "Biển số chốt": muc["bien_so"],
                "Độ tin cậy OCR (%)": muc["ocr_phan_tram"],
                "Số lần xác nhận": muc["so_lan_xac_nhan"],
                "Trạng thái CSDL": (
                    f"✅ Đã lưu (ID: {bien_da_luu_vid[muc['bien_so']]})"
                    if muc["bien_so"] in bien_da_luu_vid
                    else "⏳ Chờ lưu"
                ),
            }
            for muc in ds_cho_luu
        ])

        df_sua = st.data_editor(
            df_vid,
            disabled=["Mã xe (#ID)", "Loại xe", "Biển số chốt", "Độ tin cậy OCR (%)", "Số lần xác nhận", "Trạng thái CSDL"],
            use_container_width=True,
            hide_index=True,
            key="bang_chon_luu_video",
        )

        ds_bien_duoc_tich = [
            hang["Biển số chốt"]
            for _, hang in df_sua.iterrows()
            if bool(hang["Chọn lưu"]) and hang["Biển số chốt"] not in bien_da_luu_vid
        ]

        cv1, cv2_col = st.columns([1.4, 2.0])
        with cv1:
            if st.button(
                f"💾 Lưu {len(ds_bien_duoc_tich)} phương tiện đã chọn vào CSDL",
                type="primary",
                disabled=(len(ds_bien_duoc_tich) == 0),
                use_container_width=True,
            ):
                map_muc = {m["bien_so"]: m for m in ds_cho_luu}
                lo_luu_vid = []
                for bs in ds_bien_duoc_tich:
                    if bs not in map_muc:
                        continue
                    muc_luu = map_muc[bs]
                    anh_goc_giai_nen = doc_anh_tu_bytes(muc_luu["anh_goc_bytes"]) if muc_luu.get("anh_goc_bytes") else None
                    if anh_goc_giai_nen is None:
                        anh_goc_giai_nen = muc_luu["anh_crop"]
                    lo_luu_vid.append((muc_luu["du_lieu_db"], anh_goc_giai_nen, muc_luu["anh_crop"]))
                try:
                    ds_id_vid = luu_nhieu_ket_qua_bien_so(lo_luu_vid)
                except (sqlite3.Error, OSError, ValueError, TypeError) as exc:
                    st.error(f"Chưa lưu được danh sách video vào CSDL: {exc}")
                else:
                    for bs, rid in zip(ds_bien_duoc_tich, ds_id_vid):
                        bien_da_luu_vid[bs] = rid
                    st.rerun()
        with cv2_col:
            if bien_da_luu_vid:
                st.success(f"✅ Đã lưu **{len(bien_da_luu_vid)}/{len(ds_cho_luu)}** phương tiện của video này vào CSDL!")

        with st.expander("🔍 Xem trước ảnh Crop & Toàn cảnh của phương tiện trong danh sách"):
            bien_xem_vid = st.selectbox("Chọn biển số muốn kiểm tra:", [m["bien_so"] for m in ds_cho_luu])
            muc_xem = next(m for m in ds_cho_luu if m["bien_so"] == bien_xem_vid)
            px1, px2 = st.columns([1.0, 2.2])
            with px1:
                st.image(cv2.cvtColor(muc_xem["anh_crop"], cv2.COLOR_BGR2RGB), caption=f"Crop: {bien_xem_vid}", width=240)
                st.json(muc_xem["du_lieu_db"])
            with px2:
                anh_toan_canh = doc_anh_tu_bytes(muc_xem["anh_goc_bytes"]) if muc_xem.get("anh_goc_bytes") else None
                if anh_toan_canh is not None:
                    st.image(cv2.cvtColor(anh_toan_canh, cv2.COLOR_BGR2RGB), caption="Khung hình toàn cảnh tương ứng", use_container_width=True)
