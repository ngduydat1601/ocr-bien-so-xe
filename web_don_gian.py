"""Giao diện Web Demo Nhận diện Biển số Xe Việt Nam (Streamlit).

Sử dụng cố định cặp mô hình tối ưu nhất:
- Phát hiện biển số: litealpr_yolov8n_eff.pt
- Đọc ký tự OCR: best.onnx (SVTR26-Tiny Fine-tuned VN)
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

from luu_csdl import luu_ket_qua_bien_so
from nguon.cau_hinh import (
    MO_HINH_DETECT,
    MO_HINH_OCR,
    PHIEN_BAN_MO_HINH,
    THU_MUC_GOC,
)
from nguon.mo_hinh_ai import nap_mo_hinh_yolo, nap_svtr26_onnx
from nguon.tien_xu_ly import doc_anh_tu_bytes, doc_anh_unicode, trich_nhan_tu_ten_file
from nguon.xu_ly_khung_hinh import xu_ly_mot_khung_hinh


DUOI_ANH_HOP_LE = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}

st.set_page_config(page_title="Nhận diện Biển số Xe Việt Nam", page_icon="🚘", layout="wide")
st.title("🚘 Nhận diện Biển số Xe Việt Nam (YOLO + SVTR26)")
st.session_state.setdefault("cac_ban_ghi_da_luu", {})


@st.cache_resource
def tai_yolo_cache(duong_dan: str):
    return nap_mo_hinh_yolo(duong_dan)


@st.cache_resource
def tai_svtr_cache(duong_dan: str):
    return nap_svtr26_onnx(duong_dan)


# ----------------- THANH CẤU HÌNH BÊN TRÁI -----------------
co_cuda = torch.cuda.is_available()
thiet_bi = "0" if co_cuda else "cpu"

with st.sidebar:
    st.header("⚙️ Cấu hình")
    if co_cuda:
        st.success(f"GPU CUDA: **{torch.cuda.get_device_name(0)}**")
    else:
        st.warning("Đang chạy CPU")

    st.info(
        f"**Model đang dùng:**\n"
        f"- Detect: `{MO_HINH_DETECT.name}`\n"
        f"- OCR: `{MO_HINH_OCR.name}`"
    )

    nguong_conf = st.slider("Ngưỡng tin cậy YOLO:", 0.10, 0.90, 0.25, 0.05)
    imgsz = st.select_slider("Kích thước ảnh YOLO (imgsz):", [416, 640, 960], value=640)

    st.markdown("---")
    st.subheader("🔧 Tối ưu cho ảnh bãi/hầm xe")
    bo_qua_goc_pip = st.checkbox("Bỏ qua góc dưới phải (ảnh PiP phụ)", value=True)
    bat_chuan_hoa = st.checkbox("Chuẩn hóa biển số VN (0↔O, 1↔I, 6↔G)", value=True)

# Kiểm tra và nạp cố định 2 model
if not MO_HINH_DETECT.exists() or not MO_HINH_OCR.exists():
    st.error(f"Thiếu file model bắt buộc: `{MO_HINH_DETECT}` hoặc `{MO_HINH_OCR}`.")
    st.stop()

mo_hinh_yolo = tai_yolo_cache(str(MO_HINH_DETECT))
bo_svtr_onnx = tai_svtr_cache(str(MO_HINH_OCR))

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
            ds_anh = sorted(p for p in thu_muc.iterdir() if p.is_file() and p.suffix.lower() in DUOI_ANH_HOP_LE)
            st.caption(f"Tìm thấy **{len(ds_anh)}** ảnh trong thư mục `{thu_muc}`.")

            if ds_anh and st.button(f"🚀 Xử lý toàn bộ {len(ds_anh)} ảnh", type="primary"):
                thanh_tien_do = st.progress(0.0, text="Đang bắt đầu xử lý...")
                bang_ket_qua = []
                tong_thoi_gian_ai_ms = 0.0
                so_phat_hien = 0
                so_co_nhan = 0
                so_dung_nhan = 0
                ds_ocr_conf = []

                t_bat_dau = time.perf_counter()
                for idx, duong_dan_anh in enumerate(ds_anh, 1):
                    anh_bgr = doc_anh_unicode(duong_dan_anh)
                    if anh_bgr is None:
                        continue

                    kq = xu_ly_mot_khung_hinh(
                        anh_bgr, mo_hinh_yolo, bo_svtr_onnx, thiet_bi,
                        nguong_conf, imgsz, bo_qua_goc_pip, bat_chuan_hoa,
                        model_version=PHIEN_BAN_MO_HINH,
                    )
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
                        ds_ocr_conf.append(ocr_conf)
                    else:
                        bien_doc, bien_doc_sach = "", ""
                        det_conf, ocr_conf, trang_thai = 0.0, 0.0, "KHONG_THAY"

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
                        "Đối chiếu": ket_luan_khop,
                        "Trạng thái": trang_thai,
                        "Det (%)": det_conf,
                        "OCR (%)": ocr_conf,
                        "Thời gian (ms)": kq["tong_ms"],
                        "_duong_dan": str(duong_dan_anh),
                    })
                    thanh_tien_do.progress(idx / len(ds_anh), text=f"Đang xử lý {idx}/{len(ds_anh)}: {duong_dan_anh.name}")

                tong_thoi_gian_thuc_s = time.perf_counter() - t_bat_dau
                thanh_tien_do.empty()

                st.session_state["kq_thu_muc"] = {
                    "bang": bang_ket_qua,
                    "tong_anh": len(bang_ket_qua),
                    "so_phat_hien": so_phat_hien,
                    "so_co_nhan": so_co_nhan,
                    "so_dung_nhan": so_dung_nhan,
                    "tong_thuc_s": tong_thoi_gian_thuc_s,
                    "tong_ai_ms": tong_thoi_gian_ai_ms,
                    "ocr_tb": (sum(ds_ocr_conf) / len(ds_ocr_conf)) if ds_ocr_conf else 0.0,
                }

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
                    c2.metric("🎯 Độ tin cậy OCR TB", f"{tk['ocr_tb']:.1f}%", "Tên file không chứa nhãn")
                c3.metric("🔍 Tỷ lệ phát hiện biển", f"{ty_le_det:.1f}%", f"{tk['so_phat_hien']}/{tk['tong_anh']} ảnh")
                c4.metric("🔤 Độ tin cậy OCR TB", f"{tk['ocr_tb']:.1f}%")

                df_kq = pd.DataFrame(tk["bang"])
                chi_xem_sai = st.checkbox("Chỉ lọc các ảnh nhận sai / không thấy biển", value=False)
                df_hien = df_kq[df_kq["Đối chiếu"] == "❌ Sai"] if (chi_xem_sai and tk["so_co_nhan"] > 0) else (
                    df_kq[df_kq["Trạng thái"] == "KHONG_THAY"] if chi_xem_sai else df_kq
                )
                st.dataframe(df_hien.drop(columns=["_duong_dan"]), use_container_width=True, hide_index=True)

                if not df_hien.empty:
                    ten_xem = st.selectbox("Xem chi tiết ảnh kết quả:", df_hien["Tên file"].tolist())
                    duong_dan_xem = df_hien.loc[df_hien["Tên file"] == ten_xem, "_duong_dan"].iloc[0]
                    anh_xem = doc_anh_unicode(Path(duong_dan_xem))
                    if anh_xem is not None:
                        kq_xem = xu_ly_mot_khung_hinh(
                            anh_xem, mo_hinh_yolo, bo_svtr_onnx, thiet_bi,
                            nguong_conf, imgsz, bo_qua_goc_pip, bat_chuan_hoa,
                            model_version=PHIEN_BAN_MO_HINH,
                        )
                        st.image(cv2.cvtColor(kq_xem["anh_ve"], cv2.COLOR_BGR2RGB), caption=ten_xem, width=650)

else:
    f_vid = st.file_uploader("Tải lên video (.mp4, .avi, .mov):", type=["mp4", "avi", "mov"])
    buoc_frame = st.slider("Xử lý mỗi N frame (giúp video chạy mượt realtime):", 1, 10, 3)

    if "vid_dang_chay" not in st.session_state:
        st.session_state.vid_dang_chay = False

    b1, b2, _ = st.columns([1, 1, 4])
    if b1.button("▶️ Bắt đầu", type="primary", use_container_width=True):
        st.session_state.vid_dang_chay = True
    if b2.button("⏹️ Dừng", use_container_width=True):
        st.session_state.vid_dang_chay = False

    if f_vid is not None and st.session_state.vid_dang_chay:
        tam = tempfile.NamedTemporaryFile(delete=False, suffix=Path(f_vid.name).suffix)
        tam.write(f_vid.read())
        cap = cv2.VideoCapture(tam.name)

        t1, t2, t3, t4 = st.columns(4)
        the_yolo, the_ocr, the_tong, the_fps = t1.empty(), t2.empty(), t3.empty(), t4.empty()

        c_vid, c_crop = st.columns([1.45, 1.0])
        khung_vid = c_vid.empty()
        khung_text = c_crop.empty()
        khung_crop = c_crop.empty()

        chi_so = 0
        while cap.isOpened() and st.session_state.vid_dang_chay:
            ok, frame = cap.read()
            if not ok:
                st.session_state.vid_dang_chay = False
                break
            chi_so += 1
            if chi_so % buoc_frame != 0:
                continue

            kq = xu_ly_mot_khung_hinh(
                frame, mo_hinh_yolo, bo_svtr_onnx, thiet_bi,
                nguong_conf, imgsz, bo_qua_goc_pip, bat_chuan_hoa,
                model_version=PHIEN_BAN_MO_HINH,
            )

            the_yolo.metric("⚡ YOLO Detect", f"{kq['yolo_ms']} ms")
            the_ocr.metric("🔤 SVTR26 ONNX", f"{kq['ocr_ms']} ms")
            the_tong.metric("⏱️ Tổng thời gian", f"{kq['tong_ms']} ms")
            the_fps.metric("🚀 FPS xử lý", f"{1000.0 / max(1.0, kq['tong_ms']):.1f}")

            h_g, w_g = kq["anh_ve"].shape[:2]
            anh_nho = cv2.resize(kq["anh_ve"], (800, int(h_g * 800 / max(1, w_g))))
            khung_vid.image(cv2.cvtColor(anh_nho, cv2.COLOR_BGR2RGB), use_container_width=True)

            if kq["cac_ket_qua"]:
                b0 = kq["cac_ket_qua"][0]
                khung_text.success(f"### `{b0['text_chuan'] or b0['raw_text']}`\nDet: {b0['det_conf']}% | OCR: {b0['ocr_conf']}%")
                khung_crop.image(cv2.cvtColor(b0["crop_goc"], cv2.COLOR_BGR2RGB), caption="Ảnh crop biển số", width=300)

        cap.release()
    else:
        st.info("💡 Tải video lên rồi bấm **▶️ Bắt đầu** để nhận diện.")
