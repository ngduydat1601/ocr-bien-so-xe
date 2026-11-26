"""Giao diện Web Demo Nhận diện Biển số Xe Việt Nam (Streamlit).

Sử dụng mặc định cặp mô hình tối ưu nhất cho cả Ô tô & Xe máy:
- Phát hiện biển số: litealpr_yolov8n_eff.pt (GPU FP16)
- Đọc ký tự OCR: svtr26_hon_hop.onnx (SVTR26-Tiny Đa năng Ô tô + Xe máy)
"""

import re
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

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

from nguon.cau_hinh import (
    LOP_PHUONG_TIEN_COCO,
    MO_HINH_DETECT,
    MO_HINH_OCR,
    MO_HINH_PHUONG_TIEN,
    NHAN_HIEN_THI_OCR,
    THU_MUC_GOC,
    THU_MUC_MO_HINH,
)
from nguon.mo_hinh_ai import BoGopHaiDetector, nap_mo_hinh_yolo, nap_svtr26_onnx
from nguon.tien_xu_ly import doc_anh_tu_bytes, doc_anh_unicode, trich_nhan_tu_ten_file
from nguon.xu_ly_khung_hinh import xu_ly_danh_sach_anh_nhanh, xu_ly_mot_khung_hinh
from nguon.xu_ly_video_hud import BoTheoDoiVideoHUD


DUOI_ANH_HOP_LE = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}

st.set_page_config(page_title="Nhận diện Biển số Xe Việt Nam", page_icon="🚘", layout="wide")
st.title("🚘 Nhận diện Biển số Xe Việt Nam (YOLO + SVTR26)")


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
    for co_n in (1, 8):
        sess.run(None, {ten_vao: np.zeros((co_n, 3, 64, 64), dtype=np.float32)})
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

    # Quét danh sách các model OCR SVTR26 ONNX trong thư mục mo_hinh/ (bỏ qua file detector 416/yolo)
    ds_file_ocr = sorted(
        p.name for p in THU_MUC_MO_HINH.glob("*.onnx")
        if "yolo" not in p.name.lower() and "416" not in p.name.lower()
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

    # Lựa chọn model phát hiện biển số
    LUA_CHON_DETECT = {
        "litealpr_yolov8n_eff.pt": "⚡ LiteALPR YOLOv8n-Eff (nhanh nhất)",
        "yolo_bien_so_vn.pt": "🎯 YOLO Biển số VN (bắt cả ô biển góc màn hình)",
        "GOP_CA_HAI": "🔗 Gộp cả 2 model (không bỏ sót, chậm hơn ~2 lần)",
    }
    ten_detect_chon = st.selectbox(
        "Model phát hiện biển số (GPU):",
        list(LUA_CHON_DETECT.keys()),
        index=0,
        format_func=lambda k: LUA_CHON_DETECT[k],
    )

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
duong_dan_detect_vn = THU_MUC_MO_HINH / "yolo_bien_so_vn.pt"
can_file_detect = {
    "litealpr_yolov8n_eff.pt": [MO_HINH_DETECT],
    "yolo_bien_so_vn.pt": [duong_dan_detect_vn],
    "GOP_CA_HAI": [MO_HINH_DETECT, duong_dan_detect_vn],
}[ten_detect_chon]
thieu = [str(p) for p in can_file_detect + [duong_dan_ocr_chon] if not p.exists()]
if thieu:
    st.error(f"Thiếu file model bắt buộc: {thieu}")
    st.stop()

if ten_detect_chon == "GOP_CA_HAI":
    mo_hinh_yolo = BoGopHaiDetector(
        tai_yolo_cache(str(MO_HINH_DETECT), thiet_bi),
        tai_yolo_cache(str(duong_dan_detect_vn), thiet_bi),
    )
    ten_detect_hien = "litealpr_eff + yolo_bien_so_vn"
else:
    mo_hinh_yolo = tai_yolo_cache(str(can_file_detect[0]), thiet_bi)
    ten_detect_hien = ten_detect_chon
bo_svtr_onnx = tai_svtr_cache(str(duong_dan_ocr_chon))
dang_chay_ocr_gpu = "CUDAExecutionProvider" in bo_svtr_onnx[0].get_providers()
st.sidebar.caption(
    f"🚀 Detect: `{ten_detect_hien}` | OCR: `{ten_ocr_chon}` ({'GPU CUDA' if dang_chay_ocr_gpu else 'CPU'})"
)
PHIEN_BAN_MO_HINH = f"{ten_detect_hien} + {ten_ocr_chon}"

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
            anh_bgr = doc_anh_tu_bytes(f_anh.read())
            kq = xu_ly_mot_khung_hinh(
                anh_bgr, mo_hinh_yolo, bo_svtr_onnx, thiet_bi,
                nguong_conf, imgsz, bo_qua_goc_pip, bat_chuan_hoa,
                model_version=PHIEN_BAN_MO_HINH,
            )

            m1, m2, m3, m4 = st.columns(4)
            m1.metric("⚡ YOLO Detect", f"{kq['yolo_ms']} ms")
            m2.metric("🔤 SVTR26 ONNX", f"{kq['ocr_ms']} ms")
            m3.metric("⏱️ Tổng thời gian", f"{kq['tong_ms']} ms")
            m4.metric("🎯 Số biển số", len(kq["cac_ket_qua"]))

            c_trai, c_phai = st.columns([1.45, 1.0])
            with c_trai:
                st.subheader(f"Ảnh kết quả: {f_anh.name}")
                st.image(cv2.cvtColor(kq["anh_ve"], cv2.COLOR_BGR2RGB), use_container_width=True)

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
                da_xu_ly = 0
                for ket_qua_lo in xu_ly_danh_sach_anh_nhanh(
                    ds_anh, mo_hinh_yolo, bo_svtr_onnx, thiet_bi,
                    nguong_conf, imgsz, bo_qua_goc_pip, bat_chuan_hoa,
                    kich_thuoc_lo=8, so_luong_doc=4,
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

                    thanh_tien_do.progress(
                        min(1.0, da_xu_ly / len(ds_anh)),
                        text=f"Đang xử lý {da_xu_ly}/{len(ds_anh)} ảnh (Đa luồng đọc + Batch YOLO)...",
                    )

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
    thu_muc_video_test = THU_MUC_GOC / "video test"
    ds_video_san = (
        sorted(p for p in thu_muc_video_test.iterdir() if p.is_file() and p.suffix.lower() in {".mp4", ".avi", ".mov"})
        if thu_muc_video_test.exists()
        else []
    )

    lua_chon_nguon_vid = (
        st.radio(
            "Nguồn video:",
            ("📂 Chọn nhanh từ thư mục video test", "⬆️ Tải file video từ máy"),
            horizontal=True,
        )
        if ds_video_san
        else "⬆️ Tải file video từ máy"
    )

    duong_dan_video_chay: str | None = None
    f_vid_tai = None
    if lua_chon_nguon_vid == "📂 Chọn nhanh từ thư mục video test":
        c_chon, c_buoc = st.columns([1.6, 1.0])
        with c_chon:
            ten_clip = st.selectbox(
                f"Chọn video ({len(ds_video_san)} clip trong thư mục `video test`):",
                [p.name for p in ds_video_san],
            )
            duong_dan_video_chay = str(thu_muc_video_test / ten_clip)
        with c_buoc:
            buoc_frame = st.slider("Xử lý mỗi N frame (60 FPS khuyên dùng 3):", 1, 10, 3)
    else:
        f_vid_tai = st.file_uploader("Tải lên video (.mp4, .avi, .mov):", type=["mp4", "avi", "mov"])
        buoc_frame = st.slider("Xử lý mỗi N frame (giúp video chạy mượt realtime):", 1, 10, 3)

    ds_ma_lop_xe = [ma for ma, ten in LOP_PHUONG_TIEN_COCO.items() if ten in cac_loai_xe_chon]

    if "vid_dang_chay" not in st.session_state:
        st.session_state.vid_dang_chay = False

    b1, b2, _ = st.columns([1, 1, 4])
    if b1.button("▶️ Bắt đầu nhận diện HUD", type="primary", use_container_width=True):
        st.session_state.vid_dang_chay = True
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
            tam.write(f_vid_tai.read())
            tam.close()
            duong_mo_cap = tam.name

        cap = cv2.VideoCapture(duong_mo_cap)
        tong_frame_vid = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        t1, t2, t3, t4 = st.columns(4)
        the_xe, the_bien, the_fps, the_chot = t1.empty(), t2.empty(), t3.empty(), t4.empty()
        thanh_vid = st.progress(0.0, text="Đang khởi chạy luồng Video HUD...")
        khung_vid = st.empty()

        chi_so = 0
        while cap.isOpened() and st.session_state.vid_dang_chay:
            ok, frame = cap.read()
            if not ok:
                st.session_state.vid_dang_chay = False
                break
            chi_so += 1
            if chi_so % buoc_frame != 0:
                continue

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
            )

            the_xe.metric("🚗 Track Xe (ByteTrack)", f"{kq_v['xe_ms']} ms", f"{kq_v['so_xe_khung_hinh']} xe trong hình")
            the_bien.metric("🔤 Detect + SVTR26", f"{kq_v['bien_ocr_ms']} ms")
            the_fps.metric("🚀 Tốc độ AI", f"{1000.0 / max(1.0, kq_v['tong_ms']):.1f} FPS", f"Tổng {kq_v['tong_ms']} ms")
            the_chot.metric("🎯 Đã chốt biển số", f"{kq_v['tong_xe_da_chot_bien']} xe")

            if tong_frame_vid > 0:
                thanh_vid.progress(min(1.0, chi_so / tong_frame_vid), text=f"Frame {chi_so}/{tong_frame_vid}")

            khung_vid.image(cv2.cvtColor(kq_v["anh_ve"], cv2.COLOR_BGR2RGB), use_container_width=True)

        cap.release()
        thanh_vid.empty()

        # Lưu bảng tổng hợp các phương tiện đã chốt biển số trong phiên chạy
        ds_tong_ket = [
            {
                "Mã xe (#ID)": f"#{v['id']}",
                "Loại xe": v["loai_xe"],
                "Biển số chốt": v["bien_chot"],
                "Độ tin cậy OCR (%)": round(v["ocr_tot_nhat"] * 100, 1),
                "Số lần xác nhận": len(v["diem_bau_chon"]) and round(v["diem_bau_chon"][v["bien_chot"]] / max(0.01, v["ocr_tot_nhat"]), 1),
            }
            for v in sorted(bo_hud.ho_so_xe.values(), key=lambda x: x["id"])
            if v["bien_chot"]
        ]
        st.session_state["bang_xe_video"] = ds_tong_ket

    else:
        st.info("💡 Chọn một clip trong `video test` (hoặc tải video lên) rồi bấm **▶️ Bắt đầu nhận diện HUD**.")

    if st.session_state.get("bang_xe_video"):
        st.subheader("📋 Danh sách phương tiện đã chốt biển số trong video")
        st.dataframe(pd.DataFrame(st.session_state["bang_xe_video"]), use_container_width=True, hide_index=True)
