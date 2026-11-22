"""Web Nhận diện Biển số Xe Việt Nam đơn giản (YOLO .pt + EasyOCR trên GPU CUDA).

Chức năng gọn nhẹ, tập trung đúng nhu cầu:
- Đầu vào: Ảnh (tải lên hoặc chọn từ thư mục test thực tế) hoặc Video.
- Suy luận GPU CUDA: YOLO (.pt) phát hiện biển số + EasyOCR đọc ký tự.
- Tối ưu riêng cho dữ liệu camera hầm/bãi đỗ xe: lọc ảnh chèn góc dưới phải (PiP),
  phóng to crop (320px), xoay thẳng nhẹ (Deskew), sắp xếp chuẩn 2 dòng, chuẩn hóa biển VN.
- Hiển thị trực quan: Ảnh/Video có khung, Ảnh crop biển số, Kết quả Text và Thời gian xử lý (ms).
"""

import re
import sys
import tempfile
import time
from pathlib import Path

import cv2
import easyocr
import numpy as np
import onnxruntime as ort
import streamlit as st
import torch
from PIL import Image
from torchvision import transforms as T
from torchvision.transforms import functional as F
from ultralytics import YOLO

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

THU_MUC_GOC = Path(__file__).resolve().parent
THU_MUC_MO_HINH = THU_MUC_GOC / "mo_hinh"
THU_MUC_TEST_MAC_DINH = Path(r"C:\Users\Admin\Downloads\drive-download-20261121T081239Z-1-001")
TU_DIEN_LITEALPR = ["<blank>"] + list("0123456789ABCDĐEFGHIJKLMNOPQRSTUVWXYZ")

# Bảng ánh xạ sửa lỗi nhầm ký tự theo ngữ cảnh vị trí biển số Việt Nam
SANG_SO = {"O": "0", "D": "0", "Q": "0", "I": "1", "L": "1", "T": "1", "Z": "2", "A": "4", "S": "5", "G": "6", "B": "8"}
SANG_CHU = {"0": "D", "1": "I", "2": "Z", "4": "A", "5": "S", "6": "G", "8": "B"}
# Vị trí thứ 4 trên biển 9 ký tự có thể là chữ cái (xe máy 50cc/điện: AB, MD, AA...)
# nên chỉ sửa các ký tự KHÔNG bao giờ cấp làm seri (I, O) sang số (1, 0)
SUA_VI_TRI_3 = {"I": "1", "O": "0"}

st.set_page_config(page_title="Nhận diện Biển số Xe Việt Nam (YOLO + SVTR26 Fine-tuned)", page_icon="🚘", layout="wide")
st.title("🚘 Nhận diện Biển số Xe Việt Nam (YOLO + SVTR26 Fine-tuned)")


@st.cache_resource
def tai_yolo(duong_dan_pt: str) -> YOLO:
    """Nạp và giữ mô hình YOLO .pt trong bộ nhớ."""
    return YOLO(duong_dan_pt)


@st.cache_resource
def tai_svtr26_onnx(duong_dan_onnx: str) -> tuple[ort.InferenceSession, str]:
    """Nạp và giữ mô hình LiteALPR SVTR26-Tiny ONNX (Fine-tuned VN)."""
    sess = ort.InferenceSession(duong_dan_onnx, providers=["CUDAExecutionProvider", "CPUExecutionProvider"])
    return sess, sess.get_inputs()[0].name


@st.cache_resource
def tai_easyocr(dung_gpu: bool) -> easyocr.Reader:
    """Nạp và giữ EasyOCR Reader trên GPU CUDA."""
    return easyocr.Reader(["en"], gpu=dung_gpu)


def doc_ky_tu_bang_svtr26(anh_crop: np.ndarray, sess: ort.InferenceSession, input_name: str) -> tuple[str, float]:
    """Chạy suy luận SVTR26-Tiny ONNX chuẩn RatioRecTVResize + CTC Decode."""
    base_shape = [[64, 64], [96, 48], [112, 40], [128, 32]]
    img = Image.fromarray(cv2.cvtColor(anh_crop, cv2.COLOR_BGR2RGB))
    w, h = img.size
    ratio = max(1, min(4, int(round(float(w) / max(1.0, float(h))))))
    img_w, img_h = base_shape[ratio - 1]
    resized = F.resize(img, (img_h, img_w), interpolation=T.InterpolationMode.BICUBIC)
    tensor = T.Compose([T.ToTensor(), T.Normalize(0.5, 0.5)])(resized).unsqueeze(0).numpy()

    preds = sess.run(None, {input_name: tensor})[0]
    probs = preds[0]
    if float(np.max(probs)) > 1.05 or float(np.min(probs)) < -0.05:
        exp_l = np.exp(probs - np.max(probs, axis=-1, keepdims=True))
        probs = exp_l / np.sum(exp_l, axis=-1, keepdims=True)
    indices = np.argmax(probs, axis=-1)
    max_probs = np.max(probs, axis=-1)

    ky_tu, diem = [], []
    truoc = -1
    for idx, sc in zip(indices, max_probs):
        if idx != 0 and idx != truoc and idx < len(TU_DIEN_LITEALPR):
            ky_tu.append(TU_DIEN_LITEALPR[idx])
            diem.append(float(sc))
        truoc = idx
    return "".join(ky_tu), (float(np.mean(diem)) if diem else 0.0)


def doc_ky_tu_bang_yolo(
    anh_crop: np.ndarray,
    mo_hinh_ky_tu: YOLO,
    thiet_bi: str,
    nguong_conf: float = 0.25,
) -> tuple[str, float, np.ndarray]:
    """Nhận diện từng ký tự bằng YOLO Character Detector, vẽ khung ký tự và sắp xếp 2 dòng."""
    valid_cls = [k for k, v in mo_hinh_ky_tu.names.items() if len(str(v).strip()) == 1]
    kqc = mo_hinh_ky_tu.predict(
        source=anh_crop,
        imgsz=320,
        conf=nguong_conf,
        iou=0.45,
        agnostic_nms=True,
        classes=valid_cls,
        device=thiet_bi,
        verbose=False,
    )[0]

    anh_ve_kt = anh_crop.copy()
    ds_kt = []
    if kqc.boxes is not None and len(kqc.boxes) > 0:
        for b, sc, cl in zip(
            kqc.boxes.xyxy.cpu().numpy(),
            kqc.boxes.conf.cpu().numpy(),
            kqc.boxes.cls.cpu().numpy().astype(int),
        ):
            x1, y1, x2, y2 = [int(v) for v in b]
            ten_kt = str(mo_hinh_ky_tu.names[cl]).upper()
            ds_kt.append((float((x1 + x2) / 2.0), float((y1 + y2) / 2.0), ten_kt, float(sc)))
            cv2.rectangle(anh_ve_kt, (x1, y1), (x2, y2), (0, 220, 255), 2)
            cv2.putText(anh_ve_kt, ten_kt, (x1, max(14, y1 - 3)), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 220, 255), 2)

    if not ds_kt:
        return "", 0.0, anh_ve_kt

    hh, ww = anh_crop.shape[:2]
    if ww / max(1, hh) < 2.2:
        ys = [k[1] for k in ds_kt]
        y_chia = (min(ys) + max(ys)) / 2.0
        dong1 = sorted([k for k in ds_kt if k[1] < y_chia], key=lambda k: k[0])
        dong2 = sorted([k for k in ds_kt if k[1] >= y_chia], key=lambda k: k[0])
        ds_kt = dong1 + dong2
    else:
        ds_kt.sort(key=lambda k: k[0])

    raw_text = "".join(k[2] for k in ds_kt)
    diem_tb = float(np.mean([k[3] for k in ds_kt]))
    return raw_text, diem_tb, anh_ve_kt


def doc_anh_unicode(duong_dan: Path) -> np.ndarray | None:
    """Đọc ảnh BGR hỗ trợ đường dẫn có dấu tiếng Việt trên Windows."""
    if not duong_dan.exists():
        return None
    return cv2.imdecode(np.frombuffer(duong_dan.read_bytes(), dtype=np.uint8), cv2.IMREAD_COLOR)


def xoay_thang_bien_so(anh_crop: np.ndarray) -> np.ndarray:
    """Ước lượng góc nghiêng cạnh ngang và xoay thẳng nhẹ vùng biển số."""
    xam = cv2.cvtColor(anh_crop, cv2.COLOR_BGR2GRAY) if anh_crop.ndim == 3 else anh_crop
    bien = cv2.Canny(xam, 50, 150)
    duong = cv2.HoughLinesP(
        bien, 1, np.pi / 180,
        threshold=max(15, xam.shape[1] // 4),
        minLineLength=max(15, xam.shape[1] // 3),
        maxLineGap=10,
    )
    if duong is None:
        return anh_crop
    cac_goc = []
    for x1, y1, x2, y2 in np.asarray(duong).reshape(-1, 4):
        g = float(np.degrees(np.arctan2(y2 - y1, x2 - x1)))
        if -25.0 <= g <= 25.0:
            cac_goc.append(g)
    if not cac_goc:
        return anh_crop
    goc = float(np.median(cac_goc))
    if abs(goc) < 1.2:
        return anh_crop
    h, w = anh_crop.shape[:2]
    ma_tran = cv2.getRotationMatrix2D((w / 2.0, h / 2.0), goc, 1.0)
    return cv2.warpAffine(anh_crop, ma_tran, (w, h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)


def chuan_hoa_bien_so_vn(van_ban_tho: str) -> str:
    """Chuẩn hóa định dạng và sửa nhầm ký tự theo ngữ cảnh vị trí biển số Việt Nam."""
    s = re.sub(r"[^A-Z0-9]", "", (van_ban_tho or "").upper())
    if len(s) < 7 or len(s) > 9:
        return s
    arr = list(s)
    # 2 ký tự đầu (mã tỉnh): bắt buộc là chữ số
    for idx in (0, 1):
        if arr[idx] in SANG_SO:
            arr[idx] = SANG_SO[arr[idx]]
    # Ký tự thứ 3 (seri chữ): bắt buộc là chữ cái
    if arr[2] in SANG_CHU:
        arr[2] = SANG_CHU[arr[2]]
    # Biển 9 ký tự xe máy VN: vị trí thứ 4 hay bị nhầm 1<->I/L/T, 0<->O
    if len(arr) == 9 and arr[3] in SUA_VI_TRI_3:
        arr[3] = SUA_VI_TRI_3[arr[3]]
    # Các ký tự số thứ tự phía sau: bắt buộc là chữ số
    bat_dau_so = 4 if len(arr) == 9 else 3
    for idx in range(bat_dau_so, len(arr)):
        if arr[idx] in SANG_SO:
            arr[idx] = SANG_SO[arr[idx]]
    sach = "".join(arr)
    if len(sach) == 9:
        return f"{sach[:4]}-{sach[4:7]}.{sach[7:]}"
    if len(sach) == 8:
        return f"{sach[:3]}-{sach[3:6]}.{sach[6:]}"
    return sach


def danh_gia_trang_thai_bien_so(van_ban_chuan: str, ocr_conf: float, nguong_hop_le: float = 0.60) -> str:
    """Đánh giá trạng thái hợp lệ định dạng biển số Việt Nam: VALID / UNCERTAIN / INVALID."""
    sach = re.sub(r"[^A-Z0-9]", "", (van_ban_chuan or "").upper())
    if not sach:
        return "INVALID"
    khop_chuan = bool(re.match(r"^[0-9]{2}[A-Z][0-9A-Z]?[0-9]{4,5}$", sach)) and len(sach) in (7, 8, 9)
    if khop_chuan and ocr_conf >= nguong_hop_le:
        return "VALID"
    if khop_chuan or (7 <= len(sach) <= 10):
        return "UNCERTAIN"
    return "INVALID"


def xu_ly_mot_khung_hinh(
    anh_bgr: np.ndarray,
    mo_hinh_yolo: YOLO,
    che_do_ocr: str,
    bo_svtr_onnx: tuple | None,
    bo_ocr_easy: easyocr.Reader | None,
    mo_hinh_ky_tu: YOLO | None,
    thiet_bi: str,
    nguong_conf: float,
    imgsz: int,
    bo_qua_goc_pip: bool,
    bat_toi_uu_crop: bool,
    bat_chuan_hoa: bool,
    model_version: str = "",
) -> dict:
    """Chạy YOLO phát hiện biển số + (SVTR26 Fine-tuned / YOLO Char / EasyOCR)."""
    h, w = anh_bgr.shape[:2]

    # 1. YOLO phát hiện biển số trên GPU
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
            if bo_qua_goc_pip and x1 > 0.72 * w and y1 > 0.62 * h:
                continue

            px, py = int((x2 - x1) * 0.05), int((y2 - y1) * 0.05)
            x1p, y1p = max(0, x1 - px), max(0, y1 - py)
            x2p, y2p = min(w, x2 + px), min(h, y2 + py)
            anh_crop = anh_bgr[y1p:y2p, x1p:x2p].copy()
            if anh_crop.size == 0:
                continue

            # Với SVTR26-Tiny: dùng trực tiếp crop gốc (khớp 100% lúc fine-tune)
            anh_vao_ocr = anh_crop
            if che_do_ocr != "svtr26_vn" and bat_toi_uu_crop:
                ch_h, ch_w = anh_crop.shape[:2]
                ty_le = 320.0 / max(1, ch_w)
                anh_vao_ocr = cv2.resize(anh_crop, (320, max(1, int(ch_h * ty_le))), interpolation=cv2.INTER_CUBIC)
                anh_vao_ocr = xoay_thang_bien_so(anh_vao_ocr)

            # 2. Đọc ký tự
            t1 = time.perf_counter()
            if che_do_ocr == "svtr26_vn" and bo_svtr_onnx is not None:
                raw_text, ocr_conf = doc_ky_tu_bang_svtr26(anh_vao_ocr, bo_svtr_onnx[0], bo_svtr_onnx[1])
                anh_hien_crop = anh_vao_ocr
            elif che_do_ocr == "yolo_char" and mo_hinh_ky_tu is not None:
                raw_text, ocr_conf, anh_hien_crop = doc_ky_tu_bang_yolo(anh_vao_ocr, mo_hinh_ky_tu, thiet_bi)
            else:
                ds_chu = bo_ocr_easy.readtext(anh_vao_ocr, allowlist="ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789-.")
                hh, ww = anh_vao_ocr.shape[:2]
                if ww / max(1, hh) < 2.2:
                    dong1 = sorted([r for r in ds_chu if np.mean([p[1] for p in r[0]]) < hh * 0.52], key=lambda r: min(p[0] for p in r[0]))
                    dong2 = sorted([r for r in ds_chu if np.mean([p[1] for p in r[0]]) >= hh * 0.52], key=lambda r: min(p[0] for p in r[0]))
                    ds_sap_xep = dong1 + dong2
                else:
                    ds_sap_xep = sorted(ds_chu, key=lambda r: min(p[0] for p in r[0]))
                raw_text = " ".join([r[1].strip() for r in ds_sap_xep if r[1].strip()])
                ocr_conf = float(np.mean([r[2] for r in ds_sap_xep])) if ds_sap_xep else 0.0
                anh_hien_crop = anh_vao_ocr
            t_ocr_bien_ms = (time.perf_counter() - t1) * 1000.0
            t_ocr_ms += t_ocr_bien_ms

            text_cuoi = chuan_hoa_bien_so_vn(raw_text) if bat_chuan_hoa else raw_text
            trang_thai_format = danh_gia_trang_thai_bien_so(text_cuoi, ocr_conf)

            cv2.rectangle(anh_ve, (x1, y1), (x2, y2), (0, 230, 80), 3)
            nhan = f"{text_cuoi or '?'} ({ocr_conf*100:.0f}%)"
            (tw, th), _ = cv2.getTextSize(nhan, cv2.FONT_HERSHEY_SIMPLEX, 0.8, 2)
            y_nen = max(0, y1 - th - 10)
            cv2.rectangle(anh_ve, (x1, y_nen), (x1 + tw + 10, y_nen + th + 10), (0, 230, 80), -1)
            cv2.putText(anh_ve, nhan, (x1 + 5, y_nen + th + 4), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 2)

            # Gói dữ liệu chuẩn tên cột Database sẵn sàng bàn giao cho tầng lưu CSDL
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
                "crop_xu_ly": anh_hien_crop,
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


# ----------------- THANH CẤU HÌNH BÊN TRÁI -----------------
co_cuda = torch.cuda.is_available()
thiet_bi = "0" if co_cuda else "cpu"

with st.sidebar:
    st.header("⚙️ Cấu hình")
    if co_cuda:
        st.success(f"GPU CUDA: **{torch.cuda.get_device_name(0)}**")
    else:
        st.warning("Đang chạy CPU")

    ds_pt = sorted(p for p in THU_MUC_MO_HINH.glob("*.pt") if "ky_tu" not in p.name)
    ten_pt_list = [p.name for p in ds_pt]
    mac_dinh_idx = ten_pt_list.index("yolo_bien_so_vn.pt") if "yolo_bien_so_vn.pt" in ten_pt_list else 0
    ten_pt_chon = st.selectbox("1. Model YOLO phát hiện biển số (.pt):", ten_pt_list, index=mac_dinh_idx)
    duong_dan_pt = str(THU_MUC_MO_HINH / ten_pt_chon)

    ten_ocr_chon = st.radio(
        "2. Bộ đọc ký tự (OCR):",
        (
            "🏆 SVTR26 Fine-tuned VN (95–98% Acc, ~14ms)",
            "YOLO Character Detector (GPU, ~22ms)",
            "EasyOCR (GPU, ~35ms)",
        ),
        index=0,
    )
    if "SVTR26" in ten_ocr_chon:
        che_do_ocr = "svtr26_vn"
        nhan_ocr_ui = "SVTR26 Fine-tuned"
    elif "YOLO" in ten_ocr_chon:
        che_do_ocr = "yolo_char"
        nhan_ocr_ui = "YOLO Char GPU"
    else:
        che_do_ocr = "easyocr"
        nhan_ocr_ui = "EasyOCR GPU"

    nguong_conf = st.slider("Ngưỡng tin cậy YOLO:", 0.10, 0.90, 0.25, 0.05)
    imgsz = st.select_slider("Kích thước ảnh YOLO (imgsz):", [640, 960], value=640)

    st.markdown("---")
    st.subheader("🔧 Tối ưu cho ảnh bãi/hầm xe")
    bo_qua_goc_pip = st.checkbox("Bỏ qua góc dưới phải (ảnh PiP phụ)", value=True)
    bat_toi_uu_crop = st.checkbox("Phóng to 320px + Deskew (cho YOLO Char / EasyOCR)", value=True)
    bat_chuan_hoa = st.checkbox("Chuẩn hóa biển số VN (0↔O, 1↔I, 6↔G)", value=True)

mo_hinh_yolo = tai_yolo(duong_dan_pt)
duong_dan_svtr = THU_MUC_MO_HINH / "best.onnx"
bo_svtr_onnx = tai_svtr26_onnx(str(duong_dan_svtr)) if (che_do_ocr == "svtr26_vn" and duong_dan_svtr.exists()) else None
duong_dan_ky_tu = THU_MUC_MO_HINH / "yolo_ky_tu_hf.pt"
mo_hinh_ky_tu = tai_yolo(str(duong_dan_ky_tu)) if (che_do_ocr == "yolo_char" and duong_dan_ky_tu.exists()) else None
bo_ocr_easy = tai_easyocr(dung_gpu=co_cuda) if che_do_ocr == "easyocr" else None
phien_ban_mo_hinh = f"{ten_pt_chon} + {che_do_ocr}"

# ----------------- CHỌN ĐẦU VÀO: ẢNH HOẶC VIDEO -----------------
loai_vao = st.radio("Chọn loại đầu vào:", ("🖼️ Ảnh", "🎬 Video"), horizontal=True)

if loai_vao == "🖼️ Ảnh":
    nguon_anh = st.radio(
        "Nguồn ảnh:",
        ("Tải ảnh lên từ máy", "Chọn nhanh từ thư mục test Downloads"),
        horizontal=True,
    )
    anh_bgr = None
    ten_hien_thi = ""

    if nguon_anh == "Tải ảnh lên từ máy":
        f_anh = st.file_uploader("Tải lên ảnh (.jpg, .jpeg, .png, .webp):", type=["jpg", "jpeg", "png", "webp"])
        if f_anh is not None:
            anh_bgr = cv2.imdecode(np.frombuffer(f_anh.read(), dtype=np.uint8), cv2.IMREAD_COLOR)
            ten_hien_thi = f_anh.name
    else:
        if THU_MUC_TEST_MAC_DINH.exists():
            ds_anh_test = sorted(THU_MUC_TEST_MAC_DINH.glob("*.jpg"))
            if ds_anh_test:
                chon_anh = st.selectbox("Chọn ảnh trong thư mục test:", [p.name for p in ds_anh_test])
                anh_bgr = doc_anh_unicode(THU_MUC_TEST_MAC_DINH / chon_anh)
                ten_hien_thi = chon_anh
        else:
            st.warning(f"Không tìm thấy thư mục `{THU_MUC_TEST_MAC_DINH}`.")

    if anh_bgr is not None:
        kq = xu_ly_mot_khung_hinh(
            anh_bgr, mo_hinh_yolo, che_do_ocr, bo_svtr_onnx, bo_ocr_easy, mo_hinh_ky_tu, thiet_bi,
            nguong_conf, imgsz, bo_qua_goc_pip, bat_toi_uu_crop, bat_chuan_hoa,
            model_version=phien_ban_mo_hinh,
        )

        # Thanh thời gian xử lý
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("⚡ YOLO Detect", f"{kq['yolo_ms']} ms")
        m2.metric(f"🔤 {nhan_ocr_ui}", f"{kq['ocr_ms']} ms")
        m3.metric("⏱️ Tổng thời gian", f"{kq['tong_ms']} ms")
        m4.metric("🎯 Số biển số", len(kq["cac_ket_qua"]))

        c_trai, c_phai = st.columns([1.45, 1.0])
        with c_trai:
            st.subheader(f"Ảnh kết quả: {ten_hien_thi}")
            st.image(cv2.cvtColor(kq["anh_ve"], cv2.COLOR_BGR2RGB), use_container_width=True)

        with c_phai:
            st.subheader("✂️ Vùng Crop & Kết quả Text")
            if not kq["cac_ket_qua"]:
                st.warning("Không phát hiện được biển số nào. Thử giảm ngưỡng tin cậy YOLO.")
            for idx, item in enumerate(kq["cac_ket_qua"], 1):
                with st.container(border=True):
                    st.markdown(f"### Biển #{idx}: `{item['text_chuan'] or '(Không đọc được)'}` — `{item['format_status']}`")
                    st.caption(
                        f"Raw: `{item['raw_text']}` | "
                        f"Det: **{item['det_conf']}%** | OCR: **{item['ocr_conf']}%**"
                    )
                    cc1, cc2 = st.columns(2)
                    cc1.image(cv2.cvtColor(item["crop_goc"], cv2.COLOR_BGR2RGB), caption="Crop gốc", use_container_width=True)
                    cc2.image(cv2.cvtColor(item["crop_xu_ly"], cv2.COLOR_BGR2RGB), caption="Crop đưa vào OCR", use_container_width=True)
                    with st.expander("📦 Output chuẩn bàn giao CSDL (du_lieu_db)"):
                        st.json(item["du_lieu_db"])
    else:
        st.info("💡 Hãy tải ảnh lên hoặc chọn ảnh trong thư mục test để chạy nhận diện.")

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
        the_yolo = t1.empty()
        the_ocr = t2.empty()
        the_tong = t3.empty()
        the_fps = t4.empty()

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
                frame, mo_hinh_yolo, che_do_ocr, bo_svtr_onnx, bo_ocr_easy, mo_hinh_ky_tu, thiet_bi,
                nguong_conf, imgsz, bo_qua_goc_pip, bat_toi_uu_crop, bat_chuan_hoa,
                model_version=phien_ban_mo_hinh,
            )

            the_yolo.metric("⚡ YOLO Detect", f"{kq['yolo_ms']} ms")
            the_ocr.metric(f"🔤 {nhan_ocr_ui}", f"{kq['ocr_ms']} ms")
            the_tong.metric("⏱️ Tổng thời gian", f"{kq['tong_ms']} ms")
            the_fps.metric("🚀 FPS xử lý", f"{1000.0 / max(1.0, kq['tong_ms']):.1f}")

            h_g, w_g = kq["anh_ve"].shape[:2]
            anh_nho = cv2.resize(kq["anh_ve"], (800, int(h_g * 800 / max(1, w_g))))
            khung_vid.image(cv2.cvtColor(anh_nho, cv2.COLOR_BGR2RGB), use_container_width=True)

            if kq["cac_ket_qua"]:
                b0 = kq["cac_ket_qua"][0]
                khung_text.success(f"### `{b0['text_chuan'] or b0['raw_text']}`\nDet: {b0['det_conf']}% | OCR: {b0['ocr_conf']}%")
                khung_crop.image(cv2.cvtColor(b0["crop_xu_ly"], cv2.COLOR_BGR2RGB), caption="Ảnh crop biển số", width=300)

        cap.release()
    else:
        st.info("💡 Tải video lên rồi bấm **▶️ Bắt đầu** để nhận diện.")
