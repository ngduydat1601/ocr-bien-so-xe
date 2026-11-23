"""Khởi tạo và chạy suy luận 2 mô hình AI: YOLO Detector (litealpr_yolov8n_eff.pt) và SVTR26 ONNX (best.onnx)."""

import cv2
import numpy as np
import onnxruntime as ort
from PIL import Image
from torchvision import transforms as T
from torchvision.transforms import functional as F
from ultralytics import YOLO

from nguon.cau_hinh import KICH_THUOC_SVTR_THEO_TY_LE, TU_DIEN_LITEALPR


def nap_mo_hinh_yolo(duong_dan: str) -> YOLO:
    """Nạp mô hình YOLO phát hiện biển số (litealpr_yolov8n_eff.pt)."""
    return YOLO(duong_dan, task="detect")


def nap_svtr26_onnx(duong_dan_onnx: str) -> tuple[ort.InferenceSession, str]:
    """Nạp mô hình SVTR26-Tiny ONNX đọc ký tự biển số Việt Nam (best.onnx)."""
    san_co = ort.get_available_providers()
    providers = [p for p in ("CUDAExecutionProvider", "CPUExecutionProvider") if p in san_co]
    sess = ort.InferenceSession(duong_dan_onnx, providers=providers or ["CPUExecutionProvider"])
    return sess, sess.get_inputs()[0].name


def doc_ky_tu_bang_svtr26(
    anh_crop: np.ndarray,
    sess: ort.InferenceSession,
    input_name: str,
) -> tuple[str, float]:
    """Chạy suy luận SVTR26-Tiny ONNX chuẩn RatioRecTVResize + CTC Decode."""
    img = Image.fromarray(cv2.cvtColor(anh_crop, cv2.COLOR_BGR2RGB))
    w, h = img.size
    ratio = max(1, min(4, int(round(float(w) / max(1.0, float(h))))))
    img_w, img_h = KICH_THUOC_SVTR_THEO_TY_LE[ratio - 1]
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
