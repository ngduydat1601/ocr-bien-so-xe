"""Khởi tạo và chạy suy luận 2 mô hình AI: YOLO Detector (litealpr_yolov8n_eff.pt) và SVTR26 ONNX (best.onnx)."""

from collections import defaultdict
import os
from pathlib import Path

import cv2
import numpy as np
import torch

# Đăng ký thư mục DLL CUDA/cuDNN có sẵn của PyTorch trên Windows cho onnxruntime-gpu
_THU_MUC_TORCH_LIB = Path(torch.__file__).resolve().parent / "lib"
if _THU_MUC_TORCH_LIB.exists():
    os.environ["PATH"] = str(_THU_MUC_TORCH_LIB) + os.pathsep + os.environ.get("PATH", "")
    if hasattr(os, "add_dll_directory"):
        os.add_dll_directory(str(_THU_MUC_TORCH_LIB))

import onnxruntime as ort
import logging

from PIL import Image
from torchvision import transforms as T
from torchvision.transforms import functional as F
from ultralytics import YOLO
from ultralytics.utils import LOGGER

from nguon.cau_hinh import KICH_THUOC_SVTR_THEO_TY_LE, TU_DIEN_LITEALPR

LOGGER.setLevel(logging.ERROR)


_CHUAN_HOA_TENSOR = T.Compose([T.ToTensor(), T.Normalize(0.5, 0.5)])


def nap_mo_hinh_yolo(duong_dan: str) -> YOLO:
    """Nạp mô hình YOLO phát hiện biển số (litealpr_yolov8n_eff.pt)."""
    return YOLO(duong_dan, task="detect")


class _HopBienGop:
    """Kết quả hộp biển số gộp (giả lập giao diện .boxes của Ultralytics: xyxy, conf, len)."""

    def __init__(self, xyxy: torch.Tensor, conf: torch.Tensor):
        self.xyxy = xyxy
        self.conf = conf

    def __len__(self) -> int:
        return int(self.xyxy.shape[0])


class _KetQuaGop:
    def __init__(self, boxes: _HopBienGop):
        self.boxes = boxes


class BoGopHaiDetector:
    """Chạy song song 2 model detect biển số rồi gộp hộp bằng NMS (không bỏ sót biển trên xe lẫn ô biển góc màn hình)."""

    def __init__(self, mo_hinh_a: YOLO, mo_hinh_b: YOLO, nguong_iou_gop: float = 0.5):
        self.mo_hinh_a = mo_hinh_a
        self.mo_hinh_b = mo_hinh_b
        self.nguong_iou_gop = nguong_iou_gop

    def predict(self, source, **kwargs) -> list[_KetQuaGop]:
        from torchvision.ops import nms

        kq_a = self.mo_hinh_a.predict(source=source, **kwargs)
        kq_b = self.mo_hinh_b.predict(source=source, **kwargs)
        ket_qua = []
        for ra, rb in zip(kq_a, kq_b):
            ds_xyxy = [r.boxes.xyxy.float().cpu() for r in (ra, rb) if r.boxes is not None and len(r.boxes) > 0]
            ds_conf = [r.boxes.conf.float().cpu() for r in (ra, rb) if r.boxes is not None and len(r.boxes) > 0]
            if not ds_xyxy:
                ket_qua.append(_KetQuaGop(_HopBienGop(torch.zeros((0, 4)), torch.zeros((0,)))))
                continue
            xyxy = torch.cat(ds_xyxy)
            conf = torch.cat(ds_conf)
            giu = nms(xyxy, conf, self.nguong_iou_gop)  # đã sắp xếp theo conf giảm dần
            ket_qua.append(_KetQuaGop(_HopBienGop(xyxy[giu], conf[giu])))
        return ket_qua


def nap_svtr26_onnx(duong_dan_onnx: str) -> tuple[ort.InferenceSession, str]:
    """Nạp mô hình SVTR26-Tiny ONNX đọc ký tự biển số Việt Nam (best.onnx)."""
    san_co = ort.get_available_providers()
    providers = [p for p in ("CUDAExecutionProvider", "CPUExecutionProvider") if p in san_co]
    tuy_chon = ort.SessionOptions()
    # Giới hạn 2 luồng intra-op giúp tránh tranh chấp nhân E-core trên CPU Intel Gen 12+
    tuy_chon.intra_op_num_threads = 2
    sess = ort.InferenceSession(duong_dan_onnx, sess_options=tuy_chon, providers=providers or ["CPUExecutionProvider"])
    return sess, sess.get_inputs()[0].name


def _tien_xu_ly_svtr26(anh_crop: np.ndarray) -> tuple[tuple[int, int], np.ndarray]:
    """Chuyển ảnh BGR crop sang tensor chuẩn hóa (3, H, W) theo tỷ lệ khung biển số."""
    img = Image.fromarray(cv2.cvtColor(anh_crop, cv2.COLOR_BGR2RGB))
    w, h = img.size
    ratio = max(1, min(4, int(round(float(w) / max(1.0, float(h))))))
    img_w, img_h = KICH_THUOC_SVTR_THEO_TY_LE[ratio - 1]
    resized = F.resize(img, (img_h, img_w), interpolation=T.InterpolationMode.BICUBIC)
    return (img_w, img_h), _CHUAN_HOA_TENSOR(resized).numpy()


def _giai_ma_ctc(probs: np.ndarray) -> tuple[str, float]:
    """Giải mã CTC cho 1 ma trận xác suất (T, C)."""
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


def doc_ky_tu_bang_svtr26(
    anh_crop: np.ndarray,
    sess: ort.InferenceSession,
    input_name: str,
) -> tuple[str, float]:
    """Chạy suy luận SVTR26-Tiny ONNX cho 1 ảnh crop đơn lẻ."""
    _, tensor_3d = _tien_xu_ly_svtr26(anh_crop)
    preds = sess.run(None, {input_name: tensor_3d[np.newaxis, ...]})[0]
    return _giai_ma_ctc(preds[0])


def doc_lo_ky_tu_svtr26(
    ds_anh_crop: list[np.ndarray],
    sess: ort.InferenceSession,
    input_name: str,
) -> list[tuple[str, float]]:
    """Chạy suy luận SVTR26-Tiny ONNX theo lô (gom nhóm các biển cùng kích thước chạy 1 lần)."""
    if not ds_anh_crop:
        return []

    ket_qua: list[tuple[str, float]] = [("", 0.0)] * len(ds_anh_crop)
    nhom_theo_co: dict[tuple[int, int], list[tuple[int, np.ndarray]]] = defaultdict(list)

    for vi_tri, crop in enumerate(ds_anh_crop):
        kich_co, tensor_3d = _tien_xu_ly_svtr26(crop)
        nhom_theo_co[kich_co].append((vi_tri, tensor_3d))

    for danh_sach in nhom_theo_co.values():
        batch_tensor = np.stack([muc[1] for muc in danh_sach], axis=0)
        preds = sess.run(None, {input_name: batch_tensor})[0]
        for (vi_tri, _), pred_don in zip(danh_sach, preds):
            ket_qua[vi_tri] = _giai_ma_ctc(pred_don)

    return ket_qua
