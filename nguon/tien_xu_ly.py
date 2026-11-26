"""Các hàm tiện ích đọc ảnh Unicode và chuẩn hóa, kiểm tra định dạng biển số Việt Nam."""

import re
from pathlib import Path

import cv2
import numpy as np

from nguon.cau_hinh import (
    NGUONG_HOP_LE_FORMAT,
    SANG_CHU,
    SANG_SO,
    SERI_KHONG_HOP_LE,
    SUA_VI_TRI_3,
)


def _lam_sach_chuoi_bien_so(van_ban: str) -> str:
    """Chuyển in hoa, đổi Đ→D và loại bỏ ký tự không phải chữ/số."""
    return re.sub(r"[^A-Z0-9]", "", (van_ban or "").upper().replace("Đ", "D"))


def doc_anh_unicode(duong_dan: Path) -> np.ndarray | None:
    """Đọc ảnh BGR hỗ trợ đường dẫn có dấu tiếng Việt trên Windows."""
    if not duong_dan.exists():
        return None
    return doc_anh_tu_bytes(duong_dan.read_bytes())


def doc_anh_tu_bytes(du_lieu_bytes: bytes) -> np.ndarray | None:
    """Giải mã bytes ảnh tải lên thành ma trận BGR (trả None nếu dữ liệu rỗng/hỏng)."""
    if not du_lieu_bytes:
        return None
    return cv2.imdecode(np.frombuffer(du_lieu_bytes, dtype=np.uint8), cv2.IMREAD_COLOR)


def chuan_hoa_bien_so_vn(van_ban_tho: str, ty_le_khung: float | None = None) -> str:
    """Chuẩn hóa định dạng và sửa nhầm ký tự cho cả biển số Ô tô & Xe máy Việt Nam.

    - 9 ký tự (xe máy 5 số / ô tô seri 2 chữ 5 số): NNLX-NNN.NN (VD: 29B1-123.45, 29LD-012.34).
    - 8 ký tự:
      + Nếu vị trí thứ 4 là chữ cái (seri 2 chữ 4 số): NNLX-NNNN (VD: 29AB-1234, 29LD-1234).
      + Nếu vị trí thứ 4 là chữ số: NNL-NNN.NN (VD: 76H-028.77, 15A-709.34).
    - 7 ký tự (ô tô 4 số): NNL-NNNN (VD: 30A-1234).
    """
    _ = ty_le_khung
    s = _lam_sach_chuoi_bien_so(van_ban_tho)
    if len(s) < 7 or len(s) > 9:
        return s

    arr = list(s)
    # 2 ký tự đầu (mã tỉnh): bắt buộc là chữ số
    for idx in (0, 1):
        if arr[idx] in SANG_SO:
            arr[idx] = SANG_SO[arr[idx]]
    # Ký tự thứ 3 (seri chữ đầu): bắt buộc là chữ cái
    if arr[2] in SANG_CHU:
        arr[2] = SANG_CHU[arr[2]]

    # Trên biển 8-9 ký tự, vị trí thứ 4 có thể là chữ cái (AB, MD, LD...) hoặc số;
    # chỉ sửa các ký tự không bao giờ cấp làm seri (I→1, O/Q→0) để không làm hỏng seri 2 chữ.
    if len(arr) in (8, 9) and arr[3] in SUA_VI_TRI_3:
        arr[3] = SUA_VI_TRI_3[arr[3]]

    # Các ký tự số thứ tự phía sau: bắt buộc là chữ số
    bat_dau_so = 3 if len(arr) == 7 else 4
    for idx in range(bat_dau_so, len(arr)):
        if arr[idx] in SANG_SO:
            arr[idx] = SANG_SO[arr[idx]]

    sach = "".join(arr)
    if len(sach) == 9:
        return f"{sach[:4]}-{sach[4:7]}.{sach[7:]}"
    if len(sach) == 8:
        if sach[3].isalpha():
            return f"{sach[:4]}-{sach[4:]}"
        return f"{sach[:3]}-{sach[3:6]}.{sach[6:]}"
    # 7 ký tự: biển ô tô 4 số (VD: 30A-1234)
    return f"{sach[:3]}-{sach[3:]}"


def danh_gia_trang_thai_bien_so(
    van_ban_chuan: str,
    ocr_conf: float,
    nguong_hop_le: float = NGUONG_HOP_LE_FORMAT,
) -> str:
    """Đánh giá trạng thái hợp lệ định dạng biển số Việt Nam: VALID / UNCERTAIN / INVALID."""
    sach = _lam_sach_chuoi_bien_so(van_ban_chuan)
    if not sach:
        return "INVALID"
    khop_chuan = (
        len(sach) in (7, 8, 9)
        and bool(re.match(r"^[0-9]{2}[A-Z][0-9A-Z]?[0-9]{4,5}$", sach))
        and sach[2] not in SERI_KHONG_HOP_LE
    )
    if khop_chuan and ocr_conf >= nguong_hop_le:
        return "VALID"
    if khop_chuan or (7 <= len(sach) <= 10):
        return "UNCERTAIN"
    return "INVALID"


def trich_nhan_tu_ten_file(ten_file: str) -> str | None:
    """Tách nhãn biển số thật từ tên file nếu có dạng '<BIEN_SO>_...jpg' hoặc '<BIEN_SO>.jpg'."""
    phan_dau = Path(ten_file).stem.split("_")[0]
    sach = _lam_sach_chuoi_bien_so(phan_dau)
    if len(sach) in (7, 8, 9) and re.match(r"^[0-9]{2}[A-Z][0-9A-Z]?[0-9]{4,5}$", sach):
        return sach
    return None
