"""Xuất báo cáo kết quả kiểm tra ra file Excel tổng hợp (.xlsx) và CSV riêng từng mục."""

from datetime import datetime
from pathlib import Path
import re

import pandas as pd

from nguon.cau_hinh import THU_MUC_GOC

THU_MUC_BAO_CAO = THU_MUC_GOC / "du_lieu" / "ket_qua_kiem_tra"
DUONG_DAN_EXCEL_TONG_HOP = THU_MUC_BAO_CAO / "BAO_CAO_KIEM_TRA_TONG_HOP.xlsx"
TEN_SHEET_TONG_HOP = "TONG_HOP_CAC_MUC"


def _chuan_hoa_ten_sheet(ten_muc: str) -> str:
    """Làm sạch tên Sheet hợp lệ theo giới hạn tối đa 31 ký tự của Excel."""
    sach = re.sub(r"[\\/*?:\[\]]+", "_", (ten_muc or "Muc_Test").strip())
    sach = re.sub(r"\s+", "_", sach).strip("_")
    return (sach[:31] or "Muc_Test")


def luu_ket_qua_muc_kiem_tra(
    nhom_kiem_tra: str,
    ten_muc: str,
    quy_mo_do_phan_giai: str,
    so_phat_hien: int,
    so_hop_le: int,
    tong_thoi_gian_s: float,
    ai_trung_binh_ms: float,
    thong_luong_hien_thi: str,
    phien_ban_model: str,
    df_chi_tiet: pd.DataFrame,
) -> tuple[Path, Path]:
    """Lưu kết quả 1 mục kiểm tra (Ảnh hoặc Video) vào:
    1) File Excel tổng hợp `BAO_CAO_KIEM_TRA_TONG_HOP.xlsx` (cập nhật bảng Tổng hợp + Sheet riêng của mục này).
    2) File CSV riêng của mục này (mã UTF-8-BOM mở trực tiếp bằng Excel không lỗi dấu).
    """
    THU_MUC_BAO_CAO.mkdir(parents=True, exist_ok=True)
    thoi_diem = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    ten_sheet = _chuan_hoa_ten_sheet(ten_muc)

    dong_tong_hop = {
        "Nhóm kiểm tra": nhom_kiem_tra,
        "Tên mục kiểm tra": ten_muc,
        "Sheet chi tiết": ten_sheet,
        "Quy mô / Độ phân giải": quy_mo_do_phan_giai,
        "Số biển phát hiện": int(so_phat_hien),
        "Số biển hợp lệ (VALID)": int(so_hop_le),
        "Tổng thời gian chạy (giây)": round(float(tong_thoi_gian_s), 2),
        "Thời gian AI TB (ms)": round(float(ai_trung_binh_ms), 2),
        "Thông lượng": thong_luong_hien_thi,
        "Model sử dụng": phien_ban_model,
        "Thời điểm chạy": thoi_diem,
    }

    # Đọc các Sheet sẵn có trong file Excel tổng hợp (nếu đã tồn tại) để không làm mất các mục trước đó
    cac_sheet_cu: dict[str, pd.DataFrame] = {}
    if DUONG_DAN_EXCEL_TONG_HOP.exists():
        try:
            cac_sheet_cu = pd.read_excel(DUONG_DAN_EXCEL_TONG_HOP, sheet_name=None)
        except Exception:
            cac_sheet_cu = {}

    if TEN_SHEET_TONG_HOP in cac_sheet_cu and not cac_sheet_cu[TEN_SHEET_TONG_HOP].empty:
        df_th = cac_sheet_cu[TEN_SHEET_TONG_HOP]
        df_th = df_th[df_th["Tên mục kiểm tra"] != ten_muc]
        df_th_moi = pd.concat([df_th, pd.DataFrame([dong_tong_hop])], ignore_index=True)
    else:
        df_th_moi = pd.DataFrame([dong_tong_hop])

    cac_sheet_cu[TEN_SHEET_TONG_HOP] = df_th_moi
    cac_sheet_cu[ten_sheet] = df_chi_tiet

    # Đảm bảo Sheet TONG_HOP_CAC_MUC luôn nằm đầu tiên
    thu_tu_sheet = [TEN_SHEET_TONG_HOP] + [k for k in cac_sheet_cu if k != TEN_SHEET_TONG_HOP]
    with pd.ExcelWriter(DUONG_DAN_EXCEL_TONG_HOP, engine="openpyxl") as writer:
        for k in thu_tu_sheet:
            cac_sheet_cu[k].to_excel(writer, sheet_name=k, index=False)

    # Xuất kèm 1 file CSV riêng của đúng mục này để tra cứu nhanh
    duong_dan_csv_rieng = THU_MUC_BAO_CAO / f"{ten_sheet}.csv"
    df_chi_tiet.to_csv(duong_dan_csv_rieng, index=False, encoding="utf-8-sig")

    return DUONG_DAN_EXCEL_TONG_HOP, duong_dan_csv_rieng
