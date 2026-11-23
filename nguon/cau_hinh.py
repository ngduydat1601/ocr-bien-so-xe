"""Cấu hình đường dẫn, từ điển OCR, bảng sửa ký tự và tham số mặc định của hệ thống LPR."""

from pathlib import Path


# Đường dẫn thư mục dự án
THU_MUC_GOC = Path(__file__).resolve().parent.parent
THU_MUC_MO_HINH = THU_MUC_GOC / "mo_hinh"

# Đường dẫn cố định 2 mô hình duy nhất đang sử dụng
MO_HINH_DETECT = THU_MUC_MO_HINH / "litealpr_yolov8n_eff.pt"
MO_HINH_OCR = THU_MUC_MO_HINH / "best.onnx"
PHIEN_BAN_MO_HINH = "litealpr_yolov8n_eff.pt + best.onnx"

# Từ điển chính chủ của SVTR26-Tiny (index 0 là CTC blank)
TU_DIEN_LITEALPR = ["<blank>"] + list("0123456789ABCDĐEFGHIJKLMNOPQRSTUVWXYZ")
KICH_THUOC_SVTR_THEO_TY_LE = [[64, 64], [96, 48], [112, 40], [128, 32]]

# Bảng ánh xạ sửa lỗi nhầm ký tự theo ngữ cảnh vị trí biển số Việt Nam
SANG_SO = {
    "O": "0", "D": "0", "Q": "0", "I": "1", "L": "1", "T": "1",
    "Z": "2", "A": "4", "S": "5", "G": "6", "B": "8",
}
SANG_CHU = {
    "0": "D", "2": "Z", "4": "A", "5": "S", "6": "G", "8": "B",
}
# Các chữ cái không dùng làm ký tự seri biển số Việt Nam
SERI_KHONG_HOP_LE = {"I", "O", "Q", "W"}
# Vị trí thứ 4 trên biển 9 ký tự có thể là chữ cái (xe máy 50cc/điện: AB, MD, AA...)
# nên chỉ sửa các ký tự không bao giờ cấp làm seri (I, O) sang số (1, 0)
SUA_VI_TRI_3 = {"I": "1", "O": "0"}

# Tham số xử lý mặc định
TY_LE_PADDING_CROP = 0.05
NGUONG_HOP_LE_FORMAT = 0.60
VUNG_PIP_X_MIN = 0.72
VUNG_PIP_Y_MIN = 0.62
