import json
import sqlite3
import tempfile
import unittest
from contextlib import closing
from pathlib import Path
from unittest.mock import patch

import cv2
import numpy as np

import luu_csdl


class LuuCSDLTests(unittest.TestCase):
    def setUp(self):
        self.thu_muc = tempfile.TemporaryDirectory()
        self.addCleanup(self.thu_muc.cleanup)
        root = Path(self.thu_muc.name)
        self.db = root / "du_lieu" / "lich_su_bien_so.sqlite3"
        self.anh_root = root / "anh_luu"
        for ten, gia_tri in (
            ("DUONG_DAN_CSDL", self.db),
            ("THU_MUC_LUU_ANH", self.anh_root),
        ):
            patcher = patch.object(luu_csdl, ten, gia_tri)
            patcher.start()
            self.addCleanup(patcher.stop)
        self.anh = np.full((32, 80, 3), 128, dtype=np.uint8)
        self.du_lieu = {
            "plate_raw": "29G216139",
            "plate_normalized": "29G2-161.39",
            "format_status": "VALID",
            "det_confidence": 0.95,
            "ocr_confidence": 0.98,
            "det_latency_ms": 5.0,
            "ocr_latency_ms": 10.0,
            "total_latency_ms": 15.0,
            "model_version": "yolo + svtr26_vn",
            "plate_bbox": [10, 20, 90, 52],
        }

    def luu(self):
        return luu_csdl.luu_ket_qua_bien_so(self.du_lieu, self.anh, self.anh)

    def test_luu_du_15_truong_anh_va_index(self):
        record_id = self.luu()
        with closing(sqlite3.connect(self.db)) as conn:
            conn.row_factory = sqlite3.Row
            row = conn.execute("SELECT * FROM lich_su_bien_so").fetchone()
            self.assertEqual(len(row), 15)
            self.assertEqual(row["id"], record_id)
            self.assertIsNotNone(row["captured_at"])
            self.assertIsNone(row["is_correct"])
            for key, value in self.du_lieu.items():
                actual = json.loads(row[key]) if key == "plate_bbox" else row[key]
                self.assertEqual(actual, value)
            for key in ("image_full_path", "image_crop_path"):
                image = self.anh_root / row[key]
                self.assertTrue(image.is_file())
                self.assertIsNotNone(cv2.imdecode(np.frombuffer(image.read_bytes(), np.uint8), 1))
            indexes = conn.execute("PRAGMA index_list(lich_su_bien_so)").fetchall()
            self.assertIn("idx_plate_time", [item[1] for item in indexes])
        self.assertEqual(self.du_lieu["plate_bbox"], [10, 20, 90, 52])

    def test_doi_soat_chi_cap_nhat_dung_ban_ghi(self):
        first, second = self.luu(), self.luu()
        self.assertNotEqual(first, second)
        for value in (True, False):
            self.assertTrue(luu_csdl.cap_nhat_doi_soat(first, value))
            with closing(sqlite3.connect(self.db)) as conn:
                rows = conn.execute("SELECT id, is_correct FROM lich_su_bien_so ORDER BY id").fetchall()
            self.assertEqual(rows, [(first, int(value)), (second, None)])
        self.assertFalse(luu_csdl.cap_nhat_doi_soat(second + 100, True))

    def test_doi_soat_khi_chua_co_csdl_va_kieu_sai(self):
        self.assertFalse(luu_csdl.cap_nhat_doi_soat(1, True))
        for value in (None, "False", 1):
            with self.assertRaises(TypeError):
                luu_csdl.cap_nhat_doi_soat(1, value)

    def test_chuoi_sql_duoc_luu_nhu_du_lieu(self):
        raw = "'; DROP TABLE lich_su_bien_so; --"
        self.du_lieu["plate_raw"] = raw
        self.luu()
        with closing(sqlite3.connect(self.db)) as conn:
            self.assertEqual(conn.execute("SELECT plate_raw FROM lich_su_bien_so").fetchone()[0], raw)


if __name__ == "__main__":
    unittest.main()
