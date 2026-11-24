"""Testy jednostkowe evaluate.py i predict.py na danych syntetycznych."""

from __future__ import annotations

import struct
import tempfile
import unittest
import zlib
from pathlib import Path

from src.evaluate import evaluate_predictions
from src.models import KNNModel, save_model
from src.predict import predict_batch, predict_single


def _make_simple_png_8x8(fill_value: int) -> bytes:
    ihdr = struct.pack(">IIBBBBB", 8, 8, 8, 0, 0, 0, 0)
    raw = b"".join(b"\x00" + bytes([fill_value] * 8) for _ in range(8))
    comp = zlib.compress(raw)

    def chunk(ctype: bytes, data: bytes) -> bytes:
        crc = zlib.crc32(ctype + data) & 0xFFFFFFFF
        return struct.pack(">I", len(data)) + ctype + data + struct.pack(">I", crc)

    return b"\x89PNG\r\n\x1a\n" + chunk(b"IHDR", ihdr) + chunk(b"IDAT", comp) + chunk(b"IEND", b"")


class TestPredictAndEvaluate(unittest.TestCase):
    def test_predict_single_batch_and_reject_label_csv(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            tmp_dir = Path(tmp)
            img_dir = tmp_dir / "images"
            img_dir.mkdir()
            (img_dir / "dark.png").write_bytes(_make_simple_png_8x8(0))
            (img_dir / "bright.png").write_bytes(_make_simple_png_8x8(255))

            model = KNNModel(k=1, metric="l2").fit(
                [[0.0] * 64, [1.0] * 64],
                [0, 8],
            )
            model_path = tmp_dir / "model.json"
            save_model(model, model_path, train_csv_sha256="dummy")

            self.assertEqual(predict_single(model_path, img_dir / "dark.png"), 0)
            self.assertEqual(predict_single(model_path, img_dir / "bright.png"), 8)

            in_csv = tmp_dir / "in.csv"
            in_csv.write_text(
                "image_id,image_path\nid-a,images/dark.png\nid-b,images/bright.png\n",
                encoding="utf-8",
            )
            out_csv = tmp_dir / "preds.csv"
            count = predict_batch(model_path, in_csv, tmp_dir, out_csv)
            self.assertEqual(count, 2)
            lines = out_csv.read_text(encoding="utf-8").splitlines()
            self.assertEqual(
                lines,
                [
                    "image_id,image_path,predicted_label",
                    "id-a,images/dark.png,0",
                    "id-b,images/bright.png,8",
                ],
            )

            bad_csv = tmp_dir / "with_label.csv"
            bad_csv.write_text(
                "image_id,image_path,label\nid-a,images/dark.png,0\n",
                encoding="utf-8",
            )
            with self.assertRaisesRegex(ValueError, "label"):
                predict_batch(model_path, bad_csv, tmp_dir, tmp_dir / "never.csv")

    def test_evaluate_predictions_join_by_image_id(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            tmp_dir = Path(tmp)
            labels_csv = tmp_dir / "labels.csv"
            preds_csv = tmp_dir / "preds.csv"
            # Celowo inna kolejnosc wierszy, zeby sprawdzic laczenie po image_id
            labels_csv.write_text(
                "image_id,image_path,label\n"
                "id-1,images/a.png,3\n"
                "id-2,images/b.png,5\n"
                "id-3,images/c.png,5\n",
                encoding="utf-8",
            )
            preds_csv.write_text(
                "image_id,image_path,predicted_label\n"
                "id-3,images/c.png,8\n"
                "id-1,images/a.png,3\n"
                "id-2,images/b.png,8\n",
                encoding="utf-8",
            )
            report = evaluate_predictions(preds_csv, labels_csv)
            self.assertEqual(report["total"], 3)
            self.assertEqual(report["hits"], 1)
            self.assertEqual(report["errors"], 2)
            self.assertEqual(report["confusion_matrix"][3][3], 1)
            self.assertEqual(report["confusion_matrix"][5][8], 2)
            self.assertEqual(
                report["most_confused_pairs"],
                [{"count": 2, "predicted_label": 8, "true_label": 5}],
            )
            self.assertEqual(
                [e["image_id"] for e in report["misclassified"]],
                ["id-2", "id-3"],
            )


if __name__ == "__main__":
    unittest.main()
