"""Testy behawioralne dla Czesci 4: predykcja w osobnym procesie bez label, ocena po message_id, baseline'y."""

from __future__ import annotations

import csv
import json
import random
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

from src.data_io import compute_sha256, load_csv, verify_data_manifest
from src.evaluate import evaluate_validation, load_predictions_csv
from src.models import SpamClassifierArtifact
from src.preprocess import tokenize_text
from src.run_all import run_predict_subprocess, write_unlabeled_csv


PROJECT_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_DIR / "data"
VAL_CSV = DATA_DIR / "validation.csv"
MODEL_JSON = PROJECT_DIR / "models" / "spam_classifier.json"
PREDICT_SCRIPT = PROJECT_DIR / "src" / "predict.py"


class TestPredictAndEvaluate(unittest.TestCase):
    def test_predict_subprocess_without_label_matches_in_memory_and_handles_shuffling(
        self,
    ) -> None:
        """Sprawdzenie (a): osobny proces na CSV bez label daje oceny zgodne < 1e-12 i niezalezne od kolejnosci."""
        self.assertTrue(MODEL_JSON.exists(), "Brak zapisanego modelu spam_classifier.json")

        artifact = SpamClassifierArtifact.load_json(MODEL_JSON)
        mids, texts, _ = load_csv(VAL_CSV, include_label=False)
        in_memory = {
            mid: artifact.predict_one(txt) for mid, txt in zip(mids, texts)
        }

        with tempfile.TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)
            unlabeled_csv = tmp / "val_no_label.csv"
            write_unlabeled_csv(VAL_CSV, unlabeled_csv)

            # Potwierdzenie, ze CSV przekazywany do predict.py NIE ma kolumny label.
            with open(unlabeled_csv, "r", encoding="utf-8", newline="") as f:
                header = next(csv.reader(f))
            self.assertEqual(header, ["message_id", "text"])

            # Przetasowana kopia bez kolumny label (test laczenia po message_id)
            shuffled_csv = tmp / "val_shuffled_no_label.csv"
            pairs = list(zip(mids, texts))
            random.Random(2026).shuffle(pairs)
            with open(shuffled_csv, "w", encoding="utf-8", newline="") as f:
                w = csv.writer(f, lineterminator="\n")
                w.writerow(["message_id", "text"])
                for mid, txt in pairs:
                    w.writerow([mid, txt])

            out_normal = tmp / "preds.csv"
            out_shuffled = tmp / "preds_shuffled.csv"
            run_predict_subprocess(
                predict_script=PREDICT_SCRIPT,
                model_json=MODEL_JSON,
                unlabeled_csv=unlabeled_csv,
                output_csv=out_normal,
            )
            run_predict_subprocess(
                predict_script=PREDICT_SCRIPT,
                model_json=MODEL_JSON,
                unlabeled_csv=shuffled_csv,
                output_csv=out_shuffled,
            )

            preds_normal = load_predictions_csv(out_normal)
            preds_shuffled = load_predictions_csv(out_shuffled)
            self.assertEqual(len(preds_normal), 60)
            self.assertEqual(set(preds_normal.keys()), set(mids))

            for mid in mids:
                mem_sc, mem_lbl = in_memory[mid]
                sub_sc, sub_lbl = preds_normal[mid]
                shuf_sc, shuf_lbl = preds_shuffled[mid]
                self.assertEqual(sub_lbl, mem_lbl)
                self.assertAlmostEqual(sub_sc, mem_sc, delta=1e-12)
                self.assertEqual(shuf_lbl, mem_lbl)
                self.assertAlmostEqual(shuf_sc, mem_sc, delta=1e-12)

            # Test trybu pojedynczego tekstu w osobnym procesie
            sample_text = "Subject: FREE $500 CASH OFFER!! Click http://example.com remove now"
            proc = subprocess.run(
                [
                    sys.executable,
                    str(PREDICT_SCRIPT),
                    "--model",
                    str(MODEL_JSON),
                    "--text",
                    sample_text,
                ],
                capture_output=True,
                text=True,
                check=False,
            )
            self.assertEqual(proc.returncode, 0, msg=proc.stderr)
            single_res = json.loads(proc.stdout)
            mem_single_sc, mem_single_lbl = artifact.predict_one(sample_text)
            self.assertEqual(single_res["predicted_label"], mem_single_lbl)
            self.assertAlmostEqual(
                float(single_res["score"]), mem_single_sc, delta=1e-12
            )

            # Ocena dla normalnej i przetasowanej kolejnosci musi byc identyczna po zlaczeniu po message_id.
            m1, e1 = evaluate_validation(
                validation_csv=VAL_CSV,
                predictions_csv=out_normal,
                model_json=MODEL_JSON,
            )
            m2, e2 = evaluate_validation(
                validation_csv=VAL_CSV,
                predictions_csv=out_shuffled,
                model_json=MODEL_JSON,
            )
            self.assertEqual(m1["selected_model"], m2["selected_model"])
            self.assertEqual(e1["n_fp"], e2["n_fp"])
            self.assertEqual(e1["n_fn"], e2["n_fn"])

    def test_validation_matrix_sums_and_baselines(self) -> None:
        """Sprawdzenie (f): macierz pomylek 60 = 40 + 20, 'zawsze ham' = 40/60, niezalezne zliczenie reguly."""
        val_metrics_path = PROJECT_DIR / "results" / "validation_metrics.json"
        if not val_metrics_path.exists():
            self.skipTest("Brak results/validation_metrics.json przed uruchomieniem oceny")

        data = json.loads(val_metrics_path.read_text(encoding="utf-8"))
        sel = data["selected_model"]
        cm = sel["confusion_matrix"]
        self.assertEqual(cm["tp"] + cm["fp"] + cm["tn"] + cm["fn"], 60)
        self.assertEqual(cm["tp"] + cm["fn"], 20)
        self.assertEqual(cm["fp"] + cm["tn"], 40)

        # Punkt odniesienia: 'zawsze ham' daje dokladnie TP=0, FP=0, FN=20, TN=40 (40/60)
        ah = data["baselines"]["always_ham"]
        self.assertEqual(ah["confusion_matrix"], {"tp": 0, "fp": 0, "tn": 40, "fn": 20})
        self.assertEqual(ah["accuracy_fraction"], "40/60")

        # Niezalezne zliczenie zamrozonej reguly '<MONEY> i remove -> spam' na validation.csv
        _, texts, labels = load_csv(VAL_CSV, include_label=True)
        assert labels is not None
        ind_tp = ind_fp = ind_tn = ind_fn = 0
        for txt, lbl in zip(texts, labels):
            toks = set(tokenize_text(txt, apply_masking=True))
            pred_spam = ("<MONEY>" in toks) and ("remove" in toks)
            if lbl == "spam" and pred_spam:
                ind_tp += 1
            elif lbl == "ham" and pred_spam:
                ind_fp += 1
            elif lbl == "ham" and not pred_spam:
                ind_tn += 1
            else:
                ind_fn += 1

        rule_cm = data["baselines"]["rule_money_and_remove"]["confusion_matrix"]
        self.assertEqual(
            rule_cm, {"tp": ind_tp, "fp": ind_fp, "tn": ind_tn, "fn": ind_fn}
        )

        # Dane w project/data/ pozostaja nienaruszone
        verify_data_manifest(DATA_DIR)


if __name__ == "__main__":
    unittest.main()
