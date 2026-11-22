"""Testy integracyjne Czesci 5: powtarzalnosc pelnego przebiegu (e), diagnostyka, zamrozenie modelu."""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from src.data_io import compute_sha256, verify_data_manifest
from src.run_all import run_full_pipeline


PROJECT_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_DIR / "data"


class TestEndToEndReproducibility(unittest.TestCase):
    def test_two_full_runs_produce_identical_models_and_predictions(self) -> None:
        """Sprawdzenie (e): dwa pelne przebiegi daja identyczne pliki modelu i przewidywan, data/ bez zmian."""
        before_manifest = verify_data_manifest(DATA_DIR)

        with tempfile.TemporaryDirectory() as tmp1, tempfile.TemporaryDirectory() as tmp2:
            p1, p2 = Path(tmp1), Path(tmp2)
            for p in (p1, p2):
                (p / "data").symlink_to(DATA_DIR)

            run_full_pipeline(p1)
            run_full_pipeline(p2)

            for rel in (
                "models/spam_classifier.json",
                "models/alternative_model.json",
                "results/cv_results.json",
                "results/validation_predictions.csv",
                "results/validation_predictions_alternative.csv",
                "results/validation_metrics.json",
                "results/validation_errors.json",
                "results/diagnostics_section5.json",
            ):
                b1 = (p1 / rel).read_bytes()
                b2 = (p2 / rel).read_bytes()
                self.assertEqual(b1, b2, msg=f"Brak powtarzalnosci pliku {rel}")

            # Sprawdzenie diagnostyki z sekcji 5 (c1: ILUG = 0 zmian, d: suma 60)
            diag = json.loads(
                (p1 / "results" / "diagnostics_section5.json").read_text(encoding="utf-8")
            )
            ilug = diag["c_validation_perturbations"]["ilug_footer_perturbation"]
            self.assertEqual(ilug["score_changes_gt_1e_12"], 0)
            self.assertEqual(ilug["label_flips"], 0)

            thr = diag["d_shared_threads_validation"]
            self.assertEqual(thr["n_shared"] + thr["n_other"], 60)
            self.assertEqual(
                thr["n_shared_spam"] + thr["n_other_spam"], 20
            )
            self.assertEqual(
                thr["n_shared_ham"] + thr["n_other_ham"], 40
            )

        after_manifest = verify_data_manifest(DATA_DIR)
        self.assertEqual(before_manifest, after_manifest)

    def test_frozen_models_sha256_unchanged(self) -> None:
        """Sprawdza, ze zapisane modele w project/models/ maja te same SHA-256 co przed otwarciem validation.csv."""
        frozen_path = PROJECT_DIR / "results" / "frozen_models_sha256.json"
        if not frozen_path.exists():
            self.skipTest("Brak results/frozen_models_sha256.json")

        frozen = json.loads(frozen_path.read_text(encoding="utf-8"))["models"]
        for fname, expected_sha in frozen.items():
            actual_sha = compute_sha256(PROJECT_DIR / "models" / fname)
            self.assertEqual(
                actual_sha,
                expected_sha,
                msg=f"Model {fname} zmienil sume SHA-256 po zamrozeniu!",
            )


if __name__ == "__main__":
    unittest.main()
