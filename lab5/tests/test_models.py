"""Testy jednostkowe modeli, regul remisu, stratyfikacji CV i odrzucania kolumny label."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from src.data_io import load_unlabeled_csv, make_stratified_folds
from src.models import (
    KNNModel,
    NearestCentroidModel,
    SoftmaxRegressionModel,
    load_model,
    save_model,
)


class TestModelsAndDataIO(unittest.TestCase):
    def test_knn_vote_tie_broken_by_nearest_neighbor(self) -> None:
        # k=3, sasiedzi: klasa 5 (d=1), klasa 2 (d=2), klasa 8 (d=3) -> remis glosow 1:1:1
        # Najblizszy sasiad ma klase 5.
        X_train = [[1.0], [2.0], [3.0], [10.0]]
        y_train = [5, 2, 8, 0]
        model = KNNModel(k=3, metric="l1").fit(X_train, y_train)
        pred, info = model.predict_one_with_ties([0.0])
        self.assertEqual(pred, 5)
        self.assertTrue(info["vote_tie"])
        self.assertFalse(info["distance_tie_among_tied_classes"])

    def test_knn_tied_class_equal_distance_broken_by_lower_label(self) -> None:
        # k=3, sasiedzi w tej samej odleglosci d=1 dla klasy 7 i klasy 2, trzeci d=5 klasa 9
        X_train = [[1.0], [-1.0], [5.0]]
        y_train = [7, 2, 9]
        model = KNNModel(k=3, metric="l1").fit(X_train, y_train)
        pred, info = model.predict_one_with_ties([0.0])
        self.assertEqual(pred, 2)
        self.assertTrue(info["vote_tie"])
        self.assertTrue(info["distance_tie_among_tied_classes"])

    def test_knn_k_boundary_distance_tie_broken_by_train_order(self) -> None:
        # k=1, oba punkty maja odleglosc 1.0 od [0.0]; pierwszy w train.csv ma klase 9
        X_train = [[1.0], [-1.0]]
        y_train = [9, 1]
        model = KNNModel(k=1, metric="l2").fit(X_train, y_train)
        pred, info = model.predict_one_with_ties([0.0])
        self.assertEqual(pred, 9)
        self.assertTrue(info["distance_tie_at_k_boundary"])

    def test_nearest_centroid_l2(self) -> None:
        X_train = [[float(c), 0.0] for c in range(10)]
        y_train = list(range(10))
        model = NearestCentroidModel().fit(X_train, y_train)
        self.assertEqual(model.predict_one([3.1, 0.0]), 3)
        self.assertEqual(model.predict_one([8.9, 0.0]), 9)

    def test_softmax_gradient_matches_finite_difference(self) -> None:
        X = [[0.2, 0.8], [0.9, 0.1], [0.5, 0.5], [0.1, 0.3]]
        X_T = [[row[j] for row in X] for j in range(2)]
        y = [0, 1, 2, 9]
        W = [[0.05 * (c - 4) * (j + 1) for j in range(2)] for c in range(10)]
        b = [0.02 * (c - 5) for c in range(10)]
        lam = 0.03

        _, grad_W, grad_b, _ = SoftmaxRegressionModel._eval_loss_and_grads(
            X, X_T, y, W, b, lam
        )
        eps = 1e-6
        for c in (0, 2, 9):
            for j in range(2):
                W[c][j] += eps
                lp, _, _, _ = SoftmaxRegressionModel._eval_loss_and_grads(
                    X, X_T, y, W, b, lam
                )
                W[c][j] -= 2 * eps
                lm, _, _, _ = SoftmaxRegressionModel._eval_loss_and_grads(
                    X, X_T, y, W, b, lam
                )
                W[c][j] += eps
                num_g = (lp - lm) / (2 * eps)
                self.assertAlmostEqual(grad_W[c][j], num_g, places=6)

            b[c] += eps
            lp, _, _, _ = SoftmaxRegressionModel._eval_loss_and_grads(
                X, X_T, y, W, b, lam
            )
            b[c] -= 2 * eps
            lm, _, _, _ = SoftmaxRegressionModel._eval_loss_and_grads(
                X, X_T, y, W, b, lam
            )
            b[c] += eps
            num_gb = (lp - lm) / (2 * eps)
            self.assertAlmostEqual(grad_b[c], num_gb, places=6)

    def test_stratified_folds_balanced_and_deterministic(self) -> None:
        labels = [c for c in range(10) for _ in range(60)]
        folds1 = make_stratified_folds(labels, n_folds=5, seed=20261114)
        folds2 = make_stratified_folds(labels, n_folds=5, seed=20261114)
        self.assertEqual(folds1, folds2)
        self.assertEqual(len(folds1), 5)
        seen = set()
        for fold in folds1:
            self.assertEqual(len(fold), 120)
            self.assertEqual(fold, sorted(fold))
            seen.update(fold)
            counts = {c: 0 for c in range(10)}
            for idx in fold:
                counts[labels[idx]] += 1
            self.assertTrue(all(v == 12 for v in counts.values()))
        self.assertEqual(len(seen), 600)

    def test_unlabeled_csv_rejects_label_column(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            bad_csv = Path(tmp) / "with_label.csv"
            bad_csv.write_text(
                "image_id,image_path,label\nid1,images/a.png,3\n",
                encoding="utf-8",
            )
            with self.assertRaisesRegex(ValueError, "label"):
                load_unlabeled_csv(bad_csv)

    def test_model_save_load_roundtrip_deterministic(self) -> None:
        X = [[0.0, 1.0], [1.0, 0.0]]
        y = [0, 1]
        model = KNNModel(k=1, metric="l2").fit(X, y)
        with tempfile.TemporaryDirectory() as tmp:
            p1 = Path(tmp) / "m1.json"
            p2 = Path(tmp) / "m2.json"
            save_model(model, p1, train_csv_sha256="abc123")
            save_model(model, p2, train_csv_sha256="abc123")
            self.assertEqual(p1.read_bytes(), p2.read_bytes())
            loaded = load_model(p1)
            self.assertEqual(loaded.predict(X), [0, 1])


if __name__ == "__main__":
    unittest.main()
