"""Testy behawioralne dla Czesci 3: stratyfikacja CV, izolacja foldow, prog, powtarzalnosc.

Testy operuja wylacznie na project/data/train.csv i nigdy nie otwieraja validation.csv.
"""

from __future__ import annotations

import json
import tempfile
import time
import unittest
from pathlib import Path

from src.cv_train import (
    CV_FOLDS,
    CV_SEED,
    MAX_FP_OOF,
    MIN_DF,
    make_stratified_folds,
    run_cv_and_train,
)
from src.data_io import compute_sha256, load_csv
from src.metrics import confusion_matrix, labels_to_binary
from src.preprocess import tokenize_corpus
from src.tfidf import TfidfVectorizer


PROJECT_DIR = Path(__file__).resolve().parents[1]
TRAIN_CSV = PROJECT_DIR / "data" / "train.csv"
MANIFEST_JSON = PROJECT_DIR / "data" / "manifest.json"


class TestCvPipeline(unittest.TestCase):
    def test_stratified_kfold_structure_and_determinism(self) -> None:
        """5 rozlacznych foldow po 36 wiadomosci (24 ham, 12 spam), deterministyczne dla ziarna 2026."""
        _, _, labels = load_csv(TRAIN_CSV, include_label=True)
        assert labels is not None
        y = labels_to_binary(labels)

        folds_a = make_stratified_folds(y, n_splits=CV_FOLDS, seed=CV_SEED)
        folds_b = make_stratified_folds(y, n_splits=CV_FOLDS, seed=CV_SEED)
        self.assertEqual(folds_a, folds_b)
        self.assertEqual(len(folds_a), 5)

        seen = []
        for fold in folds_a:
            self.assertEqual(len(fold), 36)
            n_ham = sum(1 for i in fold if y[i] == 0)
            n_spam = sum(1 for i in fold if y[i] == 1)
            self.assertEqual(n_ham, 24)
            self.assertEqual(n_spam, 12)
            seen.extend(fold)

        self.assertEqual(sorted(seen), list(range(180)))

    def test_cv_fold_tfidf_isolation_no_leakage(self) -> None:
        """Sprawdzenie (b): slownik i IDF w kazdym foldzie CV pochodza tylko z 144 wiadomosci uczacych."""
        _, texts, labels = load_csv(TRAIN_CSV, include_label=True)
        assert labels is not None
        y = labels_to_binary(labels)
        folds = make_stratified_folds(y, n_splits=CV_FOLDS, seed=CV_SEED)
        base_tokenized = tokenize_corpus(texts, apply_masking=True)

        sentinel = "zzuniquefoldleaktoken"
        for fold_idx, val_indices in enumerate(folds):
            val_set = set(val_indices)
            train_indices = [i for i in range(len(y)) if i not in val_set]

            # Wstrzykujemy wielokrotnie unikalny token wylacznie do czesci walidacyjnej tego foldu.
            poisoned_tokenized = [list(toks) for toks in base_tokenized]
            for vi in val_indices:
                poisoned_tokenized[vi].extend([sentinel, sentinel, sentinel])

            vec_clean = TfidfVectorizer(min_df=MIN_DF).fit(
                [base_tokenized[i] for i in train_indices]
            )
            vec_poisoned = TfidfVectorizer(min_df=MIN_DF).fit(
                [poisoned_tokenized[i] for i in train_indices]
            )

            self.assertNotIn(
                sentinel,
                vec_poisoned.vocab,
                msg=f"Wyciek tokenu walidacyjnego do slownika w foldzie {fold_idx + 1}",
            )
            self.assertEqual(vec_clean.vocab, vec_poisoned.vocab)
            self.assertEqual(vec_clean.idf, vec_poisoned.idf)

    def test_cv_train_threshold_optimality_reproducibility_and_speed(self) -> None:
        """Sprawdzenia (d) i (e): optymalnosc progu niezalezna petla, powtarzalnosc bajt-w-bajt, czas < 30s."""
        with open(MANIFEST_JSON, "r", encoding="utf-8") as f:
            expected_train_sha = json.load(f)["files"]["train.csv"]["sha256"]

        with tempfile.TemporaryDirectory() as tmp1, tempfile.TemporaryDirectory() as tmp2:
            # Podpinamy read-only link do project/data w katalogach tymczasowych,
            # zeby nie nadpisywac artefaktow produkcyjnych w trakcie testu.
            for tmp in (tmp1, tmp2):
                (Path(tmp) / "data").symlink_to(PROJECT_DIR / "data")

            t0 = time.monotonic()
            out1 = run_cv_and_train(tmp1, apply_masking=True, save_outputs=True)
            elapsed = time.monotonic() - t0
            out2 = run_cv_and_train(tmp2, apply_masking=True, save_outputs=True)

            # Kontrola budzetu czasu (< 30 s)
            self.assertLess(elapsed, 30.0, msg=f"CV + uczenie trwalo za dlugo: {elapsed:.2f}s")

            # Sprawdzenie (e): dwa kolejne przebiegi daja bajtowo identyczne pliki JSON.
            for rel in (
                "models/spam_classifier.json",
                "models/alternative_model.json",
                "results/cv_results.json",
                "results/train_top_weights.json",
            ):
                b1 = (Path(tmp1) / rel).read_bytes()
                b2 = (Path(tmp2) / rel).read_bytes()
                self.assertEqual(b1, b2, msg=f"Brak powtarzalnosci dla pliku {rel}")

            # Sprawdzenie (d): niezalezna petla potwierdza optymalnosc wybranego progu na OOF (180 wiad.).
            cv_res = out1["cv_results"]
            chosen_thr = float(cv_res["decision"]["winner_threshold"])  # type: ignore[index]
            oof_rows = cv_res["winner_oof_predictions"]  # type: ignore[index]
            self.assertEqual(len(oof_rows), 180)

            y_true = [1 if r["label"] == "spam" else 0 for r in oof_rows]
            y_scores = [float(r["oof_score"]) for r in oof_rows]
            chosen_cm = confusion_matrix(
                y_true, [1 if sc >= chosen_thr else 0 for sc in y_scores]
            )
            self.assertLessEqual(chosen_cm["fp"], MAX_FP_OOF)

            unique_scores = sorted(set(y_scores))
            cand_thresholds = [0.0, 1.0] + unique_scores + [
                0.5 * (unique_scores[i] + unique_scores[i + 1])
                for i in range(len(unique_scores) - 1)
            ]
            best_tp = -1
            best_fp_at_tp = 10**9
            for thr in cand_thresholds:
                cm = confusion_matrix(y_true, [1 if sc >= thr else 0 for sc in y_scores])
                if cm["fp"] <= MAX_FP_OOF:
                    if cm["tp"] > best_tp or (
                        cm["tp"] == best_tp and cm["fp"] < best_fp_at_tp
                    ):
                        best_tp = cm["tp"]
                        best_fp_at_tp = cm["fp"]

            self.assertEqual(chosen_cm["tp"], best_tp)
            self.assertEqual(chosen_cm["fp"], best_fp_at_tp)

            # Kontrola orientacyjna rozmiaru slownika (2000-3500)
            vocab_sz = int(cv_res["train_summary"]["full_vocab_size"])  # type: ignore[index]
            self.assertGreaterEqual(vocab_sz, 2000)
            self.assertLessEqual(vocab_sz, 3500)

        # Niezmiennosc train.csv po przebiegu
        self.assertEqual(compute_sha256(TRAIN_CSV), expected_train_sha)


if __name__ == "__main__":
    unittest.main()
