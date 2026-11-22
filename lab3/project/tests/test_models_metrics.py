"""Testy matematyczne i behawioralne dla Czesci 2: regresja L2, Naiwny Bayes, miary, JSON."""

from __future__ import annotations

import math
import tempfile
import time
import unittest
from pathlib import Path

from src.metrics import (
    average_precision,
    compute_metrics,
    confusion_matrix,
    select_threshold,
)
from src.models import (
    LogisticRegressionL2,
    MultinomialNaiveBayes,
    SpamClassifierArtifact,
)
from src.preprocess import tokenize_corpus
from src.tfidf import TfidfVectorizer


class TestLogisticRegressionL2(unittest.TestCase):
    def test_gradient_matches_finite_differences(self) -> None:
        """Porownuje gradient analityczny regresji L2 z roznicami skonczonymi."""
        X = [
            {0: 0.8, 1: 0.6},
            {0: 0.2, 2: 0.9},
            {1: 0.5, 2: 0.5},
            {0: 0.7, 1: 0.1, 2: 0.4},
        ]
        y = [1, 0, 1, 0]
        w = [0.35, -0.42, 0.18]
        b = -0.25
        eps = 1e-6

        for C in (0.1, 1.0, 10.0, 100.0):
            _, grad_w, grad_b = LogisticRegressionL2.objective_and_gradient(
                X, y, w, b, C
            )
            for j in range(len(w)):
                w_plus = list(w)
                w_minus = list(w)
                w_plus[j] += eps
                w_minus[j] -= eps
                l_plus, _, _ = LogisticRegressionL2.objective_and_gradient(
                    X, y, w_plus, b, C
                )
                l_minus, _, _ = LogisticRegressionL2.objective_and_gradient(
                    X, y, w_minus, b, C
                )
                num_grad = (l_plus - l_minus) / (2.0 * eps)
                self.assertAlmostEqual(
                    grad_w[j],
                    num_grad,
                    delta=1e-6,
                    msg=f"Rozbieznosc gradientu w[{j}] dla C={C}",
                )

            lb_plus, _, _ = LogisticRegressionL2.objective_and_gradient(
                X, y, w, b + eps, C
            )
            lb_minus, _, _ = LogisticRegressionL2.objective_and_gradient(
                X, y, w, b - eps, C
            )
            num_grad_b = (lb_plus - lb_minus) / (2.0 * eps)
            self.assertAlmostEqual(
                grad_b,
                num_grad_b,
                delta=1e-6,
                msg=f"Rozbieznosc gradientu b dla C={C}",
            )

    def test_kkt_optimality_regularization_monotonicity_and_separation(self) -> None:
        """Po dopasowaniu ||grad|| < 1e-6, mniejsze C zmniejsza ||w||_2, a klasy sa rozdzielone."""
        X = [
            {0: 0.9, 1: 0.1, 2: 0.0},
            {0: 0.8, 1: 0.2, 2: 0.0},
            {0: 0.7, 1: 0.3, 2: 0.1},
            {0: 0.1, 1: 0.8, 2: 0.5},
            {0: 0.0, 1: 0.9, 2: 0.4},
            {0: 0.0, 1: 0.7, 2: 0.7},
        ]
        y = [1, 1, 1, 0, 0, 0]

        norms = []
        for C in (0.1, 1.0, 10.0, 100.0):
            clf = LogisticRegressionL2(C=C).fit(X, y, n_features=3)
            self.assertTrue(clf.converged_, msg=f"Solver nie zbiegl dla C={C}")
            _, gw, gb = LogisticRegressionL2.objective_and_gradient(
                X, y, clf.weights, clf.intercept, C
            )
            grad_norm = math.sqrt(sum(v * v for v in gw) + gb * gb)
            self.assertLess(
                grad_norm,
                1e-6,
                msg=f"Norma gradientu po dopasowaniu za duza dla C={C}: {grad_norm}",
            )
            w_norm = math.sqrt(sum(v * v for v in clf.weights))
            norms.append(w_norm)

            probs = clf.predict_proba(X)
            for p, yi in zip(probs, y):
                if yi == 1:
                    self.assertGreater(p, 0.5)
                else:
                    self.assertLess(p, 0.5)

        # Monotonicznosc regularyzacji L2: wieksze C -> mniejsza kara -> wieksza norma ||w||_2
        for i in range(len(norms) - 1):
            self.assertLess(norms[i], norms[i + 1])


class TestMultinomialNaiveBayes(unittest.TestCase):
    def test_hand_calculated_probabilities(self) -> None:
        """Porownuje wielomianowy Naiwny Bayes z recznym wyliczeniem na 2 klasach i 3 slowach."""
        # 4 dokumenty: 2 ham (y=0), 2 spam (y=1), 3 cechy (V=3)
        # ham:  doc0 = {0: 2.0, 1: 1.0}, doc1 = {0: 1.0}        -> sums_0 = [3.0, 1.0, 0.0], S_0 = 4.0
        # spam: doc2 = {1: 1.0, 2: 2.0}, doc3 = {2: 1.0}        -> sums_1 = [0.0, 1.0, 3.0], S_1 = 4.0
        X = [
            {0: 2.0, 1: 1.0},
            {0: 1.0},
            {1: 1.0, 2: 2.0},
            {2: 1.0},
        ]
        y = [0, 0, 1, 1]
        x_test = {0: 1.0, 2: 2.0}

        for alpha in (1.0, 0.1):
            nb = MultinomialNaiveBayes(alpha=alpha).fit(X, y, n_features=3)

            # Reczne obliczenie
            denom_0 = 4.0 + alpha * 3.0
            denom_1 = 4.0 + alpha * 3.0
            lp0 = [
                math.log((3.0 + alpha) / denom_0),
                math.log((1.0 + alpha) / denom_0),
                math.log((0.0 + alpha) / denom_0),
            ]
            lp1 = [
                math.log((0.0 + alpha) / denom_1),
                math.log((1.0 + alpha) / denom_1),
                math.log((3.0 + alpha) / denom_1),
            ]
            prior = math.log(0.5)
            ll0 = prior + 1.0 * lp0[0] + 2.0 * lp0[2]
            ll1 = prior + 1.0 * lp1[0] + 2.0 * lp1[2]
            expected_p_spam = 1.0 / (1.0 + math.exp(ll0 - ll1))

            for j in range(3):
                self.assertAlmostEqual(nb.feature_log_prob[0][j], lp0[j], places=12)
                self.assertAlmostEqual(nb.feature_log_prob[1][j], lp1[j], places=12)

            p_spam = nb.predict_proba_one(x_test)
            self.assertAlmostEqual(p_spam, expected_p_spam, places=12)
            self.assertTrue(0.0 <= p_spam <= 1.0)
            self.assertFalse(math.isnan(p_spam) or math.isinf(p_spam))

            # Zgodnosc reprezentacji log-odds (w^T x + b) z predict_proba_one
            w = nb.feature_weights()
            z = nb.intercept + sum(w[j] * val for j, val in x_test.items())
            p_from_weights = 1.0 / (1.0 + math.exp(-z))
            self.assertAlmostEqual(p_spam, p_from_weights, places=12)


class TestMetricsAndThreshold(unittest.TestCase):
    def test_hand_calculated_metrics_and_pr_auc(self) -> None:
        """Recznie sprawdzona macierz pomylek, miary i Average Precision."""
        # 5 przykladow: 2 spam, 3 ham
        y_true = [1, 0, 1, 0, 0]
        y_scores = [0.9, 0.7, 0.6, 0.4, 0.1]
        # Dla progu 0.5: pred = [1, 1, 1, 0, 0] -> TP=2, FP=1, TN=2, FN=0
        m = compute_metrics(y_true, y_scores, threshold=0.5)
        self.assertEqual(m["confusion_matrix"], {"tp": 2, "fp": 1, "tn": 2, "fn": 0})
        self.assertAlmostEqual(m["fpr"], 1.0 / 3.0, places=12)
        self.assertAlmostEqual(m["recall"], 1.0, places=12)
        self.assertAlmostEqual(m["precision"], 2.0 / 3.0, places=12)
        self.assertAlmostEqual(m["accuracy"], 4.0 / 5.0, places=12)

        # Reczny PR-AUC (Average Precision):
        # score=0.9 (y=1): rec=0.5, prec=1/1=1.0 -> delta=0.5 * 1.0 = 0.5
        # score=0.7 (y=0): rec=0.5, prec=1/2=0.5 -> delta=0.0
        # score=0.6 (y=1): rec=1.0, prec=2/3     -> delta=0.5 * 2/3 = 1/3
        # Razem AP = 0.5 + 1/3 = 5/6
        self.assertAlmostEqual(average_precision(y_true, y_scores), 5.0 / 6.0, places=12)

    def test_select_threshold_rule_and_midpoint(self) -> None:
        """Sprawdza regule doboru progu (maks. TP przy FP <= max_fp, mniej FP, srodek przedzialu)."""
        # Posortowane malejaco oceny:
        # 0.95 (spam), 0.85 (spam), 0.75 (ham - FP=1), 0.65 (spam), 0.55 (ham - FP=2), 0.30 (ham)
        y_true = [1, 1, 0, 1, 0, 0]
        y_scores = [0.95, 0.85, 0.75, 0.65, 0.55, 0.30]

        # Przy max_fp=1 optymalne jest TP=3, FP=1 (oceny >= 0.65), przedzial (0.55, 0.65] -> srodek 0.60
        thr, m = select_threshold(y_true, y_scores, max_fp=1)
        self.assertAlmostEqual(thr, 0.60, places=12)
        self.assertEqual(m["confusion_matrix"]["tp"], 3)
        self.assertEqual(m["confusion_matrix"]["fp"], 1)

        # Przy max_fp=0 optymalne jest TP=2, FP=0 (oceny >= 0.85), przedzial (0.75, 0.85] -> srodek 0.80
        thr0, m0 = select_threshold(y_true, y_scores, max_fp=0)
        self.assertAlmostEqual(thr0, 0.80, places=12)
        self.assertEqual(m0["confusion_matrix"]["tp"], 2)
        self.assertEqual(m0["confusion_matrix"]["fp"], 0)

        # Niezalezna petla po wszystkich progach potwierdza optymalnosc
        for cand in [0.0, 0.30, 0.55, 0.65, 0.75, 0.85, 0.95, 1.0]:
            cm = confusion_matrix(y_true, [1 if s >= cand else 0 for s in y_scores])
            if cm["fp"] <= 1:
                self.assertLessEqual(cm["tp"], m["confusion_matrix"]["tp"])


class TestJsonArtifactRoundTrip(unittest.TestCase):
    def test_save_and_reload_without_training_for_both_models(self) -> None:
        """Po zapisie i wczytaniu z JSON oceny sa identyczne (< 1e-12) bez ponownego uczenia."""
        texts = [
            "Subject: FREE money $1000!! Click http://spam.example remove now",
            "Subject: FREE offer $500!! Click http://spam2.example remove",
            "Subject: Win $200 cash FREE!!",
            "Subject: Linux kernel patch discussion",
            "Subject: Meeting schedule for kernel review",
            "Subject: Patch review notes for project",
        ]
        y = [1, 1, 1, 0, 0, 0]
        tokenized = tokenize_corpus(texts, apply_masking=True)
        vec = TfidfVectorizer(min_df=2).fit(tokenized)
        X = vec.transform(tokenized)

        probe_texts = [
            "Subject: FREE $300 offer! remove",
            "Subject: Kernel patch meeting",
            "Subject: Completely unknown vocabulary xyzzy",
        ]

        models = [
            LogisticRegressionL2(C=1.0).fit(X, y, n_features=vec.vocab_size),
            MultinomialNaiveBayes(alpha=0.1).fit(X, y, n_features=vec.vocab_size),
        ]

        with tempfile.TemporaryDirectory() as tmpdir:
            for idx, model in enumerate(models):
                artifact = SpamClassifierArtifact(
                    vectorizer=vec,
                    model=model,
                    threshold=0.5,
                    apply_masking=True,
                )
                before_preds = artifact.predict_texts(probe_texts)
                json_path = Path(tmpdir) / f"model_{idx}.json"
                artifact.save_json(json_path)

                reloaded = SpamClassifierArtifact.load_json(json_path)
                after_preds = reloaded.predict_texts(probe_texts)

                for (s1, l1), (s2, l2) in zip(before_preds, after_preds):
                    self.assertEqual(l1, l2)
                    self.assertAlmostEqual(s1, s2, delta=1e-12)

    def test_solver_speed_on_realistic_size(self) -> None:
        """Kontrola szybkosci i zbieznosci solvera Newton-Woodbury na rozmiarze train (180x2637)."""
        from src.data_io import load_csv
        from src.metrics import labels_to_binary

        train_csv = Path(__file__).resolve().parents[1] / "data" / "train.csv"
        _, texts, labels = load_csv(train_csv, include_label=True)
        assert labels is not None
        y = labels_to_binary(labels)
        tokenized = tokenize_corpus(texts, apply_masking=True)
        vec = TfidfVectorizer(min_df=2).fit(tokenized)
        X = vec.transform(tokenized)

        t0 = time.monotonic()
        for C in (0.1, 1.0, 10.0, 100.0):
            clf = LogisticRegressionL2(C=C).fit(X, y, n_features=vec.vocab_size)
            self.assertTrue(
                clf.converged_,
                msg=f"Solver Newton-Woodbury nie zbiegl na train dla C={C}",
            )
            self.assertLess(clf.grad_norm_, 1e-6)
        elapsed = time.monotonic() - t0
        # 4 dopasowania na calym train musza byc duzo szybsze niz budzet 30 s dla calej 5-krotnej CV.
        self.assertLess(elapsed, 10.0, msg=f"Zbyt wolne dopasowanie: {elapsed:.2f}s")


if __name__ == "__main__":
    unittest.main()
