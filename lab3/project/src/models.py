"""Wlasne implementacje modeli (regresja logistyczna L2, Naiwny Bayes) i serializacja JSON."""

from __future__ import annotations

import json
import math
from operator import mul
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

from src.preprocess import get_preprocessing_spec, tokenize_text
from src.tfidf import SparseVector, TfidfVectorizer


def sigmoid(z: float) -> float:
    """Numerycznie stabilna funkcja sigmoidalna."""
    if z >= 0.0:
        ez = math.exp(-z)
        return 1.0 / (1.0 + ez)
    ez = math.exp(z)
    return ez / (1.0 + ez)


def softplus(z: float) -> float:
    """Numerycznie stabilne log(1 + exp(z))."""
    if z > 0.0:
        return z + math.log1p(math.exp(-z))
    return math.log1p(math.exp(z))


def _cholesky_decompose(mat: List[List[float]]) -> List[List[float]]:
    """Rozklad Cholesky'ego A = L L^T dla macierzy symetrycznej dodatnio okreslonej."""
    n = len(mat)
    L: List[List[float]] = [[0.0] * n for _ in range(n)]
    for i in range(n):
        row_i = L[i]
        mat_i = mat[i]
        for j in range(i):
            row_j = L[j]
            s = sum(map(mul, row_i[:j], row_j[:j]))
            row_i[j] = (mat_i[j] - s) / row_j[j]
        s_diag = sum(map(mul, row_i[:i], row_i[:i]))
        diag_val = mat_i[i] - s_diag
        if diag_val <= 0.0:
            raise ValueError("Macierz w kroku Newtona nie jest dodatnio okreslona")
        row_i[i] = math.sqrt(diag_val)
    return L


def _cholesky_solve(L: List[List[float]], rhs: Sequence[float]) -> List[float]:
    """Rozwiazuje uklad (L L^T) x = rhs majac dany czynnik Cholesky'ego L."""
    n = len(L)
    y = [0.0] * n
    for i in range(n):
        row_i = L[i]
        s = sum(map(mul, row_i[:i], y[:i]))
        y[i] = (rhs[i] - s) / row_i[i]

    x = [0.0] * n
    for i in range(n - 1, -1, -1):
        s = 0.0
        for k in range(i + 1, n):
            s += L[k][i] * x[k]
        x[i] = (y[i] - s) / L[i][i]
    return x


def _sparse_gram_matrix(X: Sequence[SparseVector]) -> List[List[float]]:
    """Oblicza symetryczna macierz Grama K = X X^T dla rzadkich wektorow."""
    n = len(X)
    K = [[0.0] * n for _ in range(n)]
    for i in range(n):
        xi = X[i]
        if not xi:
            continue
        K[i][i] = sum(v * v for v in xi.values())
        for j in range(i + 1, n):
            xj = X[j]
            if not xj:
                continue
            if len(xi) <= len(xj):
                dot_val = sum(v * xj[k] for k, v in xi.items() if k in xj)
            else:
                dot_val = sum(v * xi[k] for k, v in xj.items() if k in xi)
            K[i][j] = dot_val
            K[j][i] = dot_val
    return K


class LogisticRegressionL2:
    """Binarna regresja logistyczna z regularyzacja L2 uczona dokladna metoda Newtona.

    Funkcja celu (zgodna z konwencja scikit-learn, bez kary dla wyrazu wolnego b):
        J(w, b) = (1 / (2 * C)) * ||w||_2^2 + sum_i (softplus(z_i) - y_i * z_i)
    gdzie z_i = w^T x_i + b, y_i in {0, 1} (1 = spam, 0 = ham).

    Krok Newtona wyznaczany jest dokladnie poprzez tozsamosc Woodbury'ego
    w przestrzeni N x N (A = I + C * S K S, zawsze scisle dodatnio okreslona).
    """

    def __init__(
        self,
        *,
        C: float = 1.0,
        max_iter: int = 100,
        tol: float = 1e-8,
    ) -> None:
        if C <= 0.0:
            raise ValueError("Parametr C musi byc > 0")
        self.C = float(C)
        self.max_iter = int(max_iter)
        self.tol = float(tol)
        self.weights: List[float] = []
        self.intercept: float = 0.0
        self.n_iter_: int = 0
        self.converged_: bool = False
        self.grad_norm_: float = 0.0

    @staticmethod
    def objective_and_gradient(
        X: Sequence[SparseVector],
        y: Sequence[int],
        weights: Sequence[float],
        intercept: float,
        C: float,
    ) -> Tuple[float, List[float], float]:
        """Oblicza wartosc funkcji celu J(w, b) oraz gradient (dJ/dw, dJ/db)."""
        inv_c = 1.0 / float(C)
        grad_w = [inv_c * wj for wj in weights]
        grad_b = 0.0
        loss = 0.5 * inv_c * sum(wj * wj for wj in weights)

        for vec, yi in zip(X, y):
            z = intercept
            for j, val in vec.items():
                z += weights[j] * val
            loss += softplus(z) - yi * z
            diff = sigmoid(z) - yi
            grad_b += diff
            for j, val in vec.items():
                grad_w[j] += diff * val

        return loss, grad_w, grad_b

    def fit(
        self,
        X: Sequence[SparseVector],
        y: Sequence[int],
        *,
        n_features: int,
    ) -> "LogisticRegressionL2":
        n = len(X)
        if n != len(y) or n == 0:
            raise ValueError("X i y musza miec ta sama, niezerowa dlugosc")

        v_size = int(n_features)
        w = [0.0] * v_size
        b = 0.0
        C = self.C
        K = _sparse_gram_matrix(X)

        loss, gw, gb = self.objective_and_gradient(X, y, w, b, C)
        converged = False
        n_iter = 0

        for it in range(1, self.max_iter + 1):
            grad_l2 = math.sqrt(sum(g * g for g in gw) + gb * gb)
            if grad_l2 <= self.tol:
                converged = True
                break

            n_iter = it

            # Wyznaczenie p_i oraz s_i = sqrt(p_i * (1 - p_i))
            s = [0.0] * n
            for i, vec in enumerate(X):
                z = b
                for j, val in vec.items():
                    z += w[j] * val
                p = sigmoid(z)
                s[i] = math.sqrt(max(p * (1.0 - p), 0.0))

            # Budowa ukladu A = I_N + C * diag(s) * K * diag(s)
            A = [[0.0] * n for _ in range(n)]
            for i in range(n):
                si = s[i]
                Ki = K[i]
                Ai = A[i]
                csi = C * si
                for j in range(n):
                    Ai[j] = csi * Ki[j] * s[j]
                Ai[i] += 1.0

            # Prawa strona: rhs0 = -C * s * (X g_w), rhs1 = s
            rhs0 = [0.0] * n
            for i, vec in enumerate(X):
                xg = 0.0
                for j, val in vec.items():
                    xg += val * gw[j]
                rhs0[i] = -C * s[i] * xg

            L = _cholesky_decompose(A)
            u0 = _cholesky_solve(L, rhs0)
            u1 = _cholesky_solve(L, s)

            denom = sum(map(mul, s, u1))
            if denom <= 1e-30:
                break
            db = (-gb - sum(map(mul, s, u0))) / denom

            # v = s * (u0 + db * u1), dw = -C * (g_w + X^T v)
            dw = [-C * g for g in gw]
            for i, vec in enumerate(X):
                vi = s[i] * (u0[i] + db * u1[i])
                cvi = C * vi
                for j, val in vec.items():
                    dw[j] -= cvi * val

            dg = sum(map(mul, gw, dw)) + gb * db
            if dg >= 0.0:
                break

            # Backtracking z warunkiem Armijo (z zabezpieczeniem na blad zaokraglen float64
            # blisko minimum, gdzie zmiana funkcji celu jest rzedu ulp(loss), a norma gradientu maleje).
            step = 1.0
            c1 = 1e-4
            loss_tol = 1e-12 * (1.0 + abs(loss))
            accepted = False
            for _ in range(40):
                w_cand = [w[j] + step * dw[j] for j in range(v_size)]
                b_cand = b + step * db
                loss_cand, gw_cand, gb_cand = self.objective_and_gradient(
                    X, y, w_cand, b_cand, C
                )
                if math.isfinite(loss_cand):
                    grad_cand = math.sqrt(sum(g * g for g in gw_cand) + gb_cand * gb_cand)
                    armijo_ok = loss_cand <= loss + c1 * step * dg
                    near_min_ok = loss_cand <= loss + loss_tol and grad_cand < grad_l2
                    if armijo_ok or near_min_ok:
                        w = w_cand
                        b = b_cand
                        loss = loss_cand
                        gw = gw_cand
                        gb = gb_cand
                        accepted = True
                        break
                step *= 0.5

            if not accepted:
                break

        grad_l2 = math.sqrt(sum(g * g for g in gw) + gb * gb)
        if grad_l2 <= self.tol:
            converged = True

        self.weights = w
        self.intercept = b
        self.n_iter_ = n_iter
        self.converged_ = converged
        self.grad_norm_ = grad_l2
        return self

    def predict_proba_one(self, vec: SparseVector) -> float:
        z = self.intercept
        weights = self.weights
        for j, val in vec.items():
            z += weights[j] * val
        return sigmoid(z)

    def predict_proba(self, X: Sequence[SparseVector]) -> List[float]:
        return [self.predict_proba_one(vec) for vec in X]

    def feature_weights(self) -> List[float]:
        return list(self.weights)

    def to_dict(self) -> Dict[str, object]:
        return {
            "model_type": "logistic_regression",
            "hyperparams": {"C": self.C},
            "weights": list(self.weights),
            "intercept": self.intercept,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, object]) -> "LogisticRegressionL2":
        hp = data["hyperparams"]  # type: ignore[index]
        model = cls(C=float(hp["C"]))  # type: ignore[index]
        model.weights = [float(v) for v in data["weights"]]  # type: ignore[arg-type]
        model.intercept = float(data["intercept"])
        model.converged_ = True
        return model


class MultinomialNaiveBayes:
    """Wielomianowy Naiwny Bayes z wygladzaniem addytywnym alfa (stdlib Pythona)."""

    def __init__(self, *, alpha: float = 1.0) -> None:
        if alpha <= 0.0:
            raise ValueError("Parametr alpha musi byc > 0")
        self.alpha = float(alpha)
        self.class_log_prior: List[float] = [0.0, 0.0]
        self.feature_log_prob: List[List[float]] = [[], []]

    def fit(
        self,
        X: Sequence[SparseVector],
        y: Sequence[int],
        *,
        n_features: int,
    ) -> "MultinomialNaiveBayes":
        n_total = len(y)
        if len(X) != n_total or n_total == 0:
            raise ValueError("X i y musza miec ta sama, niezerowa dlugosc")

        n_spam = sum(1 for yi in y if yi == 1)
        n_ham = n_total - n_spam
        if n_spam == 0 or n_ham == 0:
            raise ValueError("Obie klasy (ham i spam) musza wystepowac w zbiorze uczacym")

        self.class_log_prior = [
            math.log(n_ham / n_total),
            math.log(n_spam / n_total),
        ]

        v_size = int(n_features)
        sums = [[0.0] * v_size, [0.0] * v_size]
        for vec, yi in zip(X, y):
            cls_sums = sums[yi]
            for j, val in vec.items():
                cls_sums[j] += val

        alpha = self.alpha
        log_probs: List[List[float]] = [[], []]
        for cls_idx in (0, 1):
            total_weight = sum(sums[cls_idx])
            denom = total_weight + alpha * v_size
            log_probs[cls_idx] = [
                math.log((sums[cls_idx][j] + alpha) / denom) for j in range(v_size)
            ]

        self.feature_log_prob = log_probs
        return self

    def predict_proba_one(self, vec: SparseVector) -> float:
        ll0 = self.class_log_prior[0]
        ll1 = self.class_log_prior[1]
        lp0 = self.feature_log_prob[0]
        lp1 = self.feature_log_prob[1]
        for j, val in vec.items():
            ll0 += val * lp0[j]
            ll1 += val * lp1[j]

        m = ll0 if ll0 >= ll1 else ll1
        e0 = math.exp(ll0 - m)
        e1 = math.exp(ll1 - m)
        return e1 / (e0 + e1)

    def predict_proba(self, X: Sequence[SparseVector]) -> List[float]:
        return [self.predict_proba_one(vec) for vec in X]

    def feature_weights(self) -> List[float]:
        """Zwraca wagi log-odds: log P(j|spam) - log P(j|ham)."""
        lp0, lp1 = self.feature_log_prob
        return [a - b for a, b in zip(lp1, lp0)]

    @property
    def intercept(self) -> float:
        return self.class_log_prior[1] - self.class_log_prior[0]

    def to_dict(self) -> Dict[str, object]:
        return {
            "model_type": "multinomial_nb",
            "hyperparams": {"alpha": self.alpha},
            "class_log_prior": list(self.class_log_prior),
            "feature_log_prob": [
                list(self.feature_log_prob[0]),
                list(self.feature_log_prob[1]),
            ],
            "weights": self.feature_weights(),
            "intercept": self.intercept,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, object]) -> "MultinomialNaiveBayes":
        hp = data["hyperparams"]  # type: ignore[index]
        model = cls(alpha=float(hp["alpha"]))  # type: ignore[index]
        model.class_log_prior = [float(v) for v in data["class_log_prior"]]  # type: ignore[arg-type]
        flp = data["feature_log_prob"]  # type: ignore[index]
        model.feature_log_prob = [
            [float(v) for v in flp[0]],  # type: ignore[index]
            [float(v) for v in flp[1]],  # type: ignore[index]
        ]
        return model


ModelType = LogisticRegressionL2 | MultinomialNaiveBayes


class SpamClassifierArtifact:
    """Kompletny, serializowalny artefakt klasyfikatora do uzycia bez ponownego uczenia."""

    def __init__(
        self,
        *,
        vectorizer: TfidfVectorizer,
        model: ModelType,
        threshold: float,
        apply_masking: bool = True,
    ) -> None:
        self.vectorizer = vectorizer
        self.model = model
        self.threshold = float(threshold)
        self.apply_masking = bool(apply_masking)

    def predict_one(self, text: str) -> Tuple[float, str]:
        tokens = tokenize_text(text, apply_masking=self.apply_masking)
        vec = self.vectorizer.transform_one(tokens)
        score = self.model.predict_proba_one(vec)
        label = "spam" if score >= self.threshold else "ham"
        return score, label

    def predict_texts(self, texts: Sequence[str]) -> List[Tuple[float, str]]:
        return [self.predict_one(t) for t in texts]

    def explain_one(self, text: str, *, top_k: int = 10) -> List[Dict[str, object]]:
        """Zwraca tokeny o najwiekszym wkladzie (waga * wartosc TF-IDF) w ocene wiadomosci."""
        tokens = tokenize_text(text, apply_masking=self.apply_masking)
        vec = self.vectorizer.transform_one(tokens)
        weights = self.model.feature_weights()
        terms = self.vectorizer.index_to_term()
        contributions = []
        for idx, tfidf_val in vec.items():
            w = weights[idx]
            contributions.append(
                {
                    "token": terms[idx],
                    "tfidf": tfidf_val,
                    "weight": w,
                    "contribution": tfidf_val * w,
                }
            )
        contributions.sort(key=lambda item: abs(float(item["contribution"])), reverse=True)
        return contributions[:top_k]

    def top_features(self, *, top_k: int = 30) -> Dict[str, List[Dict[str, object]]]:
        """Zwraca top_k slow najbardziej wspierajacych spam oraz ham."""
        weights = self.model.feature_weights()
        terms = self.vectorizer.index_to_term()
        pairs = list(zip(terms, weights))
        top_spam = sorted(pairs, key=lambda x: (-x[1], x[0]))[:top_k]
        top_ham = sorted(pairs, key=lambda x: (x[1], x[0]))[:top_k]
        return {
            "spam": [{"token": t, "weight": w} for t, w in top_spam],
            "ham": [{"token": t, "weight": w} for t, w in top_ham],
        }

    def to_dict(self) -> Dict[str, object]:
        return {
            "artifact_version": "1.0",
            "preprocessing": get_preprocessing_spec(apply_masking=self.apply_masking),
            "tfidf": self.vectorizer.to_dict(),
            "model": self.model.to_dict(),
            "threshold": self.threshold,
        }

    def save_json(self, path: str | Path) -> None:
        out_path = Path(path)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        payload = json.dumps(self.to_dict(), indent=2, sort_keys=True, ensure_ascii=False)
        out_path.write_text(payload + "\n", encoding="utf-8")

    @classmethod
    def from_dict(cls, data: Dict[str, object]) -> "SpamClassifierArtifact":
        prep = data["preprocessing"]  # type: ignore[index]
        apply_masking = bool(prep.get("apply_masking", True))  # type: ignore[union-attr]
        vectorizer = TfidfVectorizer.from_dict(data["tfidf"])  # type: ignore[arg-type]
        model_data = data["model"]  # type: ignore[index]
        mtype = model_data["model_type"]  # type: ignore[index]
        if mtype == "logistic_regression":
            model: ModelType = LogisticRegressionL2.from_dict(model_data)  # type: ignore[arg-type]
        elif mtype == "multinomial_nb":
            model = MultinomialNaiveBayes.from_dict(model_data)  # type: ignore[arg-type]
        else:
            raise ValueError(f"Nieznany typ modelu w JSON: {mtype!r}")
        return cls(
            vectorizer=vectorizer,
            model=model,
            threshold=float(data["threshold"]),
            apply_masking=apply_masking,
        )

    @classmethod
    def load_json(cls, path: str | Path) -> "SpamClassifierArtifact":
        raw = Path(path).read_text(encoding="utf-8")
        return cls.from_dict(json.loads(raw))
