"""Modele klasyfikacji w czystym Pythonie 3.12 oraz deterministyczny zapis JSON."""

from __future__ import annotations

import json
import math
from operator import mul
from pathlib import Path
from typing import Any

CLASSES = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
MODEL_FORMAT_VERSION = "1.0"
FEATURE_TRANSFORM_SPEC = {
    "description": "64 grayscale pixels (8x8 row-major) divided by 255.0",
    "input_shape": [8, 8],
    "name": "pixel_div_255",
    "scale": 255.0,
}


def _sq_l2_distance(a: list[float], b: list[float]) -> float:
    total = 0.0
    for u, v in zip(a, b):
        diff = u - v
        total += diff * diff
    return total


def _l1_distance(a: list[float], b: list[float]) -> float:
    total = 0.0
    for u, v in zip(a, b):
        total += abs(u - v)
    return total


class NearestCentroidModel:
    """Punkt odniesienia: najblizszy centroid klasy w odleglosci L2."""

    model_type = "nearest_centroid"

    def __init__(self) -> None:
        self.centroids: dict[int, list[float]] = {}

    def fit(self, X: list[list[float]], y: list[int]) -> "NearestCentroidModel":
        if not X:
            raise ValueError("Pusty zbior uczacy.")
        dim = len(X[0])
        sums: dict[int, list[float]] = {c: [0.0] * dim for c in CLASSES}
        counts: dict[int, int] = {c: 0 for c in CLASSES}

        for vec, label in zip(X, y):
            counts[label] += 1
            acc = sums[label]
            for j, val in enumerate(vec):
                acc[j] += val

        self.centroids = {}
        for c in CLASSES:
            if counts[c] == 0:
                raise ValueError(f"Brak przykladow klasy {c} w zbiorze uczacym.")
            inv_cnt = 1.0 / counts[c]
            self.centroids[c] = [s * inv_cnt for s in sums[c]]
        return self

    def predict_one(self, x: list[float]) -> int:
        best_class = CLASSES[0]
        best_dist = _sq_l2_distance(x, self.centroids[best_class])
        for c in CLASSES[1:]:
            dist = _sq_l2_distance(x, self.centroids[c])
            if dist < best_dist:
                best_dist = dist
                best_class = c
        return best_class

    def predict(self, X: list[list[float]]) -> list[int]:
        return [self.predict_one(x) for x in X]


class KNNModel:
    """Kandydat A: k-NN z rownymi glosami i deterministycznymi regulami remisu."""

    model_type = "knn"

    def __init__(self, k: int, metric: str) -> None:
        if k < 1:
            raise ValueError("k musi byc >= 1.")
        if metric not in {"l2", "l1"}:
            raise ValueError(f"Nieobslugiwana metryka: {metric} (dozwolone: l2, l1).")
        self.k = int(k)
        self.metric = metric
        self.X_train: list[list[float]] = []
        self.y_train: list[int] = []

    def fit(self, X: list[list[float]], y: list[int]) -> "KNNModel":
        if len(X) != len(y) or not X:
            raise ValueError("Nieprawidlowy zbior uczacy dla k-NN.")
        if self.k > len(X):
            raise ValueError("k nie moze przekraczac liczby przykladow uczacych.")
        self.X_train = [list(vec) for vec in X]
        self.y_train = [int(lbl) for lbl in y]
        return self

    def _dist(self, a: list[float], b: list[float]) -> float:
        if self.metric == "l2":
            return _sq_l2_distance(a, b)
        return _l1_distance(a, b)

    def predict_one_with_ties(self, x: list[float]) -> tuple[int, dict[str, bool]]:
        dist_fn = self._dist
        neighbors = [
            (dist_fn(x, tx), idx, ty)
            for idx, (tx, ty) in enumerate(zip(self.X_train, self.y_train))
        ]
        # Remis odleglosci przy wyborze k sasiadow rozstrzyga kolejnosc w train.csv (idx).
        neighbors.sort(key=lambda item: (item[0], item[1]))

        distance_tie_at_k = (
            self.k < len(neighbors)
            and neighbors[self.k - 1][0] == neighbors[self.k][0]
        )

        top_k = neighbors[: self.k]
        vote_counts: dict[int, int] = {}
        min_dist_per_class: dict[int, float] = {}
        for dist, _idx, label in top_k:
            vote_counts[label] = vote_counts.get(label, 0) + 1
            if label not in min_dist_per_class or dist < min_dist_per_class[label]:
                min_dist_per_class[label] = dist

        max_votes = max(vote_counts.values())
        tied_classes = [c for c, cnt in vote_counts.items() if cnt == max_votes]

        vote_tie = len(tied_classes) > 1
        distance_tie_among_tied_classes = False

        if not vote_tie:
            predicted = tied_classes[0]
        else:
            # Remis glosow rozstrzyga najblizszy sasiad sposrod remisujacych klas;
            # rowna odleglosc rozstrzyga nizsza etykieta.
            best_min_dist = min(min_dist_per_class[c] for c in tied_classes)
            closest_tied = sorted(
                c for c in tied_classes if min_dist_per_class[c] == best_min_dist
            )
            distance_tie_among_tied_classes = len(closest_tied) > 1
            predicted = closest_tied[0]

        tie_info = {
            "distance_tie_at_k_boundary": distance_tie_at_k,
            "vote_tie": vote_tie,
            "distance_tie_among_tied_classes": distance_tie_among_tied_classes,
        }
        return predicted, tie_info

    def predict_one(self, x: list[float]) -> int:
        pred, _ = self.predict_one_with_ties(x)
        return pred

    def predict(self, X: list[list[float]]) -> list[int]:
        return [self.predict_one(x) for x in X]

    def predict_with_tie_stats(
        self, X: list[list[float]]
    ) -> tuple[list[int], dict[str, int]]:
        preds: list[int] = []
        stats = {
            "distance_tie_at_k_boundary": 0,
            "vote_tie": 0,
            "distance_tie_among_tied_classes": 0,
        }
        for x in X:
            pred, info = self.predict_one_with_ties(x)
            preds.append(pred)
            for key, triggered in info.items():
                if triggered:
                    stats[key] += 1
        return preds, stats

    def to_dict(self, train_csv_sha256: str) -> dict[str, Any]:
        return {
            "classes": CLASSES,
            "feature_transform": FEATURE_TRANSFORM_SPEC,
            "format_version": MODEL_FORMAT_VERSION,
            "model_type": self.model_type,
            "parameters": {
                "train_labels": self.y_train,
                "train_vectors": self.X_train,
            },
            "settings": {
                "k": self.k,
                "metric": self.metric,
                "tie_breaking": {
                    "k_boundary_distance_tie": "train_csv_order",
                    "vote_tie": "nearest_neighbor_among_tied_classes",
                    "tied_classes_distance_tie": "lower_label",
                },
                "weights": "uniform",
            },
            "train_csv_sha256": train_csv_sha256,
        }

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "KNNModel":
        settings = payload["settings"]
        params = payload["parameters"]
        model = cls(k=int(settings["k"]), metric=str(settings["metric"]))
        model.X_train = [[float(v) for v in row] for row in params["train_vectors"]]
        model.y_train = [int(lbl) for lbl in params["train_labels"]]
        if len(model.X_train) != len(model.y_train) or not model.X_train:
            raise ValueError("Uszkodzone parametry zapisanego modelu k-NN.")
        return model


class SoftmaxRegressionModel:
    """Kandydat B: wieloklasowa regresja softmax z kara L2 (lambda/2)*||W||^2."""

    model_type = "softmax"

    def __init__(
        self,
        reg_lambda: float,
        learning_rate: float = 1.0,
        max_epochs: int = 1000,
        rel_tol: float = 1e-6,
    ) -> None:
        if reg_lambda < 0.0:
            raise ValueError("Lambda musi byc >= 0.")
        if learning_rate <= 0.0:
            raise ValueError("Krok uczenia musi byc > 0.")
        if max_epochs < 1:
            raise ValueError("Maksymalna liczba epok musi byc >= 1.")
        if rel_tol < 0.0:
            raise ValueError("Prog wzglednej zmiany straty musi byc >= 0.")
        self.reg_lambda = float(reg_lambda)
        self.learning_rate = float(learning_rate)
        self.max_epochs = int(max_epochs)
        self.rel_tol = float(rel_tol)
        self.W: list[list[float]] = []
        self.b: list[float] = []
        self.training_info: dict[str, Any] = {}

    @staticmethod
    def _eval_loss_and_grads(
        X: list[list[float]],
        X_T: list[list[float]],
        y: list[int],
        W: list[list[float]],
        b: list[float],
        reg_lambda: float,
    ) -> tuple[float, list[list[float]], list[float], float]:
        n_samples = len(X)
        n_classes = len(CLASSES)
        n_features = len(X_T)
        inv_n = 1.0 / n_samples

        err_T: list[list[float]] = [[0.0] * n_samples for _ in range(n_classes)]
        grad_b = [0.0] * n_classes
        ce_sum = 0.0

        for i in range(n_samples):
            x_i = X[i]
            logits = [b[c] + sum(map(mul, W[c], x_i)) for c in range(n_classes)]
            max_logit = max(logits)
            exps = [math.exp(z - max_logit) for z in logits]
            sum_exp = sum(exps)
            inv_sum = 1.0 / sum_exp
            target = y[i]

            for c in range(n_classes):
                prob = exps[c] * inv_sum
                if c == target:
                    ce_sum -= math.log(max(prob, 1e-300))
                    diff = prob - 1.0
                else:
                    diff = prob
                err_T[c][i] = diff
                grad_b[c] += diff

        l2_sq = 0.0
        for c in range(n_classes):
            grad_b[c] *= inv_n
            w_c = W[c]
            l2_sq += sum(map(mul, w_c, w_c))

        loss = ce_sum * inv_n + 0.5 * reg_lambda * l2_sq

        grad_W: list[list[float]] = [[0.0] * n_features for _ in range(n_classes)]
        grad_norm_sq = sum(g * g for g in grad_b)
        for c in range(n_classes):
            err_c = err_T[c]
            w_c = W[c]
            gw_c = grad_W[c]
            for j in range(n_features):
                g_val = sum(map(mul, err_c, X_T[j])) * inv_n + reg_lambda * w_c[j]
                gw_c[j] = g_val
                grad_norm_sq += g_val * g_val

        return loss, grad_W, grad_b, math.sqrt(grad_norm_sq)

    def fit(self, X: list[list[float]], y: list[int]) -> "SoftmaxRegressionModel":
        if len(X) != len(y) or not X:
            raise ValueError("Nieprawidlowy zbior uczacy dla regresji softmax.")
        n_features = len(X[0])
        n_classes = len(CLASSES)
        W = [[0.0] * n_features for _ in range(n_classes)]
        b = [0.0] * n_classes
        X_T = [[row[j] for row in X] for j in range(n_features)]

        prev_loss: float | None = None
        initial_loss: float | None = None
        final_loss = 0.0
        final_grad_norm = 0.0
        stopped_by = "max_epochs"
        epochs_run = 0
        lr = self.learning_rate

        for ep in range(self.max_epochs):
            loss, grad_W, grad_b, grad_norm = self._eval_loss_and_grads(
                X, X_T, y, W, b, self.reg_lambda
            )
            if not math.isfinite(loss):
                raise RuntimeError(
                    f"Rozbieznosc uczenia softmax w epoce {ep} (loss={loss})."
                )
            if initial_loss is None:
                initial_loss = loss

            if prev_loss is not None:
                rel_change = abs(prev_loss - loss) / max(abs(prev_loss), 1e-12)
                if rel_change < self.rel_tol:
                    stopped_by = "threshold"
                    epochs_run = ep
                    final_loss = loss
                    final_grad_norm = grad_norm
                    break

            prev_loss = loss
            for c in range(n_classes):
                b[c] -= lr * grad_b[c]
                w_c = W[c]
                gw_c = grad_W[c]
                for j in range(n_features):
                    w_c[j] -= lr * gw_c[j]
            epochs_run = ep + 1
        else:
            final_loss, _, _, final_grad_norm = self._eval_loss_and_grads(
                X, X_T, y, W, b, self.reg_lambda
            )
            stopped_by = "max_epochs"

        self.W = W
        self.b = b
        self.training_info = {
            "epochs_run": epochs_run,
            "final_grad_norm": final_grad_norm,
            "final_loss": final_loss,
            "initial_loss": initial_loss,
            "stopped_by": stopped_by,
        }
        return self

    def predict_one(self, x: list[float]) -> int:
        best_class = CLASSES[0]
        best_logit = self.b[0] + sum(map(mul, self.W[0], x))
        for c in range(1, len(CLASSES)):
            logit = self.b[c] + sum(map(mul, self.W[c], x))
            if logit > best_logit:
                best_logit = logit
                best_class = CLASSES[c]
        return best_class

    def predict(self, X: list[list[float]]) -> list[int]:
        return [self.predict_one(x) for x in X]

    def to_dict(self, train_csv_sha256: str) -> dict[str, Any]:
        return {
            "classes": CLASSES,
            "feature_transform": FEATURE_TRANSFORM_SPEC,
            "format_version": MODEL_FORMAT_VERSION,
            "model_type": self.model_type,
            "parameters": {
                "b": self.b,
                "W": self.W,
            },
            "settings": {
                "init": "zeros",
                "learning_rate": self.learning_rate,
                "loss": "mean_cross_entropy_plus_half_lambda_l2_W",
                "max_epochs": self.max_epochs,
                "reg_lambda": self.reg_lambda,
                "rel_tol": self.rel_tol,
            },
            "train_csv_sha256": train_csv_sha256,
            "training_info": self.training_info,
        }

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "SoftmaxRegressionModel":
        settings = payload["settings"]
        params = payload["parameters"]
        model = cls(
            reg_lambda=float(settings["reg_lambda"]),
            learning_rate=float(settings["learning_rate"]),
            max_epochs=int(settings["max_epochs"]),
            rel_tol=float(settings["rel_tol"]),
        )
        model.W = [[float(v) for v in row] for row in params["W"]]
        model.b = [float(v) for v in params["b"]]
        model.training_info = dict(payload.get("training_info", {}))
        if len(model.W) != len(CLASSES) or len(model.b) != len(CLASSES):
            raise ValueError("Uszkodzone parametry zapisanego modelu softmax.")
        return model


def save_model(
    model: KNNModel | SoftmaxRegressionModel,
    output_path: str | Path,
    train_csv_sha256: str,
) -> Path:
    """Zapisuje model jako deterministyczny JSON (sort_keys=True, LF, UTF-8)."""
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = model.to_dict(train_csv_sha256=train_csv_sha256)
    serialized = (
        json.dumps(
            payload,
            sort_keys=True,
            indent=2,
            ensure_ascii=False,
            allow_nan=False,
        )
        + "\n"
    )
    path.write_text(serialized, encoding="utf-8", newline="\n")
    return path


def load_model(model_path: str | Path) -> KNNModel | SoftmaxRegressionModel:
    """Wczytuje zapisany model z JSON bez ponownego uczenia."""
    path = Path(model_path)
    if not path.is_file():
        raise FileNotFoundError(f"Plik modelu nie istnieje: {path}")
    payload = json.loads(path.read_text(encoding="utf-8"))
    version = payload.get("format_version")
    if version != MODEL_FORMAT_VERSION:
        raise ValueError(
            f"Nieobslugiwana wersja formatu modelu: {version} "
            f"(oczekiwano {MODEL_FORMAT_VERSION})."
        )
    model_type = payload.get("model_type")
    if model_type == "knn":
        return KNNModel.from_dict(payload)
    if model_type == "softmax":
        return SoftmaxRegressionModel.from_dict(payload)
    raise ValueError(f"Nieznany typ modelu: {model_type}")
