"""S7: Norma gradientu koncowego modelu (spam_classifier.json) na calym train.csv.

Odczytuje wagi, intercept, C, slownik i IDF wprost z lab3/project/models/spam_classifier.json.
Przetwarza teksty train.csv przez tokenize_text (zgodnie z apply_masking z JSON), buduje
wektory TF-IDF wlasna funkcja (z porownaniem do TfidfVectorizer.from_dict) i liczy strate L2
oraz gradient niezaleznym wzorem (bez objective_and_gradient). Dla porownania wylicza tez
gradient funkcja projektu LogisticRegressionL2.objective_and_gradient.
Kryterium: ||grad||_2 < 1e-6.
"""

from __future__ import annotations

import csv
import json
import math
import sys
from collections import Counter
from pathlib import Path
from typing import Dict, List, Sequence, Tuple


REPO_ROOT = Path(__file__).resolve().parents[2]
LAB3_PROJECT = REPO_ROOT / "lab3" / "project"
MODEL_JSON = LAB3_PROJECT / "models" / "spam_classifier.json"
TRAIN_CSV = LAB3_PROJECT / "data" / "train.csv"
OUTPUT_JSON = REPO_ROOT / "lab4" / "wyniki" / "s7_final_gradient.json"

sys.path.insert(0, str(LAB3_PROJECT))

from src.models import LogisticRegressionL2  # noqa: E402
from src.preprocess import tokenize_text  # noqa: E402
from src.tfidf import TfidfVectorizer  # noqa: E402


def independent_sigmoid(z: float) -> float:
    if z >= 0.0:
        return 1.0 / (1.0 + math.exp(-z))
    ez = math.exp(z)
    return ez / (1.0 + ez)


def independent_softplus(z: float) -> float:
    if z > 0.0:
        return z + math.log1p(math.exp(-z))
    return math.log1p(math.exp(z))


def independent_tfidf_transform(
    tokenized_docs: Sequence[Sequence[str]],
    vocab: Dict[str, int],
    idf: Sequence[float],
) -> List[Dict[int, float]]:
    vecs: List[Dict[int, float]] = []
    for toks in tokenized_docs:
        counts: Counter[int] = Counter()
        for t in toks:
            idx = vocab.get(t)
            if idx is not None:
                counts[idx] += 1
        if not counts:
            vecs.append({})
            continue
        raw = {idx: cnt * idf[idx] for idx, cnt in counts.items()}
        norm = math.sqrt(sum(v * v for v in raw.values()))
        if norm == 0.0:
            vecs.append({})
        else:
            vecs.append({idx: val / norm for idx, val in sorted(raw.items())})
    return vecs


def independent_l2_loss_and_gradient(
    X: Sequence[Dict[int, float]],
    y: Sequence[int],
    weights: Sequence[float],
    intercept: float,
    C: float,
) -> Tuple[float, List[float], float]:
    """Niezalezna implementacja funkcji celu i gradientu regresji logistycznej L2:
    J(w, b) = (1 / (2 * C)) * ||w||_2^2 + sum_i (log(1 + exp(z_i)) - y_i * z_i)
    """
    n_features = len(weights)
    gw = [weights[j] / C for j in range(n_features)]
    gb = 0.0
    loss = 0.5 * sum(wj * wj for wj in weights) / C

    for vec, yi in zip(X, y):
        z = intercept + sum(weights[j] * v for j, v in vec.items())
        loss += independent_softplus(z) - yi * z
        err = independent_sigmoid(z) - yi
        gb += err
        for j, v in vec.items():
            gw[j] += err * v

    return loss, gw, gb


def main() -> None:
    model_data = json.loads(MODEL_JSON.read_text(encoding="utf-8"))
    apply_masking = bool(model_data["preprocessing"]["apply_masking"])
    terms: List[str] = list(model_data["tfidf"]["terms"])
    idf: List[float] = [float(v) for v in model_data["tfidf"]["idf"]]
    vocab = {term: idx for idx, term in enumerate(terms)}
    weights: List[float] = [float(v) for v in model_data["model"]["weights"]]
    intercept = float(model_data["model"]["intercept"])
    C = float(model_data["model"]["hyperparams"]["C"])

    texts: List[str] = []
    y: List[int] = []
    with open(TRAIN_CSV, "r", encoding="utf-8", newline="") as f:
        for row in csv.DictReader(f):
            texts.append(row["text"])
            y.append(1 if row["label"] == "spam" else 0)

    tokenized = [tokenize_text(t, apply_masking=apply_masking) for t in texts]
    X_ind = independent_tfidf_transform(tokenized, vocab, idf)

    # Porownanie wektorow z TfidfVectorizer.from_dict
    vec_proj = TfidfVectorizer.from_dict(model_data["tfidf"])
    X_proj = vec_proj.transform(tokenized)
    max_vec_diff = 0.0
    for va, vb in zip(X_ind, X_proj):
        assert set(va.keys()) == set(vb.keys())
        for k in va:
            d = abs(va[k] - vb[k])
            if d > max_vec_diff:
                max_vec_diff = d

    # Niezalezne obliczenie straty i gradientu
    ind_loss, ind_gw, ind_gb = independent_l2_loss_and_gradient(
        X_ind, y, weights, intercept, C
    )
    ind_grad_l2 = math.sqrt(sum(g * g for g in ind_gw) + ind_gb * ind_gb)
    ind_grad_max = max(max(abs(g) for g in ind_gw), abs(ind_gb))

    # Dla porownania: funkcja projektu LogisticRegressionL2.objective_and_gradient
    proj_loss, proj_gw, proj_gb = LogisticRegressionL2.objective_and_gradient(
        X_proj, y, weights, intercept, C
    )
    proj_grad_l2 = math.sqrt(sum(g * g for g in proj_gw) + proj_gb * proj_gb)
    proj_grad_max = max(max(abs(g) for g in proj_gw), abs(proj_gb))

    passed = (ind_grad_l2 < 1e-6) and (proj_grad_l2 < 1e-6)

    payload: Dict[str, object] = {
        "check": "S7",
        "description": "Norma gradientu koncowego modelu (spam_classifier.json) na calym train.csv",
        "expected": {"grad_l2_norm_lt": 1e-6},
        "model_params": {
            "model_type": model_data["model"]["model_type"],
            "C": C,
            "n_features": len(weights),
            "n_train_docs": len(y),
            "intercept": intercept,
            "weights_l2_norm": math.sqrt(sum(w * w for w in weights)),
        },
        "tfidf_vector_max_abs_diff_vs_project": max_vec_diff,
        "independent_formula": {
            "loss": ind_loss,
            "grad_l2_norm": ind_grad_l2,
            "grad_max_abs": ind_grad_max,
            "grad_intercept": ind_gb,
        },
        "project_objective_and_gradient": {
            "loss": proj_loss,
            "grad_l2_norm": proj_grad_l2,
            "grad_max_abs": proj_grad_max,
            "grad_intercept": proj_gb,
        },
        "comparison_independent_vs_project": {
            "loss_abs_diff": abs(ind_loss - proj_loss),
            "grad_l2_abs_diff": abs(ind_grad_l2 - proj_grad_l2),
        },
        "what_s7_does_not_cover": [
            "Przetwarzanie tekstu (maskowanie i tokenizacja) pochodzi z kodu projektu (src/preprocess.py), wiec S7 weryfikuje zbieznosc optymalizacji regresji L2 dla tej macierzy X, a nie poprawnosc samej tokenizacji ani maskowania.",
            "Nie weryfikuje wyboru hiperparametru C ani progu decyzji (to dotyczy CV, a nie minimum funkcji celu na calym train).",
            "Nie dotyczy modelu alternatywnego (MultinomialNaiveBayes nie uzywa optymalizacji gradientowej).",
        ],
        "passed": passed,
    }

    OUTPUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT_JSON.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )

    print(
        f"S7 zakonczone: passed={passed} | "
        f"independent ||grad||_2={ind_grad_l2:.3e} (max={ind_grad_max:.3e}) | "
        f"project ||grad||_2={proj_grad_l2:.3e} | loss={ind_loss:.6f}"
    )
    if not passed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
