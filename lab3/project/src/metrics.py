"""Miary oceny klasyfikacji binarnej (spam=1, ham=0) i dobor progu decyzji."""

from __future__ import annotations

from typing import Dict, List, Sequence, Tuple


def labels_to_binary(labels: Sequence[str]) -> List[int]:
    out: List[int] = []
    for lbl in labels:
        if lbl == "spam":
            out.append(1)
        elif lbl == "ham":
            out.append(0)
        else:
            raise ValueError(f"Nieznana etykieta: {lbl!r}")
    return out


def confusion_matrix(y_true: Sequence[int], y_pred: Sequence[int]) -> Dict[str, int]:
    """Oblicza macierz pomylek (klasa pozytywna: spam=1)."""
    if len(y_true) != len(y_pred):
        raise ValueError("y_true i y_pred musza miec ta sama dlugosc")
    tp = fp = tn = fn = 0
    for yt, yp in zip(y_true, y_pred):
        if yt == 1 and yp == 1:
            tp += 1
        elif yt == 0 and yp == 1:
            fp += 1
        elif yt == 0 and yp == 0:
            tn += 1
        elif yt == 1 and yp == 0:
            fn += 1
        else:
            raise ValueError(f"Nieprawidlowe wartosci etykiet: yt={yt}, yp={yp}")
    return {"tp": tp, "fp": fp, "tn": tn, "fn": fn}


def average_precision(y_true: Sequence[int], y_scores: Sequence[float]) -> float:
    """Pole pod krzywa precyzja-czulosc (Average Precision) z poprawna obsluga remisow."""
    if len(y_true) != len(y_scores):
        raise ValueError("y_true i y_scores musza miec ta sama dlugosc")
    n_pos = sum(1 for y in y_true if y == 1)
    if n_pos == 0:
        return 0.0

    # Grupowanie remisow wedlug unikalnych ocen malejaco (niezalezne od kolejnosci wierszy).
    grouped: Dict[float, List[int]] = {}
    for yt, sc in zip(y_true, y_scores):
        grouped.setdefault(float(sc), []).append(int(yt))

    tp = 0
    fp = 0
    prev_recall = 0.0
    ap = 0.0
    for score in sorted(grouped.keys(), reverse=True):
        bucket = grouped[score]
        tp += sum(1 for y in bucket if y == 1)
        fp += sum(1 for y in bucket if y == 0)
        recall = tp / n_pos
        precision = tp / (tp + fp)
        ap += (recall - prev_recall) * precision
        prev_recall = recall

    return ap


def compute_metrics_from_predictions(
    y_true: Sequence[int],
    y_pred: Sequence[int],
    *,
    y_scores: Sequence[float] | None = None,
) -> Dict[str, object]:
    """Oblicza komplet miar z planu (z liczbami bezwzglednymi obok ulamkow)."""
    cm = confusion_matrix(y_true, y_pred)
    tp, fp, tn, fn = cm["tp"], cm["fp"], cm["tn"], cm["fn"]
    n_total = tp + fp + tn + fn
    n_spam = tp + fn
    n_ham = fp + tn
    n_pred_spam = tp + fp

    fpr = (fp / n_ham) if n_ham > 0 else 0.0
    recall = (tp / n_spam) if n_spam > 0 else 0.0
    precision = (tp / n_pred_spam) if n_pred_spam > 0 else 0.0
    accuracy = ((tp + tn) / n_total) if n_total > 0 else 0.0
    pr_auc = average_precision(y_true, y_scores) if y_scores is not None else None

    return {
        "confusion_matrix": cm,
        "n_total": n_total,
        "n_spam": n_spam,
        "n_ham": n_ham,
        "n_pred_spam": n_pred_spam,
        "fpr": fpr,
        "fpr_fraction": f"{fp}/{n_ham}",
        "recall": recall,
        "recall_fraction": f"{tp}/{n_spam}",
        "precision": precision,
        "precision_fraction": f"{tp}/{n_pred_spam}",
        "accuracy": accuracy,
        "accuracy_fraction": f"{tp + tn}/{n_total}",
        "pr_auc": pr_auc,
    }


def compute_metrics(
    y_true: Sequence[int],
    y_scores: Sequence[float],
    threshold: float,
) -> Dict[str, object]:
    """Oblicza miary dla zadanych ocen i progu (spam gdy score >= threshold)."""
    y_pred = [1 if sc >= threshold else 0 for sc in y_scores]
    metrics = compute_metrics_from_predictions(y_true, y_pred, y_scores=y_scores)
    metrics["threshold"] = float(threshold)
    return metrics


def select_threshold(
    y_true: Sequence[int],
    y_scores: Sequence[float],
    *,
    max_fp: int = 3,
) -> Tuple[float, Dict[str, object]]:
    """Dobiera prog dajacy najwieksza czulosc przy FP <= max_fp.

    Reguly rozstrzygania remisow:
    1. Najwieksze TP (czulosc),
    2. Najmniejsze FP,
    3. Srodek przedzialu ocen dajacego dokladnie to optymalne (TP, FP).
    """
    if len(y_true) != len(y_scores) or len(y_true) == 0:
        raise ValueError("y_true i y_scores musza miec ta sama, niezerowa dlugosc")

    grouped: Dict[float, List[int]] = {}
    for yt, sc in zip(y_true, y_scores):
        grouped.setdefault(float(sc), []).append(int(yt))

    unique_desc = sorted(grouped.keys(), reverse=True)
    m = len(unique_desc)

    # Punkt pracy 0: nic nie jest spamem (prog powyzej maksimum ocen).
    max_score = unique_desc[0]
    above_max = 0.5 * (max_score + 1.0) if max_score < 1.0 else max_score + 1e-6
    best_tp = 0
    best_fp = 0
    best_threshold = above_max

    cum_tp = 0
    cum_fp = 0
    for idx, s_hi in enumerate(unique_desc):
        bucket = grouped[s_hi]
        cum_tp += sum(1 for y in bucket if y == 1)
        cum_fp += sum(1 for y in bucket if y == 0)

        if cum_fp > max_fp:
            break

        if idx + 1 < m:
            s_lo = unique_desc[idx + 1]
            cand_threshold = 0.5 * (s_lo + s_hi)
            if cand_threshold <= s_lo:
                cand_threshold = s_hi
        else:
            cand_threshold = 0.5 * s_hi

        if (cum_tp > best_tp) or (cum_tp == best_tp and cum_fp < best_fp):
            best_tp = cum_tp
            best_fp = cum_fp
            best_threshold = cand_threshold

    metrics = compute_metrics(y_true, y_scores, best_threshold)
    return best_threshold, metrics
