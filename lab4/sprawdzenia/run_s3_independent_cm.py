"""S3: Niezalezna macierz pomylek bez importowania czegokolwiek z lab3/project/src/.

Czyta validation.csv (kolumna label), validation_predictions.csv oraz
validation_predictions_alternative.csv, laczy po message_id, liczy TP/FP/FN/TN,
miary i porownuje z lab3/project/results/validation_metrics.json.
"""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Dict, List, Sequence, Tuple


REPO_ROOT = Path(__file__).resolve().parents[2]
LAB3_PROJECT = REPO_ROOT / "lab3" / "project"
VAL_CSV = LAB3_PROJECT / "data" / "validation.csv"
PRED_MAIN_CSV = LAB3_PROJECT / "results" / "validation_predictions.csv"
PRED_ALT_CSV = LAB3_PROJECT / "results" / "validation_predictions_alternative.csv"
VAL_METRICS_JSON = LAB3_PROJECT / "results" / "validation_metrics.json"
OUTPUT_JSON = REPO_ROOT / "lab4" / "wyniki" / "s3_independent_cm.json"


def read_validation_labels(path: Path) -> Tuple[Dict[str, str], int, List[str]]:
    labels: Dict[str, str] = {}
    duplicates: List[str] = []
    n_rows = 0
    with open(path, "r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            n_rows += 1
            mid = row["message_id"]
            lbl = row["label"]
            if mid in labels:
                duplicates.append(mid)
            labels[mid] = lbl
    return labels, n_rows, duplicates


def read_predictions(path: Path) -> Tuple[Dict[str, Tuple[float, str]], int, List[str]]:
    preds: Dict[str, Tuple[float, str]] = {}
    duplicates: List[str] = []
    n_rows = 0
    with open(path, "r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            n_rows += 1
            mid = row["message_id"]
            sc = float(row["score"])
            lbl = row["predicted_label"]
            if mid in preds:
                duplicates.append(mid)
            preds[mid] = (sc, lbl)
    return preds, n_rows, duplicates


def independent_average_precision(
    y_true: Sequence[int], y_scores: Sequence[float]
) -> float:
    n_pos = sum(1 for y in y_true if y == 1)
    if n_pos == 0:
        return 0.0
    buckets: Dict[float, List[int]] = {}
    for yt, sc in zip(y_true, y_scores):
        buckets.setdefault(sc, []).append(yt)
    tp = fp = 0
    prev_rec = 0.0
    ap = 0.0
    for sc in sorted(buckets.keys(), reverse=True):
        grp = buckets[sc]
        tp += sum(1 for y in grp if y == 1)
        fp += sum(1 for y in grp if y == 0)
        rec = tp / n_pos
        prec = tp / (tp + fp)
        ap += (rec - prev_rec) * prec
        prev_rec = rec
    return ap


def evaluate_pair(
    val_labels: Dict[str, str],
    preds: Dict[str, Tuple[float, str]],
    pred_rows: int,
    pred_dups: List[str],
    recorded_metrics: Dict[str, object],
) -> Dict[str, object]:
    val_ids = set(val_labels.keys())
    pred_ids = set(preds.keys())
    missing_ids = sorted(val_ids - pred_ids)
    extra_ids = sorted(pred_ids - val_ids)
    joined_ids = sorted(val_ids & pred_ids)

    tp = fp = fn = tn = 0
    y_true_bin: List[int] = []
    y_scores: List[float] = []
    for mid in joined_ids:
        t_lbl = val_labels[mid]
        sc, p_lbl = preds[mid]
        yt = 1 if t_lbl == "spam" else 0
        yp = 1 if p_lbl == "spam" else 0
        y_true_bin.append(yt)
        y_scores.append(sc)
        if yt == 1 and yp == 1:
            tp += 1
        elif yt == 0 and yp == 1:
            fp += 1
        elif yt == 1 and yp == 0:
            fn += 1
        else:
            tn += 1

    n_total = tp + fp + fn + tn
    n_spam = tp + fn
    n_ham = fp + tn
    n_pred_spam = tp + fp
    fpr = fp / n_ham if n_ham else 0.0
    recall = tp / n_spam if n_spam else 0.0
    precision = tp / n_pred_spam if n_pred_spam else 0.0
    accuracy = (tp + tn) / n_total if n_total else 0.0
    pr_auc = independent_average_precision(y_true_bin, y_scores)

    rec_cm = recorded_metrics["confusion_matrix"]  # type: ignore[index]
    cm_match = (
        tp == rec_cm["tp"]
        and fp == rec_cm["fp"]
        and fn == rec_cm["fn"]
        and tn == rec_cm["tn"]
    )
    metrics_diffs = {
        "fpr_abs_diff": abs(fpr - float(recorded_metrics["fpr"])),
        "recall_abs_diff": abs(recall - float(recorded_metrics["recall"])),
        "precision_abs_diff": abs(precision - float(recorded_metrics["precision"])),
        "accuracy_abs_diff": abs(accuracy - float(recorded_metrics["accuracy"])),
        "pr_auc_abs_diff": abs(pr_auc - float(recorded_metrics["pr_auc"])),
    }
    floats_match = all(d < 1e-12 for d in metrics_diffs.values())

    ok = (
        len(joined_ids) == 60
        and pred_rows == 60
        and len(missing_ids) == 0
        and len(extra_ids) == 0
        and len(pred_dups) == 0
        and cm_match
        and floats_match
    )

    return {
        "pred_rows": pred_rows,
        "joined_rows": len(joined_ids),
        "missing_message_ids": missing_ids,
        "extra_message_ids": extra_ids,
        "duplicate_message_ids": pred_dups,
        "computed": {
            "confusion_matrix": {"tp": tp, "fp": fp, "fn": fn, "tn": tn},
            "n_total": n_total,
            "n_spam": n_spam,
            "n_ham": n_ham,
            "fpr": fpr,
            "recall": recall,
            "precision": precision,
            "accuracy": accuracy,
            "pr_auc": pr_auc,
        },
        "recorded_confusion_matrix": rec_cm,
        "cm_match": cm_match,
        "metrics_diffs": metrics_diffs,
        "floats_match_lt_1e_12": floats_match,
        "passed": ok,
    }


def main() -> None:
    val_labels, val_rows, val_dups = read_validation_labels(VAL_CSV)
    recorded = json.loads(VAL_METRICS_JSON.read_text(encoding="utf-8"))

    main_preds, main_rows, main_dups = read_predictions(PRED_MAIN_CSV)
    alt_preds, alt_rows, alt_dups = read_predictions(PRED_ALT_CSV)

    main_res = evaluate_pair(
        val_labels, main_preds, main_rows, main_dups, recorded["selected_model"]
    )
    alt_res = evaluate_pair(
        val_labels,
        alt_preds,
        alt_rows,
        alt_dups,
        recorded["baselines"]["alternative_model_informational"],
    )

    # Niezalezne sprawdzenie baseline'u 'zawsze ham'
    n_ham_val = sum(1 for v in val_labels.values() if v == "ham")
    n_spam_val = sum(1 for v in val_labels.values() if v == "spam")
    always_ham_expected_cm = {"tp": 0, "fp": 0, "fn": n_spam_val, "tn": n_ham_val}
    always_ham_match = (
        recorded["baselines"]["always_ham"]["confusion_matrix"] == always_ham_expected_cm
    )

    # Niezalezne sprawdzenie progow informacyjnych (0.25, 0.40, 0.50) dla modelu glownego
    info_thr_checks: Dict[str, Dict[str, object]] = {}
    all_info_ok = True
    for thr_str, rec_m in recorded["selected_model_informational_thresholds"].items():
        thr_val = float(thr_str)
        tp = fp = fn = tn = 0
        for mid, t_lbl in val_labels.items():
            sc, _ = main_preds[mid]
            yt = 1 if t_lbl == "spam" else 0
            yp = 1 if sc >= thr_val else 0
            if yt == 1 and yp == 1:
                tp += 1
            elif yt == 0 and yp == 1:
                fp += 1
            elif yt == 1 and yp == 0:
                fn += 1
            else:
                tn += 1
        cm = {"tp": tp, "fp": fp, "fn": fn, "tn": tn}
        ok = cm == rec_m["confusion_matrix"]
        all_info_ok = all_info_ok and ok
        info_thr_checks[thr_str] = {
            "computed_cm": cm,
            "recorded_cm": rec_m["confusion_matrix"],
            "match": ok,
        }

    passed = (
        val_rows == 60
        and len(val_dups) == 0
        and bool(main_res["passed"])
        and bool(alt_res["passed"])
        and always_ham_match
        and all_info_ok
    )

    payload: Dict[str, object] = {
        "check": "S3",
        "description": "Niezalezna macierz pomylek bez importow z src/ (laczenie po message_id dla modelu glownego i alternatywnego)",
        "imports_from_src": False,
        "validation_csv": {
            "rows": val_rows,
            "n_ham": n_ham_val,
            "n_spam": n_spam_val,
            "duplicate_message_ids": val_dups,
        },
        "selected_model": main_res,
        "alternative_model": alt_res,
        "always_ham_match": always_ham_match,
        "informational_thresholds_check": info_thr_checks,
        "passed": passed,
    }

    OUTPUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT_JSON.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )

    print(
        f"S3 zakonczone: passed={passed} | "
        f"glowny={main_res['computed']['confusion_matrix']} (joined={main_res['joined_rows']}, match={main_res['passed']}) | "
        f"alt={alt_res['computed']['confusion_matrix']} (joined={alt_res['joined_rows']}, match={alt_res['passed']})"
    )
    if not passed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
