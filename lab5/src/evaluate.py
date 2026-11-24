"""Ocena przewidywan poprzez laczenie z CSV etykiet po kolumnie image_id."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.data_io import load_labeled_csv


def load_predictions_csv(predictions_path: Path) -> list[dict[str, object]]:
    if not predictions_path.is_file():
        raise FileNotFoundError(f"Plik przewidywan nie istnieje: {predictions_path}")

    with predictions_path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        fieldnames = reader.fieldnames or []
        required = {"image_id", "image_path", "predicted_label"}
        missing = required - set(fieldnames)
        if missing:
            raise ValueError(
                f"Plik {predictions_path} nie ma wymaganych kolumn: {sorted(missing)}."
            )
        rows: list[dict[str, object]] = []
        for row in reader:
            pred = int(row["predicted_label"])
            if not (0 <= pred <= 9):
                raise ValueError(f"Nieprawidlowa przewidywana etykieta: {pred}")
            rows.append(
                {
                    "image_id": row["image_id"],
                    "image_path": row["image_path"],
                    "predicted_label": pred,
                }
            )
    return rows


def evaluate_predictions(
    predictions_path: Path, labels_path: Path
) -> dict[str, Any]:
    preds = load_predictions_csv(predictions_path)
    truths = load_labeled_csv(labels_path)

    pred_by_id: dict[str, dict[str, object]] = {}
    for row in preds:
        img_id = str(row["image_id"])
        if img_id in pred_by_id:
            raise ValueError(f"Zduplikowany image_id w przewidywaniach: {img_id}")
        pred_by_id[img_id] = row

    truth_by_id: dict[str, dict[str, object]] = {}
    for row in truths:
        img_id = str(row["image_id"])
        if img_id in truth_by_id:
            raise ValueError(f"Zduplikowany image_id w etykietach: {img_id}")
        truth_by_id[img_id] = row

    pred_ids = set(pred_by_id)
    truth_ids = set(truth_by_id)
    if pred_ids != truth_ids:
        missing_in_preds = sorted(truth_ids - pred_ids)[:5]
        extra_in_preds = sorted(pred_ids - truth_ids)[:5]
        raise ValueError(
            "Niezgodnosc zbiorow image_id miedzy przewidywaniami a etykietami: "
            f"brakujace={missing_in_preds}, nadmiarowe={extra_in_preds}."
        )

    matrix = [[0] * 10 for _ in range(10)]
    pair_counts: Counter[tuple[int, int]] = Counter()
    errors: list[dict[str, object]] = []
    hits = 0

    for row in truths:
        img_id = str(row["image_id"])
        true_lbl = int(row["label"])
        pred_row = pred_by_id[img_id]
        pred_lbl = int(pred_row["predicted_label"])
        matrix[true_lbl][pred_lbl] += 1
        if pred_lbl == true_lbl:
            hits += 1
        else:
            pair_counts[(true_lbl, pred_lbl)] += 1
            errors.append(
                {
                    "image_id": img_id,
                    "image_path": str(row["image_path"]),
                    "predicted_label": pred_lbl,
                    "true_label": true_lbl,
                }
            )

    total = len(truths)
    error_count = total - hits
    most_confused = [
        {
            "count": count,
            "predicted_label": pred_lbl,
            "true_label": true_lbl,
        }
        for (true_lbl, pred_lbl), count in sorted(
            pair_counts.items(), key=lambda kv: (-kv[1], kv[0][0], kv[0][1])
        )
    ]

    return {
        "accuracy": hits / total if total else 0.0,
        "confusion_matrix": matrix,
        "errors": error_count,
        "hits": hits,
        "misclassified": errors,
        "most_confused_pairs": most_confused,
        "total": total,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Ocenia plik przewidywan wzgledem pliku CSV z etykietami."
    )
    parser.add_argument(
        "--predictions",
        type=Path,
        required=True,
        help="Sciezka do pliku CSV z przewidywaniami.",
    )
    parser.add_argument(
        "--labels",
        type=Path,
        required=True,
        help="Sciezka do pliku CSV z etykietami.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Opcjonalna sciezka wyjsciowego pliku JSON z ocena.",
    )
    args = parser.parse_args(argv)

    report = evaluate_predictions(args.predictions, args.labels)
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(
            json.dumps(report, sort_keys=True, indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8",
            newline="\n",
        )

    print(
        f"Trafienia: {report['hits']}/{report['total']} "
        f"({report['accuracy'] * 100:.2f}%), pomylki: {report['errors']}"
    )
    if report["most_confused_pairs"]:
        top_str = ", ".join(
            f"{p['true_label']}->{p['predicted_label']} ({p['count']})"
            for p in report["most_confused_pairs"][:10]
        )
        print(f"Najczesciej mylone pary: {top_str}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
