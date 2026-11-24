"""Niezalezny skrypt S5: macierz pomylek bez importow z src/."""

from __future__ import annotations

import argparse
import csv
import json
from collections import Counter
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Niezaleznie liczy macierz pomylek 10x10 i porownuje z evaluate.py."
    )
    parser.add_argument("--predictions", type=Path, required=True)
    parser.add_argument("--labels", type=Path, required=True)
    parser.add_argument("--eval-json", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    with args.predictions.open("r", encoding="utf-8", newline="") as handle:
        pred_map = {
            row["image_id"]: int(row["predicted_label"])
            for row in csv.DictReader(handle)
        }

    with args.labels.open("r", encoding="utf-8", newline="") as handle:
        label_rows = list(csv.DictReader(handle))

    if set(pred_map.keys()) != {r["image_id"] for r in label_rows}:
        raise ValueError("Niezgodnosc zbiorow image_id w S5.")

    matrix = [[0] * 10 for _ in range(10)]
    hits = 0
    errors = 0
    pairs: Counter[tuple[int, int]] = Counter()
    bad_ids: list[str] = []

    for row in label_rows:
        img_id = row["image_id"]
        y_true = int(row["label"])
        y_pred = pred_map[img_id]
        matrix[y_true][y_pred] += 1
        if y_true == y_pred:
            hits += 1
        else:
            errors += 1
            pairs[(y_true, y_pred)] += 1
            bad_ids.append(img_id)

    eval_report = json.loads(args.eval_json.read_text(encoding="utf-8"))
    eval_bad_ids = [item["image_id"] for item in eval_report["misclassified"]]

    matches = {
        "confusion_matrix_equal": matrix == eval_report["confusion_matrix"],
        "errors_equal": errors == eval_report["errors"],
        "hits_equal": hits == eval_report["hits"],
        "misclassified_ids_equal": sorted(bad_ids) == sorted(eval_bad_ids),
        "total_equal": len(label_rows) == eval_report["total"],
    }
    all_ok = all(matches.values())

    report = {
        "all_checks_passed": all_ok,
        "comparisons_with_evaluate_py": matches,
        "independent_result": {
            "confusion_matrix": matrix,
            "errors": errors,
            "hits": hits,
            "misclassified_image_ids": bad_ids,
            "most_confused_pairs": [
                {"count": c, "predicted_label": p, "true_label": t}
                for (t, p), c in sorted(
                    pairs.items(), key=lambda kv: (-kv[1], kv[0][0], kv[0][1])
                )
            ],
            "total": len(label_rows),
        },
    }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(report, sort_keys=True, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
        newline="\n",
    )

    if not all_ok:
        print(f"BLAD S5: niezgodnosc z evaluate.py: {matches}")
        return 1
    print(f"OK S5: niezalezna macierz 10x10 identyczna z evaluate.py ({hits}/{len(label_rows)}).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
