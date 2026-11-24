"""Punkt odniesienia na tescie: najblizszy centroid L2 uczony wylacznie na train.csv.

Wynik sluzy wylacznie jako punkt odniesienia (nie do wyboru modelu).
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from collections import Counter
from pathlib import Path

LAB5_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(LAB5_DIR))

from src.data_io import load_dataset
from src.models import NearestCentroidModel

Z_95 = 1.959963984540054


def wilson_ci_95(hits: int, total: int) -> dict[str, float]:
    p = hits / total
    z2 = Z_95 * Z_95
    denom = 1.0 + z2 / total
    center = (p + z2 / (2.0 * total)) / denom
    half = (Z_95 * math.sqrt((p * (1.0 - p) + z2 / (4.0 * total)) / total)) / denom
    return {
        "center": center,
        "lower": max(0.0, center - half),
        "upper": min(1.0, center + half),
        "z": Z_95,
    }


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Ocenia punkt odniesienia (najblizszy centroid L2) na test.csv."
    )
    parser.add_argument("--train-csv", type=Path, default=Path("data/train.csv"))
    parser.add_argument("--test-csv", type=Path, default=Path("data/test.csv"))
    parser.add_argument("--images-dir", type=Path, default=Path("data"))
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("wyniki/test_odniesienie_centroid.json"),
    )
    args = parser.parse_args()

    X_train, y_train, _, _ = load_dataset(args.train_csv, args.images_dir)
    X_test, y_test, test_ids, test_paths = load_dataset(args.test_csv, args.images_dir)

    t0 = time.perf_counter()
    model = NearestCentroidModel().fit(X_train, y_train)
    t_fit = time.perf_counter() - t0

    t0 = time.perf_counter()
    preds = model.predict(X_test)
    t_pred = time.perf_counter() - t0

    matrix = [[0] * 10 for _ in range(10)]
    pair_counts: Counter[tuple[int, int]] = Counter()
    errors = []
    hits = 0
    for img_id, img_path, y_true, y_pred in zip(test_ids, test_paths, y_test, preds):
        matrix[y_true][y_pred] += 1
        if y_true == y_pred:
            hits += 1
        else:
            pair_counts[(y_true, y_pred)] += 1
            errors.append(
                {
                    "image_id": img_id,
                    "image_path": img_path,
                    "predicted_label": y_pred,
                    "true_label": y_true,
                }
            )

    total = len(y_test)
    report = {
        "accuracy": hits / total,
        "confusion_matrix": matrix,
        "errors": total - hits,
        "hits": hits,
        "misclassified_count": len(errors),
        "model_type": "nearest_centroid",
        "most_confused_pairs": [
            {"count": c, "predicted_label": p, "true_label": t}
            for (t, p), c in sorted(
                pair_counts.items(), key=lambda kv: (-kv[1], kv[0][0], kv[0][1])
            )
        ],
        "predict_time_s": t_pred,
        "settings": {"metric": "l2"},
        "total": total,
        "train_time_s": t_fit,
        "wilson_95": wilson_ci_95(hits, total),
    }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(report, sort_keys=True, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    w = report["wilson_95"]
    print(
        f"Centroid (test): {hits}/{total} ({report['accuracy']*100:.2f}%), "
        f"Wilson 95%: [{w['lower']*100:.2f}%, {w['upper']*100:.2f}%]"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
