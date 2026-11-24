"""Sprawdzenie S4 na zbiorze testowym (czesc B Kroku 5).

Porownuje przewidywania modelu z pamieci (uczonego na train.csv jak w train.py)
z istniejacym wyniki/test_predictions.csv oraz weryfikuje tryb pojedynczej sciezki
predict.py w 3 osobnych procesach. Nie nadpisuje model/model.json ani test_predictions.csv.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import subprocess
import sys
import time
from pathlib import Path

LAB5_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(LAB5_DIR))

from src.data_io import load_dataset
from src.models import KNNModel


def run_single_predict(model_rel: Path, image_rel: str) -> tuple[int, float]:
    env = dict(os.environ)
    env["PYTHONDONTWRITEBYTECODE"] = "1"
    t0 = time.perf_counter()
    proc = subprocess.run(
        [
            sys.executable,
            "-B",
            "src/predict.py",
            "--model",
            str(model_rel),
            "--image",
            f"data/{image_rel}",
        ],
        cwd=str(LAB5_DIR),
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )
    dt = time.perf_counter() - t0
    if proc.returncode != 0:
        raise RuntimeError(
            f"Blad predict.py dla obrazu {image_rel} (rc={proc.returncode}): {proc.stderr}"
        )
    return int(proc.stdout.strip()), dt


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Sprawdzenie S4 na test.csv: model w pamieci vs test_predictions.csv i 3 pojedyncze obrazy."
    )
    parser.add_argument("--model", type=Path, default=Path("model/model.json"))
    parser.add_argument("--train-csv", type=Path, default=Path("data/train.csv"))
    parser.add_argument("--test-csv", type=Path, default=Path("data/test.csv"))
    parser.add_argument("--images-dir", type=Path, default=Path("data"))
    parser.add_argument(
        "--test-predictions",
        type=Path,
        default=Path("wyniki/test_predictions.csv"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("wyniki/s4_test.json"),
    )
    args = parser.parse_args()

    # Odczytaj ustawienia zamrozonego modelu (bez nadpisywania model/model.json)
    model_meta = json.loads((LAB5_DIR / args.model).read_text(encoding="utf-8"))
    if model_meta["model_type"] != "knn":
        raise ValueError(f"Oczekiwano modelu knn, otrzymano {model_meta['model_type']}.")
    k = int(model_meta["settings"]["k"])
    metric = str(model_meta["settings"]["metric"])

    # Ucz model w pamieci dokladnie tak jak w src/train.py
    X_train, y_train, _, _ = load_dataset(LAB5_DIR / args.train_csv, LAB5_DIR / args.images_dir)
    X_test, _, test_ids, test_paths = load_dataset(
        LAB5_DIR / args.test_csv, LAB5_DIR / args.images_dir
    )
    mem_model = KNNModel(k=k, metric=metric).fit(X_train, y_train)
    mem_preds = mem_model.predict(X_test)

    # Wczytaj istniejacy wyniki/test_predictions.csv
    with (LAB5_DIR / args.test_predictions).open("r", encoding="utf-8", newline="") as handle:
        pred_rows = list(csv.DictReader(handle))

    if len(pred_rows) != len(test_ids):
        raise ValueError(
            f"Nieoczekiwana liczba wierszy w {args.test_predictions}: {len(pred_rows)}"
        )

    ids_match = [r["image_id"] for r in pred_rows] == test_ids
    paths_match = [r["image_path"] for r in pred_rows] == test_paths
    file_preds = [int(r["predicted_label"]) for r in pred_rows]
    diffs_count = sum(1 for a, b in zip(mem_preds, file_preds) if a != b)

    # 3 pojedyncze obrazy testowe w osobnych procesach predict.py
    sample_indices = [0, 100, 199]
    single_checks = []
    for idx in sample_indices:
        single_pred, elapsed = run_single_predict(args.model, test_paths[idx])
        single_checks.append(
            {
                "elapsed_s": elapsed,
                "file_pred": file_preds[idx],
                "image_id": test_ids[idx],
                "image_path": test_paths[idx],
                "in_memory_pred": mem_preds[idx],
                "match": single_pred == file_preds[idx] == mem_preds[idx],
                "single_pred": single_pred,
            }
        )

    single_matches = sum(1 for item in single_checks if item["match"])
    all_passed = ids_match and paths_match and (diffs_count == 0) and (single_matches == 3)

    report = {
        "all_passed": all_passed,
        "batch_file": args.test_predictions.as_posix(),
        "batch_image_ids_match": ids_match,
        "batch_image_paths_match": paths_match,
        "batch_total": len(file_preds),
        "batch_vs_in_memory_diffs": diffs_count,
        "model_settings": {"k": k, "metric": metric, "model_type": "knn"},
        "single_image_checks": single_checks,
        "single_image_matches": single_matches,
        "single_image_total": len(single_checks),
    }

    out_path = LAB5_DIR / args.output
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(
        json.dumps(report, sort_keys=True, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
        newline="\n",
    )

    print(
        f"S4 (test): roznice wsadowy vs pamiec = {diffs_count}/{len(file_preds)}, "
        f"pojedyncze obrazy = {single_matches}/{len(single_checks)}, all_passed={all_passed}"
    )
    return 0 if all_passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
