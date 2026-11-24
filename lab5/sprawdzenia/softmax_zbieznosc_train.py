"""Sprawdzenie (czesc A): zbieznosc straty regresji softmax wylacznie na train.csv.

Zapisuje wyniki dla lr in {0.5, 1.0, 1.5, 2.0} i lambda in {0, 1e-4, 1e-3, 1e-2, 1e-1}
do pliku JSON (domyslnie wyniki/softmax_zbieznosc_train.json).
Nie uzywa validation.csv ani test.csv.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any

LAB5_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(LAB5_DIR))

from src.data_io import load_dataset
from src.models import SoftmaxRegressionModel

LEARNING_RATES = [0.5, 1.0, 1.5, 2.0]
LAMBDAS = [0.0, 1e-4, 1e-3, 1e-2, 1e-1]
MAX_EPOCHS = 1500
REL_TOL = 1e-5


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Bada zbieznosc straty regresji softmax wylacznie na train.csv."
    )
    parser.add_argument(
        "--train-csv",
        type=Path,
        default=Path("data/train.csv"),
        help="Sciezka do train.csv (domyslnie: data/train.csv).",
    )
    parser.add_argument(
        "--images-dir",
        type=Path,
        default=Path("data"),
        help="Katalog bazowy obrazow (domyslnie: data).",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("wyniki/softmax_zbieznosc_train.json"),
        help="Sciezka wyjsciowego pliku JSON.",
    )
    args = parser.parse_args()

    X_train, y_train, _, _ = load_dataset(args.train_csv, args.images_dir)
    n_train = len(y_train)
    runs: list[dict[str, Any]] = []

    for lr in LEARNING_RATES:
        print(f"\n=== lr={lr} (max_epochs={MAX_EPOCHS}, rel_tol={REL_TOL}) ===")
        for lam in LAMBDAS:
            t0 = time.perf_counter()
            model = SoftmaxRegressionModel(
                reg_lambda=lam,
                learning_rate=lr,
                max_epochs=MAX_EPOCHS,
                rel_tol=REL_TOL,
            )
            model.fit(X_train, y_train)
            elapsed = time.perf_counter() - t0
            preds = model.predict(X_train)
            train_hits = sum(1 for p, y in zip(preds, y_train) if p == y)
            info = model.training_info
            record = {
                "elapsed_s": elapsed,
                "epochs_run": int(info["epochs_run"]),
                "final_grad_norm": float(info["final_grad_norm"]),
                "final_loss": float(info["final_loss"]),
                "initial_loss": float(info["initial_loss"]),
                "learning_rate": lr,
                "max_epochs": MAX_EPOCHS,
                "reg_lambda": lam,
                "rel_tol": REL_TOL,
                "stopped_by": str(info["stopped_by"]),
                "train_hits": train_hits,
                "train_total": n_train,
            }
            runs.append(record)
            print(
                f"  lam={lam:<6g} epochs={record['epochs_run']:4d} "
                f"stopped={record['stopped_by']:<10s} "
                f"loss={record['final_loss']:.6f} "
                f"grad_norm={record['final_grad_norm']:.6f} "
                f"train_hits={train_hits}/{n_train} ({elapsed:.2f}s)"
            )

    payload = {
        "lambdas": LAMBDAS,
        "learning_rates": LEARNING_RATES,
        "max_epochs": MAX_EPOCHS,
        "rel_tol": REL_TOL,
        "runs": runs,
        "train_samples": n_train,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(payload, sort_keys=True, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    print(f"\nZapisano {len(runs)} wynikow zbieznosci do {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
