"""Skrypt uczenia i zapisu koncowego modelu na zbiorze train.csv."""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.data_io import load_dataset, sha256_file
from src.models import KNNModel, SoftmaxRegressionModel, save_model


def train_and_save(
    train_csv: Path,
    images_dir: Path,
    output_model: Path,
    model_type: str,
    k: int = 1,
    metric: str = "l2",
    reg_lambda: float = 0.0,
    learning_rate: float = 1.0,
    max_epochs: int = 1000,
    rel_tol: float = 1e-6,
) -> tuple[KNNModel | SoftmaxRegressionModel, float]:
    t0 = time.perf_counter()
    X_train, y_train, _, _ = load_dataset(train_csv, images_dir)
    train_sha = sha256_file(train_csv)

    if model_type == "knn":
        model: KNNModel | SoftmaxRegressionModel = KNNModel(k=k, metric=metric)
    elif model_type == "softmax":
        model = SoftmaxRegressionModel(
            reg_lambda=reg_lambda,
            learning_rate=learning_rate,
            max_epochs=max_epochs,
            rel_tol=rel_tol,
        )
    else:
        raise ValueError(f"Nieobslugiwany typ modelu: {model_type}")

    model.fit(X_train, y_train)
    save_model(model, output_model, train_csv_sha256=train_sha)
    elapsed = time.perf_counter() - t0
    return model, elapsed


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Uczy wskazany model na train.csv i zapisuje go jako JSON."
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
        help="Katalog bazowy dla sciezek image_path (domyslnie: data).",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("model/model.json"),
        help="Sciezka wyjsciowego pliku modelu JSON (domyslnie: model/model.json).",
    )
    parser.add_argument(
        "--model-type",
        choices=["knn", "softmax"],
        required=True,
        help="Typ modelu do nauczenia: knn lub softmax.",
    )
    parser.add_argument("--k", type=int, default=1, help="Parametr k dla k-NN.")
    parser.add_argument(
        "--metric",
        choices=["l2", "l1"],
        default="l2",
        help="Metryka dla k-NN.",
    )
    parser.add_argument(
        "--reg-lambda",
        type=float,
        default=0.0,
        help="Parametr regularyzacji L2 dla softmax.",
    )
    parser.add_argument(
        "--learning-rate",
        type=float,
        default=1.0,
        help="Krok uczenia dla softmax.",
    )
    parser.add_argument(
        "--max-epochs",
        type=int,
        default=1000,
        help="Maksymalna liczba epok dla softmax.",
    )
    parser.add_argument(
        "--rel-tol",
        type=float,
        default=1e-6,
        help="Prog wzglednej zmiany straty dla softmax.",
    )
    args = parser.parse_args(argv)

    model, elapsed = train_and_save(
        train_csv=args.train_csv,
        images_dir=args.images_dir,
        output_model=args.output,
        model_type=args.model_type,
        k=args.k,
        metric=args.metric,
        reg_lambda=args.reg_lambda,
        learning_rate=args.learning_rate,
        max_epochs=args.max_epochs,
        rel_tol=args.rel_tol,
    )
    print(
        f"Zapisano model {model.model_type} do {args.output} "
        f"(czas: {elapsed:.4f} s)"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
