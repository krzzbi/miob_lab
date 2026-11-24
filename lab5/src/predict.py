"""Skrypt inferencji: przewidywanie dla pojedynczego PNG lub wsadowo z CSV bez etykiet.

Nie wczytuje zadnego pliku z etykietami i nie wykonuje ponownego uczenia.
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.data_io import load_unlabeled_csv
from src.models import load_model
from src.png_decoder import extract_features


def predict_single(model_path: Path, image_path: Path) -> int:
    model = load_model(model_path)
    features = extract_features(image_path)
    return model.predict_one(features)


def predict_batch(
    model_path: Path,
    input_csv: Path,
    images_dir: Path,
    output_csv: Path,
) -> int:
    model = load_model(model_path)
    rows = load_unlabeled_csv(input_csv)

    output_csv.parent.mkdir(parents=True, exist_ok=True)
    with output_csv.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle, lineterminator="\n")
        writer.writerow(["image_id", "image_path", "predicted_label"])
        for row in rows:
            rel_path = row["image_path"]
            features = extract_features(images_dir / rel_path)
            pred = model.predict_one(features)
            writer.writerow([row["image_id"], rel_path, pred])
    return len(rows)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Przewiduje cyfre 0-9 z zapisanego modelu dla pojedynczego PNG "
            "lub pliku CSV bez kolumny label."
        )
    )
    parser.add_argument(
        "--model",
        type=Path,
        required=True,
        help="Sciezka do zapisanego pliku modelu JSON.",
    )
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument(
        "--image",
        type=Path,
        help="Sciezka do pojedynczego obrazu PNG 8x8.",
    )
    mode.add_argument(
        "--input-csv",
        type=Path,
        help="Sciezka do pliku CSV z kolumnami image_id,image_path (bez label).",
    )
    parser.add_argument(
        "--images-dir",
        type=Path,
        default=None,
        help="Katalog bazowy dla sciezek image_path w trybie wsadowym (wymagany z --input-csv).",
    )
    parser.add_argument(
        "--output-csv",
        type=Path,
        default=None,
        help="Sciezka wyjsciowego CSV w trybie wsadowym (wymagany z --input-csv).",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    if args.image is not None:
        pred = predict_single(args.model, args.image)
        print(pred)
        return 0

    if args.images_dir is None or args.output_csv is None:
        parser.error("Tryb --input-csv wymaga podania --images-dir oraz --output-csv.")

    count = predict_batch(
        model_path=args.model,
        input_csv=args.input_csv,
        images_dir=args.images_dir,
        output_csv=args.output_csv,
    )
    print(f"Zapisano {count} przewidywan do {args.output_csv}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
