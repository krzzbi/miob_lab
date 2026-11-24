"""Wczytywanie plikow CSV, obrazow i warstwowy podzial CV."""

from __future__ import annotations

import csv
import hashlib
import random
from pathlib import Path

from src.png_decoder import extract_features


def sha256_file(path: str | Path) -> str:
    """Oblicza SHA-256 pojedynczego pliku."""
    hasher = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(65536), b""):
            hasher.update(chunk)
    return hasher.hexdigest()


def load_labeled_csv(csv_path: str | Path) -> list[dict[str, object]]:
    """Wczytuje plik CSV z kolumnami image_id, image_path, label."""
    path = Path(csv_path)
    if not path.is_file():
        raise FileNotFoundError(f"Plik CSV nie istnieje: {path}")

    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        fieldnames = reader.fieldnames or []
        required = {"image_id", "image_path", "label"}
        missing = required - set(fieldnames)
        if missing:
            raise ValueError(
                f"Plik {path} nie ma wymaganych kolumn: {sorted(missing)}."
            )

        rows: list[dict[str, object]] = []
        for idx, row in enumerate(reader, start=2):
            label_int = int(row["label"])
            if not (0 <= label_int <= 9):
                raise ValueError(
                    f"Nieprawidlowa etykieta w wierszu {idx} pliku {path}: {label_int}."
                )
            rows.append(
                {
                    "image_id": row["image_id"],
                    "image_path": row["image_path"],
                    "label": label_int,
                }
            )
    return rows


def load_unlabeled_csv(csv_path: str | Path) -> list[dict[str, str]]:
    """Wczytuje plik CSV dla predict.py (wymaga image_id, image_path; odrzuca label)."""
    path = Path(csv_path)
    if not path.is_file():
        raise FileNotFoundError(f"Plik CSV nie istnieje: {path}")

    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        fieldnames = reader.fieldnames or []

        if any(col.strip().lower() == "label" for col in fieldnames):
            raise ValueError(
                f"Plik wejsciowy {path} zawiera kolumne 'label' — "
                "predict.py nie wczytuje plikow z etykietami."
            )

        required = {"image_id", "image_path"}
        missing = required - set(fieldnames)
        if missing:
            raise ValueError(
                f"Plik {path} nie ma wymaganych kolumn: {sorted(missing)}."
            )

        rows: list[dict[str, str]] = []
        for row in reader:
            rows.append(
                {
                    "image_id": row["image_id"],
                    "image_path": row["image_path"],
                }
            )
    return rows


def load_dataset(
    csv_path: str | Path, images_dir: str | Path
) -> tuple[list[list[float]], list[int], list[str], list[str]]:
    """Wczytuje cechy (64 floaty), etykiety, image_id i image_path dla pliku CSV."""
    rows = load_labeled_csv(csv_path)
    base_dir = Path(images_dir)
    features: list[list[float]] = []
    labels: list[int] = []
    image_ids: list[str] = []
    image_paths: list[str] = []

    for row in rows:
        rel_path = str(row["image_path"])
        features.append(extract_features(base_dir / rel_path))
        labels.append(int(row["label"]))
        image_ids.append(str(row["image_id"]))
        image_paths.append(rel_path)

    return features, labels, image_ids, image_paths


def make_stratified_folds(
    labels: list[int], n_folds: int = 5, seed: int = 20261114
) -> list[list[int]]:
    """Tworzy deterministyczne, warstwowe po klasach foldy walidacji krzyzowej.

    Indeksy wewnatrz kazdego foldu sa posortowane rosnaco, aby zachowac kolejnosc train.csv.
    """
    if n_folds < 2:
        raise ValueError("Liczba foldow musi wynosic co najmniej 2.")

    by_class: dict[int, list[int]] = {}
    for idx, label in enumerate(labels):
        by_class.setdefault(label, []).append(idx)

    rng = random.Random(seed)
    folds: list[list[int]] = [[] for _ in range(n_folds)]

    for label in sorted(by_class):
        indices = list(by_class[label])
        rng.shuffle(indices)
        for pos, sample_idx in enumerate(indices):
            folds[pos % n_folds].append(sample_idx)

    return [sorted(fold) for fold in folds]
