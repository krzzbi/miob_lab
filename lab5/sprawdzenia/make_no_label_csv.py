"""Tworzy kopie pliku CSV z samymi kolumnami image_id,image_path (bez label).

Bez importow z src/.
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Kopiuje CSV zachowujac wylacznie kolumny image_id i image_path."
    )
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    with args.input.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        fieldnames = reader.fieldnames or []
        if "image_id" not in fieldnames or "image_path" not in fieldnames:
            raise ValueError(f"Brak wymaganych kolumn w {args.input}.")
        rows = [(r["image_id"], r["image_path"]) for r in reader]

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle, lineterminator="\n")
        writer.writerow(["image_id", "image_path"])
        writer.writerows(rows)

    print(f"Zapisano {len(rows)} wierszy bez kolumny label do {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
