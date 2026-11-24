"""Niezalezny skrypt S7: SHA-256 bajtow wszystkich plikow w lab5/data/.

Bez importow z src/, bez parsowania CSV i bez dekodowania PNG.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


def sha256_bytes(path: Path) -> str:
    hasher = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(65536), b""):
            hasher.update(chunk)
    return hasher.hexdigest()


def collect_data_hashes(data_dir: Path) -> dict[str, object]:
    if not data_dir.is_dir():
        raise FileNotFoundError(f"Katalog danych nie istnieje: {data_dir}")
    files: dict[str, str] = {}
    for entry in sorted(data_dir.rglob("*")):
        if entry.is_file():
            rel = entry.relative_to(data_dir).as_posix()
            files[rel] = sha256_bytes(entry)
    return {
        "file_count": len(files),
        "files": files,
    }


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Liczy SHA-256 wszystkich plikow w katalogu danych."
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=Path("data"),
        help="Sciezka do katalogu danych (domyslnie: data).",
    )
    parser.add_argument(
        "--output",
        type=Path,
        required=True,
        help="Sciezka wyjsciowego pliku JSON.",
    )
    parser.add_argument(
        "--compare",
        type=Path,
        default=None,
        help="Opcjonalny plik JSON do porownania (np. sha_data_przed.json).",
    )
    args = parser.parse_args()

    result = collect_data_hashes(args.data_dir)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(result, sort_keys=True, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    print(f"Zapisano SHA-256 dla {result['file_count']} plikow do {args.output}")

    if args.compare is not None:
        other = json.loads(args.compare.read_text(encoding="utf-8"))
        if other != result:
            print(f"BLAD: niezgodnosc haszy miedzy {args.compare} a {args.output}!")
            return 1
        print(f"OK: {args.compare} i {args.output} sa identyczne ({result['file_count']} plikow).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
