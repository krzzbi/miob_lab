"""Niezalezny skrypt bramki (czesc B i D8): weryfikuje SHA-256 modelu i plikow src/.

Bez importow z src/. Porownuje biezace hasze z wyniki/sha_przed_testem.json.
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


def collect_current(model_path: Path, src_dir: Path) -> dict[str, object]:
    if not model_path.is_file():
        raise FileNotFoundError(f"Brak pliku modelu: {model_path}")
    if not src_dir.is_dir():
        raise FileNotFoundError(f"Brak katalogu src: {src_dir}")

    src_files: dict[str, str] = {}
    for entry in sorted(src_dir.rglob("*")):
        if entry.is_file():
            if "__pycache__" in entry.parts or entry.suffix == ".pyc":
                raise RuntimeError(f"Wykryto plik bytecode w src/: {entry}")
            rel = entry.relative_to(src_dir.parent).as_posix()
            src_files[rel] = sha256_bytes(entry)

    return {
        "model": {
            "path": model_path.as_posix(),
            "sha256": sha256_bytes(model_path),
        },
        "src_file_count": len(src_files),
        "src_files": src_files,
    }


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Porownuje biezace SHA-256 modelu i src/ z plikiem zamrozenia."
    )
    parser.add_argument("--model", type=Path, default=Path("model/model.json"))
    parser.add_argument("--src-dir", type=Path, default=Path("src"))
    parser.add_argument(
        "--expected",
        type=Path,
        default=Path("wyniki/sha_przed_testem.json"),
    )
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()

    expected = json.loads(args.expected.read_text(encoding="utf-8"))
    current = collect_current(args.model, args.src_dir)
    match = current == expected

    report = {
        "all_match": match,
        "current": current,
        "expected_file": args.expected.as_posix(),
        "model_match": current["model"] == expected.get("model"),
        "src_files_match": current["src_files"] == expected.get("src_files"),
    }

    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(
            json.dumps(report, sort_keys=True, indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8",
            newline="\n",
        )

    if not match:
        print("BLAD BRAMKI: niezgodnosc SHA-256 z sha_przed_testem.json!")
        return 1

    print(
        f"OK BRAMKA: model ({current['model']['sha256'][:12]}...) oraz "
        f"{current['src_file_count']} plikow src/ identyczne z {args.expected}."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
