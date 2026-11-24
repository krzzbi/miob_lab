"""Niezalezny skrypt S8: zamrozenie SHA-256 modelu i wszystkich plikow src/."""

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


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Zapisuje SHA-256 pliku modelu oraz wszystkich plikow w src/."
    )
    parser.add_argument("--model", type=Path, default=Path("model/model.json"))
    parser.add_argument("--src-dir", type=Path, default=Path("src"))
    parser.add_argument("--output", type=Path, default=Path("wyniki/sha_przed_testem.json"))
    args = parser.parse_args()

    if not args.model.is_file():
        raise FileNotFoundError(f"Brak pliku modelu: {args.model}")
    if not args.src_dir.is_dir():
        raise FileNotFoundError(f"Brak katalogu src: {args.src_dir}")

    src_files: dict[str, str] = {}
    for entry in sorted(args.src_dir.rglob("*")):
        if entry.is_file():
            if "__pycache__" in entry.parts or entry.suffix == ".pyc":
                raise RuntimeError(f"Wykryto plik bytecode w src/: {entry}")
            rel = entry.relative_to(args.src_dir.parent).as_posix()
            src_files[rel] = sha256_bytes(entry)

    payload = {
        "model": {
            "path": args.model.as_posix(),
            "sha256": sha256_bytes(args.model),
        },
        "src_file_count": len(src_files),
        "src_files": src_files,
    }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(payload, sort_keys=True, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    print(
        f"S8 zamrozenie: zapisano SHA-256 modelu ({payload['model']['sha256'][:12]}...) "
        f"oraz {len(src_files)} plikow src/ do {args.output}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
