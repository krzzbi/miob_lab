"""Liczenie i porownywanie sum SHA-256 wszystkich plikow w katalogu lab3/.

Uzycie:
  PYTHONDONTWRITEBYTECODE=1 /Users/jasminetea/.pyenv/shims/python3 -B \
    lab4/sprawdzenia/check_lab3_sha.py --output lab4/wyniki/sha_przed.json

  PYTHONDONTWRITEBYTECODE=1 /Users/jasminetea/.pyenv/shims/python3 -B \
    lab4/sprawdzenia/check_lab3_sha.py --compare lab4/wyniki/sha_przed.json
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
from typing import Dict, List, Tuple


REPO_ROOT = Path(__file__).resolve().parents[2]
LAB3_DIR = REPO_ROOT / "lab3"


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def snapshot_lab3(lab3_dir: Path) -> Dict[str, object]:
    files: Dict[str, Dict[str, object]] = {}
    pycache_files: Dict[str, Dict[str, object]] = {}

    for root, dirnames, filenames in os.walk(lab3_dir):
        dirnames.sort()
        filenames.sort()
        root_path = Path(root)
        for fname in filenames:
            if fname == ".DS_Store":
                continue
            fpath = root_path / fname
            rel = fpath.relative_to(lab3_dir).as_posix()
            entry = {
                "sha256": sha256_file(fpath),
                "size_bytes": fpath.stat().st_size,
            }
            if "__pycache__" in fpath.relative_to(lab3_dir).parts:
                pycache_files[rel] = entry
            else:
                files[rel] = entry

    files_sorted = dict(sorted(files.items()))
    pycache_sorted = dict(sorted(pycache_files.items()))

    return {
        "root": "lab3",
        "ignored": [".DS_Store"],
        "summary": {
            "n_files": len(files_sorted),
            "n_pycache_files": len(pycache_sorted),
            "n_total": len(files_sorted) + len(pycache_sorted),
        },
        "files": files_sorted,
        "pycache_file_list": list(pycache_sorted.keys()),
        "pycache_files": pycache_sorted,
    }


def compare_snapshots(
    before: Dict[str, object], after: Dict[str, object]
) -> Dict[str, object]:
    def diff_section(key: str) -> Dict[str, object]:
        b_map: Dict[str, Dict[str, object]] = before[key]  # type: ignore[assignment]
        a_map: Dict[str, Dict[str, object]] = after[key]  # type: ignore[assignment]
        b_keys = set(b_map.keys())
        a_keys = set(a_map.keys())
        added = sorted(a_keys - b_keys)
        removed = sorted(b_keys - a_keys)
        changed = sorted(
            k
            for k in (b_keys & a_keys)
            if b_map[k]["sha256"] != a_map[k]["sha256"]
        )
        return {
            "identical": not (added or removed or changed),
            "added": added,
            "removed": removed,
            "changed": changed,
        }

    files_diff = diff_section("files")
    pycache_diff = diff_section("pycache_files")
    return {
        "identical": bool(files_diff["identical"] and pycache_diff["identical"]),
        "files": files_diff,
        "pycache_files": pycache_diff,
    }


def main(argv: List[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description="Migawka i porownanie SHA-256 calego katalogu lab3/."
    )
    parser.add_argument(
        "--lab3-dir",
        default=str(LAB3_DIR),
        help="Sciezka do katalogu lab3/.",
    )
    parser.add_argument(
        "--output",
        default=None,
        help="Sciezka zapisu migawki JSON.",
    )
    parser.add_argument(
        "--compare",
        default=None,
        help="Sciezka do bazowej migawki JSON (np. sha_przed.json) do porownania.",
    )
    args = parser.parse_args(argv)

    snap = snapshot_lab3(Path(args.lab3_dir).resolve())

    if args.output:
        out_path = Path(args.output)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(
            json.dumps(snap, indent=2, sort_keys=True, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )
        print(
            f"Zapisano migawke SHA-256 do {args.output} "
            f"(pliki={snap['summary']['n_files']}, __pycache__={snap['summary']['n_pycache_files']})"
        )

    if args.compare:
        before = json.loads(Path(args.compare).read_text(encoding="utf-8"))
        diff = compare_snapshots(before, snap)
        print(json.dumps(diff, indent=2, ensure_ascii=False))
        if not diff["identical"]:
            raise SystemExit(1)


if __name__ == "__main__":
    main()
