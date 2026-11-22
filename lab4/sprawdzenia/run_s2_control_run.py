"""S2: Przebieg kontrolny run_all.py na kopii danych w lab4/proby/przebieg/.

Kroki:
1. Kopiuje lab3/project/data/ do lab4/proby/przebieg/data/ i weryfikuje SHA-256 kopii z manifest.json.
2. Uruchamia lab3/project/src/run_all.py --project-dir lab4/proby/przebieg (z PYTHONDONTWRITEBYTECODE=1 i -B).
3. Porownuje SHA-256 wszystkich 9 wygenerowanych plikow z lab3/project/models/ i results/
   oraz oba wygenerowane modele z lab3/project/results/frozen_models_sha256.json.
4. Zapisuje wynik w lab4/wyniki/s2_control_run.json.
"""

from __future__ import annotations

import csv
import hashlib
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Dict, List


REPO_ROOT = Path(__file__).resolve().parents[2]
LAB3_PROJECT_DIR = REPO_ROOT / "lab3" / "project"
WORK_DIR = REPO_ROOT / "lab4" / "proby" / "przebieg"
OUTPUT_JSON = REPO_ROOT / "lab4" / "wyniki" / "s2_control_run.json"

EXPECTED_GENERATED_FILES: List[str] = [
    "models/spam_classifier.json",
    "models/alternative_model.json",
    "results/cv_results.json",
    "results/train_top_weights.json",
    "results/validation_predictions.csv",
    "results/validation_predictions_alternative.csv",
    "results/validation_metrics.json",
    "results/validation_errors.json",
    "results/diagnostics_section5.json",
]


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def find_first_json_diff(a: Any, b: Any, prefix: str = "") -> str | None:
    if type(a) is not type(b):
        return f"{prefix or '<root>'}: rozne typy ({type(a).__name__} vs {type(b).__name__})"
    if isinstance(a, dict):
        keys_a, keys_b = set(a.keys()), set(b.keys())
        if keys_a != keys_b:
            return (
                f"{prefix or '<root>'}: rozne klucze "
                f"(tylko_lab3={sorted(keys_a - keys_b)}, tylko_s2={sorted(keys_b - keys_a)})"
            )
        for k in sorted(a.keys()):
            d = find_first_json_diff(a[k], b[k], f"{prefix}.{k}" if prefix else str(k))
            if d is not None:
                return d
        return None
    if isinstance(a, list):
        if len(a) != len(b):
            return f"{prefix}: rozna dlugosc listy ({len(a)} vs {len(b)})"
        for i, (x, y) in enumerate(zip(a, b)):
            d = find_first_json_diff(x, y, f"{prefix}[{i}]")
            if d is not None:
                return d
        return None
    if a != b:
        return f"{prefix}: {a!r} != {b!r}"
    return None


def describe_file_diff(lab3_path: Path, s2_path: Path) -> str:
    if lab3_path.suffix == ".json":
        a = json.loads(lab3_path.read_text(encoding="utf-8"))
        b = json.loads(s2_path.read_text(encoding="utf-8"))
        d = find_first_json_diff(a, b)
        return d if d is not None else "Identyczna struktura JSON, roznica w formatowaniu/bialych znakach"
    if lab3_path.suffix == ".csv":
        with open(lab3_path, "r", encoding="utf-8", newline="") as fa, open(
            s2_path, "r", encoding="utf-8", newline=""
        ) as fb:
            ra = list(csv.reader(fa))
            rb = list(csv.reader(fb))
        if len(ra) != len(rb):
            return f"Rozna liczba wierszy CSV ({len(ra)} vs {len(rb)})"
        for idx, (row_a, row_b) in enumerate(zip(ra, rb)):
            if row_a != row_b:
                return f"Pierwsza roznica w wierszu {idx}: {row_a!r} != {row_b!r}"
        return "Identyczne wiersze CSV, roznica w koncach linii"
    return "Roznica binarna"


def main() -> None:
    if WORK_DIR.exists():
        shutil.rmtree(WORK_DIR)
    WORK_DIR.mkdir(parents=True, exist_ok=True)

    # 1. Kopiowanie data/ i weryfikacja SHA-256 kopii z manifest.json
    src_data = LAB3_PROJECT_DIR / "data"
    dst_data = WORK_DIR / "data"
    shutil.copytree(src_data, dst_data)

    manifest = json.loads((dst_data / "manifest.json").read_text(encoding="utf-8"))
    data_copy_check: Dict[str, Dict[str, object]] = {}
    all_data_ok = True
    for fname, spec in manifest["files"].items():
        orig_sha = sha256_file(src_data / fname)
        copy_sha = sha256_file(dst_data / fname)
        expected_sha = spec["sha256"]
        ok = (orig_sha == expected_sha) and (copy_sha == expected_sha)
        all_data_ok = all_data_ok and ok
        data_copy_check[fname] = {
            "expected_manifest_sha256": expected_sha,
            "lab3_sha256": orig_sha,
            "copy_sha256": copy_sha,
            "ok": ok,
        }

    # 2. Uruchomienie run_all.py w osobnym procesie z PYTHONDONTWRITEBYTECODE=1 i -B
    env = dict(os.environ)
    env["PYTHONDONTWRITEBYTECODE"] = "1"
    run_all_script = LAB3_PROJECT_DIR / "src" / "run_all.py"
    cmd = [
        sys.executable,
        "-B",
        str(run_all_script),
        "--project-dir",
        str(WORK_DIR),
    ]

    t0 = time.monotonic()
    proc = subprocess.run(
        cmd,
        cwd=str(LAB3_PROJECT_DIR),
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )
    elapsed_s = time.monotonic() - t0

    # 3. Porownanie SHA-256 wszystkich 9 wygenerowanych plikow
    comparisons: Dict[str, Dict[str, object]] = {}
    all_files_match = proc.returncode == 0
    for rel in EXPECTED_GENERATED_FILES:
        lab3_f = LAB3_PROJECT_DIR / rel
        s2_f = WORK_DIR / rel
        exists_lab3 = lab3_f.exists()
        exists_s2 = s2_f.exists()
        if exists_lab3 and exists_s2:
            sha_lab3 = sha256_file(lab3_f)
            sha_s2 = sha256_file(s2_f)
            match = sha_lab3 == sha_s2
            diff_detail = None if match else describe_file_diff(lab3_f, s2_f)
        else:
            sha_lab3 = sha256_file(lab3_f) if exists_lab3 else None
            sha_s2 = sha256_file(s2_f) if exists_s2 else None
            match = False
            diff_detail = f"Brak pliku (exists_lab3={exists_lab3}, exists_s2={exists_s2})"

        all_files_match = all_files_match and match
        comparisons[rel] = {
            "lab3_sha256": sha_lab3,
            "s2_sha256": sha_s2,
            "lab3_size_bytes": lab3_f.stat().st_size if exists_lab3 else None,
            "s2_size_bytes": s2_f.stat().st_size if exists_s2 else None,
            "identical": match,
            "diff_detail": diff_detail,
        }

    # Sprawdzenie, czy run_all.py nie utworzyl nieoczekiwanych plikow w models/ lub results/
    actual_generated = sorted(
        p.relative_to(WORK_DIR).as_posix()
        for sub in ("models", "results")
        if (WORK_DIR / sub).exists()
        for p in (WORK_DIR / sub).rglob("*")
        if p.is_file()
    )
    unexpected_files = sorted(set(actual_generated) - set(EXPECTED_GENERATED_FILES))

    # 4. Porownanie modeli z frozen_models_sha256.json
    frozen_data = json.loads(
        (LAB3_PROJECT_DIR / "results" / "frozen_models_sha256.json").read_text(
            encoding="utf-8"
        )
    )
    frozen_check: Dict[str, Dict[str, object]] = {}
    all_frozen_match = True
    for fname, frozen_sha in frozen_data["models"].items():
        rel = f"models/{fname}"
        s2_sha = comparisons[rel]["s2_sha256"]
        lab3_sha = comparisons[rel]["lab3_sha256"]
        ok = (s2_sha == frozen_sha) and (lab3_sha == frozen_sha)
        all_frozen_match = all_frozen_match and ok
        frozen_check[fname] = {
            "frozen_sha256": frozen_sha,
            "lab3_sha256": lab3_sha,
            "s2_sha256": s2_sha,
            "identical": ok,
        }

    passed = (
        all_data_ok
        and proc.returncode == 0
        and all_files_match
        and all_frozen_match
        and len(unexpected_files) == 0
    )

    payload: Dict[str, object] = {
        "check": "S2",
        "description": "Pelny przebieg kontrolny run_all.py w lab4/proby/przebieg i porownanie SHA-256 9 plikow oraz frozen_models_sha256.json",
        "expected": {
            "data_copy_matches_manifest": True,
            "returncode": 0,
            "n_generated_files_identical": len(EXPECTED_GENERATED_FILES),
            "frozen_models_identical": True,
            "unexpected_files": [],
        },
        "command": cmd,
        "cwd": "lab3/project",
        "returncode": proc.returncode,
        "wall_seconds": round(elapsed_s, 4),
        "data_copy_check": data_copy_check,
        "file_comparisons": comparisons,
        "n_identical": sum(1 for v in comparisons.values() if v["identical"]),
        "n_total_checked": len(EXPECTED_GENERATED_FILES),
        "unexpected_files": unexpected_files,
        "frozen_models_check": frozen_check,
        "passed": passed,
        "stdout": proc.stdout,
        "stderr": proc.stderr,
    }

    OUTPUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT_JSON.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )

    print(
        f"S2 zakonczone: passed={passed} | returncode={proc.returncode} | "
        f"czas={elapsed_s:.3f}s | zgodne pliki={payload['n_identical']}/{payload['n_total_checked']} | "
        f"frozen_ok={all_frozen_match} | data_ok={all_data_ok}"
    )
    if not passed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
