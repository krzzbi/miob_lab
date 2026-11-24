"""Sprawdzenia S3 (odtwarzalnosc bajtowa) i S4 (predict.py w osobnym procesie + odrzucenie label)."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import subprocess
import sys
import time
from pathlib import Path

LAB5_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(LAB5_DIR))

from src.data_io import load_dataset
from src.models import KNNModel, SoftmaxRegressionModel


def sha256_bytes(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def run_python(cmd_args: list[str]) -> tuple[subprocess.CompletedProcess[str], float]:
    env = dict(os.environ)
    env["PYTHONDONTWRITEBYTECODE"] = "1"
    t0 = time.perf_counter()
    proc = subprocess.run(
        [sys.executable, "-B", *cmd_args],
        cwd=str(LAB5_DIR),
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )
    dt = time.perf_counter() - t0
    return proc, dt


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Uruchamia sprawdzenia S3 oraz S4 (w tym B2 odrzucenie kolumny label)."
    )
    parser.add_argument("--experiment-json", type=Path, default=Path("wyniki/eksperymenty_validation.json"))
    parser.add_argument("--model-out", type=Path, default=Path("model/model.json"))
    parser.add_argument("--repro-model-out", type=Path, default=Path("proby/model_repro_check.json"))
    parser.add_argument("--val-nolabel-csv", type=Path, default=Path("proby/validation_no_label.csv"))
    parser.add_argument("--val-preds-csv", type=Path, default=Path("wyniki/validation_predictions.csv"))
    parser.add_argument("--output", type=Path, default=Path("wyniki/s3_s4_sprawdzenia.json"))
    args = parser.parse_args()

    exp = json.loads((LAB5_DIR / args.experiment_json).read_text(encoding="utf-8"))
    sel = exp["selection"]
    model_type = str(sel["selected_model_type"])
    settings = sel["selected_settings"]

    train_flags: list[str] = ["--model-type", model_type]
    if model_type == "knn":
        train_flags += ["--k", str(settings["k"]), "--metric", str(settings["metric"])]
    elif model_type == "softmax":
        train_flags += [
            "--reg-lambda", str(settings["reg_lambda"]),
            "--learning-rate", str(settings["learning_rate"]),
            "--max-epochs", str(settings["max_epochs"]),
            "--rel-tol", str(settings["rel_tol"]),
        ]
    else:
        raise ValueError(f"Nieznany typ modelu: {model_type}")

    # S3: dwa niezalezne uczenia w osobnych procesach
    p1, t_train_1 = run_python(["src/train.py", *train_flags, "--output", str(args.model_out)])
    if p1.returncode != 0:
        raise RuntimeError(f"Blad pierwszego uczenia S3: {p1.stderr}")

    p2, t_train_2 = run_python(["src/train.py", *train_flags, "--output", str(args.repro_model_out)])
    if p2.returncode != 0:
        raise RuntimeError(f"Blad drugiego uczenia S3: {p2.stderr}")

    m1_path = LAB5_DIR / args.model_out
    m2_path = LAB5_DIR / args.repro_model_out
    b1 = m1_path.read_bytes()
    b2 = m2_path.read_bytes()
    s3_identical = b1 == b2
    s3_sha1 = sha256_bytes(m1_path)
    s3_sha2 = sha256_bytes(m2_path)

    # Przygotuj przewidywania w pamieci (przed zapisem) dla porownania w S4
    X_train, y_train, _, _ = load_dataset(LAB5_DIR / "data/train.csv", LAB5_DIR / "data")
    X_val, _, val_ids, val_paths = load_dataset(LAB5_DIR / "data/validation.csv", LAB5_DIR / "data")
    if model_type == "knn":
        mem_model: KNNModel | SoftmaxRegressionModel = KNNModel(
            k=int(settings["k"]), metric=str(settings["metric"])
        ).fit(X_train, y_train)
    else:
        mem_model = SoftmaxRegressionModel(
            reg_lambda=float(settings["reg_lambda"]),
            learning_rate=float(settings["learning_rate"]),
            max_epochs=int(settings["max_epochs"]),
            rel_tol=float(settings["rel_tol"]),
        ).fit(X_train, y_train)
    in_mem_preds = mem_model.predict(X_val)

    # Utworz kopie validation bez kolumny label w lab5/proby/
    nolabel_path = LAB5_DIR / args.val_nolabel_csv
    nolabel_path.parent.mkdir(parents=True, exist_ok=True)
    with nolabel_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle, lineterminator="\n")
        writer.writerow(["image_id", "image_path"])
        for img_id, img_path in zip(val_ids, val_paths):
            writer.writerow([img_id, img_path])

    # S4a: predict.py wsadowo w osobnym procesie
    p_batch, t_batch = run_python([
        "src/predict.py",
        "--model", str(args.model_out),
        "--input-csv", str(args.val_nolabel_csv),
        "--images-dir", "data",
        "--output-csv", str(args.val_preds_csv),
    ])
    if p_batch.returncode != 0:
        raise RuntimeError(f"Blad predict.py (wsadowy S4): {p_batch.stderr}")

    preds_path = LAB5_DIR / args.val_preds_csv
    with preds_path.open("r", encoding="utf-8", newline="") as handle:
        batch_rows = list(csv.DictReader(handle))

    batch_preds = [int(r["predicted_label"]) for r in batch_rows]
    batch_diffs = sum(1 for a, b in zip(in_mem_preds, batch_preds) if a != b)
    ids_match = [r["image_id"] for r in batch_rows] == val_ids
    paths_match = [r["image_path"] for r in batch_rows] == val_paths

    # S4b: predict.py dla 3 pojedynczych obrazow w osobnych procesach
    sample_indices = [0, 100, 199]
    single_checks = []
    for idx in sample_indices:
        rel_img = val_paths[idx]
        p_single, t_single = run_python([
            "src/predict.py",
            "--model", str(args.model_out),
            "--image", f"data/{rel_img}",
        ])
        if p_single.returncode != 0:
            raise RuntimeError(f"Blad predict.py (pojedynczy obraz {rel_img}): {p_single.stderr}")
        single_pred = int(p_single.stdout.strip())
        single_checks.append({
            "batch_pred": batch_preds[idx],
            "elapsed_s": t_single,
            "image_id": val_ids[idx],
            "image_path": rel_img,
            "match": single_pred == batch_preds[idx],
            "single_pred": single_pred,
        })
    single_all_match = all(item["match"] for item in single_checks)

    # S4c (B2): odrzucenie CSV z kolumna label
    p_bad, t_bad = run_python([
        "src/predict.py",
        "--model", str(args.model_out),
        "--input-csv", "data/validation.csv",
        "--images-dir", "data",
        "--output-csv", "proby/should_not_exist.csv",
    ])
    bad_file_exists = (LAB5_DIR / "proby/should_not_exist.csv").exists()
    label_rejected = (p_bad.returncode != 0) and ("label" in (p_bad.stderr + p_bad.stdout).lower()) and (not bad_file_exists)

    all_passed = s3_identical and (batch_diffs == 0) and ids_match and paths_match and single_all_match and label_rejected

    report = {
        "all_passed": all_passed,
        "s3_reproducibility": {
            "byte_identical": s3_identical,
            "model_bytes": len(b1),
            "sha256_run1": s3_sha1,
            "sha256_run2": s3_sha2,
            "train_time_run1_s": t_train_1,
            "train_time_run2_s": t_train_2,
        },
        "s4_inference": {
            "batch_elapsed_s": t_batch,
            "batch_image_ids_preserved": ids_match,
            "batch_image_paths_preserved": paths_match,
            "batch_vs_in_memory_diffs": batch_diffs,
            "batch_total": len(batch_preds),
            "label_column_rejected": label_rejected,
            "label_rejection_returncode": p_bad.returncode,
            "label_rejection_stderr": p_bad.stderr.strip(),
            "single_image_checks": single_checks,
            "single_vs_batch_all_match": single_all_match,
        },
    }

    out_path = LAB5_DIR / args.output
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(
        json.dumps(report, sort_keys=True, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
        newline="\n",
    )

    print(f"S3 (odtwarzalnosc bajtowa): identical={s3_identical}, sha256={s3_sha1}")
    print(f"S4a (wsadowy vs pamiec): roznice={batch_diffs}/{len(batch_preds)}, czas={t_batch:.4f}s")
    print(f"S4b (3 pojedyncze vs wsadowy): all_match={single_all_match}")
    print(f"S4c (odrzucenie kolumny label): rejected={label_rejected} (rc={p_bad.returncode})")

    return 0 if all_passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
