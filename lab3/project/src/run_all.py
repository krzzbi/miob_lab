"""Pelny, powtarzalny przebieg prototypu: dane -> CV/uczenie -> predykcja (bez label) -> ocena -> diagnostyka."""

from __future__ import annotations

import argparse
import csv
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Sequence

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.cv_train import run_cv_and_train
from src.data_io import load_csv, verify_data_manifest
from src.diagnostics import run_diagnostics
from src.evaluate import evaluate_validation


def write_unlabeled_csv(source_csv: str | Path, dest_csv: str | Path) -> None:
    """Tworzy kopie CSV zawierajaca wylacznie kolumny message_id i text (bez label)."""
    mids, texts, _ = load_csv(source_csv, include_label=False)
    with open(dest_csv, "w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f, lineterminator="\n")
        writer.writerow(["message_id", "text"])
        for mid, txt in zip(mids, texts):
            writer.writerow([mid, txt])


def run_predict_subprocess(
    *,
    predict_script: Path,
    model_json: Path,
    unlabeled_csv: Path,
    output_csv: Path,
) -> None:
    """Uruchamia predict.py w osobnym procesie Pythona bez stanu uczacego w pamieci."""
    cmd = [
        sys.executable,
        str(predict_script),
        "--model",
        str(model_json),
        "--input-csv",
        str(unlabeled_csv),
        "--output-csv",
        str(output_csv),
    ]
    proc = subprocess.run(cmd, capture_output=True, text=True, check=False)
    if proc.returncode != 0:
        raise RuntimeError(
            f"Blad procesu predict.py (kod {proc.returncode}):\n{proc.stderr}\n{proc.stdout}"
        )


def run_full_pipeline(project_dir: str | Path, *, skip_train: bool = False) -> None:
    proj = Path(project_dir).resolve()
    data_dir = proj / "data"
    predict_script = Path(__file__).resolve().parent / "predict.py"

    # 1. Weryfikacja integralnosci danych przed startem
    verify_data_manifest(data_dir)

    # 2. Czesc 3: CV na train.csv, wybor modelu i uczenie koncowe
    if not skip_train:
        run_cv_and_train(proj, apply_masking=True, save_outputs=True)

    # 3. Czesc 4: Przewidywanie w osobnym procesie na kopii validation.csv BEZ kolumny label
    with tempfile.TemporaryDirectory() as tmpdir:
        unlabeled_val = Path(tmpdir) / "validation_unlabeled.csv"
        write_unlabeled_csv(data_dir / "validation.csv", unlabeled_val)

        run_predict_subprocess(
            predict_script=predict_script,
            model_json=proj / "models" / "spam_classifier.json",
            unlabeled_csv=unlabeled_val,
            output_csv=proj / "results" / "validation_predictions.csv",
        )
        run_predict_subprocess(
            predict_script=predict_script,
            model_json=proj / "models" / "alternative_model.json",
            unlabeled_csv=unlabeled_val,
            output_csv=proj / "results" / "validation_predictions_alternative.csv",
        )

    # 4. Czesc 4: Jednorazowa ocena po message_id + punkty odniesienia
    evaluate_validation(
        validation_csv=data_dir / "validation.csv",
        predictions_csv=proj / "results" / "validation_predictions.csv",
        model_json=proj / "models" / "spam_classifier.json",
        alternative_predictions_csv=proj / "results" / "validation_predictions_alternative.csv",
        alternative_model_json=proj / "models" / "alternative_model.json",
        output_metrics_json=proj / "results" / "validation_metrics.json",
        output_errors_json=proj / "results" / "validation_errors.json",
    )

    # 5. Czesc 5: Diagnostyka po ocenie (sekcja 5 planu)
    run_diagnostics(proj)

    # 6. Koncowa weryfikacja nienaruszalnosci project/data/
    verify_data_manifest(data_dir)


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description="Uruchamia pelny przebieg prototypu rozpoznawania spamu."
    )
    parser.add_argument(
        "--project-dir",
        default=str(Path(__file__).resolve().parents[1]),
        help="Sciezka do folderu project/.",
    )
    parser.add_argument(
        "--skip-train",
        action="store_true",
        help="Uzyj istniejacych zamrozonych modeli z models/ bez ponownego uczenia.",
    )
    args = parser.parse_args(argv)
    run_full_pipeline(args.project_dir, skip_train=args.skip_train)
    print("Pelny przebieg zakonczony pomyslnie; dane wejsciowe i artefakty zweryfikowane.")


if __name__ == "__main__":
    main()
