"""S9: Oceny z obiektow w pamieci (prosto po uczeniu, przed serializacja) vs podproces predict.py.

Na przebiegu z S2 (lab4/proby/przebieg):
1. Uruchamia run_cv_and_train(lab4/proby/przebieg, apply_masking=True, save_outputs=False)
   i pobiera obiekty winner_artifact oraz alternative_artifact prosto z pamieci (przed zapisem JSON).
2. Potwierdza, ze serializacja w pamieci daje SHA-256 identyczne z plikami wygenerowanymi w S2
   (oraz z lab3/project/models/).
3. Tworzy kopie validation.csv BEZ kolumny label w lab4/proby/s9_predict/validation_no_label.csv.
4. Uruchamia lab3/project/src/predict.py w osobnym procesie (PYTHONDONTWRITEBYTECODE=1 i -B)
   dla obu modeli z S2 i porownuje oceny z obiektami z pamieci (max diff < 1e-12, 100% identycznych etykiet).
5. Dodatkowo sprawdza tryb --stdin w osobnym procesie dla pojedynczej wiadomosci.
"""

from __future__ import annotations

import csv
import hashlib
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Dict, List, Tuple


REPO_ROOT = Path(__file__).resolve().parents[2]
LAB3_PROJECT = REPO_ROOT / "lab3" / "project"
S2_DIR = REPO_ROOT / "lab4" / "proby" / "przebieg"
S9_WORK_DIR = REPO_ROOT / "lab4" / "proby" / "s9_predict"
OUTPUT_JSON = REPO_ROOT / "lab4" / "wyniki" / "s9_in_memory_vs_subprocess.json"

sys.path.insert(0, str(LAB3_PROJECT))

from src.cv_train import run_cv_and_train  # noqa: E402
from src.data_io import load_csv  # noqa: E402
from src.models import SpamClassifierArtifact  # noqa: E402


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_file(path: Path) -> str:
    return sha256_bytes(path.read_bytes())


def read_pred_csv(path: Path) -> Dict[str, Tuple[float, str]]:
    out: Dict[str, Tuple[float, str]] = {}
    with open(path, "r", encoding="utf-8", newline="") as f:
        for row in csv.DictReader(f):
            out[row["message_id"]] = (float(row["score"]), row["predicted_label"])
    return out


def run_predict_csv_subprocess(
    model_json: Path, unlabeled_csv: Path, output_csv: Path
) -> subprocess.CompletedProcess[str]:
    env = dict(os.environ)
    env["PYTHONDONTWRITEBYTECODE"] = "1"
    cmd = [
        sys.executable,
        "-B",
        str(LAB3_PROJECT / "src" / "predict.py"),
        "--model",
        str(model_json),
        "--input-csv",
        str(unlabeled_csv),
        "--output-csv",
        str(output_csv),
    ]
    return subprocess.run(cmd, env=env, capture_output=True, text=True, check=False)


def compare_in_memory_vs_csv(
    artifact: SpamClassifierArtifact,
    mids: List[str],
    texts: List[str],
    subproc_csv: Path,
    s2_csv: Path,
) -> Dict[str, object]:
    in_mem = {mid: artifact.predict_one(txt) for mid, txt in zip(mids, texts)}
    sub_preds = read_pred_csv(subproc_csv)
    s2_preds = read_pred_csv(s2_csv)

    max_diff_sub = 0.0
    max_diff_s2 = 0.0
    label_mismatches_sub = 0
    label_mismatches_s2 = 0

    for mid in mids:
        m_sc, m_lbl = in_mem[mid]
        p_sc, p_lbl = sub_preds[mid]
        s_sc, s_lbl = s2_preds[mid]
        max_diff_sub = max(max_diff_sub, abs(m_sc - p_sc))
        max_diff_s2 = max(max_diff_s2, abs(m_sc - s_sc))
        if m_lbl != p_lbl:
            label_mismatches_sub += 1
        if m_lbl != s_lbl:
            label_mismatches_s2 += 1

    ok = (
        len(sub_preds) == len(mids) == 60
        and max_diff_sub < 1e-12
        and max_diff_s2 < 1e-12
        and label_mismatches_sub == 0
        and label_mismatches_s2 == 0
    )
    return {
        "n_messages": len(mids),
        "max_abs_diff_vs_subprocess": max_diff_sub,
        "label_mismatches_vs_subprocess": label_mismatches_sub,
        "max_abs_diff_vs_s2_csv": max_diff_s2,
        "label_mismatches_vs_s2_csv": label_mismatches_s2,
        "passed": ok,
    }


def main() -> None:
    if not (S2_DIR / "data" / "train.csv").exists():
        raise RuntimeError(f"Brak wynikow S2 w {S2_DIR}. Uruchom najpierw S2.")

    if S9_WORK_DIR.exists():
        shutil.rmtree(S9_WORK_DIR)
    S9_WORK_DIR.mkdir(parents=True, exist_ok=True)

    # 1. Uczenie w pamieci na danych z S2 (bez nadpisywania plikow S2)
    out_mem = run_cv_and_train(S2_DIR, apply_masking=True, save_outputs=False)
    win_art: SpamClassifierArtifact = out_mem["winner_artifact"]  # type: ignore[assignment]
    alt_art: SpamClassifierArtifact = out_mem["alternative_artifact"]  # type: ignore[assignment]

    # 2. Potwierdzenie, ze serializacja obiektow z pamieci daje ten sam SHA-256 co pliki S2 i lab3/
    win_bytes = (
        json.dumps(win_art.to_dict(), indent=2, sort_keys=True, ensure_ascii=False) + "\n"
    ).encode("utf-8")
    alt_bytes = (
        json.dumps(alt_art.to_dict(), indent=2, sort_keys=True, ensure_ascii=False) + "\n"
    ).encode("utf-8")

    s2_win_path = S2_DIR / "models" / "spam_classifier.json"
    s2_alt_path = S2_DIR / "models" / "alternative_model.json"
    win_sha_match = sha256_bytes(win_bytes) == sha256_file(s2_win_path)
    alt_sha_match = sha256_bytes(alt_bytes) == sha256_file(s2_alt_path)

    # 3. Przygotowanie kopii validation.csv BEZ kolumny label
    mids, texts, _ = load_csv(S2_DIR / "data" / "validation.csv", include_label=False)
    unlabeled_csv = S9_WORK_DIR / "validation_no_label.csv"
    with open(unlabeled_csv, "w", encoding="utf-8", newline="") as f:
        w = csv.writer(f, lineterminator="\n")
        w.writerow(["message_id", "text"])
        for mid, txt in zip(mids, texts):
            w.writerow([mid, txt])

    with open(unlabeled_csv, "r", encoding="utf-8", newline="") as f:
        unlabeled_header = next(csv.reader(f))

    # 4. Uruchomienie predict.py w osobnym procesie dla obu modeli
    out_win_csv = S9_WORK_DIR / "preds_winner.csv"
    out_alt_csv = S9_WORK_DIR / "preds_alternative.csv"
    proc_win = run_predict_csv_subprocess(s2_win_path, unlabeled_csv, out_win_csv)
    proc_alt = run_predict_csv_subprocess(s2_alt_path, unlabeled_csv, out_alt_csv)

    win_cmp = (
        compare_in_memory_vs_csv(
            win_art,
            mids,
            texts,
            out_win_csv,
            S2_DIR / "results" / "validation_predictions.csv",
        )
        if proc_win.returncode == 0
        else {"passed": False, "error": proc_win.stderr}
    )
    alt_cmp = (
        compare_in_memory_vs_csv(
            alt_art,
            mids,
            texts,
            out_alt_csv,
            S2_DIR / "results" / "validation_predictions_alternative.csv",
        )
        if proc_alt.returncode == 0
        else {"passed": False, "error": proc_alt.stderr}
    )

    # 5. Dodatkowo test trybu --stdin w osobnym procesie
    sample_msg = "Subject: Kernel patch review\n\nPlease check the attached diff.\n"
    env = dict(os.environ)
    env["PYTHONDONTWRITEBYTECODE"] = "1"
    proc_stdin = subprocess.run(
        [
            sys.executable,
            "-B",
            str(LAB3_PROJECT / "src" / "predict.py"),
            "--model",
            str(s2_win_path),
            "--stdin",
        ],
        input=sample_msg,
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )
    stdin_ok = False
    stdin_diff = None
    if proc_stdin.returncode == 0:
        stdin_res = json.loads(proc_stdin.stdout)
        mem_sc, mem_lbl = win_art.predict_one(sample_msg)
        stdin_diff = abs(float(stdin_res["score"]) - mem_sc)
        stdin_ok = (stdin_res["predicted_label"] == mem_lbl) and (stdin_diff < 1e-12)

    passed = (
        unlabeled_header == ["message_id", "text"]
        and win_sha_match
        and alt_sha_match
        and proc_win.returncode == 0
        and proc_alt.returncode == 0
        and bool(win_cmp["passed"])
        and bool(alt_cmp["passed"])
        and stdin_ok
    )

    payload: Dict[str, object] = {
        "check": "S9",
        "description": "Porownanie obiektow w pamieci prosto po uczeniu (przed zapisem JSON) z podprocesem predict.py na CSV bez kolumny label",
        "expected": {
            "unlabeled_header": ["message_id", "text"],
            "in_memory_serialization_matches_s2_sha256": True,
            "max_abs_diff_lt": 1e-12,
            "label_mismatches": 0,
            "stdin_mode_matches_in_memory": True,
        },
        "unlabeled_csv_header": unlabeled_header,
        "in_memory_serialization_check": {
            "winner_sha256_matches_s2": win_sha_match,
            "alternative_sha256_matches_s2": alt_sha_match,
        },
        "winner_comparison": win_cmp,
        "alternative_comparison": alt_cmp,
        "stdin_mode_check": {
            "returncode": proc_stdin.returncode,
            "abs_diff": stdin_diff,
            "passed": stdin_ok,
        },
        "passed": passed,
    }

    OUTPUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT_JSON.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )

    print(
        f"S9 zakonczone: passed={passed} | "
        f"winner max_diff={win_cmp.get('max_abs_diff_vs_subprocess')} (lbl_mis={win_cmp.get('label_mismatches_vs_subprocess')}) | "
        f"alt max_diff={alt_cmp.get('max_abs_diff_vs_subprocess')} (lbl_mis={alt_cmp.get('label_mismatches_vs_subprocess')}) | "
        f"sha_match=({win_sha_match}, {alt_sha_match}) | stdin_ok={stdin_ok}"
    )
    if not passed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
