"""S4: Spojnosc progu dla obu plikow przewidywan bez importow z src/.

Sprawdza dla validation_predictions.csv (vs spam_classifier.json) oraz
validation_predictions_alternative.csv (vs alternative_model.json), ile wierszy ma
predicted_label != ('spam' if score >= threshold else 'ham'). Oczekiwane: 0.
"""

from __future__ import annotations

import csv
import json
import math
from pathlib import Path
from typing import Dict, List


REPO_ROOT = Path(__file__).resolve().parents[2]
LAB3_PROJECT = REPO_ROOT / "lab3" / "project"
OUTPUT_JSON = REPO_ROOT / "lab4" / "wyniki" / "s4_threshold_consistency.json"


def check_pair(pred_csv: Path, model_json: Path) -> Dict[str, object]:
    model_data = json.loads(model_json.read_text(encoding="utf-8"))
    threshold = float(model_data["threshold"])

    mismatches: List[Dict[str, object]] = []
    non_finite_or_oob: List[Dict[str, object]] = []
    n_rows = 0
    min_margin = float("inf")
    closest_row: Dict[str, object] | None = None
    max_ham_score = -1.0
    min_spam_score = 2.0

    with open(pred_csv, "r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            n_rows += 1
            mid = row["message_id"]
            score = float(row["score"])
            recorded_lbl = row["predicted_label"]
            if not math.isfinite(score) or score < 0.0 or score > 1.0:
                non_finite_or_oob.append({"message_id": mid, "score": score})

            expected_lbl = "spam" if score >= threshold else "ham"
            if recorded_lbl != expected_lbl:
                mismatches.append(
                    {
                        "message_id": mid,
                        "score": score,
                        "threshold": threshold,
                        "recorded_label": recorded_lbl,
                        "expected_label": expected_lbl,
                    }
                )
            margin = abs(score - threshold)
            if margin < min_margin:
                min_margin = margin
                closest_row = {
                    "message_id": mid,
                    "score": score,
                    "predicted_label": recorded_lbl,
                    "abs_margin": margin,
                }
            if recorded_lbl == "ham" and score > max_ham_score:
                max_ham_score = score
            if recorded_lbl == "spam" and score < min_spam_score:
                min_spam_score = score

    return {
        "predictions_csv": pred_csv.relative_to(REPO_ROOT).as_posix(),
        "model_json": model_json.relative_to(REPO_ROOT).as_posix(),
        "threshold": threshold,
        "n_rows": n_rows,
        "n_mismatches": len(mismatches),
        "mismatches": mismatches,
        "n_non_finite_or_oob": len(non_finite_or_oob),
        "min_margin_to_threshold": min_margin,
        "closest_to_threshold": closest_row,
        "max_predicted_ham_score": max_ham_score,
        "min_predicted_spam_score": min_spam_score,
        "passed": len(mismatches) == 0 and len(non_finite_or_oob) == 0 and n_rows == 60,
    }


def main() -> None:
    main_res = check_pair(
        LAB3_PROJECT / "results" / "validation_predictions.csv",
        LAB3_PROJECT / "models" / "spam_classifier.json",
    )
    alt_res = check_pair(
        LAB3_PROJECT / "results" / "validation_predictions_alternative.csv",
        LAB3_PROJECT / "models" / "alternative_model.json",
    )
    passed = bool(main_res["passed"]) and bool(alt_res["passed"])

    payload: Dict[str, object] = {
        "check": "S4",
        "description": "Spojnosc predicted_label z (score >= threshold z JSON) dla obu modeli",
        "expected": {"n_mismatches_main": 0, "n_mismatches_alternative": 0},
        "selected_model": main_res,
        "alternative_model": alt_res,
        "passed": passed,
    }

    OUTPUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT_JSON.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )

    print(
        f"S4 zakonczone: passed={passed} | "
        f"glowny mismatches={main_res['n_mismatches']}/{main_res['n_rows']} | "
        f"alt mismatches={alt_res['n_mismatches']}/{alt_res['n_rows']}"
    )
    if not passed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
