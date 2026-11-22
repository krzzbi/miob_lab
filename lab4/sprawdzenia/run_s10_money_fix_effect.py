"""S10: Skutek probnej poprawki U1 (_MONEY_RE) w izolowanej kopii lab4/proby/v2/.

NIE zmienia niczego w lab3/.
Kroki:
1. Kopiuje lab3/project/src/ i data/ do lab4/proby/v2/.
2. W lab4/proby/v2/src/preprocess.py poprawia wylacznie _MONEY_RE (usuniecie blednego \\b po [$£€]).
3. Sprawdza z gory zapisane testy syntetyczne (4 sufiksalne z S8 -> <MONEY>, 8 dotychczasowych -> bez zmian).
4. Liczy, w ilu wiadomosciach train.csv i validation.csv zmienila sie lista tokenow (oczekiwane: 1 w train, 0 w validation).
5. Uruchamia pelny lab4/proby/v2/src/run_all.py --project-dir lab4/proby/v2 (PYTHONDONTWRITEBYTECODE=1, -B)
   i porownuje z wynikami Labu 3.
6. Zapisuje wynik w lab4/wyniki/s10_money_fix_effect.json.
"""

from __future__ import annotations

import csv
import importlib.util
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path
from typing import Dict, List, Tuple


REPO_ROOT = Path(__file__).resolve().parents[2]
LAB3_PROJECT = REPO_ROOT / "lab3" / "project"
V2_DIR = REPO_ROOT / "lab4" / "proby" / "v2"
OUTPUT_JSON = REPO_ROOT / "lab4" / "wyniki" / "s10_money_fix_effect.json"

OLD_MONEY_RE_BLOCK = (
    "_MONEY_RE = re.compile(\n"
    '    r"(?:(?:US\\$|USD|EUR|GBP|[\\$£€])\\s*\\d[\\d,]*(?:\\.\\d+)?(?:\\s*(?:million|billion|thousand))?"\n'
    '    r"|\\b\\d[\\d,]*(?:\\.\\d+)?\\s*(?:[\\$£€]|USD|EUR|GBP|dollars?|cents?)\\b)",\n'
    "    flags=re.IGNORECASE,\n"
    ")"
)

NEW_MONEY_RE_BLOCK = (
    "_MONEY_RE = re.compile(\n"
    '    r"(?:(?:US\\$|USD|EUR|GBP|[\\$£€])\\s*\\d[\\d,]*(?:\\.\\d+)?(?:\\s*(?:million|billion|thousand))?"\n'
    '    r"|\\b\\d[\\d,]*(?:\\.\\d+)?\\s*(?:[\\$£€]|(?:USD|EUR|GBP|dollars?|cents?)\\b))",\n'
    "    flags=re.IGNORECASE,\n"
    ")"
)

# Oczekiwane wyniki syntetyczne zapisane PRZED uruchomieniem:
EXPECTED_SUFFIX_CASES = [
    {"input": "Subject: Offer 100$ now", "expected_contains": "<MONEY>"},
    {"input": "Subject: Win 100.000$ today", "expected_contains": "<MONEY>"},
    {"input": "Subject: Pay 50 € now", "expected_contains": "<MONEY>"},
    {"input": "Subject: Cost 25£ only", "expected_contains": "<MONEY>"},
]

EXPECTED_EXISTING_CASES = [
    {"input": "Subject: Price $100 today", "expected_contains": "<MONEY>"},
    {"input": "Subject: Only $14.95 now", "expected_contains": "<MONEY>"},
    {"input": "Subject: Pay US$ 100 immediately", "expected_contains": "<MONEY>"},
    {"input": "Subject: Earn 100 dollars fast", "expected_contains": "<MONEY>"},
    {"input": "Subject: Send USD 50 here", "expected_contains": "<MONEY>"},
    {"input": "Subject: Total 20 USD due", "expected_contains": "<MONEY>"},
    {"input": "Subject: Fee £50 applied", "expected_contains": "<MONEY>"},
    {"input": "Subject: Only 50 cents each", "expected_contains": "<MONEY>"},
    {"input": "Subject: Cost EUR 20 net", "expected_contains": "<MONEY>"},
    {"input": "Subject: Win $5 million now", "expected_contains": "<MONEY>"},
]


def load_module_from_path(mod_name: str, path: Path):
    spec = importlib.util.spec_from_file_location(mod_name, str(path))
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def read_csv_rows(path: Path) -> List[Tuple[str, str, str]]:
    rows: List[Tuple[str, str, str]] = []
    with open(path, "r", encoding="utf-8", newline="") as f:
        for r in csv.DictReader(f):
            rows.append((r["message_id"], r["text"], r["label"]))
    return rows


def read_pred_csv(path: Path) -> Dict[str, Tuple[float, str]]:
    out: Dict[str, Tuple[float, str]] = {}
    with open(path, "r", encoding="utf-8", newline="") as f:
        for r in csv.DictReader(f):
            out[r["message_id"]] = (float(r["score"]), r["predicted_label"])
    return out


def main() -> None:
    if V2_DIR.exists():
        shutil.rmtree(V2_DIR)
    V2_DIR.mkdir(parents=True, exist_ok=True)

    # 1. Kopia src/ (bez __pycache__) i data/ do lab4/proby/v2/
    shutil.copytree(
        LAB3_PROJECT / "src",
        V2_DIR / "src",
        ignore=shutil.ignore_patterns("__pycache__", "*.pyc"),
    )
    shutil.copytree(LAB3_PROJECT / "data", V2_DIR / "data")

    # 2. Poprawka wylacznie _MONEY_RE w kopii lab4/proby/v2/src/preprocess.py
    v2_prep_path = V2_DIR / "src" / "preprocess.py"
    src_text = v2_prep_path.read_text(encoding="utf-8")
    if OLD_MONEY_RE_BLOCK not in src_text:
        raise RuntimeError("Nie znaleziono oczekiwanego bloku _MONEY_RE w kopii preprocess.py")
    patched_text = src_text.replace(OLD_MONEY_RE_BLOCK, NEW_MONEY_RE_BLOCK, 1)
    v2_prep_path.write_text(patched_text, encoding="utf-8")

    # Wczytanie oryginalnego i poprawionego modulu preprocess (bez zapisu .pyc)
    orig_prep = load_module_from_path("orig_preprocess", LAB3_PROJECT / "src" / "preprocess.py")
    v2_prep = load_module_from_path("v2_preprocess", v2_prep_path)

    # 3. Testy syntetyczne
    suffix_results = []
    for case in EXPECTED_SUFFIX_CASES:
        orig_toks = orig_prep.tokenize_text(case["input"], apply_masking=True)
        v2_toks = v2_prep.tokenize_text(case["input"], apply_masking=True)
        ok = case["expected_contains"] in v2_toks
        suffix_results.append(
            {
                "input": case["input"],
                "expected_contains": case["expected_contains"],
                "orig_tokens": orig_toks,
                "v2_tokens": v2_toks,
                "passed": ok,
            }
        )

    existing_results = []
    for case in EXPECTED_EXISTING_CASES:
        orig_toks = orig_prep.tokenize_text(case["input"], apply_masking=True)
        v2_toks = v2_prep.tokenize_text(case["input"], apply_masking=True)
        ok = (case["expected_contains"] in v2_toks) and (v2_toks == orig_toks)
        existing_results.append(
            {
                "input": case["input"],
                "expected_contains": case["expected_contains"],
                "orig_tokens": orig_toks,
                "v2_tokens": v2_toks,
                "identical_to_orig": v2_toks == orig_toks,
                "passed": ok,
            }
        )

    synthetic_ok = all(r["passed"] for r in suffix_results) and all(
        r["passed"] for r in existing_results
    )

    # 4. Zliczenie zmienionych list tokenow w train.csv i validation.csv
    token_changes: Dict[str, Dict[str, object]] = {}
    for split in ("train", "validation"):
        rows = read_csv_rows(V2_DIR / "data" / f"{split}.csv")
        changed_msgs = []
        for mid, txt, lbl in rows:
            t_orig = orig_prep.tokenize_text(txt, apply_masking=True)
            t_v2 = v2_prep.tokenize_text(txt, apply_masking=True)
            if t_orig != t_v2:
                changed_msgs.append(
                    {
                        "message_id": mid,
                        "label": lbl,
                        "orig_money_count": t_orig.count("<MONEY>"),
                        "v2_money_count": t_v2.count("<MONEY>"),
                        "orig_num_count": t_orig.count("<NUM>"),
                        "v2_num_count": t_v2.count("<NUM>"),
                        "orig_len": len(t_orig),
                        "v2_len": len(t_v2),
                    }
                )
        token_changes[split] = {
            "n_total_messages": len(rows),
            "n_changed_messages": len(changed_msgs),
            "changed_messages": changed_msgs,
        }

    # 5. Pelny przebieg run_all.py z kodu v2 w osobnym procesie
    env = dict(os.environ)
    env["PYTHONDONTWRITEBYTECODE"] = "1"
    cmd = [
        sys.executable,
        "-B",
        str(V2_DIR / "src" / "run_all.py"),
        "--project-dir",
        str(V2_DIR),
    ]
    t0 = time.monotonic()
    proc = subprocess.run(
        cmd,
        cwd=str(V2_DIR),
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )
    elapsed_s = time.monotonic() - t0
    if proc.returncode != 0:
        raise RuntimeError(f"Blad run_all.py w v2:\n{proc.stderr}\n{proc.stdout}")

    # 6. Porownanie wynikow v2 z Labem 3
    lab3_cv = json.loads((LAB3_PROJECT / "results" / "cv_results.json").read_text(encoding="utf-8"))
    v2_cv = json.loads((V2_DIR / "results" / "cv_results.json").read_text(encoding="utf-8"))

    lab3_val_m = json.loads((LAB3_PROJECT / "results" / "validation_metrics.json").read_text(encoding="utf-8"))
    v2_val_m = json.loads((V2_DIR / "results" / "validation_metrics.json").read_text(encoding="utf-8"))

    def extract_winner_cv(cv_json: Dict[str, object]) -> Dict[str, object]:
        dec = cv_json["decision"]  # type: ignore[index]
        mtype = dec["winner_model_type"]
        param = dec["winner_param"]
        cands = (
            cv_json["logistic_regression_candidates"]  # type: ignore[index]
            if mtype == "logistic_regression"
            else cv_json["multinomial_nb_candidates"]  # type: ignore[index]
        )
        wc = next(c for c in cands if {c["param_name"]: c["param_value"]} == param)
        oof = wc["oof_metrics"]
        return {
            "model_type": mtype,
            "param": param,
            "threshold": float(dec["winner_threshold"]),
            "full_vocab_size": int(cv_json["train_summary"]["full_vocab_size"]),  # type: ignore[index]
            "cv_cm": oof["confusion_matrix"],
            "cv_pr_auc": float(oof["pr_auc"]),
        }

    c_lab3 = extract_winner_cv(lab3_cv)
    c_v2 = extract_winner_cv(v2_cv)

    # Zmiana oceny OOF dla samej wiadomosci 52c16dcd1605d5e8
    oof_lab3_map = {r["message_id"]: float(r["oof_score"]) for r in lab3_cv["winner_oof_predictions"]}
    oof_v2_map = {r["message_id"]: float(r["oof_score"]) for r in v2_cv["winner_oof_predictions"]}
    target_mid = "52c16dcd1605d5e8"
    max_oof_diff = max(abs(oof_lab3_map[m] - oof_v2_map[m]) for m in oof_lab3_map)

    # Diagnostycznie na validation.csv: liczba zmienionych decyzji i maksymalna zmiana oceny
    pred_lab3 = read_pred_csv(LAB3_PROJECT / "results" / "validation_predictions.csv")
    pred_v2 = read_pred_csv(V2_DIR / "results" / "validation_predictions.csv")

    val_decision_changes = []
    max_val_diff = 0.0
    max_val_diff_mid = None
    for mid in sorted(pred_lab3.keys()):
        s1, l1 = pred_lab3[mid]
        s2, l2 = pred_v2[mid]
        diff = abs(s2 - s1)
        if diff > max_val_diff:
            max_val_diff = diff
            max_val_diff_mid = mid
        if l1 != l2:
            val_decision_changes.append(
                {"message_id": mid, "lab3_label": l1, "v2_label": l2, "lab3_score": s1, "v2_score": s2}
            )

    same_model_and_c = (c_lab3["model_type"] == c_v2["model_type"]) and (
        c_lab3["param"] == c_v2["param"]
    )
    same_cv_tp_fp = (
        c_lab3["cv_cm"]["tp"] == c_v2["cv_cm"]["tp"]  # type: ignore[index]
        and c_lab3["cv_cm"]["fp"] == c_v2["cv_cm"]["fp"]  # type: ignore[index]
    )
    zero_val_flips = len(val_decision_changes) == 0
    no_decision_change = same_model_and_c and same_cv_tp_fp and zero_val_flips

    payload: Dict[str, object] = {
        "check": "S10",
        "description": "Skutek probnej poprawki U1 (_MONEY_RE) w kopii lab4/proby/v2/ vs Lab 3",
        "patch_applied_in_v2_only": {
            "old": OLD_MONEY_RE_BLOCK,
            "new": NEW_MONEY_RE_BLOCK,
        },
        "synthetic_tests": {
            "suffix_currency_cases": suffix_results,
            "existing_money_cases": existing_results,
            "all_passed": synthetic_ok,
        },
        "token_list_changes": token_changes,
        "expected_token_changes_match": (
            token_changes["train"]["n_changed_messages"] == 1
            and token_changes["validation"]["n_changed_messages"] == 0
        ),
        "v2_run_wall_seconds": round(elapsed_s, 4),
        "comparison_lab3_vs_v2": {
            "lab3": c_lab3,
            "v2": c_v2,
            "same_model_and_param": same_model_and_c,
            "same_cv_tp_fp": same_cv_tp_fp,
            "threshold_diff": c_v2["threshold"] - c_lab3["threshold"],
            "vocab_size_diff": c_v2["full_vocab_size"] - c_lab3["full_vocab_size"],
            "cv_pr_auc_diff": c_v2["cv_pr_auc"] - c_lab3["cv_pr_auc"],
            "cv_max_oof_score_diff": max_oof_diff,
            "affected_train_msg_oof": {
                "message_id": target_mid,
                "lab3_oof_score": oof_lab3_map[target_mid],
                "v2_oof_score": oof_v2_map[target_mid],
                "diff": oof_v2_map[target_mid] - oof_lab3_map[target_mid],
            },
            "validation_diagnostic": {
                "lab3_cm": lab3_val_m["selected_model"]["confusion_matrix"],
                "v2_cm": v2_val_m["selected_model"]["confusion_matrix"],
                "n_changed_decisions": len(val_decision_changes),
                "changed_decisions": val_decision_changes,
                "max_abs_score_diff": max_val_diff,
                "max_abs_score_diff_message_id": max_val_diff_mid,
            },
        },
        "criteria_result": {
            "no_decision_change": no_decision_change,
            "classification": (
                "POTWIERDZONY BLAD, wplyw W (bez zmiany decyzji)"
                if no_decision_change
                else "POTWIERDZONY BLAD, wplyw W (ZMIANA DECYZJI LUB WYBORU MODELU/CV)"
            ),
        },
    }

    OUTPUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT_JSON.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )

    print(
        f"S10 zakonczone | synthetic_ok={synthetic_ok} | "
        f"token_changes: train={token_changes['train']['n_changed_messages']}, val={token_changes['validation']['n_changed_messages']} | "
        f"same_model_C={same_model_and_c}, same_cv_tp_fp={same_cv_tp_fp}, val_flips={len(val_decision_changes)} | "
        f"max_val_score_diff={max_val_diff:.3e} (msg={max_val_diff_mid}) | "
        f"thr_diff={c_v2['threshold'] - c_lab3['threshold']:+.3e} | "
        f"czas_v2={elapsed_s:.2f}s"
    )
    if not synthetic_ok:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
