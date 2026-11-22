"""S5: Izolacja slownika i IDF w kodzie produkcyjnym CV (evaluate_candidate_cv i run_cv_and_train).

Sprawdza na kopiach train.csv w lab4/proby/s5_cv_isolation/ dwojako:
1. Czarnoskrzynkowo (bez zadnej podmiany w src/):
   - token-straznik 'zqxsentinelzqx' (tylko male litery, nieusuwany przez maskowanie)
     dopisany do 3 wiadomosci z czesci walidacyjnej foldu 1 (folds[0]) -> w foldzie 1
     vocab_size bez zmian (+0) i oof_scores dla wszystkich 36 wiadomosci foldu 1 identyczne (diff = 0.0);
   - ten sam straznik dopisany do 3 wiadomosci z czesci uczacej foldu 1 (folds[1]) -> w foldzie 1
     vocab_size wzrasta dokladnie o +1.
2. Z lekkim owinieciem cv_train.TfidfVectorizer w pamieci (bez modyfikacji plikow w lab3/):
   - przechwytuje slowniki vocab i wektory idf we wszystkich 5 foldach podczas wywolania
     produkcyjnego evaluate_candidate_cv() i run_cv_and_train(), potwierdzajac wprost:
     brak straznika w vocab foldu k i identyczne idf, gdy straznik jest tylko w walidacji foldu k,
     oraz obecnosc straznika w vocab, gdy jest w czesci uczacej.
"""

from __future__ import annotations

import csv
import json
import re
import shutil
import sys
from pathlib import Path
from typing import Dict, List, Sequence, Tuple


REPO_ROOT = Path(__file__).resolve().parents[2]
LAB3_PROJECT = REPO_ROOT / "lab3" / "project"
WORK_DIR = REPO_ROOT / "lab4" / "proby" / "s5_cv_isolation"
OUTPUT_JSON = REPO_ROOT / "lab4" / "wyniki" / "s5_cv_isolation.json"

sys.path.insert(0, str(LAB3_PROJECT))

import src.cv_train as cv_train  # noqa: E402
from src.data_io import load_csv  # noqa: E402
from src.metrics import labels_to_binary  # noqa: E402
from src.preprocess import (  # noqa: E402
    BODY_MASKING_PATTERNS,
    SUBJECT_PREFIX_PATTERN,
    SUBJECT_TAG_PATTERNS,
    mask_source_artifacts,
    tokenize_corpus,
    tokenize_text,
)
from src.tfidf import TfidfVectorizer as OriginalTfidfVectorizer  # noqa: E402


SENTINEL = "zqxsentinelzqx"


def write_train_copy(
    dest_proj: Path,
    mids: Sequence[str],
    texts: Sequence[str],
    labels: Sequence[str],
) -> None:
    data_dir = dest_proj / "data"
    data_dir.mkdir(parents=True, exist_ok=True)
    with open(data_dir / "train.csv", "w", encoding="utf-8", newline="") as f:
        w = csv.writer(f, lineterminator="\n")
        w.writerow(["message_id", "text", "label"])
        for mid, txt, lbl in zip(mids, texts, labels):
            w.writerow([mid, txt, lbl])


def fold_message_id_assignment(
    mids: Sequence[str], labels: Sequence[str]
) -> List[List[str]]:
    y = labels_to_binary(labels)
    folds = cv_train.make_stratified_folds(
        y, n_splits=cv_train.CV_FOLDS, seed=cv_train.CV_SEED
    )
    return [[mids[i] for i in fold] for fold in folds]


def run_with_vectorizer_capture(
    tokenized: Sequence[Sequence[str]],
    y: Sequence[int],
    folds: Sequence[Sequence[int]],
) -> Tuple[Dict[str, object], List[Dict[str, object]]]:
    captured: List[Dict[str, object]] = []

    class CapturingTfidfVectorizer(OriginalTfidfVectorizer):
        def fit(self, tokenized_docs: Sequence[Sequence[str]]) -> "CapturingTfidfVectorizer":
            super().fit(tokenized_docs)
            captured.append(
                {
                    "n_docs_fit": self.n_docs_fit,
                    "vocab_size": self.vocab_size,
                    "has_sentinel": SENTINEL in self.vocab,
                    "vocab": dict(self.vocab),
                    "idf": list(self.idf),
                }
            )
            return self

    orig = cv_train.TfidfVectorizer
    cv_train.TfidfVectorizer = CapturingTfidfVectorizer  # type: ignore[misc]
    try:
        res = cv_train.evaluate_candidate_cv(
            tokenized,
            y,
            folds,
            model_type="logistic_regression",
            param_name="C",
            param_value=10.0,
            min_df=cv_train.MIN_DF,
            max_fp=cv_train.MAX_FP_OOF,
        )
    finally:
        cv_train.TfidfVectorizer = orig  # type: ignore[misc]
    return res, captured


def main() -> None:
    if WORK_DIR.exists():
        shutil.rmtree(WORK_DIR)
    WORK_DIR.mkdir(parents=True, exist_ok=True)

    # (a) Weryfikacja wlasnosci tokenu-straznika
    sentinel_is_lower = SENTINEL.islower() and SENTINEL.isalpha() and len(SENTINEL) >= 2
    matching_patterns = []
    probe = f"Subject: neutral message\n\nhello {SENTINEL} world"
    for name, pat in [
        ("subject_prefix", SUBJECT_PREFIX_PATTERN),
        *SUBJECT_TAG_PATTERNS,
        *BODY_MASKING_PATTERNS,
    ]:
        if re.search(pat, SENTINEL, flags=re.IGNORECASE):
            matching_patterns.append(name)
    masked_probe = mask_source_artifacts(probe)
    tokens_probe = tokenize_text(probe, apply_masking=True)
    sentinel_survives_masking = (SENTINEL in masked_probe) and (SENTINEL in tokens_probe)

    # Wczytanie oryginalnego train.csv (tylko odczyt)
    mids, texts, labels = load_csv(LAB3_PROJECT / "data" / "train.csv", include_label=True)
    assert labels is not None
    y = labels_to_binary(labels)
    sentinel_in_orig_train = any(
        SENTINEL in toks for toks in tokenize_corpus(texts, apply_masking=True)
    )

    folds = cv_train.make_stratified_folds(
        y, n_splits=cv_train.CV_FOLDS, seed=cv_train.CV_SEED
    )
    base_fold_mids = fold_message_id_assignment(mids, labels)

    # (b) Wybor >= 2 wiadomosci (wybieramy 3) w foldzie 1 (folds[0]) i w foldzie 2 (folds[1])
    val_only_indices = folds[0][:3]      # tylko w czesci walidacyjnej foldu 1 (a uczacej foldow 2-5)
    train_part_indices = folds[1][:3]    # w czesci uczacej foldu 1 (a walidacyjnej foldu 2)

    texts_val_only = list(texts)
    for idx in val_only_indices:
        texts_val_only[idx] = texts_val_only[idx] + f"\n{SENTINEL}\n"

    texts_train_part = list(texts)
    for idx in train_part_indices:
        texts_train_part[idx] = texts_train_part[idx] + f"\n{SENTINEL}\n"

    # Zapis kopii danych w lab4/proby/s5_cv_isolation/
    proj_base = WORK_DIR / "base"
    proj_val_only = WORK_DIR / "val_only_fold1"
    proj_train_part = WORK_DIR / "train_part_fold1"
    write_train_copy(proj_base, mids, texts, labels)
    write_train_copy(proj_val_only, mids, texts_val_only, labels)
    write_train_copy(proj_train_part, mids, texts_train_part, labels)

    # (c) Potwierdzenie identycznego przydzialu message_id do foldow
    mids_a, texts_a, labels_a = load_csv(proj_val_only / "data" / "train.csv", include_label=True)
    mids_b, texts_b, labels_b = load_csv(proj_train_part / "data" / "train.csv", include_label=True)
    assert labels_a is not None and labels_b is not None
    folds_mids_identical_a = fold_message_id_assignment(mids_a, labels_a) == base_fold_mids
    folds_mids_identical_b = fold_message_id_assignment(mids_b, labels_b) == base_fold_mids

    # ------------------------------------------------------------------
    # Metoda 1 (czarna skrzynka): niezmodyfikowane run_cv_and_train()
    # ------------------------------------------------------------------
    out_base = cv_train.run_cv_and_train(proj_base, apply_masking=True, save_outputs=False)
    out_val_only = cv_train.run_cv_and_train(proj_val_only, apply_masking=True, save_outputs=False)
    out_train_part = cv_train.run_cv_and_train(proj_train_part, apply_masking=True, save_outputs=False)

    def get_lr10_per_fold(out: Dict[str, object]) -> List[Dict[str, object]]:
        cands = out["cv_results"]["logistic_regression_candidates"]  # type: ignore[index]
        c10 = next(c for c in cands if c["param_value"] == 10.0)
        return c10["per_fold"]  # type: ignore[return-value]

    def get_oof_map(out: Dict[str, object]) -> Dict[str, float]:
        return {
            r["message_id"]: float(r["oof_score"])
            for r in out["cv_results"]["winner_oof_predictions"]  # type: ignore[index]
        }

    pf_base = get_lr10_per_fold(out_base)
    pf_a = get_lr10_per_fold(out_val_only)
    pf_b = get_lr10_per_fold(out_train_part)

    oof_base = get_oof_map(out_base)
    oof_a = get_oof_map(out_val_only)
    oof_b = get_oof_map(out_train_part)

    fold1_mids = base_fold_mids[0]
    fold2_mids = base_fold_mids[1]

    # W scenariuszu A (straznik tylko w walidacji foldu 1):
    # - fold 1: vocab_size bez zmian, oof_scores dla 36 wiad. foldu 1 identyczne (diff = 0.0)
    # - foldy 2..5: straznik jest w czesci uczacej -> vocab_size +1
    bb_fold1_vocab_delta_a = int(pf_a[0]["vocab_size"]) - int(pf_base[0]["vocab_size"])
    bb_fold1_max_oof_diff_a = max(abs(oof_a[m] - oof_base[m]) for m in fold1_mids)
    bb_other_folds_vocab_deltas_a = [
        int(pf_a[k]["vocab_size"]) - int(pf_base[k]["vocab_size"]) for k in range(1, 5)
    ]

    # W scenariuszu B (straznik w czesci uczacej foldu 1, czyli w walidacji foldu 2):
    # - fold 1: vocab_size +1
    # - fold 2: vocab_size bez zmian, oof_scores dla 36 wiad. foldu 2 identyczne (diff = 0.0)
    bb_fold1_vocab_delta_b = int(pf_b[0]["vocab_size"]) - int(pf_base[0]["vocab_size"])
    bb_fold2_vocab_delta_b = int(pf_b[1]["vocab_size"]) - int(pf_base[1]["vocab_size"])
    bb_fold2_max_oof_diff_b = max(abs(oof_b[m] - oof_base[m]) for m in fold2_mids)

    # ------------------------------------------------------------------
    # Metoda 2 (wrapper TfidfVectorizer na produkcyjnym evaluate_candidate_cv)
    # ------------------------------------------------------------------
    tok_base = tokenize_corpus(texts, apply_masking=True)
    tok_a = tokenize_corpus(texts_a, apply_masking=True)
    tok_b = tokenize_corpus(texts_b, apply_masking=True)

    ev_base, cap_base = run_with_vectorizer_capture(tok_base, y, folds)
    ev_a, cap_a = run_with_vectorizer_capture(tok_a, y, folds)
    ev_b, cap_b = run_with_vectorizer_capture(tok_b, y, folds)

    # Fold 1 w scenariuszu A (straznik w walidacji foldu 1):
    # brak w vocab, identyczne vocab i idf, identyczne oof_scores w foldzie 1
    cap_a_fold1_has_sentinel = bool(cap_a[0]["has_sentinel"])
    cap_a_fold1_vocab_identical = cap_a[0]["vocab"] == cap_base[0]["vocab"]
    cap_a_fold1_idf_identical = cap_a[0]["idf"] == cap_base[0]["idf"]
    cap_a_fold1_oof_diff = max(
        abs(float(ev_a["oof_scores"][i]) - float(ev_base["oof_scores"][i]))  # type: ignore[index]
        for i in folds[0]
    )
    cap_a_other_folds_have_sentinel = [bool(cap_a[k]["has_sentinel"]) for k in range(1, 5)]

    # Fold 1 w scenariuszu B (straznik w czesci uczacej foldu 1):
    cap_b_fold1_has_sentinel = bool(cap_b[0]["has_sentinel"])
    cap_b_fold1_vocab_delta = int(cap_b[0]["vocab_size"]) - int(cap_base[0]["vocab_size"])
    # Fold 2 w scenariuszu B (straznik tylko w walidacji foldu 2):
    cap_b_fold2_has_sentinel = bool(cap_b[1]["has_sentinel"])
    cap_b_fold2_vocab_identical = cap_b[1]["vocab"] == cap_base[1]["vocab"]
    cap_b_fold2_idf_identical = cap_b[1]["idf"] == cap_base[1]["idf"]
    cap_b_fold2_oof_diff = max(
        abs(float(ev_b["oof_scores"][i]) - float(ev_base["oof_scores"][i]))  # type: ignore[index]
        for i in folds[1]
    )

    # Sprawdzenie uczenia koncowego (full_vec na calym train w run_cv_and_train):
    # w obu scenariuszach straznik jest w 3 wiadomosciach train, wiec trafia do slownika koncowego (+1).
    full_vocab_base = int(out_base["cv_results"]["train_summary"]["full_vocab_size"])  # type: ignore[index]
    full_vocab_a = int(out_val_only["cv_results"]["train_summary"]["full_vocab_size"])  # type: ignore[index]
    full_vocab_b = int(out_train_part["cv_results"]["train_summary"]["full_vocab_size"])  # type: ignore[index]

    passed = (
        sentinel_is_lower
        and not matching_patterns
        and sentinel_survives_masking
        and not sentinel_in_orig_train
        and folds_mids_identical_a
        and folds_mids_identical_b
        and bb_fold1_vocab_delta_a == 0
        and bb_fold1_max_oof_diff_a == 0.0
        and bb_other_folds_vocab_deltas_a == [1, 1, 1, 1]
        and bb_fold1_vocab_delta_b == 1
        and bb_fold2_vocab_delta_b == 0
        and bb_fold2_max_oof_diff_b == 0.0
        and (not cap_a_fold1_has_sentinel)
        and cap_a_fold1_vocab_identical
        and cap_a_fold1_idf_identical
        and cap_a_fold1_oof_diff == 0.0
        and cap_a_other_folds_have_sentinel == [True, True, True, True]
        and cap_b_fold1_has_sentinel
        and cap_b_fold1_vocab_delta == 1
        and (not cap_b_fold2_has_sentinel)
        and cap_b_fold2_vocab_identical
        and cap_b_fold2_idf_identical
        and cap_b_fold2_oof_diff == 0.0
        and (full_vocab_a == full_vocab_base + 1)
        and (full_vocab_b == full_vocab_base + 1)
    )

    payload: Dict[str, object] = {
        "check": "S5",
        "description": "Izolacja slownika i IDF w produkcyjnym evaluate_candidate_cv() i run_cv_and_train()",
        "sentinel_checks": {
            "sentinel": SENTINEL,
            "is_lowercase_alpha": sentinel_is_lower,
            "masking_patterns_matching_sentinel": matching_patterns,
            "survives_masking_and_tokenization": sentinel_survives_masking,
            "already_present_in_original_train": sentinel_in_orig_train,
            "n_modified_docs_per_scenario": 3,
            "val_only_fold1_message_ids": [mids[i] for i in val_only_indices],
            "train_part_fold1_message_ids": [mids[i] for i in train_part_indices],
            "fold_assignment_identical_scenario_a": folds_mids_identical_a,
            "fold_assignment_identical_scenario_b": folds_mids_identical_b,
        },
        "method_1_black_box_run_cv_and_train": {
            "base_fold_vocab_sizes": [int(f["vocab_size"]) for f in pf_base],
            "scenario_a_val_only_fold1": {
                "fold_vocab_sizes": [int(f["vocab_size"]) for f in pf_a],
                "fold1_vocab_delta": bb_fold1_vocab_delta_a,
                "fold1_max_oof_score_diff": bb_fold1_max_oof_diff_a,
                "folds_2_to_5_vocab_deltas": bb_other_folds_vocab_deltas_a,
            },
            "scenario_b_train_part_fold1": {
                "fold_vocab_sizes": [int(f["vocab_size"]) for f in pf_b],
                "fold1_vocab_delta": bb_fold1_vocab_delta_b,
                "fold2_vocab_delta": bb_fold2_vocab_delta_b,
                "fold2_max_oof_score_diff": bb_fold2_max_oof_diff_b,
            },
            "full_train_vocab_sizes": {
                "base": full_vocab_base,
                "scenario_a": full_vocab_a,
                "scenario_b": full_vocab_b,
            },
        },
        "method_2_captured_vectorizer_evaluate_candidate_cv": {
            "note": "Tymczasowe owiniecie cv_train.TfidfVectorizer w pamieci bez modyfikacji plikow w lab3/.",
            "scenario_a_val_only_fold1": {
                "fold1_has_sentinel": cap_a_fold1_has_sentinel,
                "fold1_vocab_identical_to_base": cap_a_fold1_vocab_identical,
                "fold1_idf_identical_to_base": cap_a_fold1_idf_identical,
                "fold1_max_oof_score_diff": cap_a_fold1_oof_diff,
                "folds_2_to_5_have_sentinel": cap_a_other_folds_have_sentinel,
            },
            "scenario_b_train_part_fold1": {
                "fold1_has_sentinel": cap_b_fold1_has_sentinel,
                "fold1_vocab_delta": cap_b_fold1_vocab_delta,
                "fold2_has_sentinel": cap_b_fold2_has_sentinel,
                "fold2_vocab_identical_to_base": cap_b_fold2_vocab_identical,
                "fold2_idf_identical_to_base": cap_b_fold2_idf_identical,
                "fold2_max_oof_score_diff": cap_b_fold2_oof_diff,
            },
        },
        "what_s5_does_not_cover": [
            "Nie sprawdza potencjalnego wycieku na etapie recznego ustalania wzorcow maskowania przed CV (wzorce sa stale i ustalone na calym train.csv, nie wewnatrz foldow CV).",
            "Nie testuje bezposrednio pozostalych kandydatow z siatki w Metodzie 2 (sprawdzono dla wybranego C=10.0, a w Metodzie 1 caly run_cv_and_train); wszystkie korzystaja z tej samej funkcji evaluate_candidate_cv().",
            "Metoda 2 wykorzystuje podmiane referencji cv_train.TfidfVectorizer w pamieci (choc Metoda 1 weryfikuje izolacje bez zadnej podmiany).",
        ],
        "passed": passed,
    }

    OUTPUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT_JSON.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )

    print(
        f"S5 zakonczone: passed={passed} | "
        f"val_only fold1: has_sentinel={cap_a_fold1_has_sentinel}, vocab_delta={bb_fold1_vocab_delta_a}, oof_diff={bb_fold1_max_oof_diff_a} | "
        f"train_part fold1: has_sentinel={cap_b_fold1_has_sentinel}, vocab_delta={bb_fold1_vocab_delta_b}"
    )
    if not passed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
