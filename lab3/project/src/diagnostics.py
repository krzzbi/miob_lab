"""Czesc 5: Diagnostyka po ocenie (sekcja 5 planu).

Wylacznie diagnoza - skrypt nie modyfikuje zapisanego modelu, danych ani przewidywan.
Realizuje kolejno:
  a) kalibracje na ocenach OOF z CV (train),
  b) porownanie z maskowaniem vs bez maskowania (CV na train i informacyjnie validation),
  c) perturbacje na kopiach w pamieci wiadomosci spam z validation (stopka ILUG i cytat),
  d) rozbicie wynikow na wspolne watki (train vs validation) i pozostale wiadomosci.
"""

from __future__ import annotations

import argparse
import json
import re
import statistics
import sys
from pathlib import Path
from typing import Dict, List, Sequence

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.cv_train import run_cv_and_train
from src.data_io import load_csv
from src.evaluate import load_predictions_csv
from src.metrics import (
    compute_metrics,
    compute_metrics_from_predictions,
    labels_to_binary,
)
from src.models import SpamClassifierArtifact


# Dokladna stopka listy ILUG z train.csv (test perturbacyjny c1)
ILUG_FOOTER_FROM_TRAIN = (
    "\n\n-- \n"
    "Irish Linux Users' Group: ilug@linux.ie\n"
    "http://www.linux.ie/mailman/listinfo/ilug for (un)subscription information.\n"
    "List maintainer: listmaster@linux.ie\n"
)

# Linia przypisania cytatu (nie-maskowana) + neutralna linia cytatu '>' (test perturbacyjny c2)
UNMASKED_QUOTE_PREFIX = (
    "On Mon, Sep 23, 2002 at 10:15:00AM +0100, David wrote:\n"
    "> Here is the previous message.\n\n"
)


def normalize_subject_for_threads(raw_text: str) -> str:
    """Ustalona z gory normalizacja tematu do wykrywania wspolnych watkow (sekcja 5 planu).

    Temat bez prefiksu Subject:, bez wielokrotnych Re:/Fwd:/Fw:, bez znacznikow list [...],
    male litery i ujednolicone biale znaki.
    """
    first_line = raw_text.split("\n", 1)[0]
    s = re.sub(r"^\s*Subject:\s*", "", first_line, count=1, flags=re.IGNORECASE)
    prev = None
    while prev != s:
        prev = s
        s = re.sub(r"^\s*(?:re|fwd?)(?:\[\d+\])?:\s*", "", s, flags=re.IGNORECASE)
        s = re.sub(r"^\s*\[[^\]]+\]\s*", "", s)
    s = re.sub(r"\s*\(fwd?\)\s*$", "", s, flags=re.IGNORECASE)
    s = re.sub(r"\s+", " ", s).strip().lower()
    return s


def _insert_after_subject(raw_text: str, inserted_block: str) -> str:
    parts = raw_text.split("\n", 1)
    if len(parts) == 1:
        return parts[0] + "\n\n" + inserted_block
    return parts[0] + "\n\n" + inserted_block + parts[1].lstrip("\n")


def run_diagnostics(project_dir: str | Path) -> Dict[str, object]:
    """Wykonuje pelna diagnostyke z sekcji 5 (a-d) i zapisuje results/diagnostics_section5.json."""
    proj = Path(project_dir)
    train_csv = proj / "data" / "train.csv"
    val_csv = proj / "data" / "validation.csv"
    model_path = proj / "models" / "spam_classifier.json"
    cv_path = proj / "results" / "cv_results.json"
    val_pred_path = proj / "results" / "validation_predictions.csv"
    val_metrics_path = proj / "results" / "validation_metrics.json"

    cv_data = json.loads(cv_path.read_text(encoding="utf-8"))
    val_metrics_data = json.loads(val_metrics_path.read_text(encoding="utf-8"))
    frozen_artifact = SpamClassifierArtifact.load_json(model_path)

    train_mids, train_texts, train_labels = load_csv(train_csv, include_label=True)
    val_mids, val_texts, val_labels = load_csv(val_csv, include_label=True)
    assert train_labels is not None and val_labels is not None
    y_val = labels_to_binary(val_labels)

    # ------------------------------------------------------------------
    # (a) Kalibracja na ocenach OOF z CV (train) + rozrzut progu
    # ------------------------------------------------------------------
    win_type = cv_data["decision"]["winner_model_type"]
    win_param = cv_data["decision"]["winner_param"]
    cand_list = (
        cv_data["logistic_regression_candidates"]
        if win_type == "logistic_regression"
        else cv_data["multinomial_nb_candidates"]
    )
    winner_cand = next(
        c for c in cand_list if {c["param_name"]: c["param_value"]} == win_param
    )
    diag_a = {
        "model_type": win_type,
        "hyperparams": win_param,
        "selected_oof_threshold": winner_cand["threshold"],
        "calibration_bins": winner_cand["calibration_bins"],
        "per_fold_thresholds_max_fp_0": [
            {
                "fold": f["fold"],
                "threshold": f["fold_own_threshold_max_fp_0"]["threshold"],
                "tp": f["fold_own_threshold_max_fp_0"]["tp"],
                "fp": f["fold_own_threshold_max_fp_0"]["fp"],
            }
            for f in winner_cand["per_fold"]
        ],
        "per_fold_thresholds_max_fp_1": [
            {
                "fold": f["fold"],
                "threshold": f["fold_own_threshold_max_fp_1"]["threshold"],
                "tp": f["fold_own_threshold_max_fp_1"]["tp"],
                "fp": f["fold_own_threshold_max_fp_1"]["fp"],
            }
            for f in winner_cand["per_fold"]
        ],
    }

    # ------------------------------------------------------------------
    # (b) Porownanie: z maskowaniem vs bez maskowania (wylacznie w pamieci)
    # ------------------------------------------------------------------
    unmasked_run = run_cv_and_train(proj, apply_masking=False, save_outputs=False)
    unmasked_cv = unmasked_run["cv_results"]
    unmasked_art: SpamClassifierArtifact = unmasked_run["winner_artifact"]  # type: ignore[assignment]
    unmasked_dec = unmasked_cv["decision"]  # type: ignore[index]
    unmasked_cand_list = (
        unmasked_cv["logistic_regression_candidates"]  # type: ignore[index]
        if unmasked_dec["winner_model_type"] == "logistic_regression"
        else unmasked_cv["multinomial_nb_candidates"]  # type: ignore[index]
    )
    unmasked_winner_cand = next(
        c
        for c in unmasked_cand_list
        if {c["param_name"]: c["param_value"]} == unmasked_dec["winner_param"]
    )
    unmasked_val_preds = unmasked_art.predict_texts(val_texts)
    unmasked_val_scores = [sc for sc, _ in unmasked_val_preds]
    unmasked_val_metrics = compute_metrics(
        y_val, unmasked_val_scores, unmasked_art.threshold
    )

    diag_b = {
        "with_masking": {
            "winner_model_type": win_type,
            "winner_param": win_param,
            "vocab_size": cv_data["train_summary"]["full_vocab_size"],
            "cv_oof": winner_cand["oof_metrics"],
            "validation_informational": val_metrics_data["selected_model"],
        },
        "without_masking": {
            "winner_model_type": unmasked_dec["winner_model_type"],
            "winner_param": unmasked_dec["winner_param"],
            "vocab_size": unmasked_cv["train_summary"]["full_vocab_size"],  # type: ignore[index]
            "cv_oof": unmasked_winner_cand["oof_metrics"],
            "validation_informational": unmasked_val_metrics,
        },
    }

    # ------------------------------------------------------------------
    # (c) Perturbacje na kopiach w pamieci wiadomosci spam z validation
    # ------------------------------------------------------------------
    spam_indices = [i for i, lbl in enumerate(val_labels) if lbl == "spam"]
    ilug_details: List[Dict[str, object]] = []
    quote_details: List[Dict[str, object]] = []

    ilug_score_changes = 0
    ilug_label_flips = 0
    ilug_max_abs_diff = 0.0

    quote_label_flips = 0
    quote_drops: List[float] = []

    for idx in spam_indices:
        mid = val_mids[idx]
        orig_text = val_texts[idx]
        orig_score, orig_label = frozen_artifact.predict_one(orig_text)

        # c1: dopisanie stopki ILUG z train
        with_ilug = orig_text + ILUG_FOOTER_FROM_TRAIN
        ilug_score, ilug_label = frozen_artifact.predict_one(with_ilug)
        abs_diff = abs(ilug_score - orig_score)
        if abs_diff > ilug_max_abs_diff:
            ilug_max_abs_diff = abs_diff
        if abs_diff > 1e-12:
            ilug_score_changes += 1
        if ilug_label != orig_label:
            ilug_label_flips += 1
        ilug_details.append(
            {
                "message_id": mid,
                "orig_score": orig_score,
                "perturbed_score": ilug_score,
                "abs_diff": abs_diff,
                "orig_label": orig_label,
                "perturbed_label": ilug_label,
            }
        )

        # c2: dopisanie nie-maskowanej linii przypisania cytatu + linii cytatu '>'
        with_quote = _insert_after_subject(orig_text, UNMASKED_QUOTE_PREFIX)
        q_score, q_label = frozen_artifact.predict_one(with_quote)
        drop = orig_score - q_score
        quote_drops.append(drop)
        flipped = q_label != orig_label
        if flipped:
            quote_label_flips += 1
        quote_details.append(
            {
                "message_id": mid,
                "orig_score": orig_score,
                "perturbed_score": q_score,
                "score_drop": drop,
                "orig_label": orig_label,
                "perturbed_label": q_label,
                "flipped": flipped,
            }
        )

    diag_c = {
        "n_spam_tested": len(spam_indices),
        "ilug_footer_perturbation": {
            "appended_footer": ILUG_FOOTER_FROM_TRAIN.strip(),
            "score_changes_gt_1e_12": ilug_score_changes,
            "label_flips": ilug_label_flips,
            "max_abs_score_diff": ilug_max_abs_diff,
            "details": ilug_details,
        },
        "unmasked_quote_perturbation": {
            "inserted_prefix": UNMASKED_QUOTE_PREFIX.strip(),
            "label_flips": quote_label_flips,
            "median_score_drop": statistics.median(quote_drops),
            "mean_score_drop": statistics.mean(quote_drops),
            "min_score_drop": min(quote_drops),
            "max_score_drop": max(quote_drops),
            "details": quote_details,
        },
    }

    # ------------------------------------------------------------------
    # (d) Wspolne watki (train vs validation) — stala definicja z gory
    # ------------------------------------------------------------------
    train_norm_subjects = {normalize_subject_for_threads(t) for t in train_texts}
    train_norm_subjects.discard("")

    pred_map = load_predictions_csv(val_pred_path)
    shared_idx: List[int] = []
    other_idx: List[int] = []
    shared_rows: List[Dict[str, object]] = []

    for i, (mid, txt, lbl) in enumerate(zip(val_mids, val_texts, val_labels)):
        norm_subj = normalize_subject_for_threads(txt)
        if norm_subj and norm_subj in train_norm_subjects:
            shared_idx.append(i)
            sc, plbl = pred_map[mid]
            shared_rows.append(
                {
                    "message_id": mid,
                    "true_label": lbl,
                    "predicted_label": plbl,
                    "score": sc,
                    "normalized_subject": norm_subj,
                }
            )
        else:
            other_idx.append(i)

    def subset_metrics(indices: Sequence[int]) -> Dict[str, object]:
        yt = [y_val[i] for i in indices]
        ys = [pred_map[val_mids[i]][0] for i in indices]
        yp = [1 if pred_map[val_mids[i]][1] == "spam" else 0 for i in indices]
        return compute_metrics_from_predictions(yt, yp, y_scores=ys)

    diag_d = {
        "definition": (
            "Temat z pierwszej linii bez prefiksu Subject:, bez wielokrotnych Re:/Fwd:/Fw:, "
            "bez znacznikow list [...], male litery i ujednolicone biale znaki."
        ),
        "n_shared": len(shared_idx),
        "n_shared_spam": sum(1 for i in shared_idx if val_labels[i] == "spam"),
        "n_shared_ham": sum(1 for i in shared_idx if val_labels[i] == "ham"),
        "n_other": len(other_idx),
        "n_other_spam": sum(1 for i in other_idx if val_labels[i] == "spam"),
        "n_other_ham": sum(1 for i in other_idx if val_labels[i] == "ham"),
        "shared_messages": shared_rows,
        "metrics_shared": subset_metrics(shared_idx),
        "metrics_other": subset_metrics(other_idx),
    }

    payload: Dict[str, object] = {
        "a_cv_calibration_and_threshold_spread": diag_a,
        "b_masking_vs_no_masking": diag_b,
        "c_validation_perturbations": diag_c,
        "d_shared_threads_validation": diag_d,
    }

    out_path = proj / "results" / "diagnostics_section5.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(
        json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    return payload


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description="Diagnostyka po ocenie (sekcja 5 planu)."
    )
    parser.add_argument(
        "--project-dir",
        default=str(Path(__file__).resolve().parents[1]),
        help="Sciezka do folderu project/.",
    )
    args = parser.parse_args(argv)
    res = run_diagnostics(args.project_dir)
    c_ilug = res["c_validation_perturbations"]["ilug_footer_perturbation"]  # type: ignore[index]
    c_quote = res["c_validation_perturbations"]["unmasked_quote_perturbation"]  # type: ignore[index]
    d_thr = res["d_shared_threads_validation"]  # type: ignore[index]
    print(
        f"Diagnostyka sekcji 5 zapisana | ILUG zmiany={c_ilug['score_changes_gt_1e_12']}/20 | "
        f"cytat zmiany decyzji={c_quote['label_flips']}/20 (mediana spadku={c_quote['median_score_drop']:.4f}) | "
        f"wspolne watki={d_thr['n_shared']} (spam={d_thr['n_shared_spam']}, ham={d_thr['n_shared_ham']})"
    )


if __name__ == "__main__":
    main()
