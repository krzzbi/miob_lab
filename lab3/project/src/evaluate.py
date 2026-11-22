"""Czesc 4: Jednorazowa ocena przewidywan na validation.csv po message_id + punkty odniesienia."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.cv_train import predict_baseline_money_remove
from src.data_io import load_csv
from src.metrics import (
    compute_metrics,
    compute_metrics_from_predictions,
    labels_to_binary,
)
from src.models import SpamClassifierArtifact
from src.preprocess import tokenize_corpus


SENSITIVITY_THRESHOLDS: Tuple[float, ...] = (0.25, 0.40, 0.50)


def load_predictions_csv(path: str | Path) -> Dict[str, Tuple[float, str]]:
    """Wczytuje plik przewidywan (message_id,score,predicted_label) jako mape po message_id."""
    preds: Dict[str, Tuple[float, str]] = {}
    with open(path, "r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            mid = row["message_id"]
            if mid in preds:
                raise ValueError(f"Zduplikowane message_id w pliku przewidywan: {mid}")
            preds[mid] = (float(row["score"]), row["predicted_label"])
    return preds


def _align_predictions(
    mids: Sequence[str],
    pred_map: Dict[str, Tuple[float, str]],
) -> Tuple[List[float], List[int]]:
    """Laczy przewidywania z kolejnoscia validation.csv scisle po message_id."""
    if set(mids) != set(pred_map.keys()) or len(mids) != len(pred_map):
        raise ValueError(
            "Zbior message_id w pliku przewidywan nie zgadza sie 1:1 z validation.csv"
        )
    scores: List[float] = []
    y_pred: List[int] = []
    for mid in mids:
        sc, lbl = pred_map[mid]
        if lbl not in ("spam", "ham"):
            raise ValueError(f"Nieprawidlowa przewidywana etykieta {lbl!r} dla {mid}")
        scores.append(sc)
        y_pred.append(1 if lbl == "spam" else 0)
    return scores, y_pred


def _extract_subject(raw_text: str) -> str:
    first_line = raw_text.split("\n", 1)[0].strip()
    return first_line


def evaluate_validation(
    *,
    validation_csv: str | Path,
    predictions_csv: str | Path,
    model_json: str | Path,
    alternative_predictions_csv: str | Path | None = None,
    alternative_model_json: str | Path | None = None,
    output_metrics_json: str | Path | None = None,
    output_errors_json: str | Path | None = None,
) -> Tuple[Dict[str, object], Dict[str, object]]:
    """Ocenia zapisane przewidywania wzgledem etykiet z validation.csv (laczenie po message_id)."""
    mids, texts, labels = load_csv(validation_csv, include_label=True)
    assert labels is not None
    y_true = labels_to_binary(labels)

    artifact = SpamClassifierArtifact.load_json(model_json)
    pred_map = load_predictions_csv(predictions_csv)
    scores, y_pred = _align_predictions(mids, pred_map)

    selected_metrics = compute_metrics_from_predictions(
        y_true, y_pred, y_scores=scores
    )
    selected_metrics["threshold"] = artifact.threshold
    selected_metrics["model_type"] = artifact.model.to_dict()["model_type"]
    selected_metrics["hyperparams"] = artifact.model.to_dict()["hyperparams"]

    # Informacyjnie: wyniki wybranego modelu dla progow pomocniczych (bez zmiany progu w modelu)
    sensitivity = {
        f"{thr:.2f}": compute_metrics(y_true, scores, thr)
        for thr in SENSITIVITY_THRESHOLDS
    }

    # Punkty odniesienia
    always_ham_metrics = compute_metrics_from_predictions(y_true, [0] * len(y_true))
    tokenized = tokenize_corpus(texts, apply_masking=artifact.apply_masking)
    rule_preds = predict_baseline_money_remove(tokenized)
    rule_metrics = compute_metrics_from_predictions(y_true, rule_preds)

    alt_metrics = None
    if alternative_predictions_csv is not None and Path(alternative_predictions_csv).exists():
        alt_pred_map = load_predictions_csv(alternative_predictions_csv)
        alt_scores, alt_y_pred = _align_predictions(mids, alt_pred_map)
        alt_metrics = compute_metrics_from_predictions(
            y_true, alt_y_pred, y_scores=alt_scores
        )
        if alternative_model_json is not None and Path(alternative_model_json).exists():
            alt_art = SpamClassifierArtifact.load_json(alternative_model_json)
            alt_metrics["threshold"] = alt_art.threshold
            alt_metrics["model_type"] = alt_art.model.to_dict()["model_type"]
            alt_metrics["hyperparams"] = alt_art.model.to_dict()["hyperparams"]

    # Lista wszystkich FP i FN wybranego modelu z 5 slowami najmocniej przesuwajacymi ocene
    fp_list: List[Dict[str, object]] = []
    fn_list: List[Dict[str, object]] = []
    for mid, text, true_lbl, sc, yp in zip(mids, texts, labels, scores, y_pred):
        pred_lbl = "spam" if yp == 1 else "ham"
        if pred_lbl == true_lbl:
            continue
        err_type = "FP" if (true_lbl == "ham" and pred_lbl == "spam") else "FN"
        all_contribs = artifact.explain_one(text, top_k=50)
        top_5_abs = all_contribs[:5]
        # 5 slow najmocniej przesuwajacych ocene w strone blednej decyzji
        if err_type == "FP":
            dir_contribs = sorted(
                [c for c in all_contribs if float(c["contribution"]) > 0.0],
                key=lambda c: -float(c["contribution"]),
            )[:5]
        else:
            dir_contribs = sorted(
                [c for c in all_contribs if float(c["contribution"]) < 0.0],
                key=lambda c: float(c["contribution"]),
            )[:5]

        entry: Dict[str, object] = {
            "error_type": err_type,
            "message_id": mid,
            "true_label": true_lbl,
            "predicted_label": pred_lbl,
            "score": sc,
            "threshold": artifact.threshold,
            "subject": _extract_subject(text),
            "top_5_by_abs_contribution": top_5_abs,
            "top_5_pushing_toward_error": dir_contribs,
            "text_preview": " ".join(text.split())[:260],
        }
        if err_type == "FP":
            fp_list.append(entry)
        else:
            fn_list.append(entry)

    metrics_payload: Dict[str, object] = {
        "dataset": {
            "n_total": len(y_true),
            "n_ham": y_true.count(0),
            "n_spam": y_true.count(1),
        },
        "selected_model": selected_metrics,
        "selected_model_informational_thresholds": sensitivity,
        "baselines": {
            "always_ham": always_ham_metrics,
            "rule_money_and_remove": rule_metrics,
            "alternative_model_informational": alt_metrics,
        },
    }

    errors_payload: Dict[str, object] = {
        "threshold": artifact.threshold,
        "n_fp": len(fp_list),
        "n_fn": len(fn_list),
        "false_positives": fp_list,
        "false_negatives": fn_list,
    }

    if output_metrics_json is not None:
        p = Path(output_metrics_json)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(
            json.dumps(metrics_payload, indent=2, sort_keys=True, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )

    if output_errors_json is not None:
        p = Path(output_errors_json)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(
            json.dumps(errors_payload, indent=2, sort_keys=True, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )

    return metrics_payload, errors_payload


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description="Ocena przewidywan na validation.csv po message_id."
    )
    parser.add_argument(
        "--project-dir",
        default=str(Path(__file__).resolve().parents[1]),
        help="Sciezka do folderu project/.",
    )
    args = parser.parse_args(argv)
    proj = Path(args.project_dir)

    metrics, errors = evaluate_validation(
        validation_csv=proj / "data" / "validation.csv",
        predictions_csv=proj / "results" / "validation_predictions.csv",
        model_json=proj / "models" / "spam_classifier.json",
        alternative_predictions_csv=proj / "results" / "validation_predictions_alternative.csv",
        alternative_model_json=proj / "models" / "alternative_model.json",
        output_metrics_json=proj / "results" / "validation_metrics.json",
        output_errors_json=proj / "results" / "validation_errors.json",
    )
    sel = metrics["selected_model"]  # type: ignore[index]
    cm = sel["confusion_matrix"]  # type: ignore[index]
    print(
        f"Ocena validation (60): TP={cm['tp']}, FP={cm['fp']}, FN={cm['fn']}, TN={cm['tn']} | "
        f"FPR={sel['fpr_fraction']} ({sel['fpr']*100:.1f}%), "
        f"czulosc={sel['recall_fraction']} ({sel['recall']*100:.1f}%), "
        f"PR-AUC={sel['pr_auc']:.4f} | FP={errors['n_fp']}, FN={errors['n_fn']}"
    )


if __name__ == "__main__":
    main()
