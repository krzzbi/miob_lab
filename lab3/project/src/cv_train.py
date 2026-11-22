"""Czesc 3: 5-krotna stratyfikowana CV na train, wybor modelu/ustawien/progu i uczenie koncowe.

Skrypt operuje wylacznie na project/data/train.csv i nigdy nie otwiera validation.csv.
"""

from __future__ import annotations

import argparse
import json
import random
import re
import sys
import time
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.data_io import load_csv
from src.metrics import (
    compute_metrics,
    compute_metrics_from_predictions,
    labels_to_binary,
    select_threshold,
)
from src.models import (
    LogisticRegressionL2,
    ModelType,
    MultinomialNaiveBayes,
    SpamClassifierArtifact,
)
from src.preprocess import (
    BODY_MASKING_PATTERNS,
    SUBJECT_PREFIX_PATTERN,
    SUBJECT_TAG_PATTERNS,
    mask_source_artifacts,
    tokenize_corpus,
)
from src.tfidf import TfidfVectorizer


LR_GRID: Tuple[float, ...] = (0.1, 1.0, 10.0, 100.0)
NB_GRID: Tuple[float, ...] = (0.01, 0.1, 1.0)
CV_FOLDS: int = 5
CV_SEED: int = 2026
MAX_FP_OOF: int = 3
MIN_DF: int = 2


def make_stratified_folds(
    y: Sequence[int],
    *,
    n_splits: int = CV_FOLDS,
    seed: int = CV_SEED,
) -> List[List[int]]:
    """Tworzy deterministyczne, stratyfikowane foldy (zachowujace proporcje klas)."""
    rng = random.Random(seed)
    ham_idx = [i for i, yi in enumerate(y) if yi == 0]
    spam_idx = [i for i, yi in enumerate(y) if yi == 1]
    rng.shuffle(ham_idx)
    rng.shuffle(spam_idx)

    folds: List[List[int]] = [[] for _ in range(n_splits)]
    for pos, idx in enumerate(ham_idx):
        folds[pos % n_splits].append(idx)
    for pos, idx in enumerate(spam_idx):
        folds[pos % n_splits].append(idx)

    for fold in folds:
        fold.sort()
    return folds


def _fit_model(
    model_type: str,
    param_value: float,
    X_train: Sequence[Dict[int, float]],
    y_train: Sequence[int],
    n_features: int,
) -> ModelType:
    if model_type == "logistic_regression":
        clf = LogisticRegressionL2(C=param_value).fit(
            X_train, y_train, n_features=n_features
        )
        if not clf.converged_:
            raise RuntimeError(
                f"Solver regresji logistycznej nie zbiegl dla C={param_value} "
                f"(grad_norm={clf.grad_norm_:.3e})"
            )
        return clf
    if model_type == "multinomial_nb":
        return MultinomialNaiveBayes(alpha=param_value).fit(
            X_train, y_train, n_features=n_features
        )
    raise ValueError(f"Nieznany model_type: {model_type!r}")


def compute_calibration_bins(
    y_true: Sequence[int],
    y_scores: Sequence[float],
    *,
    n_bins: int = 5,
) -> List[Dict[str, object]]:
    """Grupuje oceny OOF w rowne przedzialy [0, 1] do oceny kalibracji (sekcja 5 planu)."""
    bins: List[Dict[str, object]] = []
    width = 1.0 / n_bins
    for b in range(n_bins):
        lo = b * width
        hi = (b + 1) * width
        idxs = [
            i
            for i, sc in enumerate(y_scores)
            if (lo <= sc < hi) or (b == n_bins - 1 and lo <= sc <= hi)
        ]
        n_bin = len(idxs)
        if n_bin > 0:
            mean_score = sum(y_scores[i] for i in idxs) / n_bin
            n_spam = sum(y_true[i] for i in idxs)
            spam_rate = n_spam / n_bin
        else:
            mean_score = None
            n_spam = 0
            spam_rate = None
        bins.append(
            {
                "bin": f"[{lo:.1f}, {hi:.1f}{']' if b == n_bins - 1 else ')'}",
                "count": n_bin,
                "spam_count": n_spam,
                "mean_score": mean_score,
                "actual_spam_rate": spam_rate,
            }
        )
    return bins


def evaluate_candidate_cv(
    tokenized_docs: Sequence[Sequence[str]],
    y: Sequence[int],
    folds: Sequence[Sequence[int]],
    *,
    model_type: str,
    param_name: str,
    param_value: float,
    min_df: int = MIN_DF,
    max_fp: int = MAX_FP_OOF,
) -> Dict[str, object]:
    """Przeprowadza 5-krotna CV z dopasowaniem slownika i IDF od nowa w kazdym foldzie."""
    n = len(y)
    oof_scores = [0.0] * n
    fold_vocab_sizes: List[int] = []
    fold_grad_norms: List[float] = []

    for fold_idx, val_indices in enumerate(folds):
        val_set = set(val_indices)
        train_indices = [i for i in range(n) if i not in val_set]

        train_toks = [tokenized_docs[i] for i in train_indices]
        val_toks = [tokenized_docs[i] for i in val_indices]
        y_train = [y[i] for i in train_indices]

        vec = TfidfVectorizer(min_df=min_df).fit(train_toks)
        fold_vocab_sizes.append(vec.vocab_size)
        X_train = vec.transform(train_toks)
        X_val = vec.transform(val_toks)

        model = _fit_model(model_type, param_value, X_train, y_train, vec.vocab_size)
        if isinstance(model, LogisticRegressionL2):
            fold_grad_norms.append(model.grad_norm_)

        val_probs = model.predict_proba(X_val)
        for i, prob in zip(val_indices, val_probs):
            oof_scores[i] = prob

    threshold, oof_metrics = select_threshold(y, oof_scores, max_fp=max_fp)

    per_fold: List[Dict[str, object]] = []
    for fold_idx, val_indices in enumerate(folds):
        y_f = [y[i] for i in val_indices]
        s_f = [oof_scores[i] for i in val_indices]
        m_global = compute_metrics(y_f, s_f, threshold)
        thr_fp0, m_fp0 = select_threshold(y_f, s_f, max_fp=0)
        thr_fp1, m_fp1 = select_threshold(y_f, s_f, max_fp=1)
        per_fold.append(
            {
                "fold": fold_idx + 1,
                "n_total": len(val_indices),
                "vocab_size": fold_vocab_sizes[fold_idx],
                "at_selected_threshold": {
                    "tp": m_global["confusion_matrix"]["tp"],  # type: ignore[index]
                    "fp": m_global["confusion_matrix"]["fp"],  # type: ignore[index]
                    "tn": m_global["confusion_matrix"]["tn"],  # type: ignore[index]
                    "fn": m_global["confusion_matrix"]["fn"],  # type: ignore[index]
                    "recall": m_global["recall"],
                    "fpr": m_global["fpr"],
                    "pr_auc": m_global["pr_auc"],
                },
                "fold_own_threshold_max_fp_0": {
                    "threshold": thr_fp0,
                    "tp": m_fp0["confusion_matrix"]["tp"],  # type: ignore[index]
                    "fp": m_fp0["confusion_matrix"]["fp"],  # type: ignore[index]
                },
                "fold_own_threshold_max_fp_1": {
                    "threshold": thr_fp1,
                    "tp": m_fp1["confusion_matrix"]["tp"],  # type: ignore[index]
                    "fp": m_fp1["confusion_matrix"]["fp"],  # type: ignore[index]
                },
            }
        )

    return {
        "model_type": model_type,
        "param_name": param_name,
        "param_value": param_value,
        "threshold": threshold,
        "oof_metrics": oof_metrics,
        "oof_scores": oof_scores,
        "per_fold": per_fold,
        "calibration_bins": compute_calibration_bins(y, oof_scores),
        "max_fold_grad_norm": max(fold_grad_norms) if fold_grad_norms else None,
    }


def predict_baseline_money_remove(tokenized_docs: Sequence[Sequence[str]]) -> List[int]:
    """Zamrozona regula odniesienia: '<MONEY> i remove -> spam'."""
    preds: List[int] = []
    for toks in tokenized_docs:
        tset = set(toks)
        preds.append(1 if ("<MONEY>" in tset and "remove" in tset) else 0)
    return preds


def evaluate_baselines_on_train(
    tokenized_docs: Sequence[Sequence[str]],
    y: Sequence[int],
    folds: Sequence[Sequence[int]],
) -> Dict[str, object]:
    """Ocenia punkty odniesienia ('zawsze ham' i '<MONEY> i remove -> spam') na train."""
    always_ham_preds = [0] * len(y)
    rule_preds = predict_baseline_money_remove(tokenized_docs)

    def summarize(preds: Sequence[int]) -> Dict[str, object]:
        overall = compute_metrics_from_predictions(y, preds)
        per_fold = []
        for idx, val_indices in enumerate(folds):
            y_f = [y[i] for i in val_indices]
            p_f = [preds[i] for i in val_indices]
            mf = compute_metrics_from_predictions(y_f, p_f)
            per_fold.append(
                {
                    "fold": idx + 1,
                    "tp": mf["confusion_matrix"]["tp"],  # type: ignore[index]
                    "fp": mf["confusion_matrix"]["fp"],  # type: ignore[index]
                    "recall": mf["recall"],
                    "fpr": mf["fpr"],
                }
            )
        return {"overall": overall, "per_fold": per_fold}

    return {
        "always_ham": summarize(always_ham_preds),
        "rule_money_and_remove": summarize(rule_preds),
    }


def _candidate_rank_key(cand: Dict[str, object]) -> Tuple[int, float, int]:
    """Kryterium wyboru w CV: maks. TP (przy FP <= 3), remis: wyzsze PR-AUC, potem mniej FP."""
    m = cand["oof_metrics"]  # type: ignore[index]
    cm = m["confusion_matrix"]  # type: ignore[index]
    return (int(cm["tp"]), float(m["pr_auc"]), -int(cm["fp"]))


def _fold_threshold_spread(cand: Dict[str, object]) -> float:
    """Rozpietosc (max - min) progow wlasnych miedzy foldami (przy max_fp=0 <= 2.5%)."""
    thrs = [
        float(f["fold_own_threshold_max_fp_0"]["threshold"])  # type: ignore[index]
        for f in cand["per_fold"]  # type: ignore[union-attr]
    ]
    return max(thrs) - min(thrs)


def select_winning_and_alternative(
    lr_results: Sequence[Dict[str, object]],
    nb_results: Sequence[Dict[str, object]],
) -> Tuple[Dict[str, object], Dict[str, object], Dict[str, object]]:
    """Wybiera najlepsza regresje, najlepszego Naiwnego Bayesa oraz zwyciezce wg planu."""
    best_lr = max(lr_results, key=_candidate_rank_key)
    best_nb = max(nb_results, key=_candidate_rank_key)

    lr_cm = best_lr["oof_metrics"]["confusion_matrix"]  # type: ignore[index]
    nb_cm = best_nb["oof_metrics"]["confusion_matrix"]  # type: ignore[index]
    lr_tp, lr_fp = int(lr_cm["tp"]), int(lr_cm["fp"])
    nb_tp, nb_fp = int(nb_cm["tp"]), int(nb_cm["fp"])
    lr_ap = float(best_lr["oof_metrics"]["pr_auc"])  # type: ignore[index]
    nb_ap = float(best_nb["oof_metrics"]["pr_auc"])  # type: ignore[index]
    lr_spread = _fold_threshold_spread(best_lr)
    nb_spread = _fold_threshold_spread(best_nb)

    # Doprecyzowana regula wyboru: Naiwny Bayes wygrywa tylko gdy wykrywa >= 3 spamy wiecej
    # (>= 5 pp czulosci) przy <= tej samej liczbie FP albo ma wyraznie stabilniejszy prog
    # miedzy foldami (rozpietosc mniejsza o co najmniej polowe przy co najmniej tej samej czulosci).
    nb_clearly_more_recall = (nb_tp >= lr_tp + 3) and (nb_fp <= lr_fp)
    nb_clearly_more_stable = (
        nb_tp >= lr_tp and nb_fp <= lr_fp and nb_spread <= 0.5 * lr_spread
    )
    if nb_clearly_more_recall or nb_clearly_more_stable:
        winner = best_nb
        alternative = best_lr
        reason = (
            f"Wybrano Naiwnego Bayesa ({nb_tp}/60 TP, {nb_fp}/120 FP, PR-AUC={nb_ap:.4f}, "
            f"rozrzut progu={nb_spread:.4f}) zamiast regresji ({lr_tp}/60 TP, {lr_fp}/120 FP, "
            f"PR-AUC={lr_ap:.4f}, rozrzut progu={lr_spread:.4f}) zgodnie z regula wyboru."
        )
    else:
        winner = best_lr
        alternative = best_nb
        reason = (
            f"Wybrano model glowny (regresja logistyczna L2: {lr_tp}/60 TP, {lr_fp}/120 FP, "
            f"PR-AUC={lr_ap:.4f}, rozrzut progu={lr_spread:.4f}) wobec Naiwnego Bayesa "
            f"({nb_tp}/60 TP, {nb_fp}/120 FP, PR-AUC={nb_ap:.4f}, rozrzut progu={nb_spread:.4f}); "
            f"roznica nie spelnia progu przelaczenia (>= 3 spamy przy <= FP lub wyraznie stabilniejszy prog)."
        )

    borderline = (nb_tp > lr_tp and not nb_clearly_more_recall) or (
        abs(nb_tp - lr_tp) <= 2 and nb_ap > lr_ap + 0.01
    )

    decision_info = {
        "winner_model_type": winner["model_type"],
        "winner_param": {winner["param_name"]: winner["param_value"]},
        "winner_threshold": winner["threshold"],
        "winner_threshold_spread_max_fp_0": _fold_threshold_spread(winner),
        "alternative_model_type": alternative["model_type"],
        "alternative_param": {alternative["param_name"]: alternative["param_value"]},
        "alternative_threshold": alternative["threshold"],
        "alternative_threshold_spread_max_fp_0": _fold_threshold_spread(alternative),
        "borderline": borderline,
        "reason": reason,
    }
    return winner, alternative, decision_info


def compute_masking_pattern_stats(
    texts: Sequence[str],
    labels: Sequence[str],
) -> List[Dict[str, object]]:
    """Liczy dopasowania kazdego wzorca maskowania na train (osobno i sekwencyjnie)."""
    all_patterns = list(SUBJECT_TAG_PATTERNS) + list(BODY_MASKING_PATTERNS)
    work = [
        re.sub(SUBJECT_PREFIX_PATTERN, "", t, count=1, flags=re.IGNORECASE)
        for t in texts
    ]
    stats: List[Dict[str, object]] = []
    for name, pat in all_patterns:
        rx = re.compile(pat, flags=re.IGNORECASE)
        s_standalone = sum(
            1 for t, lbl in zip(texts, labels) if lbl == "spam" and rx.search(t)
        )
        h_standalone = sum(
            1 for t, lbl in zip(texts, labels) if lbl == "ham" and rx.search(t)
        )

        s_seq = 0
        h_seq = 0
        next_work: List[str] = []
        for t, lbl in zip(work, labels):
            if rx.search(t):
                if lbl == "spam":
                    s_seq += 1
                else:
                    h_seq += 1
                t = rx.sub(" ", t)
            next_work.append(t)
        work = next_work

        stats.append(
            {
                "name": name,
                "pattern": pat,
                "standalone_total": s_standalone + h_standalone,
                "standalone_spam": s_standalone,
                "standalone_ham": h_standalone,
                "sequential_total": s_seq + h_seq,
                "sequential_spam": s_seq,
                "sequential_ham": h_seq,
            }
        )
    return stats


def _extract_context_snippet(masked_text: str, token: str, *, radius: int = 45) -> str:
    """Wyciaga krotki fragment kontekstu dla tokenu ze zmaskowanej wiadomosci train."""
    compact = re.sub(r"\s+", " ", masked_text).strip()
    if token.startswith("<") and token.endswith(">"):
        return compact[:120]
    m = re.search(rf"\b{re.escape(token)}\b", compact, flags=re.IGNORECASE)
    if m is None:
        return compact[:120]
    start = max(0, m.start() - radius)
    end = min(len(compact), m.end() + radius)
    prefix = "..." if start > 0 else ""
    suffix = "..." if end < len(compact) else ""
    return f"{prefix}{compact[start:end]}{suffix}"


def find_single_class_tokens(
    mids: Sequence[str],
    raw_texts: Sequence[str],
    tokenized_docs: Sequence[Sequence[str]],
    labels: Sequence[str],
    vectorizer: TfidfVectorizer,
    *,
    min_docs: int = 8,
    apply_masking: bool = True,
) -> List[Dict[str, object]]:
    """Wyszukuje tokeny ze slownika train wystepujace w >= min_docs wiadomosciach tylko jednej klasy."""
    masked_texts = [
        mask_source_artifacts(t) if apply_masking else t for t in raw_texts
    ]
    spam_docs: Dict[str, List[int]] = {}
    ham_docs: Dict[str, List[int]] = {}

    for doc_idx, (toks, lbl) in enumerate(zip(tokenized_docs, labels)):
        for tok in set(toks):
            if tok not in vectorizer.vocab:
                continue
            if lbl == "spam":
                spam_docs.setdefault(tok, []).append(doc_idx)
            else:
                ham_docs.setdefault(tok, []).append(doc_idx)

    out: List[Dict[str, object]] = []
    for tok in sorted(vectorizer.vocab.keys()):
        s_idxs = spam_docs.get(tok, [])
        h_idxs = ham_docs.get(tok, [])
        if len(s_idxs) >= min_docs and len(h_idxs) == 0:
            only_class = "spam"
            idxs = s_idxs
        elif len(h_idxs) >= min_docs and len(s_idxs) == 0:
            only_class = "ham"
            idxs = h_idxs
        else:
            continue

        examples = [
            {
                "message_id": mids[i],
                "context": _extract_context_snippet(masked_texts[i], tok),
            }
            for i in idxs[:2]
        ]
        out.append(
            {
                "token": tok,
                "only_class": only_class,
                "doc_count": len(idxs),
                "examples": examples,
            }
        )

    out.sort(key=lambda item: (-int(item["doc_count"]), str(item["only_class"]), str(item["token"])))
    return out


def run_cv_and_train(
    project_dir: str | Path,
    *,
    apply_masking: bool = True,
    save_outputs: bool = True,
) -> Dict[str, object]:
    """Uruchamia pelna Czesc 3 na project/data/train.csv (bez otwierania validation.csv)."""
    proj = Path(project_dir)
    train_csv = proj / "data" / "train.csv"
    mids, texts, labels = load_csv(train_csv, include_label=True)
    assert labels is not None
    y = labels_to_binary(labels)

    tokenized = tokenize_corpus(texts, apply_masking=apply_masking)
    folds = make_stratified_folds(y, n_splits=CV_FOLDS, seed=CV_SEED)

    lr_results = [
        evaluate_candidate_cv(
            tokenized,
            y,
            folds,
            model_type="logistic_regression",
            param_name="C",
            param_value=c_val,
            min_df=MIN_DF,
            max_fp=MAX_FP_OOF,
        )
        for c_val in LR_GRID
    ]
    nb_results = [
        evaluate_candidate_cv(
            tokenized,
            y,
            folds,
            model_type="multinomial_nb",
            param_name="alpha",
            param_value=a_val,
            min_df=MIN_DF,
            max_fp=MAX_FP_OOF,
        )
        for a_val in NB_GRID
    ]

    baselines = evaluate_baselines_on_train(tokenized, y, folds)
    winner_cv, alt_cv, decision_info = select_winning_and_alternative(
        lr_results, nb_results
    )

    # Uczenie koncowe na calym train.csv (180 wiadomosci)
    full_vec = TfidfVectorizer(min_df=MIN_DF).fit(tokenized)
    X_full = full_vec.transform(tokenized)

    winner_model = _fit_model(
        str(winner_cv["model_type"]),
        float(winner_cv["param_value"]),
        X_full,
        y,
        full_vec.vocab_size,
    )
    alt_model = _fit_model(
        str(alt_cv["model_type"]),
        float(alt_cv["param_value"]),
        X_full,
        y,
        full_vec.vocab_size,
    )

    winner_artifact = SpamClassifierArtifact(
        vectorizer=full_vec,
        model=winner_model,
        threshold=float(winner_cv["threshold"]),
        apply_masking=apply_masking,
    )
    alt_artifact = SpamClassifierArtifact(
        vectorizer=full_vec,
        model=alt_model,
        threshold=float(alt_cv["threshold"]),
        apply_masking=apply_masking,
    )

    def strip_scores(cand: Dict[str, object]) -> Dict[str, object]:
        return {k: v for k, v in cand.items() if k != "oof_scores"}

    cv_payload: Dict[str, object] = {
        "cv_config": {
            "n_splits": CV_FOLDS,
            "seed": CV_SEED,
            "max_fp_oof": MAX_FP_OOF,
            "min_df": MIN_DF,
            "apply_masking": apply_masking,
            "lr_grid": list(LR_GRID),
            "nb_grid": list(NB_GRID),
        },
        "train_summary": {
            "n_total": len(y),
            "n_ham": y.count(0),
            "n_spam": y.count(1),
            "full_vocab_size": full_vec.vocab_size,
        },
        "decision": decision_info,
        "logistic_regression_candidates": [strip_scores(c) for c in lr_results],
        "multinomial_nb_candidates": [strip_scores(c) for c in nb_results],
        "baselines_on_train": baselines,
        "winner_oof_predictions": [
            {
                "message_id": mid,
                "label": lbl,
                "oof_score": sc,
            }
            for mid, lbl, sc in zip(mids, labels, winner_cv["oof_scores"])  # type: ignore[arg-type]
        ],
    }

    weights_payload: Dict[str, object] = {
        "winner_model_type": winner_cv["model_type"],
        "winner_param": {winner_cv["param_name"]: winner_cv["param_value"]},
        "winner_top_30": winner_artifact.top_features(top_k=30),
        "alternative_model_type": alt_cv["model_type"],
        "alternative_param": {alt_cv["param_name"]: alt_cv["param_value"]},
        "alternative_top_30": alt_artifact.top_features(top_k=30),
        "masking_pattern_stats": compute_masking_pattern_stats(texts, labels),
        "single_class_tokens_ge_8": find_single_class_tokens(
            mids,
            texts,
            tokenized,
            labels,
            full_vec,
            min_docs=8,
            apply_masking=apply_masking,
        ),
    }

    if save_outputs:
        models_dir = proj / "models"
        results_dir = proj / "results"
        models_dir.mkdir(parents=True, exist_ok=True)
        results_dir.mkdir(parents=True, exist_ok=True)

        winner_artifact.save_json(models_dir / "spam_classifier.json")
        alt_artifact.save_json(models_dir / "alternative_model.json")

        (results_dir / "cv_results.json").write_text(
            json.dumps(cv_payload, indent=2, sort_keys=True, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )
        (results_dir / "train_top_weights.json").write_text(
            json.dumps(weights_payload, indent=2, sort_keys=True, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )

    return {
        "cv_results": cv_payload,
        "train_top_weights": weights_payload,
        "winner_artifact": winner_artifact,
        "alternative_artifact": alt_artifact,
    }


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description="5-krotna CV na train.csv, wybor modelu i uczenie koncowe."
    )
    parser.add_argument(
        "--project-dir",
        default=str(Path(__file__).resolve().parents[1]),
        help="Sciezka do folderu project/.",
    )
    args = parser.parse_args(argv)

    t0 = time.monotonic()
    out = run_cv_and_train(args.project_dir, apply_masking=True, save_outputs=True)
    elapsed = time.monotonic() - t0
    dec = out["cv_results"]["decision"]  # type: ignore[index]
    vocab_sz = out["cv_results"]["train_summary"]["full_vocab_size"]  # type: ignore[index]
    print(
        f"CV + uczenie zakonczone w {elapsed:.2f}s | slownik={vocab_sz} | "
        f"wybrany={dec['winner_model_type']} {dec['winner_param']} | "
        f"prog={dec['winner_threshold']:.6f}"
    )


if __name__ == "__main__":
    main()
