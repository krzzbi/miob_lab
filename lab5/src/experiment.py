"""Eksperyment wyboru modelu na train.csv (5-fold CV) i validation.csv.

Nie otwiera i nie uzywa test.csv ani obrazow testowych.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any, Callable

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.data_io import load_dataset, make_stratified_folds
from src.models import KNNModel, NearestCentroidModel, SoftmaxRegressionModel

KNN_KS = [1, 3, 5, 7]
KNN_METRICS = ["l2", "l1"]
SOFTMAX_LAMBDAS = [0.0, 1e-4, 1e-3, 1e-2, 1e-1]
CV_SEED = 20261114
CV_FOLDS = 5


def run_cv(
    model_factory: Callable[[], NearestCentroidModel | KNNModel | SoftmaxRegressionModel],
    X_train: list[list[float]],
    y_train: list[int],
    folds: list[list[int]],
) -> dict[str, Any]:
    fold_hits: list[int] = []
    fold_sizes: list[int] = []
    fold_training_info: list[dict[str, Any]] = []
    n_total = len(X_train)

    for fold_val_indices in folds:
        val_set = set(fold_val_indices)
        tr_indices = [i for i in range(n_total) if i not in val_set]
        X_tr = [X_train[i] for i in tr_indices]
        y_tr = [y_train[i] for i in tr_indices]
        X_va = [X_train[i] for i in fold_val_indices]
        y_va = [y_train[i] for i in fold_val_indices]

        model = model_factory()
        model.fit(X_tr, y_tr)
        preds = model.predict(X_va)
        hits = sum(1 for p, y in zip(preds, y_va) if p == y)
        fold_hits.append(hits)
        fold_sizes.append(len(y_va))
        if isinstance(model, SoftmaxRegressionModel):
            fold_training_info.append(dict(model.training_info))

    total_hits = sum(fold_hits)
    total_samples = sum(fold_sizes)
    fold_accs = [h / s for h, s in zip(fold_hits, fold_sizes)]
    mean_acc = total_hits / total_samples

    result: dict[str, Any] = {
        "fold_accuracies": fold_accs,
        "fold_hits": fold_hits,
        "fold_sizes": fold_sizes,
        "max_accuracy": max(fold_accs),
        "max_hits": max(fold_hits),
        "mean_accuracy": mean_acc,
        "min_accuracy": min(fold_accs),
        "min_hits": min(fold_hits),
        "n_folds": len(folds),
        "seed": CV_SEED,
        "total_hits": total_hits,
        "total_samples": total_samples,
    }
    if fold_training_info:
        result["fold_training_info"] = fold_training_info
    return result


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Porownanie punktu odniesienia oraz kandydatow A (k-NN) i B (softmax)."
    )
    parser.add_argument("--train-csv", type=Path, default=Path("data/train.csv"))
    parser.add_argument("--val-csv", type=Path, default=Path("data/validation.csv"))
    parser.add_argument("--images-dir", type=Path, default=Path("data"))
    parser.add_argument("--learning-rate", type=float, required=True)
    parser.add_argument("--max-epochs", type=int, required=True)
    parser.add_argument("--rel-tol", type=float, required=True)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("wyniki/eksperymenty_validation.json"),
    )
    args = parser.parse_args(argv)

    X_train, y_train, _, _ = load_dataset(args.train_csv, args.images_dir)
    X_val, y_val, _, _ = load_dataset(args.val_csv, args.images_dir)
    folds = make_stratified_folds(y_train, n_folds=CV_FOLDS, seed=CV_SEED)

    # 1. Punkt odniesienia: najblizszy centroid (L2)
    t0 = time.perf_counter()
    centroid = NearestCentroidModel().fit(X_train, y_train)
    t_fit = time.perf_counter() - t0
    t0 = time.perf_counter()
    centroid_val_preds = centroid.predict(X_val)
    t_pred = time.perf_counter() - t0
    centroid_val_hits = sum(p == y for p, y in zip(centroid_val_preds, y_val))
    centroid_cv = run_cv(NearestCentroidModel, X_train, y_train, folds)
    baseline_result = {
        "cv": centroid_cv,
        "model_type": "nearest_centroid",
        "settings": {"metric": "l2"},
        "train_time_s": t_fit,
        "val_hits": centroid_val_hits,
        "val_predict_time_s": t_pred,
        "val_total": len(y_val),
    }

    # 2. Kandydat A: k-NN (8 konfiguracji)
    knn_results: list[dict[str, Any]] = []
    for metric in KNN_METRICS:
        for k in KNN_KS:
            t0 = time.perf_counter()
            model = KNNModel(k=k, metric=metric).fit(X_train, y_train)
            t_fit = time.perf_counter() - t0
            t0 = time.perf_counter()
            preds, tie_stats = model.predict_with_tie_stats(X_val)
            t_pred = time.perf_counter() - t0
            val_hits = sum(p == y for p, y in zip(preds, y_val))
            knn_results.append(
                {
                    "cv": None,
                    "model_type": "knn",
                    "settings": {"k": k, "metric": metric, "weights": "uniform"},
                    "tie_stats_validation": tie_stats,
                    "train_time_s": t_fit,
                    "val_hits": val_hits,
                    "val_predict_time_s": t_pred,
                    "val_total": len(y_val),
                }
            )

    # 3. Kandydat B: regresja softmax z kara L2 (5 konfiguracji)
    softmax_results: list[dict[str, Any]] = []
    for lam in SOFTMAX_LAMBDAS:
        t0 = time.perf_counter()
        model = SoftmaxRegressionModel(
            reg_lambda=lam,
            learning_rate=args.learning_rate,
            max_epochs=args.max_epochs,
            rel_tol=args.rel_tol,
        ).fit(X_train, y_train)
        t_fit = time.perf_counter() - t0
        t0 = time.perf_counter()
        preds = model.predict(X_val)
        t_pred = time.perf_counter() - t0
        val_hits = sum(p == y for p, y in zip(preds, y_val))
        softmax_results.append(
            {
                "cv": None,
                "model_type": "softmax",
                "settings": {
                    "learning_rate": args.learning_rate,
                    "max_epochs": args.max_epochs,
                    "reg_lambda": lam,
                    "rel_tol": args.rel_tol,
                },
                "train_time_s": t_fit,
                "training_info": dict(model.training_info),
                "val_hits": val_hits,
                "val_predict_time_s": t_pred,
                "val_total": len(y_val),
            }
        )

    # 4. Kryterium 7a w obrebie A
    max_val_a = max(r["val_hits"] for r in knn_results)
    top_a = [r for r in knn_results if r["val_hits"] == max_val_a]
    for r in top_a:
        k_val = int(r["settings"]["k"])
        m_val = str(r["settings"]["metric"])
        r["cv"] = run_cv(
            lambda k=k_val, m=m_val: KNNModel(k=k, metric=m),
            X_train,
            y_train,
            folds,
        )
    top_a_sorted = sorted(
        top_a,
        key=lambda r: (
            -int(r["val_hits"]),
            -int(r["cv"]["total_hits"]),
            -int(r["settings"]["k"]),
        ),
    )
    best_a = top_a_sorted[0]
    unresolved_tie_a = (
        len(top_a_sorted) > 1
        and top_a_sorted[0]["val_hits"] == top_a_sorted[1]["val_hits"]
        and top_a_sorted[0]["cv"]["total_hits"] == top_a_sorted[1]["cv"]["total_hits"]
        and top_a_sorted[0]["settings"]["k"] == top_a_sorted[1]["settings"]["k"]
    )

    # 5. Kryterium 7a w obrebie B
    max_val_b = max(r["val_hits"] for r in softmax_results)
    top_b = [r for r in softmax_results if r["val_hits"] == max_val_b]
    for r in top_b:
        lam_val = float(r["settings"]["reg_lambda"])
        r["cv"] = run_cv(
            lambda lam=lam_val: SoftmaxRegressionModel(
                reg_lambda=lam,
                learning_rate=args.learning_rate,
                max_epochs=args.max_epochs,
                rel_tol=args.rel_tol,
            ),
            X_train,
            y_train,
            folds,
        )
    top_b_sorted = sorted(
        top_b,
        key=lambda r: (
            -int(r["val_hits"]),
            -int(r["cv"]["total_hits"]),
            -float(r["settings"]["reg_lambda"]),
        ),
    )
    best_b = top_b_sorted[0]

    # 6. Kryterium 7b miedzy A i B
    a_val = int(best_a["val_hits"])
    b_val = int(best_b["val_hits"])
    a_cv_hits = int(best_a["cv"]["total_hits"])
    b_cv_hits = int(best_b["cv"]["total_hits"])
    a_cv_mean = float(best_a["cv"]["mean_accuracy"])
    b_cv_mean = float(best_b["cv"]["mean_accuracy"])

    a_clear_adv = (a_val - b_val >= 3) and (a_cv_hits >= b_cv_hits)
    b_clear_adv = (b_val - a_val >= 3) and (b_cv_hits >= a_cv_hits)
    cv_diff_hits = abs(a_cv_hits - b_cv_hits)
    cv_diff_pp = abs(a_cv_mean - b_cv_mean) * 100.0

    if a_clear_adv:
        selected = best_a
        decision_rule = "7b_clear_advantage_A"
    elif b_clear_adv:
        selected = best_b
        decision_rule = "7b_clear_advantage_B"
    elif cv_diff_hits < 3:
        # 3/600 = dokladnie 0.50 pp; < 0.5 pp odpowiada roznicy 0, 1 lub 2 trafien na 600
        selected = best_a
        decision_rule = "7b_no_clear_advantage_cv_diff_below_0.5pp_choose_knn"
    elif a_cv_hits > b_cv_hits:
        selected = best_a
        decision_rule = "7b_no_clear_advantage_higher_cv_A"
    else:
        selected = best_b
        decision_rule = "7b_no_clear_advantage_higher_cv_B"

    # 7. Kryterium 7c: wyzsze trafienia na validation niz punkt odniesienia
    beats_baseline = int(selected["val_hits"]) > int(baseline_result["val_hits"])

    summary = {
        "baseline": baseline_result,
        "candidate_a_knn": knn_results,
        "candidate_b_softmax": softmax_results,
        "selection": {
            "beats_baseline_7c": beats_baseline,
            "best_a": {
                "cv_mean_accuracy": a_cv_mean,
                "cv_total_hits": a_cv_hits,
                "settings": best_a["settings"],
                "val_hits": a_val,
            },
            "best_b": {
                "cv_mean_accuracy": b_cv_mean,
                "cv_total_hits": b_cv_hits,
                "settings": best_b["settings"],
                "val_hits": b_val,
            },
            "clear_advantage_a": a_clear_adv,
            "clear_advantage_b": b_clear_adv,
            "cv_diff_hits": cv_diff_hits,
            "cv_diff_pp": cv_diff_pp,
            "decision_rule": decision_rule,
            "selected_model_type": selected["model_type"],
            "selected_settings": selected["settings"],
            "selected_val_hits": selected["val_hits"],
            "unresolved_tie_a": unresolved_tie_a,
        },
    }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(summary, sort_keys=True, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
        newline="\n",
    )

    print(
        f"Centroid (baseline): val={baseline_result['val_hits']}/200, "
        f"CV={centroid_cv['mean_accuracy']*100:.2f}% "
        f"({centroid_cv['min_hits']}-{centroid_cv['max_hits']}/120)"
    )
    print(
        f"Best A (k-NN): {best_a['settings']} -> val={a_val}/200, "
        f"CV={a_cv_mean*100:.2f}% ({a_cv_hits}/600)"
    )
    print(
        f"Best B (softmax): {best_b['settings']} -> val={b_val}/200, "
        f"CV={b_cv_mean*100:.2f}% ({b_cv_hits}/600)"
    )
    print(
        f"Wybor: {selected['model_type']} {selected['settings']} "
        f"(regula: {decision_rule}, 7c_ok={beats_baseline})"
    )

    if unresolved_tie_a:
        print("UWAGA: nierozstrzygniety remis miedzy metrykami dla tego samego k w A!")
        return 3
    if not beats_baseline:
        print("STOP (7c): wybrany kandydat nie przewyzsza punktu odniesienia na validation!")
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
