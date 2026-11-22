"""Czesc 4: Niezalezny skrypt przewidujacy korzystajacy wylacznie z zapisanego pliku JSON.

Skrypt NIGDY nie odczytuje kolumny 'label' (ani nie wymaga jej obecnosci w CSV).
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import List, Sequence, Tuple

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.data_io import load_csv
from src.models import SpamClassifierArtifact


def predict_csv_file(
    *,
    model_path: str | Path,
    input_csv: str | Path,
    output_csv: str | Path,
) -> List[Tuple[str, float, str]]:
    """Klasyfikuje wiadomosci z pliku CSV (bez czytania 'label') i zapisuje wynik CSV."""
    artifact = SpamClassifierArtifact.load_json(model_path)
    mids, texts, _ = load_csv(input_csv, include_label=False)
    preds = artifact.predict_texts(texts)

    out_path = Path(output_csv)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    rows: List[Tuple[str, float, str]] = []
    with open(out_path, "w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f, lineterminator="\n")
        writer.writerow(["message_id", "score", "predicted_label"])
        for mid, (score, pred_label) in zip(mids, preds):
            writer.writerow([mid, repr(float(score)), pred_label])
            rows.append((mid, float(score), pred_label))
    return rows


def predict_single_text(
    *,
    model_path: str | Path,
    text: str,
    message_id: str | None = None,
) -> dict[str, object]:
    """Klasyfikuje pojedynczy tekst wiadomosci bez ponownego uczenia."""
    artifact = SpamClassifierArtifact.load_json(model_path)
    score, pred_label = artifact.predict_one(text)
    return {
        "message_id": message_id,
        "score": float(score),
        "predicted_label": pred_label,
        "threshold": artifact.threshold,
    }


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description="Przewidywanie spamu z zapisanego pliku JSON (bez czytania kolumny label)."
    )
    parser.add_argument(
        "--model",
        required=True,
        help="Sciezka do zapisanego pliku JSON modelu (np. models/spam_classifier.json).",
    )
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument(
        "--input-csv",
        help="Sciezka do pliku CSV z kolumnami message_id,text (kolumna label nie jest czytana).",
    )
    group.add_argument(
        "--text",
        help="Pojedynczy tekst wiadomosci do klasyfikacji.",
    )
    group.add_argument(
        "--stdin",
        action="store_true",
        help="Wczytaj pojedynczy tekst wiadomosci ze standardowego wejscia.",
    )
    parser.add_argument(
        "--output-csv",
        help="Sciezka wyjsciowa CSV (wymagana dla --input-csv).",
    )
    parser.add_argument(
        "--message-id",
        default=None,
        help="Opcjonalny identyfikator wiadomosci dla trybu --text / --stdin.",
    )
    args = parser.parse_args(argv)

    if args.input_csv:
        if not args.output_csv:
            parser.error("Dla --input-csv wymagany jest parametr --output-csv.")
        rows = predict_csv_file(
            model_path=args.model,
            input_csv=args.input_csv,
            output_csv=args.output_csv,
        )
        n_spam = sum(1 for _, _, lbl in rows if lbl == "spam")
        print(
            f"Zapisano {len(rows)} przewidywan do {args.output_csv} "
            f"(spam={n_spam}, ham={len(rows) - n_spam})"
        )
    else:
        raw_text = sys.stdin.read() if args.stdin else str(args.text)
        result = predict_single_text(
            model_path=args.model,
            text=raw_text,
            message_id=args.message_id,
        )
        print(json.dumps(result, indent=2, sort_keys=True, ensure_ascii=False))


if __name__ == "__main__":
    main()
