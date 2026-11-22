"""Odczyt danych CSV i weryfikacja integralnosci względem manifest.json."""

from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path
from typing import Dict, List, Tuple


VALID_LABELS = {"spam", "ham"}


def compute_sha256(path: str | Path) -> str:
    """Oblicza skrot SHA-256 pliku."""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def load_csv(
    path: str | Path,
    *,
    include_label: bool = True,
) -> Tuple[List[str], List[str], List[str] | None]:
    """Wczytuje plik CSV z wiadomosciami i rozdziela kolumny wedlug roli.

    Gdy include_label=False, kolumna 'label' nie jest odczytywana ani wymagana.
    """
    message_ids: List[str] = []
    texts: List[str] = []
    labels: List[str] | None = [] if include_label else None

    with open(path, "r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        if reader.fieldnames is None:
            raise ValueError(f"Plik CSV jest pusty lub brak naglowka: {path}")
        if "message_id" not in reader.fieldnames or "text" not in reader.fieldnames:
            raise ValueError(
                f"Plik CSV musi zawierac kolumny 'message_id' i 'text': {path}"
            )
        if include_label and "label" not in reader.fieldnames:
            raise ValueError(f"Plik CSV musi zawierac kolumne 'label': {path}")

        for idx, row in enumerate(reader, start=1):
            mid = row["message_id"]
            txt = row["text"]
            if mid is None or txt is None:
                raise ValueError(f"Brakujace pole w wierszu {idx} pliku {path}")
            message_ids.append(mid)
            texts.append(txt)
            if include_label and labels is not None:
                lbl = row["label"]
                if lbl not in VALID_LABELS:
                    raise ValueError(
                        f"Nieznana etykieta {lbl!r} w wierszu {idx} pliku {path}"
                    )
                labels.append(lbl)

    return message_ids, texts, labels


def load_label_map(path: str | Path) -> Dict[str, str]:
    """Wczytuje mapowanie message_id -> label (do oceny po message_id)."""
    mids, _, labels = load_csv(path, include_label=True)
    assert labels is not None
    if len(set(mids)) != len(mids):
        raise ValueError(f"Zduplikowane message_id w pliku {path}")
    return dict(zip(mids, labels))


def verify_data_manifest(data_dir: str | Path) -> Dict[str, Dict[str, object]]:
    """Weryfikuje SHA-256 i licznosci train.csv oraz validation.csv z manifest.json."""
    data_path = Path(data_dir)
    manifest_path = data_path / "manifest.json"
    with open(manifest_path, "r", encoding="utf-8") as f:
        manifest = json.load(f)

    files_spec = manifest.get("files", {})
    report: Dict[str, Dict[str, object]] = {}

    for filename, expected in files_spec.items():
        file_path = data_path / filename
        actual_sha = compute_sha256(file_path)
        mids, _, labels = load_csv(file_path, include_label=True)
        assert labels is not None
        n_rows = len(mids)
        n_ham = sum(1 for lbl in labels if lbl == "ham")
        n_spam = sum(1 for lbl in labels if lbl == "spam")

        sha_ok = actual_sha == expected["sha256"]
        counts_ok = (
            n_rows == expected["rows"]
            and n_ham == expected["ham"]
            and n_spam == expected["spam"]
        )
        if not (sha_ok and counts_ok):
            raise ValueError(
                f"Niezgodnosc danych dla {filename}: "
                f"sha_ok={sha_ok}, counts_ok={counts_ok}"
            )
        report[filename] = {
            "sha256": actual_sha,
            "rows": n_rows,
            "ham": n_ham,
            "spam": n_spam,
            "ok": True,
        }

    return report
