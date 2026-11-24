"""Dokladny dwustronny test McNemara (k-NN vs najblizszy centroid) na test.csv.

Niezalezny skrypt w sprawdzenia/ BEZ importow z src/.
Poniewaz wyniki/test_odniesienie_centroid.json nie zawiera przewidywan per obraz,
skrypt niezaleznie dekoduje obrazy PNG (stdlib struct+zlib), wyznacza centroidy
klas 0-9 na data/train.csv, przewiduje na data/test.csv i weryfikuje dokladna
zgodnosc (trafienia i macierz 10x10) z wyniki/test_odniesienie_centroid.json.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import struct
import zlib
from pathlib import Path


def _paeth(a: int, b: int, c: int) -> int:
    p = a + b - c
    pa, pb, pc = abs(p - a), abs(p - b), abs(p - c)
    if pa <= pb and pa <= pc:
        return a
    if pb <= pc:
        return b
    return c


def decode_png_features(path: Path) -> list[float]:
    raw = path.read_bytes()
    if raw[:8] != b"\x89PNG\r\n\x1a\n":
        raise ValueError(f"Nieprawidlowa sygnatura PNG: {path}")
    pos = 8
    idat = bytearray()
    w = h = 0
    while pos < len(raw):
        length = struct.unpack(">I", raw[pos : pos + 4])[0]
        ctype = raw[pos + 4 : pos + 8]
        cdata = raw[pos + 8 : pos + 8 + length]
        crc_expected = struct.unpack(">I", raw[pos + 8 + length : pos + 12 + length])[0]
        crc_actual = zlib.crc32(ctype + cdata) & 0xFFFFFFFF
        if crc_actual != crc_expected:
            raise ValueError(f"Bledne CRC w {path}")
        if ctype == b"IHDR":
            w, h, bit_depth, color_type, comp, filt, interlace = struct.unpack(
                ">IIBBBBB", cdata
            )
            if (w, h, bit_depth, color_type, comp, filt, interlace) != (8, 8, 8, 0, 0, 0, 0):
                raise ValueError(f"Nieobslugiwany format PNG w {path}")
        elif ctype == b"IDAT":
            idat.extend(cdata)
        elif ctype == b"IEND":
            break
        pos += 12 + length

    decomp = zlib.decompress(bytes(idat))
    prev = [0] * w
    pixels: list[float] = []
    stride = 1 + w
    for r in range(h):
        off = r * stride
        ftype = decomp[off]
        row_bytes = decomp[off + 1 : off + stride]
        recon = [0] * w
        for c in range(w):
            x = row_bytes[c]
            a = recon[c - 1] if c > 0 else 0
            b = prev[c]
            ul = prev[c - 1] if c > 0 else 0
            if ftype == 0:
                val = x
            elif ftype == 1:
                val = (x + a) & 0xFF
            elif ftype == 2:
                val = (x + b) & 0xFF
            elif ftype == 3:
                val = (x + ((a + b) >> 1)) & 0xFF
            elif ftype == 4:
                val = (x + _paeth(a, b, ul)) & 0xFF
            else:
                raise ValueError(f"Nieobslugiwany filtr {ftype}")
            recon[c] = val
            pixels.append(val / 255.0)
        prev = recon
    return pixels


def exact_mcnemar_two_sided_p(b: int, c: int) -> float:
    """Dokladny dwustronny test McNemara (rozklad dwumianowy Bin(b+c, 0.5))."""
    n = b + c
    if n == 0:
        return 1.0
    k_min = min(b, c)
    tail_num = sum(math.comb(n, k) for k in range(k_min + 1))
    p_val = (2 * tail_num) / (2**n)
    return min(1.0, p_val)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Dokladny dwustronny test McNemara dla k-NN vs centroid na test.csv."
    )
    parser.add_argument("--train-csv", type=Path, default=Path("data/train.csv"))
    parser.add_argument("--test-csv", type=Path, default=Path("data/test.csv"))
    parser.add_argument("--images-dir", type=Path, default=Path("data"))
    parser.add_argument("--knn-predictions", type=Path, default=Path("wyniki/test_predictions.csv"))
    parser.add_argument("--knn-eval", type=Path, default=Path("wyniki/test_evaluation.json"))
    parser.add_argument(
        "--centroid-ref",
        type=Path,
        default=Path("wyniki/test_odniesienie_centroid.json"),
    )
    parser.add_argument("--output", type=Path, default=Path("wyniki/mcnemar_test.json"))
    args = parser.parse_args()

    # 1. Wczytaj przewidywania k-NN z wyniki/test_predictions.csv
    with args.knn_predictions.open("r", encoding="utf-8", newline="") as handle:
        knn_preds = {r["image_id"]: int(r["predicted_label"]) for r in csv.DictReader(handle)}

    # 2. Niezaleznie wyznacz centroidy klas 0-9 na data/train.csv
    with args.train_csv.open("r", encoding="utf-8", newline="") as handle:
        train_rows = list(csv.DictReader(handle))

    sums = {c: [0.0] * 64 for c in range(10)}
    counts = {c: 0 for c in range(10)}
    for row in train_rows:
        lbl = int(row["label"])
        vec = decode_png_features(args.images_dir / row["image_path"])
        counts[lbl] += 1
        acc = sums[lbl]
        for j in range(64):
            acc[j] += vec[j]
    centroids = {c: [s / counts[c] for s in sums[c]] for c in range(10)}

    # 3. Przewiduj centroidem na data/test.csv i zlicz tabele 2x2 z k-NN
    with args.test_csv.open("r", encoding="utf-8", newline="") as handle:
        test_rows = list(csv.DictReader(handle))

    if set(knn_preds.keys()) != {r["image_id"] for r in test_rows}:
        raise ValueError("Niezgodnosc zbiorow image_id miedzy test_predictions.csv a test.csv.")

    centroid_matrix = [[0] * 10 for _ in range(10)]
    knn_hits = 0
    centroid_hits = 0

    both_correct = 0
    knn_correct_centroid_wrong = 0
    knn_wrong_centroid_correct = 0
    both_wrong = 0

    knn_only_correct_ids: list[dict[str, object]] = []
    centroid_only_correct_ids: list[dict[str, object]] = []

    for row in test_rows:
        img_id = row["image_id"]
        y_true = int(row["label"])
        vec = decode_png_features(args.images_dir / row["image_path"])

        best_c = 0
        best_d = sum((u - v) * (u - v) for u, v in zip(vec, centroids[0]))
        for c in range(1, 10):
            d = sum((u - v) * (u - v) for u, v in zip(vec, centroids[c]))
            if d < best_d:
                best_d = d
                best_c = c

        centroid_matrix[y_true][best_c] += 1
        knn_pred = knn_preds[img_id]
        knn_ok = knn_pred == y_true
        cent_ok = best_c == y_true

        if knn_ok:
            knn_hits += 1
        if cent_ok:
            centroid_hits += 1

        if knn_ok and cent_ok:
            both_correct += 1
        elif knn_ok and not cent_ok:
            knn_correct_centroid_wrong += 1
            knn_only_correct_ids.append(
                {
                    "centroid_pred": best_c,
                    "image_id": img_id,
                    "knn_pred": knn_pred,
                    "true_label": y_true,
                }
            )
        elif not knn_ok and cent_ok:
            knn_wrong_centroid_correct += 1
            centroid_only_correct_ids.append(
                {
                    "centroid_pred": best_c,
                    "image_id": img_id,
                    "knn_pred": knn_pred,
                    "true_label": y_true,
                }
            )
        else:
            both_wrong += 1

    # 4. Weryfikacja zgodnosci z istniejacymi plikami w wyniki/
    centroid_ref = json.loads(args.centroid_ref.read_text(encoding="utf-8"))
    knn_eval = json.loads(args.knn_eval.read_text(encoding="utf-8"))

    centroid_matrix_matches = centroid_matrix == centroid_ref["confusion_matrix"]
    centroid_hits_matches = centroid_hits == centroid_ref["hits"]
    knn_hits_matches = knn_hits == knn_eval["hits"]
    if not (centroid_matrix_matches and centroid_hits_matches and knn_hits_matches):
        raise RuntimeError(
            "Niezgodnosc przeliczenia z plikami referencyjnymi: "
            f"centroid_matrix={centroid_matrix_matches}, "
            f"centroid_hits={centroid_hits_matches}, knn_hits={knn_hits_matches}"
        )

    p_exact = exact_mcnemar_two_sided_p(
        knn_correct_centroid_wrong, knn_wrong_centroid_correct
    )

    report = {
        "centroid_predictions_source": (
            "test_odniesienie_centroid.json nie zawiera przewidywan per obraz; "
            "przewidywania centroidu (L2 na pixel/255) wyznaczono niezaleznie bez importu z src/ "
            "na data/train.csv -> data/test.csv i zweryfikowano pelna zgodnosc macierzy 10x10 "
            "oraz liczby trafien z wyniki/test_odniesienie_centroid.json."
        ),
        "contingency_table": {
            "both_correct": both_correct,
            "both_wrong": both_wrong,
            "knn_correct_centroid_wrong": knn_correct_centroid_wrong,
            "knn_wrong_centroid_correct": knn_wrong_centroid_correct,
        },
        "cross_checks": {
            "centroid_hits_matches_ref": centroid_hits_matches,
            "centroid_matrix_matches_ref": centroid_matrix_matches,
            "knn_hits_matches_eval": knn_hits_matches,
        },
        "discordant_cases": {
            "centroid_only_correct": centroid_only_correct_ids,
            "knn_only_correct": knn_only_correct_ids,
        },
        "mcnemar_exact_two_sided": {
            "b_knn_correct_centroid_wrong": knn_correct_centroid_wrong,
            "c_knn_wrong_centroid_correct": knn_wrong_centroid_correct,
            "discordant_total": knn_correct_centroid_wrong + knn_wrong_centroid_correct,
            "p_value": p_exact,
        },
        "summary": {
            "centroid_hits": centroid_hits,
            "knn_hits": knn_hits,
            "total": len(test_rows),
        },
    }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(report, sort_keys=True, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
        newline="\n",
    )

    print(
        f"McNemar (dokladny dwustronny): "
        f"k-NN dobrze & centroid zle = {knn_correct_centroid_wrong}, "
        f"k-NN zle & centroid dobrze = {knn_wrong_centroid_correct}, "
        f"razem niezgodnych = {knn_correct_centroid_wrong + knn_wrong_centroid_correct}, "
        f"p = {p_exact:.6f}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
