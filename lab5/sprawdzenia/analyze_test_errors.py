"""Analiza bledow na tescie (D6) i zbiorcza wizualizacja PNG (D7, czysty stdlib).

Wczytuje zamrozony model JSON, wyniki/test_evaluation.json, data/train.csv
oraz obrazy PNG; zapisuje wyniki/bledy_test.json i wyniki/bledy_test.png.
"""

from __future__ import annotations

import argparse
import json
import math
import struct
import sys
import zlib
from pathlib import Path
from typing import Any

LAB5_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(LAB5_DIR))

from src.data_io import load_labeled_csv
from src.models import KNNModel, load_model
from src.png_decoder import decode_png_file

Z_95 = 1.959963984540054

# Krotkie opisy tego, co widac na 5 blednie rozpoznanych obrazach 8x8
# (na podstawie inspekcji macierzy 8x8 i wyniki/bledy_test.png; bez zgadywania intencji).
DEFAULT_ERROR_DESCRIPTIONS: dict[str, str] = {
    "digit-dbf4f7702a6f": (
        "Widoczna gorna pozioma belka i smukly pionowy/lekko skosny trzon schodzacy w dol "
        "bez wyraźnego dolnego lewego zawiniecia. NN1 (klasa 3, L2=0.76) ma niemal identyczny ksztalt; "
        "[przypuszczenie] rowne glosowanie w k=3 pozwolilo dwom ponad 2x dalszym sasiadom klasy 8 "
        "(L2=1.56 i 1.65) przeglosowac najblizszego sasiada."
    ),
    "digit-09e2a05d6bd2": (
        "W gornej polowie widoczne sa dwie jasne galezie (lewa i prawa) tworzace ksztalt zblizony "
        "do gornej petli oraz dluga skosna kreska w lewo w dol. Wszyscy trzej sasiedzi to klasa 9 "
        "w duzej odleglosci (L2=1.82-1.88); [przypuszczenie] jasne piksele po lewej stronie gory znaku "
        "zblizaja go w metryce pikselowej L2 do 9 przy braku bliskiego wzorca takiej 7 w train."
    ),
    "digit-7043c36778f2": (
        "Gorna czesc jest otwarta u gory (brak jasnego laczenia w wierszu 0-1), z mocna pozioma belka "
        "w wierszu 3 i pionowym trzonem po prawej (slabszym w wierszach 5-7). NN1 i NN2 (klasa 4, "
        "L2=1.33 i 1.53) maja bardzo podobny uklad otwartej gory i belki; [przypuszczenie] przy "
        "rozdzielczosci 8x8 brak wyraznego gornego domkniecia petli zbliza obraz do otwartej 4."
    ),
    "digit-4d543bd1092b": (
        "Wyrazna skosna lewa krawedz, mocna pozioma belka w wierszach 4-5 i pionowy trzon w kolumnie 5 "
        "(ciemniejszy u samego dolu). NN1 (klasa 4, L2=1.49) ma zblizona sylwetke, natomiast NN2 i NN3 "
        "(klasa 1, L2=1.73 i 1.80) maja szerokie, skosne zageszczenie pikseli w srodku; "
        "[przypuszczenie] rowne glosy w k=3 przewazyly 2:1 na korzysc dwoch dalszych sasiadow klasy 1."
    ),
    "digit-826cb4bf68f5": (
        "Pionowa jasna kreska w kolumnie 3 od wiersza 0 do 6 oraz szeroka, jasna podstawa u dolu "
        "w wierszach 5-7 (kolumny 2-6). Odleglosci wszystkich 3 sasiadow sa bardzo zblizone "
        "(NN1 klasa 1: 1.66; NN2 i NN3 klasa 6: 1.69 i 1.70); [przypuszczenie] mocna dolna podstawa "
        "w siatce 8x8 pokrywa piksele typowe dla dolnej petli cyfry 6."
    ),
}

# Minimalny font bitmapowy 5x7 (czysty Python, bez zewnetrznych bibliotek)
FONT_5X7: dict[str, list[str]] = {
    "0": ["01110", "10001", "10011", "10101", "11001", "10001", "01110"],
    "1": ["00100", "01100", "00100", "00100", "00100", "00100", "01110"],
    "2": ["01110", "10001", "00001", "00110", "01000", "10000", "11111"],
    "3": ["11110", "00001", "00001", "01110", "00001", "00001", "11110"],
    "4": ["00010", "00110", "01010", "10010", "11111", "00010", "00010"],
    "5": ["11111", "10000", "11110", "00001", "00001", "10001", "01110"],
    "6": ["01110", "10000", "11110", "10001", "10001", "10001", "01110"],
    "7": ["11111", "00001", "00010", "00100", "01000", "01000", "01000"],
    "8": ["01110", "10001", "10001", "01110", "10001", "10001", "01110"],
    "9": ["01110", "10001", "10001", "01111", "00001", "00001", "01110"],
    "A": ["01110", "10001", "10001", "11111", "10001", "10001", "10001"],
    "B": ["11110", "10001", "10001", "11110", "10001", "10001", "11110"],
    "C": ["01110", "10001", "10000", "10000", "10000", "10001", "01110"],
    "D": ["11100", "10010", "10001", "10001", "10001", "10010", "11100"],
    "E": ["11111", "10000", "10000", "11110", "10000", "10000", "11111"],
    "F": ["11111", "10000", "10000", "11110", "10000", "10000", "10000"],
    "G": ["01110", "10001", "10000", "10111", "10001", "10001", "01110"],
    "H": ["10001", "10001", "10001", "11111", "10001", "10001", "10001"],
    "I": ["01110", "00100", "00100", "00100", "00100", "00100", "01110"],
    "K": ["10001", "10010", "10100", "11000", "10100", "10010", "10001"],
    "L": ["10000", "10000", "10000", "10000", "10000", "10000", "11111"],
    "M": ["10001", "11011", "10101", "10101", "10001", "10001", "10001"],
    "N": ["10001", "11001", "10101", "10011", "10001", "10001", "10001"],
    "O": ["01110", "10001", "10001", "10001", "10001", "10001", "01110"],
    "P": ["11110", "10001", "10001", "11110", "10000", "10000", "10000"],
    "R": ["11110", "10001", "10001", "11110", "10100", "10010", "10001"],
    "S": ["01111", "10000", "10000", "01110", "00001", "00001", "11110"],
    "T": ["11111", "00100", "00100", "00100", "00100", "00100", "00100"],
    "U": ["10001", "10001", "10001", "10001", "10001", "10001", "01110"],
    "V": ["10001", "10001", "10001", "10001", "10001", "01010", "00100"],
    "W": ["10001", "10001", "10001", "10101", "10101", "11011", "10001"],
    "X": ["10001", "10001", "01010", "00100", "01010", "10001", "10001"],
    "Y": ["10001", "10001", "01010", "00100", "00100", "00100", "00100"],
    "Z": ["11111", "00001", "00010", "00100", "01000", "10000", "11111"],
    " ": ["00000", "00000", "00000", "00000", "00000", "00000", "00000"],
    ":": ["00000", "00100", "00100", "00000", "00100", "00100", "00000"],
    ".": ["00000", "00000", "00000", "00000", "00000", "01100", "01100"],
    "-": ["00000", "00000", "00000", "11111", "00000", "00000", "00000"],
    ">": ["10000", "01000", "00100", "00010", "00100", "01000", "10000"],
    "#": ["01010", "11111", "01010", "01010", "11111", "01010", "00000"],
    "(": ["00010", "00100", "01000", "01000", "01000", "00100", "00010"],
    ")": ["01000", "00100", "00010", "00010", "00010", "00100", "01000"],
    "/": ["00001", "00010", "00010", "00100", "01000", "01000", "10000"],
}


def wilson_ci_95(hits: int, total: int) -> dict[str, float]:
    p = hits / total
    z2 = Z_95 * Z_95
    denom = 1.0 + z2 / total
    center = (p + z2 / (2.0 * total)) / denom
    half = (Z_95 * math.sqrt((p * (1.0 - p) + z2 / (4.0 * total)) / total)) / denom
    return {
        "center": center,
        "lower": max(0.0, center - half),
        "upper": min(1.0, center + half),
        "z": Z_95,
    }


def encode_gray8_png(canvas: list[bytearray]) -> bytes:
    """Koder 8-bitowego PNG w skali szarosci (color_type=0, filter=0) w stdlib."""
    height = len(canvas)
    width = len(canvas[0])
    ihdr = struct.pack(">IIBBBBB", width, height, 8, 0, 0, 0, 0)
    raw = bytearray()
    for row in canvas:
        raw.append(0)  # filtr wiersza 0 (None)
        raw.extend(row)
    compressed = zlib.compress(bytes(raw), level=9)

    def make_chunk(ctype: bytes, data: bytes) -> bytes:
        crc = zlib.crc32(ctype + data) & 0xFFFFFFFF
        return struct.pack(">I", len(data)) + ctype + data + struct.pack(">I", crc)

    return (
        b"\x89PNG\r\n\x1a\n"
        + make_chunk(b"IHDR", ihdr)
        + make_chunk(b"IDAT", compressed)
        + make_chunk(b"IEND", b"")
    )


def draw_text(
    canvas: list[bytearray], x: int, y: int, text: str, color: int = 235
) -> None:
    height = len(canvas)
    width = len(canvas[0])
    cursor_x = x
    for ch in text.upper():
        glyph = FONT_5X7.get(ch, FONT_5X7[" "])
        for dy, row_bits in enumerate(glyph):
            py = y + dy
            if 0 <= py < height:
                for dx, bit in enumerate(row_bits):
                    if bit == "1":
                        px = cursor_x + dx
                        if 0 <= px < width:
                            canvas[py][px] = color
        cursor_x += 6


def draw_rect_border(
    canvas: list[bytearray], x: int, y: int, w: int, h: int, thickness: int, color: int
) -> None:
    for t in range(thickness):
        for px in range(x - t - 1, x + w + t + 1):
            if 0 <= px < len(canvas[0]):
                if 0 <= y - t - 1 < len(canvas):
                    canvas[y - t - 1][px] = color
                if 0 <= y + h + t < len(canvas):
                    canvas[y + h + t][px] = color
        for py in range(y - t - 1, y + h + t + 1):
            if 0 <= py < len(canvas):
                if 0 <= x - t - 1 < len(canvas[0]):
                    canvas[py][x - t - 1] = color
                if 0 <= x + w + t < len(canvas[0]):
                    canvas[py][x + w + t] = color


def paste_digit_nearest(
    canvas: list[bytearray], matrix_8x8: list[list[int]], x: int, y: int, scale: int = 8
) -> None:
    for r in range(8):
        for c in range(8):
            val = matrix_8x8[r][c]
            for dr in range(scale):
                row_buf = canvas[y + r * scale + dr]
                base_x = x + c * scale
                for dc in range(scale):
                    row_buf[base_x + dc] = val


def build_errors_grid_png(
    entries: list[dict[str, Any]], images_dir: Path, scale: int = 8
) -> bytes:
    """Tworzy zbiorczy obraz PNG: wiersz = 1 bledny obraz testowy + 3 najblizsi sasiedzi."""
    tile = 8 * scale  # 64 px
    border = 2
    info_w = 104
    pad_x = 12
    sep_w = 4
    col_gap = 12

    x_info = pad_x
    x_test = x_info + info_w + 8
    x_sep = x_test + tile + border + 10
    x_nn = [
        x_sep + sep_w + 10,
        x_sep + sep_w + 10 + (tile + 2 * border + col_gap),
        x_sep + sep_w + 10 + 2 * (tile + 2 * border + col_gap),
    ]
    total_w = x_nn[2] + tile + border + pad_x

    header_h = 24
    row_caption_h = 16
    row_h = row_caption_h + tile + 2 * border + 14
    total_h = header_h + max(1, len(entries)) * row_h + 8

    bg = 36
    canvas = [bytearray([bg] * total_w) for _ in range(total_h)]

    # Naglowek kolumn
    draw_text(canvas, x_info, 8, "ERR / ID", color=235)
    draw_text(canvas, x_test + 10, 8, "TEST X8", color=255)
    draw_text(canvas, x_nn[0] + 18, 8, "NN1", color=235)
    draw_text(canvas, x_nn[1] + 18, 8, "NN2", color=235)
    draw_text(canvas, x_nn[2] + 18, 8, "NN3", color=235)
    for px in range(total_w):
        canvas[header_h - 3][px] = 110

    for idx, item in enumerate(entries):
        y_top = header_h + idx * row_h
        y_tile = y_top + row_caption_h

        # Pionowy separator miedzy obrazem testowym a sasiadami
        for py in range(y_top + 2, y_top + row_h - 6):
            for sx in range(sep_w):
                canvas[py][x_sep + sx] = 115

        # Lewa kolumna opisowa
        short_hex = item["image_id"].replace("digit-", "")
        draw_text(
            canvas,
            x_info,
            y_tile + 4,
            f"#{idx + 1:02d} {item['true_label']}->{item['predicted_label']}",
            color=255,
        )
        draw_text(canvas, x_info, y_tile + 18, short_hex[:6], color=200)
        draw_text(canvas, x_info, y_tile + 28, short_hex[6:], color=200)
        nn1_tag = "NN1:OK" if item["nearest_neighbor_correct"] else "NN1:ERR"
        draw_text(
            canvas,
            x_info,
            y_tile + 46,
            nn1_tag,
            color=245 if item["nearest_neighbor_correct"] else 160,
        )

        # Obraz testowy
        draw_text(
            canvas,
            x_test,
            y_top + 4,
            f"T:{item['true_label']} P:{item['predicted_label']}",
            color=255,
        )
        test_mat = decode_png_file(images_dir / item["image_path"])
        draw_rect_border(canvas, x_test, y_tile, tile, tile, border, color=215)
        paste_digit_nearest(canvas, test_mat, x_test, y_tile, scale=scale)

        # 3 najblizsi sasiedzi
        for n_i, nb in enumerate(item["nearest_neighbors"]):
            nx = x_nn[n_i]
            draw_text(
                canvas,
                nx,
                y_top + 4,
                f"L:{nb['label']} {nb['l2_distance']:.2f}",
                color=245 if nb["label"] == item["true_label"] else 185,
            )
            nb_mat = decode_png_file(images_dir / nb["image_path"])
            b_col = 250 if nb["label"] == item["true_label"] else 120
            draw_rect_border(canvas, nx, y_tile, tile, tile, border, color=b_col)
            paste_digit_nearest(canvas, nb_mat, nx, y_tile, scale=scale)

        # Linia oddzielajaca wiersze
        y_line = y_top + row_h - 4
        if y_line < total_h:
            for px in range(total_w):
                canvas[y_line][px] = 72

    return encode_gray8_png(canvas)


def ascii_8x8(matrix: list[list[int]]) -> list[str]:
    ramp = " .:-=+*#%@"
    lines = []
    for row in matrix:
        lines.append("".join(ramp[min(9, val * 10 // 256)] * 2 for val in row))
    return lines


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Analizuje bledy modelu na test.csv i generuje bledy_test.json + bledy_test.png."
    )
    parser.add_argument("--model", type=Path, default=Path("model/model.json"))
    parser.add_argument("--train-csv", type=Path, default=Path("data/train.csv"))
    parser.add_argument("--test-csv", type=Path, default=Path("data/test.csv"))
    parser.add_argument("--images-dir", type=Path, default=Path("data"))
    parser.add_argument("--test-eval", type=Path, default=Path("wyniki/test_evaluation.json"))
    parser.add_argument("--val-eval", type=Path, default=Path("wyniki/validation_evaluation.json"))
    parser.add_argument("--descriptions-json", type=Path, default=None)
    parser.add_argument("--output-json", type=Path, default=Path("wyniki/bledy_test.json"))
    parser.add_argument("--output-png", type=Path, default=Path("wyniki/bledy_test.png"))
    args = parser.parse_args()

    model = load_model(args.model)
    if not isinstance(model, KNNModel):
        raise ValueError(f"Oczekiwano zamrozonego modelu KNNModel, otrzymano {type(model).__name__}.")

    train_rows = load_labeled_csv(args.train_csv)
    if len(train_rows) != len(model.X_train):
        raise ValueError("Niezgodnosc liczby wierszy train.csv z model.json.")

    test_eval = json.loads(args.test_eval.read_text(encoding="utf-8"))
    val_eval = json.loads(args.val_eval.read_text(encoding="utf-8"))

    descriptions: dict[str, str] = dict(DEFAULT_ERROR_DESCRIPTIONS)
    if args.descriptions_json is not None and args.descriptions_json.is_file():
        descriptions.update(json.loads(args.descriptions_json.read_text(encoding="utf-8")))

    # Statystyki remisow na calym tescie (diagnostyka B3 dla pelnosci)
    test_rows = load_labeled_csv(args.test_csv)
    X_test_all = [
        [px / 255.0 for row in decode_png_file(args.images_dir / str(r["image_path"])) for px in row]
        for r in test_rows
    ]
    _, tie_stats_test = model.predict_with_tie_stats(X_test_all)

    error_entries: list[dict[str, Any]] = []
    for err in test_eval["misclassified"]:
        img_id = str(err["image_id"])
        img_path = str(err["image_path"])
        true_lbl = int(err["true_label"])
        pred_lbl = int(err["predicted_label"])

        matrix = decode_png_file(args.images_dir / img_path)
        vec = [px / 255.0 for row in matrix for px in row]

        # Dokladnie ta sama odleglosc i kolejnosc rozstrzygania co w KNNModel
        dists = [
            (model._dist(vec, tx), idx, ty)
            for idx, (tx, ty) in enumerate(zip(model.X_train, model.y_train))
        ]
        dists.sort(key=lambda item: (item[0], item[1]))
        top3 = dists[:3]

        neighbors = []
        for rank, (sq_d, train_idx, nb_lbl) in enumerate(top3, start=1):
            tr_row = train_rows[train_idx]
            l2_d = math.sqrt(sq_d) if model.metric == "l2" else sq_d
            neighbors.append(
                {
                    "image_id": str(tr_row["image_id"]),
                    "image_path": str(tr_row["image_path"]),
                    "l2_distance": l2_d,
                    "label": int(nb_lbl),
                    "rank": rank,
                    "sq_l2_distance": sq_d,
                }
            )

        nn1_correct = neighbors[0]["label"] == true_lbl
        desc = descriptions.get(img_id, "")
        error_entries.append(
            {
                "description": desc,
                "image_id": img_id,
                "image_path": img_path,
                "nearest_neighbor_correct": nn1_correct,
                "nearest_neighbors": neighbors,
                "predicted_label": pred_lbl,
                "true_label": true_lbl,
            }
        )

        print(f"\n--- {img_id}: true={true_lbl} -> pred={pred_lbl} (NN1_ok={nn1_correct}) ---")
        for line in ascii_8x8(matrix):
            print("  " + line)
        for nb in neighbors:
            print(
                f"  NN{nb['rank']}: {nb['image_id']} label={nb['label']} "
                f"L2={nb['l2_distance']:.4f}"
            )

    png_bytes = build_errors_grid_png(error_entries, args.images_dir, scale=8)
    args.output_png.parent.mkdir(parents=True, exist_ok=True)
    args.output_png.write_bytes(png_bytes)

    nn1_correct_count = sum(1 for e in error_entries if e["nearest_neighbor_correct"])
    report = {
        "error_count": len(error_entries),
        "errors": error_entries,
        "nn1_correct_among_errors": nn1_correct_count,
        "png_layout": {
            "columns": [
                "lewa kolumna opisowa (#numer, prawdziwa->przewidziana, skrocone ID, status NN1)",
                "obraz testowy 8x8 powiekszony x8 (64x64 px, bez wygladzania)",
                "pionowy separator",
                "NN1 (64x64 px, jasna ramka gdy etykieta zgodna z prawdziwa)",
                "NN2 (64x64 px)",
                "NN3 (64x64 px)",
            ],
            "file": args.output_png.as_posix(),
            "scale": 8,
            "tile_px": 64,
        },
        "test_summary": {
            "accuracy": test_eval["accuracy"],
            "errors": test_eval["errors"],
            "hits": test_eval["hits"],
            "tie_stats_test": tie_stats_test,
            "total": test_eval["total"],
            "wilson_95": wilson_ci_95(int(test_eval["hits"]), int(test_eval["total"])),
        },
        "validation_summary": {
            "accuracy": val_eval["accuracy"],
            "errors": val_eval["errors"],
            "hits": val_eval["hits"],
            "total": val_eval["total"],
            "wilson_95": wilson_ci_95(int(val_eval["hits"]), int(val_eval["total"])),
        },
    }

    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(
        json.dumps(report, sort_keys=True, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    print(f"\nZapisano {len(error_entries)} bledow do {args.output_json} oraz {args.output_png} ({len(png_bytes)} B)")
    print(f"Remisy na tescie (k=3, L2): {tie_stats_test}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
