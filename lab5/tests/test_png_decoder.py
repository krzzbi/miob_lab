"""Testy jednostkowe dekodera PNG: sprawdzenia S1 i S2."""

from __future__ import annotations

import struct
import unittest
import zlib
from pathlib import Path

from src.png_decoder import PNGDecodeError, decode_png_bytes, decode_png_file

LAB5_DIR = Path(__file__).resolve().parent.parent


def _paeth(a: int, b: int, c: int) -> int:
    p = a + b - c
    pa = abs(p - a)
    pb = abs(p - b)
    pc = abs(p - c)
    if pa <= pb and pa <= pc:
        return a
    if pb <= pc:
        return b
    return c


def _make_chunk(chunk_type: bytes, data: bytes) -> bytes:
    crc = zlib.crc32(chunk_type + data) & 0xFFFFFFFF
    return struct.pack(">I", len(data)) + chunk_type + data + struct.pack(">I", crc)


def _encode_png_independent(
    matrix: list[list[int]],
    row_filters: list[int],
    *,
    width: int | None = None,
    height: int | None = None,
    bit_depth: int = 8,
    color_type: int = 0,
    interlace: int = 0,
) -> bytes:
    """Niezalezny koder PNG napisany w tescie (poza src/) dla sprawdzenia S2."""
    h = height if height is not None else len(matrix)
    w = width if width is not None else len(matrix[0])
    ihdr = struct.pack(">IIBBBBB", w, h, bit_depth, color_type, 0, 0, interlace)

    raw_scanlines = bytearray()
    prev_row = [0] * len(matrix[0])
    for r_idx, row in enumerate(matrix):
        ftype = row_filters[r_idx]
        raw_scanlines.append(ftype)
        for c_idx, val in enumerate(row):
            left = row[c_idx - 1] if c_idx > 0 else 0
            up = prev_row[c_idx]
            up_left = prev_row[c_idx - 1] if c_idx > 0 else 0
            if ftype == 0:
                filt = val
            elif ftype == 1:
                filt = (val - left) & 0xFF
            elif ftype == 2:
                filt = (val - up) & 0xFF
            elif ftype == 3:
                filt = (val - ((left + up) // 2)) & 0xFF
            elif ftype == 4:
                filt = (val - _paeth(left, up, up_left)) & 0xFF
            else:
                filt = val
            raw_scanlines.append(filt)
        prev_row = row

    compressed = zlib.compress(bytes(raw_scanlines))
    return (
        b"\x89PNG\r\n\x1a\n"
        + _make_chunk(b"IHDR", ihdr)
        + _make_chunk(b"IDAT", compressed)
        + _make_chunk(b"IEND", b"")
    )


class TestPNGDecoder(unittest.TestCase):
    SAMPLE_MATRIX = [
        [0, 16, 32, 48, 64, 80, 96, 112],
        [255, 239, 223, 207, 191, 175, 159, 143],
        [128, 0, 255, 128, 64, 191, 32, 223],
        [10, 200, 30, 180, 50, 160, 70, 140],
        [250, 5, 245, 15, 235, 25, 225, 35],
        [100, 100, 100, 200, 200, 50, 50, 0],
        [1, 2, 3, 4, 252, 253, 254, 255],
        [143, 175, 191, 191, 64, 0, 128, 16],
    ]

    def test_s1_manual_reference_image(self) -> None:
        """S1: reczny przyklad wzorcowy z zamrozonymi oczekiwanymi wierszami 0 i 7."""
        img_path = LAB5_DIR / "data" / "images" / "digit-17a28342e8c0.png"
        matrix = decode_png_file(img_path)
        self.assertEqual(len(matrix), 8)
        self.assertEqual(len(matrix[0]), 8)
        self.assertEqual(matrix[0], [0, 0, 128, 191, 191, 0, 0, 0])
        self.assertEqual(matrix[7], [0, 0, 143, 175, 191, 191, 64, 0])

    def test_s2_filters_1_to_4_and_mixed(self) -> None:
        """S2: syntetyczne PNG zakodowane filtrami 1, 2, 3, 4 oraz mieszanymi."""
        for ftype in [1, 2, 3, 4]:
            with self.subTest(filter_type=ftype):
                raw = _encode_png_independent(self.SAMPLE_MATRIX, [ftype] * 8)
                decoded = decode_png_bytes(raw)
                self.assertEqual(decoded, self.SAMPLE_MATRIX)

        mixed_filters = [0, 1, 2, 3, 4, 1, 3, 4]
        raw_mixed = _encode_png_independent(self.SAMPLE_MATRIX, mixed_filters)
        self.assertEqual(decode_png_bytes(raw_mixed), self.SAMPLE_MATRIX)

    def test_s2_reject_rgb(self) -> None:
        raw = _encode_png_independent(
            self.SAMPLE_MATRIX, [0] * 8, color_type=2
        )
        with self.assertRaisesRegex(PNGDecodeError, "typ koloru"):
            decode_png_bytes(raw)

    def test_s2_reject_16bit(self) -> None:
        raw = _encode_png_independent(
            self.SAMPLE_MATRIX, [0] * 8, bit_depth=16
        )
        with self.assertRaisesRegex(PNGDecodeError, "glebia bitowa"):
            decode_png_bytes(raw)

    def test_s2_reject_interlaced(self) -> None:
        raw = _encode_png_independent(
            self.SAMPLE_MATRIX, [0] * 8, interlace=1
        )
        with self.assertRaisesRegex(PNGDecodeError, "przeplot"):
            decode_png_bytes(raw)

    def test_s2_reject_wrong_dimensions(self) -> None:
        raw_7x8 = _encode_png_independent(
            [row[:7] for row in self.SAMPLE_MATRIX], [0] * 8, width=7, height=8
        )
        with self.assertRaisesRegex(PNGDecodeError, "rozmiar"):
            decode_png_bytes(raw_7x8)

        raw_8x9 = _encode_png_independent(
            self.SAMPLE_MATRIX + [[0] * 8], [0] * 9, width=8, height=9
        )
        with self.assertRaisesRegex(PNGDecodeError, "rozmiar"):
            decode_png_bytes(raw_8x9)

    def test_s2_reject_bad_crc(self) -> None:
        valid = _encode_png_independent(self.SAMPLE_MATRIX, [0] * 8)
        # Zepsuj ostatni bajt CRC w IHDR (pozycja 8 + 4 + 4 + 13 + 3 = 32)
        bad_ihdr_crc = bytearray(valid)
        bad_ihdr_crc[32] ^= 0xFF
        with self.assertRaisesRegex(PNGDecodeError, "CRC"):
            decode_png_bytes(bytes(bad_ihdr_crc))

        # Zepsuj bajt CRC w IDAT (4 bajty przed IEND, ktory ma 12 bajtow)
        bad_idat_crc = bytearray(valid)
        bad_idat_crc[-13] ^= 0x01
        with self.assertRaisesRegex(PNGDecodeError, "CRC"):
            decode_png_bytes(bytes(bad_idat_crc))

    def test_s2_reject_unknown_row_filter(self) -> None:
        raw = _encode_png_independent(self.SAMPLE_MATRIX, [5] * 8)
        with self.assertRaisesRegex(PNGDecodeError, "typ filtru"):
            decode_png_bytes(raw)


if __name__ == "__main__":
    unittest.main()
