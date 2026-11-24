"""Dekoder PNG 8x8, 8-bitowa skala szarosci (color_type=0), bez przeplotu.

Obsluguje filtry wierszy 0-4 i weryfikuje CRC kazdego chunku.
Każdy inny format jest odrzucany z czytelnym komunikatem (PNGDecodeError).
"""

from __future__ import annotations

import struct
import zlib
from pathlib import Path

PNG_SIGNATURE = b"\x89PNG\r\n\x1a\n"
EXPECTED_WIDTH = 8
EXPECTED_HEIGHT = 8
EXPECTED_BIT_DEPTH = 8
EXPECTED_COLOR_TYPE = 0
EXPECTED_COMPRESSION = 0
EXPECTED_FILTER_METHOD = 0
EXPECTED_INTERLACE = 0


class PNGDecodeError(ValueError):
    """Blad dekodowania lub niezgodnosci formatu pliku PNG."""


def _paeth_predictor(a: int, b: int, c: int) -> int:
    p = a + b - c
    pa = abs(p - a)
    pb = abs(p - b)
    pc = abs(p - c)
    if pa <= pb and pa <= pc:
        return a
    if pb <= pc:
        return b
    return c


def decode_png_bytes(raw: bytes) -> list[list[int]]:
    """Dekoduje bajty PNG do macierzy 8x8 wartosci calkowitych 0..255."""
    if len(raw) < 8 or raw[:8] != PNG_SIGNATURE:
        raise PNGDecodeError("Nieprawidlowa sygnatura pliku PNG.")

    pos = 8
    seen_ihdr = False
    seen_iend = False
    idat_parts: list[bytes] = []
    width = height = 0

    while pos < len(raw):
        if seen_iend:
            raise PNGDecodeError("Nadmiarowe dane po chunku IEND.")
        if pos + 12 > len(raw):
            raise PNGDecodeError("Niekompletny naglowek chunku PNG.")

        length = struct.unpack(">I", raw[pos : pos + 4])[0]
        chunk_type = raw[pos + 4 : pos + 8]
        data_start = pos + 8
        data_end = data_start + length
        crc_end = data_end + 4

        if crc_end > len(raw):
            raise PNGDecodeError(
                f"Niekompletne dane lub CRC w chunku {chunk_type!r}."
            )

        chunk_data = raw[data_start:data_end]
        expected_crc = struct.unpack(">I", raw[data_end:crc_end])[0]
        actual_crc = zlib.crc32(chunk_type + chunk_data) & 0xFFFFFFFF
        if actual_crc != expected_crc:
            raise PNGDecodeError(
                f"Bledna suma CRC w chunku {chunk_type!r}: "
                f"oczekiwano {expected_crc:#010x}, otrzymano {actual_crc:#010x}."
            )

        if not seen_ihdr:
            if chunk_type != b"IHDR":
                raise PNGDecodeError(
                    f"Pierwszy chunk musi byc IHDR, otrzymano {chunk_type!r}."
                )
            if length != 13:
                raise PNGDecodeError(
                    f"Nieprawidlowa dlugosc chunku IHDR: {length} (wymagane 13)."
                )
            (
                width,
                height,
                bit_depth,
                color_type,
                compression,
                filter_method,
                interlace,
            ) = struct.unpack(">IIBBBBB", chunk_data)

            if width != EXPECTED_WIDTH or height != EXPECTED_HEIGHT:
                raise PNGDecodeError(
                    f"Nieobslugiwany rozmiar obrazu: {width}x{height} "
                    f"(wymagane {EXPECTED_WIDTH}x{EXPECTED_HEIGHT})."
                )
            if bit_depth != EXPECTED_BIT_DEPTH:
                raise PNGDecodeError(
                    f"Nieobslugiwana glebia bitowa: {bit_depth} "
                    f"(wymagane {EXPECTED_BIT_DEPTH} bitow)."
                )
            if color_type != EXPECTED_COLOR_TYPE:
                raise PNGDecodeError(
                    f"Nieobslugiwany typ koloru: {color_type} "
                    f"(wymagany {EXPECTED_COLOR_TYPE} - skala szarosci)."
                )
            if compression != EXPECTED_COMPRESSION:
                raise PNGDecodeError(
                    f"Nieobslugiwana metoda kompresji: {compression} (wymagane 0)."
                )
            if filter_method != EXPECTED_FILTER_METHOD:
                raise PNGDecodeError(
                    f"Nieobslugiwana metoda filtracji: {filter_method} (wymagane 0)."
                )
            if interlace != EXPECTED_INTERLACE:
                raise PNGDecodeError(
                    f"Nieobslugiwany przeplot: {interlace} (wymagany 0 - brak przeplotu)."
                )
            seen_ihdr = True
        else:
            if chunk_type == b"IHDR":
                raise PNGDecodeError("Zduplikowany chunk IHDR.")
            if chunk_type == b"IDAT":
                idat_parts.append(chunk_data)
            elif chunk_type == b"IEND":
                if length != 0:
                    raise PNGDecodeError(
                        f"Chunk IEND musi miec dlugosc 0, otrzymano {length}."
                    )
                seen_iend = True

        pos = crc_end

    if not seen_ihdr:
        raise PNGDecodeError("Brak chunku IHDR.")
    if not seen_iend:
        raise PNGDecodeError("Brak chunku IEND.")
    if not idat_parts:
        raise PNGDecodeError("Brak chunkow IDAT.")

    try:
        decompressed = zlib.decompress(b"".join(idat_parts))
    except zlib.error as exc:
        raise PNGDecodeError(f"Blad dekompresji zlib danych IDAT: {exc}") from exc

    expected_len = height * (1 + width)
    if len(decompressed) != expected_len:
        raise PNGDecodeError(
            f"Nieprawidlowa dlugosc zdekompresowanych danych: "
            f"{len(decompressed)} (oczekiwano {expected_len})."
        )

    rows: list[list[int]] = []
    prev_row = [0] * width
    stride = 1 + width

    for row_idx in range(height):
        offset = row_idx * stride
        filter_type = decompressed[offset]
        raw_row = decompressed[offset + 1 : offset + stride]
        recon = [0] * width

        if filter_type == 0:
            for col in range(width):
                recon[col] = raw_row[col]
        elif filter_type == 1:
            for col in range(width):
                left = recon[col - 1] if col > 0 else 0
                recon[col] = (raw_row[col] + left) & 0xFF
        elif filter_type == 2:
            for col in range(width):
                recon[col] = (raw_row[col] + prev_row[col]) & 0xFF
        elif filter_type == 3:
            for col in range(width):
                left = recon[col - 1] if col > 0 else 0
                up = prev_row[col]
                recon[col] = (raw_row[col] + ((left + up) >> 1)) & 0xFF
        elif filter_type == 4:
            for col in range(width):
                left = recon[col - 1] if col > 0 else 0
                up = prev_row[col]
                up_left = prev_row[col - 1] if col > 0 else 0
                recon[col] = (raw_row[col] + _paeth_predictor(left, up, up_left)) & 0xFF
        else:
            raise PNGDecodeError(
                f"Nieobslugiwany typ filtru w wierszu {row_idx}: {filter_type} "
                "(dozwolone 0-4)."
            )

        rows.append(recon)
        prev_row = recon

    return rows


def decode_png_file(path: str | Path) -> list[list[int]]:
    """Odczytuje plik PNG z dysku i dekoduje go do macierzy 8x8."""
    file_path = Path(path)
    if not file_path.is_file():
        raise FileNotFoundError(f"Plik obrazu nie istnieje: {file_path}")
    return decode_png_bytes(file_path.read_bytes())


def extract_features(path: str | Path) -> list[float]:
    """Zwraca wektor 64 cech: piksele macierzy 8x8 podzielone przez 255.0."""
    matrix = decode_png_file(path)
    return [pixel / 255.0 for row in matrix for pixel in row]
