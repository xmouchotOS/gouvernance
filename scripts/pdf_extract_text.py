#!/usr/bin/env python3
"""
Minimal PDF text extractor (stdlib only).

Goal: extract readable text from simple text-based PDFs (Flate streams),
good enough for role-sheet PDFs where we mainly need headings and bullets.

Usage:
  python3 scripts/pdf_extract_text.py path/to/file.pdf
  python3 scripts/pdf_extract_text.py path/to/file.pdf --out extracted.txt
"""

from __future__ import annotations

import argparse
import re
import sys
import zlib
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Iterator, List, Sequence


def _read_literal_string(data: bytes, start: int) -> tuple[str, int]:
    # Reads PDF literal string starting at '(' and returns (decoded, next_index).
    i = start + 1
    out = bytearray()
    depth = 1
    while i < len(data) and depth > 0:
        c = data[i]
        if c == 0x5C:  # backslash
            if i + 1 >= len(data):
                i += 1
                continue
            nxt = data[i + 1]
            # Basic escapes (keep it simple)
            if nxt in b"nrtbf":
                out += {
                    ord("n"): b"\n",
                    ord("r"): b"\r",
                    ord("t"): b"\t",
                    ord("b"): b"\b",
                    ord("f"): b"\f",
                }.get(nxt, bytes([nxt]))
                i += 2
                continue
            if nxt in b"()\\":
                out.append(nxt)
                i += 2
                continue
            # Octal escapes: \ddd
            if 0x30 <= nxt <= 0x37:
                j = i + 1
                oct_digits = bytearray()
                while j < len(data) and len(oct_digits) < 3 and 0x30 <= data[j] <= 0x37:
                    oct_digits.append(data[j])
                    j += 1
                try:
                    out.append(int(oct_digits.decode("ascii"), 8) & 0xFF)
                except Exception:
                    pass
                i = j
                continue
            # Unknown escape: keep the following char
            out.append(nxt)
            i += 2
            continue
        if c == 0x28:  # '('
            depth += 1
            out.append(c)
            i += 1
            continue
        if c == 0x29:  # ')'
            depth -= 1
            if depth == 0:
                i += 1
                break
            out.append(c)
            i += 1
            continue
        out.append(c)
        i += 1
    # PDFs often encode strings in WinAnsi; latin-1 is a decent fallback.
    return out.decode("latin-1", "ignore").replace("\x00", ""), i


def _read_hex_string(data: bytes, start: int) -> tuple[str, int]:
    # Reads PDF hex string starting at '<' (and not '<<').
    i = start + 1
    hex_bytes = bytearray()
    while i < len(data):
        c = data[i]
        if c == 0x3E:  # '>'
            i += 1
            break
        if chr(c).isspace():
            i += 1
            continue
        hex_bytes.append(c)
        i += 1
    try:
        raw = bytes.fromhex(hex_bytes.decode("ascii", "ignore"))
        return raw.decode("latin-1", "ignore").replace("\x00", ""), i
    except Exception:
        return "", i


def _is_delim_byte(b: int) -> bool:
    # PDF delimiters: ()<>[]{}/%
    return b in (0x28, 0x29, 0x3C, 0x3E, 0x5B, 0x5D, 0x7B, 0x7D, 0x2F, 0x25)


def _skip_ws_and_comments(data: bytes, i: int) -> int:
    while i < len(data):
        c = data[i]
        if chr(c).isspace():
            i += 1
            continue
        if c == 0x25:  # '%'
            nl = data.find(b"\n", i)
            if nl == -1:
                return len(data)
            i = nl + 1
            continue
        return i
    return i


@dataclass(frozen=True)
class _Array:
    items: Sequence[object]


def _tokenize_content_stream(data: bytes) -> Iterator[object]:
    i = 0
    while i < len(data):
        i = _skip_ws_and_comments(data, i)
        if i >= len(data):
            break

        c = data[i]
        # Arrays
        if c == 0x5B:  # '['
            i += 1
            items: List[object] = []
            while i < len(data):
                i = _skip_ws_and_comments(data, i)
                if i >= len(data):
                    break
                if data[i] == 0x5D:  # ']'
                    i += 1
                    break
                tok, i = _read_token(data, i)
                items.append(tok)
            yield _Array(tuple(items))
            continue

        tok, i = _read_token(data, i)
        yield tok


def _read_token(data: bytes, i: int) -> tuple[object, int]:
    c = data[i]
    # Dictionaries and other delimiters - always advance at least one byte.
    if c == 0x3C and i + 1 < len(data) and data[i + 1] == 0x3C:  # '<<'
        return "<<", i + 2
    if c == 0x3E and i + 1 < len(data) and data[i + 1] == 0x3E:  # '>>'
        return ">>", i + 2
    if c == 0x28:  # '('
        s, j = _read_literal_string(data, i)
        return s, j
    if c == 0x3C and i + 1 < len(data) and data[i + 1] != 0x3C:  # '<' but not '<<'
        s, j = _read_hex_string(data, i)
        return s, j
    if c == 0x2F:  # name
        j = i + 1
        while j < len(data) and not chr(data[j]).isspace() and not _is_delim_byte(data[j]):
            j += 1
        return data[i:j].decode("latin-1", "ignore"), j
    if _is_delim_byte(c):
        # Any other delimiter (including '>', ']', '{', '}', '%') as a single-char token.
        return chr(c), i + 1

    # number or operator/keyword
    j = i
    while j < len(data) and not chr(data[j]).isspace() and not _is_delim_byte(data[j]):
        j += 1
    raw = data[i:j].decode("latin-1", "ignore")
    if j == i:
        # Safety: advance to avoid infinite loops on unexpected bytes.
        return "", i + 1

    # try number
    if re.fullmatch(r"[+-]?\d+(\.\d+)?", raw):
        try:
            if "." in raw:
                return float(raw), j
            return int(raw), j
        except Exception:
            pass

    return raw, j


def _iter_streams(pdf_bytes: bytes) -> Iterable[bytes]:
    # Finds raw stream bodies. Not a full PDF parser but enough for many simple PDFs.
    for m in re.finditer(rb"stream\r?\n", pdf_bytes):
        start = m.end()
        end = pdf_bytes.find(rb"endstream", start)
        if end == -1:
            continue
        yield pdf_bytes[start:end]


def _maybe_inflate(stream_bytes: bytes) -> bytes | None:
    s = stream_bytes.lstrip(b"\r\n")
    for wbits in (zlib.MAX_WBITS, -zlib.MAX_WBITS):
        try:
            return zlib.decompress(s, wbits=wbits)
        except Exception:
            pass
    return None


def extract_text_from_pdf(pdf_path: Path) -> str:
    pdf_bytes = pdf_path.read_bytes()

    lines: List[str] = []
    current: List[str] = []

    def decode_caesar_minus_3_if_needed(s: str) -> str:
        # Some PDFs store text with a basic Caesar shift (often +3).
        # Heuristic: only shift long tokens that are mostly uppercase letters.
        def shift_token(tok: str) -> str:
            letters = [c for c in tok if c.isalpha()]
            if len(tok) < 6 or len(letters) < 6:
                return tok
            upper = sum(1 for c in letters if c.isupper())
            if upper / len(letters) < 0.7:
                return tok

            def shift_char(c: str) -> str:
                o = ord(c)
                if 65 <= o <= 90:
                    return chr(((o - 65 - 3) % 26) + 65)
                if 97 <= o <= 122:
                    return chr(((o - 97 - 3) % 26) + 97)
                return c

            shifted = "".join(shift_char(c) for c in tok)
            # Sanity: keep if it now contains vowels.
            if sum(1 for c in shifted if c in "AEIOUYaeiouy") >= 2:
                return shifted
            return tok

        return re.sub(r"[A-Za-z]{6,}", lambda m: shift_token(m.group(0)), s)

    def flush_line() -> None:
        if not current:
            lines.append("")
            return
        joined = "".join(current)
        joined = joined.replace("\x00", "")
        joined = re.sub(r"[ \t]+", " ", joined).strip()
        joined = decode_caesar_minus_3_if_needed(joined)
        lines.append(joined)
        current.clear()

    in_text = False
    last_text_y: float | None = None

    for raw_stream in _iter_streams(pdf_bytes):
        inflated = _maybe_inflate(raw_stream)
        if not inflated:
            continue
        if b"BT" not in inflated and b"Tj" not in inflated and b"TJ" not in inflated:
            continue

        operands: List[object] = []
        for tok in _tokenize_content_stream(inflated):
            if isinstance(tok, str) and tok in {
                "BT",
                "ET",
                "Tj",
                "TJ",
                "Td",
                "TD",
                "Tm",
                "T*",
                "'",
                '"',
            }:
                op = tok
                if op == "BT":
                    in_text = True
                    operands.clear()
                    last_text_y = None
                    continue
                if op == "ET":
                    if in_text:
                        flush_line()
                        flush_line()
                    in_text = False
                    operands.clear()
                    continue
                if not in_text:
                    operands.clear()
                    continue

                if op in {"Td", "TD", "T*", "Tm"}:
                    # New line / move; heuristics:
                    # - T*: explicit new line
                    # - Td/TD: new line when Y translation is significant
                    # - Tm: new line when absolute Y changes significantly
                    if op == "T*":
                        flush_line()
                        operands.clear()
                        continue
                    if op in {"Td", "TD"}:
                        if len(operands) >= 2 and isinstance(operands[-1], (int, float)) and isinstance(
                            operands[-2], (int, float)
                        ):
                            y = float(operands[-1])
                            if abs(y) > 2:
                                flush_line()
                        operands.clear()
                        continue
                    if op == "Tm":
                        if len(operands) >= 2 and isinstance(operands[-1], (int, float)):
                            y_abs = float(operands[-1])
                            if last_text_y is not None and abs(y_abs - last_text_y) > 5:
                                flush_line()
                            last_text_y = y_abs
                    operands.clear()
                    continue

                if op == "Tj" or op in {"'", '"'}:
                    if operands:
                        s = operands[-1]
                        if isinstance(s, str) and s:
                            current.append(s)
                    operands.clear()
                    continue

                if op == "TJ":
                    if operands:
                        arr = operands[-1]
                        if isinstance(arr, _Array):
                            for item in arr.items:
                                if isinstance(item, str):
                                    current.append(item)
                                elif isinstance(item, (int, float)):
                                    # Spacing (1/1000 em). Large negative => likely a word gap.
                                    if item <= -250:
                                        current.append(" ")
                        elif isinstance(arr, str):
                            current.append(arr)
                    operands.clear()
                    continue

                operands.clear()
                continue

            operands.append(tok)

    # Flush remaining
    if current:
        flush_line()

    text = "\n".join(lines)
    # Cleanup: collapse too many blank lines.
    cleaned_lines: List[str] = []
    for line in text.splitlines():
        l = line.strip()
        if not l:
            cleaned_lines.append("")
            continue
        # Drop lines that are only control-like leftovers.
        if all((not ch.isprintable()) or ch in "\x00" for ch in l):
            continue
        cleaned_lines.append(l)

    cleaned = "\n".join(cleaned_lines)
    cleaned = re.sub(r"\n{3,}", "\n\n", cleaned).strip() + "\n"
    return cleaned


def main(argv: Sequence[str]) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("pdf", type=Path)
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args(argv)

    if not args.pdf.exists():
        print(f"File not found: {args.pdf}", file=sys.stderr)
        return 2

    text = extract_text_from_pdf(args.pdf)
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(text, encoding="utf-8")
    else:
        sys.stdout.write(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
