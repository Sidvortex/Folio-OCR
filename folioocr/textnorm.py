"""Text normalisation, edit distance and resume field extraction.

``normalize_for_eval`` and the bullet/space canonicalisation are carried over
from ResumeOCR-Lite (resumeocr/charset.py), so CER numbers stay comparable
between the two projects.
"""
from __future__ import annotations

import html as _html
import re
import unicodedata
from typing import Dict, List, Sequence, Tuple

try:
    from rapidfuzz.distance import Levenshtein as _RFLev

    def levenshtein(a: Sequence, b: Sequence) -> int:
        return int(_RFLev.distance(a, b))
except ImportError:  # pragma: no cover
    def levenshtein(a: Sequence, b: Sequence) -> int:
        if len(a) < len(b):
            a, b = b, a
        prev = list(range(len(b) + 1))
        for i, ca in enumerate(a, 1):
            cur = [i]
            for j, cb in enumerate(b, 1):
                cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (ca != cb)))
            prev = cur
        return prev[-1]

BULLET_GLYPHS = "•●▪■◦‣⁃➢►▸✓✔◆♦○◾▫□➤➔"
_SPACE_LIKE = {c: " " for c in "              　\t"}
_ZERO_WIDTH = {c: None for c in "​‌‍⁠﻿­"}
_LABEL_TABLE = str.maketrans({**_SPACE_LIKE, **_ZERO_WIDTH, **{g: "•" for g in BULLET_GLYPHS}})
_EVAL_FOLD = str.maketrans({"‘": "'", "’": "'", "‚": "'", "“": '"', "”": '"', "„": '"', "–": "-", "—": "-",
                            "‑": "-", "‒": "-", "−": "-", "•": None, "·": None})
_WS = re.compile(r"\s+")


def canonicalize(text: str) -> str:
    text = unicodedata.normalize("NFC", text).translate(_LABEL_TABLE)
    return _WS.sub(" ", text).strip()


def strip_accents(text: str) -> str:
    return "".join(c for c in unicodedata.normalize("NFKD", text) if not unicodedata.combining(c))


def normalize_for_eval(text: str, mode: str = "normalized") -> str:
    """``strict``: NFC + whitespace. ``normalized``: also folds quotes, dashes,
    bullets, ligatures and accents (ResumeOCR-Lite semantics)."""
    if mode == "strict":
        return _WS.sub(" ", unicodedata.normalize("NFC", text)).strip()
    if mode != "normalized":
        raise ValueError(f"unknown mode {mode!r}")
    text = canonicalize(unicodedata.normalize("NFKC", text))
    text = strip_accents(text.replace("…", "...").translate(_EVAL_FOLD))
    return _WS.sub(" ", text).strip()


# ----------------------------------------------------------- markdown -> text
_TAG = re.compile(r"<[^>]+>")
_CELL_END = re.compile(r"</t[dh]\s*>", re.I)
_ROW_END = re.compile(r"</tr\s*>|<br\s*/?>", re.I)
_MD_IMG = re.compile(r"!\[[^\]]*\]\([^)]*\)")
_MD_LINK = re.compile(r"\[([^\]]+)\]\([^)]*\)")
_HEADING = re.compile(r"^\s{0,3}#{1,6}\s+", re.M)
_LIST = re.compile(r"^\s*(?:[-*+]|\d{1,3}[.)])\s+", re.M)
# glyph bullets ("•Built", "➢ Built") and dash bullets followed by a space; "*"/"-"
# list markers are Markdown and handled by _LIST (a bare "*" would eat "**bold**")
_BULLET_LINE = re.compile(r"^\s*(?:[" + re.escape(BULLET_GLYPHS) + r"]\s*|[–—]\s+)", re.M)
_EMPH = re.compile(r"(\*\*|__)(.+?)\1|(?<![\w*])\*(?!\s)(.+?)(?<!\s)\*(?![\w*])")
_FENCE = re.compile(r"^```[a-zA-Z]*\s*$", re.M)


def markdown_to_text(md: str) -> str:
    """Plain reading text from model Markdown (tables flattened row by row,
    headings/list markers/emphasis/links/HTML removed, math kept as source)."""
    if not md:
        return ""
    t = _FENCE.sub("", md)
    t = _MD_IMG.sub("", t)
    t = _MD_LINK.sub(r"\1", t)
    t = _CELL_END.sub(" ", t)
    t = _ROW_END.sub("\n", t)
    t = _TAG.sub("", t)
    t = _html.unescape(t)
    t = _HEADING.sub("", t)
    t = _LIST.sub("", t)
    t = _BULLET_LINE.sub("", t)
    for _ in range(2):
        t = _EMPH.sub(lambda m: m.group(2) if m.group(2) is not None else m.group(3), t)
    return t


def strip_leading_bullet(line: str) -> str:
    return _BULLET_LINE.sub("", line, count=1)


def cer(ref: str, hyp: str, mode: str = "normalized") -> Tuple[int, int]:
    """(edit distance, reference length) after normalisation."""
    r, h = normalize_for_eval(ref, mode), normalize_for_eval(hyp, mode)
    return levenshtein(r, h), len(r)


# ------------------------------------------------------------------ fields
_MONTH = (r"(?:Jan(?:uary)?|Feb(?:ruary)?|Mar(?:ch)?|Apr(?:il)?|May|June?|July?|Aug(?:ust)?|"
          r"Sep(?:t(?:ember)?)?|Oct(?:ober)?|Nov(?:ember)?|Dec(?:ember)?)")
FIELD_PATTERNS: Dict[str, re.Pattern] = {
    "email": re.compile(r"(?<![\w.+-])[A-Za-z0-9._%+\-]+@[A-Za-z0-9\-]+(?:\.[A-Za-z0-9\-]+)*\.[A-Za-z]{2,}"),
    "url": re.compile(
        r"(?<![@\w.])(?:https?://)?(?:www\.)?"
        r"(?:(?:linkedin|github|gitlab|kaggle|leetcode|medium|behance|dribbble|twitter|x|hackerrank|codeforces)\.com"
        r"|[A-Za-z0-9\-]+\.(?:github\.io|netlify\.app|vercel\.app|dev|io|me))"
        r"(?:/[A-Za-z0-9_\-./@%~?=&#]*)?", re.I),
    "phone": re.compile(r"(?<![\w/.])(?:\+\d{1,3}[\s\-.]?)?(?:\(\d{2,4}\)[\s\-.]?)?\d[\d\s\-.]{6,14}\d(?![\w/])"),
    "date": re.compile(r"\b" + _MONTH + r"\.?,?\s*['’]?(?:19|20)?\d{2}\b|\b(?:0?[1-9]|1[0-2])/(?:19|20)\d{2}\b", re.I),
}


def normalize_field(kind: str, value: str) -> str:
    v = unicodedata.normalize("NFKC", value).strip()
    if kind == "email":
        return v.lower().rstrip(".")
    if kind == "url":
        v = v.lower().rstrip("./,;:)")
        v = re.sub(r"^https?://", "", v)
        return re.sub(r"^www\.", "", v)
    if kind == "phone":
        plus = v.startswith("+")
        digits = re.sub(r"\D", "", v)
        return ("+" if plus else "") + digits
    if kind == "date":
        v = v.replace("’", "'").replace(".", " ").replace(",", " ")
        return _WS.sub(" ", v).strip().lower()
    return v


_US_DOTTED = re.compile(r"^(?:\+?1[\s.\-]?)?\(?\d{3}\)?\.\d{3}\.\d{4}$")


def plausible_phone(raw: str) -> bool:
    """10-13 digits, and not a row of table numbers: dots only in the US
    ``555.123.4567`` style (``60.17 58.92 88`` is three decimals), and not a
    run of years (``2012 2016 2020``)."""
    if not 10 <= len(re.sub(r"\D", "", raw)) <= 13:
        return False
    if "." in raw and not _US_DOTTED.match(raw):
        return False
    groups = re.findall(r"\d+", raw)
    if len(groups) >= 3 and all(len(g) == 4 and g[:2] in ("19", "20") for g in groups):
        return False
    return True


def extract_fields(text: str) -> List[Tuple[str, str]]:
    """(kind, raw value) for every email, URL, phone number and date in ``text``.
    Phone candidates must pass ``plausible_phone``, so years, CGPAs, date ranges
    and rows of table numbers are skipped."""
    out = []
    for kind, pat in FIELD_PATTERNS.items():
        for m in pat.finditer(text):
            raw = m.group(0).strip().rstrip(".,;:")
            if kind == "phone" and not plausible_phone(raw):
                continue
            if kind == "url" and "@" in text[max(0, m.start() - 1):m.start()]:
                continue
            out.append((kind, raw))
    return out


def field_set(text: str) -> Dict[str, set]:
    found: Dict[str, set] = {}
    for kind, raw in extract_fields(text):
        found.setdefault(kind, set()).add(normalize_field(kind, raw))
    return found
