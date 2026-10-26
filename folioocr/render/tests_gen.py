"""Unit tests for rendered pages, in olmOCR-Bench's JSONL schema.

Because we wrote the page, every test is exact: text that must appear
(``present``), running headers/footers that must not appear (``absent``),
reading order between consecutive blocks of one column (``order``), table cell
neighbours (``table``), formulas (``math``) and resume fields (emails, phone
numbers, URLs, dates; ``present`` tests whose id contains ``_field_``).

The same tests score dev sets (``bench.unittests``) and serve as RL rewards.
"""
from __future__ import annotations

import random
from typing import Dict, List, Sequence

from ..schema import NON_BODY, Block
from ..textnorm import extract_fields, markdown_to_text, normalize_field

TEXTUAL = ("text", "title", "figure_caption", "table_caption", "formula_caption", "figure_footnote", "table_footnote")


def _plain(block_text: str) -> str:
    return " ".join(markdown_to_text(block_text).split())


def _olmocr_safe(text: str) -> bool:
    """olmOCR's normalize_text strips *...* and _..._ pairs anywhere in the page,
    so a test string with a lone '_' or '*' can fail on a perfect transcript
    (an email's underscore pairs up with the next one on the line). Skip those."""
    return "_" not in text and "*" not in text


def _first_line(lines: Sequence[dict], bi: int, block_text: str) -> str:
    """First visual line of a block (>= 10 chars), else the start of its first
    Markdown line -- never text joined across list items or line breaks."""
    for l in lines:
        if l["block"] == bi and not l.get("math") and len(l["text"]) >= 10:
            return l["text"]
    first = next((x.strip() for x in markdown_to_text(block_text).split("\n") if x.strip()), "")
    first = " ".join(first.split())
    return first[:40].rsplit(" ", 1)[0] if len(first) > 40 else first


def _overlap_x(a, b) -> float:
    inter = max(0, min(a[2], b[2]) - max(a[0], b[0]))
    return inter / max(1, min(a[2] - a[0], b[2] - b[0]))


def make_tests(pid: str, blocks: List[Block], lines: List[dict], rng: random.Random, *, max_present: int = 12,
               max_order: int = 6, max_table: int = 6, max_math: int = 4, max_absent: int = 4,
               fields: bool = True) -> List[Dict]:
    pdf = f"{pid}.pdf"
    tests: List[Dict] = []
    body_plain = " ".join(_plain(b.text) for b in blocks if b.category not in NON_BODY).lower()

    def add(kind: str, i: int, **kw):
        tests.append({"pdf": pdf, "page": 1, "id": f"{pid}_{kind}_{i}", "type": kw.pop("type", kind), **kw})

    # present: whole visual lines of body text
    cands = [l for l in lines if not l.get("math") and not l.get("broken") and len(l["text"]) >= 12
             and blocks[l["block"]].category in TEXTUAL]
    rng.shuffle(cands)
    seen = set()
    for l in cands:
        t = l["text"].strip()
        if t in seen or not _olmocr_safe(t):
            continue
        seen.add(t)
        add("present", len(seen) - 1, text=t, max_diffs=len(t) // 25)
        if len(seen) >= max_present:
            break

    # absent: running headers, footers, page footnotes
    n = 0
    for b in blocks:
        if b.category in NON_BODY and n < max_absent:
            t = _plain(b.text)
            if len(t) >= 5 and t.lower() not in body_plain and _olmocr_safe(t):
                add("absent", n, text=t, max_diffs=0)
                n += 1

    # order: consecutive body blocks stacked in the same column
    body = [(i, b) for i, b in enumerate(blocks) if b.category in TEXTUAL]
    pairs = []
    for (i, a), (j, b) in zip(body, body[1:]):
        if _overlap_x(a.bbox, b.bbox) >= 0.3 and b.bbox[1] >= a.bbox[1]:
            before, after = _first_line(lines, i, a.text), _first_line(lines, j, b.text)
            if len(before) >= 8 and len(after) >= 8 and before != after and _olmocr_safe(before + after):
                pairs.append((before, after))
    rng.shuffle(pairs)
    for k, (before, after) in enumerate(pairs[:max_order]):
        md = min(2, len(before) // 10, len(after) // 10)
        add("order", k, before=before, after=after, max_diffs=md)

    # table: neighbours read off olmocr's own parser, so the scorer agrees
    tbl_blocks = [b for b in blocks if b.category == "table"]
    if tbl_blocks:
        try:
            from olmocr.bench.tests import parse_html_tables
        except Exception:  # olmocr not installed: no table tests
            parse_html_tables = None
        k = 0
        for b in tbl_blocks:
            if parse_html_tables is None or k >= max_table:
                break
            for td in parse_html_tables(b.text):
                arr = td.data
                R, C = arr.shape
                counts: Dict[str, int] = {}
                for v in arr.flatten():
                    counts[str(v)] = counts.get(str(v), 0) + 1
                cells = [(r, c) for r in range(R) for c in range(C)
                         if r not in td.header_rows and str(arr[r, c]).strip() and counts[str(arr[r, c])] == 1
                         and len(str(arr[r, c])) >= 2 and _olmocr_safe(str(arr[r, c]))]
                rng.shuffle(cells)
                for r, c in cells[:max(1, max_table // max(1, len(tbl_blocks)))]:
                    rel = {}
                    if c + 1 < C and str(arr[r, c + 1]).strip():
                        rel["right"] = str(arr[r, c + 1])
                    if r + 1 < R and str(arr[r + 1, c]).strip():
                        rel["down"] = str(arr[r + 1, c])
                    if c > 0 and str(arr[r, c - 1]).strip():
                        rel["left"] = str(arr[r, c - 1])
                    heads = [str(arr[h, c]) for h in sorted(td.header_rows) if h < r and str(arr[h, c]).strip()]
                    if heads:
                        rel["top_heading"] = heads[-1]
                    rel = {kk: vv for kk, vv in rel.items() if _olmocr_safe(vv)}
                    if not rel:
                        continue
                    keys = rng.sample(sorted(rel), min(len(rel), rng.randint(1, 2)))
                    add("table", k, cell=str(arr[r, c]), max_diffs=0, **{kk: rel[kk] for kk in keys})
                    k += 1
                    if k >= max_table:
                        break
                if k >= max_table:
                    break

    # math
    m = 0
    for b in blocks:
        if b.category == "formula" and m < max_math:
            tex = b.text.strip()
            if tex.startswith("$$") and tex.endswith("$$"):
                tex = tex[2:-2].strip()
            if tex:
                add("math", m, math=tex, max_diffs=0)
                m += 1

    # resume fields
    if fields:
        body_text = "\n".join(_plain(b.text) for b in blocks if b.category not in NON_BODY and b.category != "table")
        seen_f = set()
        for kind, raw in extract_fields(body_text):
            key = (kind, normalize_field(kind, raw))
            if key in seen_f or not _olmocr_safe(raw):
                continue
            seen_f.add(key)
            add(f"field_{kind}", len(seen_f) - 1, type="present", text=raw, max_diffs=0, case_sensitive=False)
    return tests


def test_category(test: Dict) -> str:
    """Grouping used for per-category pass rates (field tests get their own group)."""
    return "field" if "_field_" in test["id"] else test["type"]
