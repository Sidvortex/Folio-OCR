"""The one label format: layout JSON in reading order.

Matches Infinity-Parser2's ``doc2json`` output so fine-tuning starts from what
the base model already writes:

    [{"bbox": [x1, y1, x2, y2], "category": "...", "text": "..."}, ...]

* ``bbox`` is normalised to 0-1000 of the image width/height (integers).
* ``category`` is one of ``CATEGORIES``.
* ``text`` is Markdown for text-like blocks, HTML for ``table``, LaTeX for
  ``formula`` (wrapped in ``$$`` so Markdown consumers and olmOCR-Bench's math
  tests find it) and ``""`` for ``figure``.

Markdown output is *derived* from the blocks by ``to_markdown`` (dropping page
headers, footers and page footnotes, like Infinity-Parser2's converter), so the
two outputs can never disagree.
"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from typing import Iterable, List, Optional, Sequence

CATEGORIES = (
    "header", "title", "text", "figure", "table", "formula",
    "figure_caption", "table_caption", "formula_caption",
    "figure_footnote", "table_footnote", "page_footnote", "footer",
)
# Running headers/footers and page footnotes are not part of the body text
# (olmOCR-Bench even tests that they are absent from the Markdown).
NON_BODY = ("header", "footer", "page_footnote")


@dataclass
class Block:
    bbox: List[int]
    category: str
    text: str

    def to_dict(self) -> dict:
        return {"bbox": [int(v) for v in self.bbox], "category": self.category, "text": self.text}

    @classmethod
    def from_dict(cls, d: dict) -> "Block":
        return cls(bbox=[int(round(float(v))) for v in d["bbox"]], category=str(d["category"]),
                   text="" if d.get("text") is None else str(d.get("text")))


# ------------------------------------------------------------------- boxes
def normalize_bbox(box_px: Sequence[float], width: float, height: float) -> List[int]:
    """Pixel box -> integer box in 0..1000 (clamped)."""
    x1, y1, x2, y2 = box_px
    def n(v, size):
        return int(round(min(max(v / max(size, 1e-6), 0.0), 1.0) * 1000))
    a, b, c, d = n(x1, width), n(y1, height), n(x2, width), n(y2, height)
    return [min(a, c), min(b, d), max(a, c), max(b, d)]


def denormalize_bbox(bbox: Sequence[float], width: float, height: float) -> List[int]:
    """0..1000 box -> pixel box (same rounding as Infinity-Parser2's converter)."""
    x1, y1, x2, y2 = bbox
    return [int(x1 / 1000.0 * width), int(y1 / 1000.0 * height),
            int(x2 / 1000.0 * width), int(y2 / 1000.0 * height)]


# --------------------------------------------------------------- serialise
def dump_target(blocks: Iterable, fence: bool = True) -> str:
    """Training target / canonical text form, laid out exactly as Infinity-Parser2-Flash
    writes it (seen on its own output): the whole list on one line, items joined by
    ``", "``, inside a ```` ```json ```` fence. ``json.dumps`` default separators give
    ``{"bbox": [...`` -- the prefix Infinity-Parser2's post-processor searches for."""
    items = [b.to_dict() if isinstance(b, Block) else Block.from_dict(b).to_dict() for b in blocks]
    body = "[" + ", ".join(json.dumps(it, ensure_ascii=False) for it in items) + "]"
    return f"```json\n{body}\n```" if fence else body


_MATH_SPAN = re.compile(r"(\$\$.+?\$\$|\$[^$\n]+\$)", re.S)
_HEADING_MARK = re.compile(r"^#{1,6}\s+")
_STRONG_MARK = re.compile(r"\*\*(.+?)\*\*", re.S)
_EM_MARK = re.compile(r"(?<![\w*])\*(?=\S)(.+?)(?<=\S)\*(?![\w*])", re.S)
_BULLET_MARK = re.compile(r"^- ", re.M)


def plain_style(block: "Block") -> "Block":
    """Text the way Flash writes it in doc2json: titles without ``#``, no ``**``/``*``
    emphasis, no ``- `` bullet markers (numbered items keep their visible numbers).
    Tables, formulas and figures are unchanged; math spans are left alone."""
    if block.category in ("table", "formula", "figure") or not block.text:
        return block
    parts = _MATH_SPAN.split(block.text)
    for i in range(0, len(parts), 2):                 # even parts are outside math
        t = parts[i]
        t = _STRONG_MARK.sub(r"\1", t)
        t = _EM_MARK.sub(r"\1", t)
        t = _BULLET_MARK.sub("", t)
        parts[i] = t
    text = "".join(parts)
    if block.category == "title":
        text = _HEADING_MARK.sub("", text)
    return Block(list(block.bbox), block.category, text)


@dataclass
class ParseResult:
    blocks: List[Block] = field(default_factory=list)
    ok: bool = False          # parsed as one valid JSON list, no repair needed
    repaired: bool = False    # recovered by truncation or per-object salvage
    truncated: bool = False   # output ended before the closing bracket
    dropped: int = 0          # objects that were not valid blocks
    unknown_categories: int = 0


_FENCE_FULL = re.compile(r"```(?:json|JSON)?\s*\n(.*?)\n?```", re.DOTALL)
_FENCE_OPEN = re.compile(r"```(?:json|JSON)?\s*\n(.*)", re.DOTALL)


def extract_json_text(raw: str) -> str:
    """Strip Markdown fences (complete or cut off) around the JSON."""
    m = _FENCE_FULL.search(raw)
    if m:
        return m.group(1).strip()
    m = _FENCE_OPEN.search(raw)
    if m:
        return m.group(1).strip()
    return raw.strip()


def _valid_block(obj) -> Optional[Block]:
    if not isinstance(obj, dict) or "bbox" not in obj or "category" not in obj:
        return None
    box = obj.get("bbox")
    if not (isinstance(box, (list, tuple)) and len(box) == 4 and all(isinstance(v, (int, float)) for v in box)):
        return None
    try:
        return Block.from_dict(obj)
    except (TypeError, ValueError):
        return None


def _salvage(text: str) -> List[dict]:
    """Decode every complete ``{...}`` object that looks like a block."""
    dec = json.JSONDecoder()
    out, i = [], 0
    while True:
        i = text.find("{", i)
        if i < 0:
            break
        try:
            obj, end = dec.raw_decode(text, i)
        except json.JSONDecodeError:
            i += 1
            continue
        if isinstance(obj, dict):
            out.append(obj)
        i = end
    return out


def parse_output(raw: str) -> ParseResult:
    """Model output -> blocks. Never raises; reports what had to be repaired."""
    res = ParseResult()
    text = extract_json_text(raw or "")
    res.truncated = not text.rstrip().endswith("]")
    items = None
    try:
        data = json.loads(text)
        if isinstance(data, dict):            # a single object, or {"blocks": [...]}
            data = data.get("blocks", [data]) if "bbox" not in data else [data]
        if isinstance(data, list):
            if data and all(isinstance(x, list) for x in data):   # list of pages
                data = [x for page in data for x in page]
            items, res.ok = data, True
    except json.JSONDecodeError:
        pass
    if items is None:
        items, res.repaired = _salvage(text), True
    for obj in items:
        b = _valid_block(obj)
        if b is None:
            res.dropped += 1
            continue
        if b.category not in CATEGORIES:
            res.unknown_categories += 1
        res.blocks.append(b)
    if res.dropped and res.ok:
        res.ok = False
    return res


# ---------------------------------------------------------------- markdown
def to_markdown(blocks: Iterable, keep_non_body: bool = False) -> str:
    """Blocks -> Markdown, Infinity-Parser2 style: texts joined by blank lines,
    page headers/footers/footnotes dropped unless ``keep_non_body``."""
    out = []
    for b in blocks:
        b = b if isinstance(b, Block) else Block.from_dict(b)
        if not b.text or not b.text.strip():
            continue
        if not keep_non_body and b.category in NON_BODY:
            continue
        out.append(b.text.strip())
    return "\n\n".join(out)


def validate_blocks(blocks: Sequence[Block]) -> List[str]:
    """Problems that make a block list unusable as a *training label*."""
    errs = []
    for i, b in enumerate(blocks):
        if b.category not in CATEGORIES:
            errs.append(f"block {i}: unknown category {b.category!r}")
        x1, y1, x2, y2 = b.bbox
        if not (0 <= x1 <= x2 <= 1000 and 0 <= y1 <= y2 <= 1000):
            errs.append(f"block {i}: bbox out of range {b.bbox}")
        if b.category == "figure" and b.text:
            errs.append(f"block {i}: figure must have empty text")
        if b.category != "figure" and not b.text.strip():
            errs.append(f"block {i}: empty text for {b.category}")
    return errs
