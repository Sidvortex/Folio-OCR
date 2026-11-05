"""Full inference pipeline: pages -> layout JSON -> Markdown, with two optional extras.

second look   blocks whose text is tiny at the model's input resolution (dense
              tables, footnotes, long small print) are cropped from the
              full-resolution page and read again with ``doc2md``; the re-read
              replaces the block only when it agrees reasonably with the first
              read (it is a sharper look, not a different answer).
cross-check   for resumes: ResumeOCR-Lite's CPU line recogniser re-reads blocks
              that contain emails, phone numbers, URLs or dates. Disagreements
              are *flagged* in the output, never silently "fixed" -- the point
              is to catch a VLM that rewrote a digit.
"""
from __future__ import annotations

import math
import os
import re
import sys
from typing import List, Optional

from PIL import Image

from .imaging import fit_image
from .schema import Block, denormalize_bbox, to_markdown
from .textnorm import field_set, markdown_to_text

_TABLE = re.compile(r"<table.*?</table>", re.S | re.I)


def est_line_height(block: Block, w: int, h: int) -> float:
    """Rough text line height in pixels for a block at image size w x h."""
    x1, y1, x2, y2 = denormalize_bbox(block.bbox, w, h)
    area = max(1, (x2 - x1) * (y2 - y1))
    n = max(1, len(markdown_to_text(block.text)))
    return math.sqrt(2.0 * area / n)


def second_look(parser, page: Image.Image, result, min_line_px: float = 14.0, max_regions: int = 4,
                min_agree: float = 0.6, region_pixels: Optional[int] = None) -> int:
    """Re-read small-text blocks of one PageResult in place. Returns blocks replaced."""
    from rapidfuzz import fuzz
    if not result.blocks:
        return 0
    small = fit_image(page, parser.max_pixels)
    sw, sh = small.size
    W, H = page.size
    cands = []
    for i, b in enumerate(result.blocks):
        if b.category not in ("text", "table", "table_footnote", "page_footnote") or not b.text.strip():
            continue
        lh = est_line_height(b, sw, sh)
        if lh < min_line_px:
            cands.append((lh, i))
    replaced = 0
    for _, i in sorted(cands)[:max_regions]:
        b = result.blocks[i]
        x1, y1, x2, y2 = denormalize_bbox(b.bbox, W, H)
        pad_x, pad_y = int(0.015 * W), int(0.008 * H)
        crop = page.crop((max(0, x1 - pad_x), max(0, y1 - pad_y), min(W, x2 + pad_x), min(H, y2 + pad_y)))
        if crop.width < 16 or crop.height < 16:
            continue
        keep = parser.max_pixels
        if region_pixels:
            parser.max_pixels = region_pixels
        try:
            r2 = parser.parse([crop], task="doc2md")[0]
        finally:
            parser.max_pixels = keep
        if r2.looped or not r2.markdown.strip():
            continue
        new = r2.markdown.strip()
        if b.category == "table":
            m = _TABLE.search(new)
            if not m:
                continue
            new = m.group(0)
        agree = fuzz.ratio(markdown_to_text(b.text), markdown_to_text(new)) / 100.0
        if agree >= min_agree and new != b.text:
            b.text = new
            replaced += 1
    if replaced:
        result.markdown = to_markdown(result.blocks, keep_non_body=parser.keep_non_body)
    return replaced


class CrossChecker:
    """ResumeOCR-Lite (ONNX detector + CRNN) as an independent reader of field-bearing blocks."""

    def __init__(self, resumeocr_lite_dir: str, det: Optional[str] = None, rec: Optional[str] = None):
        sys.path.insert(0, os.path.abspath(resumeocr_lite_dir))
        from resumeocr.pipeline import ResumeOCR
        det = det or os.path.join(resumeocr_lite_dir, "models", "det.onnx")
        rec = rec or os.path.join(resumeocr_lite_dir, "models", "rec.onnx")
        if not os.path.exists(rec):
            rec = os.path.join(resumeocr_lite_dir, "models", "rec_demo_cpu.onnx")
        self.ocr = ResumeOCR(det_model=det, rec_model=rec)

    def check(self, page: Image.Image, blocks: List[Block]) -> List[dict]:
        import cv2
        import numpy as np
        W, H = page.size
        flags = []
        for i, b in enumerate(blocks):
            vlm_fields = field_set(markdown_to_text(b.text))
            if not vlm_fields:
                continue
            x1, y1, x2, y2 = denormalize_bbox(b.bbox, W, H)
            pad = int(0.01 * W)
            crop = page.crop((max(0, x1 - pad), max(0, y1 - pad), min(W, x2 + pad), min(H, y2 + pad)))
            bgr = cv2.cvtColor(np.asarray(crop.convert("RGB")), cv2.COLOR_RGB2BGR)
            crnn_text = self.ocr.to_text(self.ocr(bgr))
            crnn_fields = field_set(crnn_text)
            for kind, vals in vlm_fields.items():
                other = crnn_fields.get(kind, set())
                for v in vals:
                    if v not in other:
                        flags.append({"block": i, "kind": kind, "vlm": v, "crnn": sorted(other),
                                      "note": "second reader disagrees" if other else "second reader found no " + kind})
        return flags


def parse_document(parser, pages: List[Image.Image], second: bool = False, checker: Optional[CrossChecker] = None,
                   keep_raw: bool = False) -> List[dict]:
    out = []
    for idx, (page, r) in enumerate(zip(pages, parser.parse(pages))):
        n2 = second_look(parser, page, r) if second else 0
        rec = {"page": idx + 1, "markdown": r.markdown, "blocks": [b.to_dict() for b in r.blocks],
               "looped": r.looped, "retried": r.retried, "parse_ok": r.parse_ok, "tokens": r.tokens,
               "seconds": round(r.seconds, 2), "second_look_replaced": n2}
        if checker is not None:
            rec["field_flags"] = checker.check(page, r.blocks)
        if keep_raw:
            rec["raw"] = r.raw
        out.append(rec)
    return out
