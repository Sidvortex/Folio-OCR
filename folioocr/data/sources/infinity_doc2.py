"""Infinity-Doc2-5M -> Folio labels.

**Licence first:** CC BY-NC-SA 4.0, gated, "solely for academic research and
educational purposes". A model trained on it should be treated as
non-commercial and share-alike. ``train_sft.py`` refuses a mixture containing
it unless ``--allow-noncommercial`` is given, and the data card says so.

Layout (after ``tar -zxvf images_labels.tar.gz`` in a sub-task folder such as
``Document_Parsing/doc2json``):

    images/part_0001/...           page images
    labels/label.jsonl             {"images": [...], "conversations": [{"from": "human", ...},
                                    {"from": "gpt", "value": <answer>}], "objects": {"ref": [...],
                                    "bbox": [[x1, y1, x2, y2], ...]}, "attributes": {"subtask": ...}}

For doc2json the answer is the layout JSON, with ``<bbox>`` and
``<ref-object>`` placeholders filled from ``objects`` in order (the README:
"the i-th item in the GPT JSON output corresponds to the i-th entry in ref and
bbox"). Boxes in ``objects`` look like pixels; they are normalised to 0-1000
like Infinity-Parser2's own output. Not run on the real files here (the data
is gated): check the first converted pages with ``--preview``.
"""
from __future__ import annotations

import json
import os
import re
from typing import Iterable, List, Optional

from PIL import Image

from ...schema import CATEGORIES, Block, normalize_bbox, parse_output, validate_blocks
from ..dedup import file_sha256
from .common import LabelWriter, normalize_markdown

_BBOX = re.compile(r'"?<bbox>"?')
_REF = re.compile(r'"?<ref-object>"?')


def fill_placeholders(answer: str, refs: List[str], boxes: List[List[float]]) -> str:
    """Substitute ``<bbox>`` / ``<ref-object>`` placeholders in order."""
    it_b, it_r = iter(boxes), iter(refs)

    def nb(_m):
        return json.dumps([float(v) for v in next(it_b)])

    def nr(_m):
        return json.dumps(next(it_r))

    out = _BBOX.sub(nb, answer)
    return _REF.sub(nr, out)


def to_unit_boxes(blocks: List[Block], w: int, h: int, scale: str = "auto") -> tuple:
    """Pixel boxes -> 0-1000. ``scale``: pixels, 1000 or auto (pixels unless every
    coordinate is <= 1000 and the page, read as pixels, would sit in its top-left corner)."""
    if not blocks:
        return blocks, "empty"
    mx = max(max(b.bbox[0], b.bbox[2]) for b in blocks)
    my = max(max(b.bbox[1], b.bbox[3]) for b in blocks)
    mode = scale
    if scale == "auto":
        mode = "pixels"
        if mx <= 1000 and my <= 1000 and (w > 1100 or h > 1100) and mx / w < 0.6 and my / h < 0.6:
            mode = "1000"
    if mode == "pixels":
        for b in blocks:
            b.bbox = normalize_bbox(b.bbox, w, h)
    else:
        for b in blocks:
            b.bbox = [int(round(min(1000, max(0, v)))) for v in b.bbox]
    return blocks, mode


def _fix_text(b: Block) -> Block:
    t = (b.text or "").strip()
    if b.category == "figure":
        b.text = ""
    elif b.category == "formula" and t:
        if t.startswith("\\[") and t.endswith("\\]"):
            t = t[2:-2].strip()
        if not (t.startswith("$$") and t.endswith("$$")):
            t = "$$" + t.strip("$").strip() + "$$"      # Folio's formula convention (see render/page.js)
        b.text = t
    elif b.category != "table":
        b.text = normalize_markdown(t)
    return b


def convert(roots: Iterable[str], out: str, limit: Optional[int] = None, longest_side: int = 1600,
            bbox_scale: str = "auto", dev_frac: float = 0.01, subtasks=("doc2json", "doc2md")) -> dict:
    roots = list(roots)
    modes = {}
    with LabelWriter(out, "infinity-doc2-5m", longest_side=longest_side, dev_frac=dev_frac,
                     args={"roots": [os.path.abspath(r) for r in roots], "bbox_scale": bbox_scale}) as w:
        for root in roots:
            lab = os.path.join(root, "labels", "label.jsonl")
            if not os.path.exists(lab):
                raise FileNotFoundError(f"{lab} (extract images_labels.tar.gz in {root} first)")
            with open(lab, encoding="utf-8") as f:
                for n, line in enumerate(f):
                    if limit and w.n_new >= limit:
                        break
                    if not line.strip():
                        continue
                    r = json.loads(line)
                    sub = (r.get("attributes") or {}).get("subtask", "")
                    if sub not in subtasks:
                        w.skip(f"subtask {sub or '?'}")
                        continue
                    if len(r.get("images") or []) != 1:
                        w.skip("not one image")
                        continue
                    rel = r["images"][0]
                    rid = f"infdoc2/{sub}/{os.path.splitext(rel)[0]}"
                    if w.has(rid):
                        continue
                    gpt = next((c["value"] for c in r.get("conversations", []) if c.get("from") == "gpt"), None)
                    if not gpt:
                        w.skip("no answer")
                        continue
                    try:
                        img = Image.open(os.path.join(root, rel))
                        img.load()
                    except Exception:
                        w.skip("bad image")
                        continue
                    W, H = img.size
                    if sub == "doc2md":
                        md = normalize_markdown(gpt.strip().removeprefix("```markdown").removesuffix("```"))
                        if md:
                            w.add(rid, img, markdown=md,
                                  meta={"subtask": sub, "sha256": file_sha256(os.path.join(root, rel))})
                        else:
                            w.skip("empty markdown")
                        continue
                    obj = r.get("objects") or {}
                    refs, boxes = obj.get("ref") or [], obj.get("bbox") or []
                    answer = gpt
                    if "<bbox>" in gpt or "<ref-object>" in gpt:
                        if gpt.count("<bbox>") != len(boxes) or len(refs) != len(boxes):
                            w.skip("placeholder count mismatch")
                            continue
                        answer = fill_placeholders(gpt, refs, boxes)
                    pr = parse_output(answer)
                    if not pr.ok or not pr.blocks:
                        w.skip("unparseable answer")
                        continue
                    blocks = [b for b in pr.blocks if b.category in CATEGORIES]
                    if len(blocks) != len(pr.blocks):
                        w.skip("unknown category")
                        continue
                    blocks, mode = to_unit_boxes(blocks, W, H, bbox_scale)
                    modes[mode] = modes.get(mode, 0) + 1
                    blocks = [_fix_text(b) for b in blocks]
                    if validate_blocks(blocks):
                        w.skip("invalid blocks")
                        continue
                    w.add(rid, img, blocks=[b.to_dict() for b in blocks],
                          meta={"subtask": sub, "bbox_mode": mode, "orig_size": [W, H],
                                "sha256": file_sha256(os.path.join(root, rel))})
        return {"written": w.n_new, "total": len(w.done), "skipped": dict(w.skipped), "bbox_modes": modes, "out": out}
