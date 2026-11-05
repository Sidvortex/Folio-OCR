"""RL items from rendered pages: whole short pages, or horizontal bands of a page.

Free GPUs cannot afford RL on full pages (thousands of output tokens times
several samples). A band of a rendered page is a smaller problem with the same
kind of answer: the band is cut only where no block crosses it, so the blocks
inside are complete, their reading order is the page's order restricted to
them (columns stay column by column), and their boxes are re-normalised to the
crop. Each item gets its own unit tests, generated from its blocks and lines
exactly as ``make_dataset`` does for pages.
"""
from __future__ import annotations

import random
from typing import Dict, Iterable, List, Optional, Tuple

from PIL import Image

from ..render.tests_gen import make_tests
from ..schema import Block


def _chars(blocks: List[Block]) -> int:
    return sum(len(b.text) for b in blocks)


def free_cuts(blocks: List[Block]) -> List[int]:
    """y positions (0-1000) that no block crosses."""
    cand = sorted({0, 1000} | {b.bbox[1] for b in blocks} | {b.bbox[3] for b in blocks})
    return [c for c in cand if not any(b.bbox[1] < c < b.bbox[3] for b in blocks)]


def bands(blocks: List[Block], min_blocks: int = 2, max_chars: int = 1500, min_height: int = 60) -> List[Tuple[int, int]]:
    cuts = free_cuts(blocks)
    out = []
    for i, a in enumerate(cuts):
        for b in cuts[i + 1:]:
            if b - a < min_height:
                continue
            inside = [x for x in blocks if x.bbox[1] >= a and x.bbox[3] <= b]
            if len(inside) < min_blocks:
                continue
            if _chars(inside) > max_chars:
                break                                   # wider bands only get longer
            out.append((a, b))
    return out


def crop_band(rec: dict, image: Image.Image, a: int, b: int, rid: str, rng: random.Random,
              fields: bool) -> Optional[dict]:
    blocks = [Block.from_dict(x) for x in rec["blocks"]]
    inside = [i for i, x in enumerate(blocks) if x.bbox[1] >= a and x.bbox[3] <= b]
    if not inside:
        return None
    W, H = image.size
    top = min(blocks[i].bbox[1] for i in inside)
    bot = max(blocks[i].bbox[3] for i in inside)
    above = max([x.bbox[3] for x in blocks if x.bbox[3] <= top] + [0])
    below = min([x.bbox[1] for x in blocks if x.bbox[1] >= bot] + [1000])
    y0 = int((top + above) / 2 / 1000 * H)
    y1 = int(min(H, (bot + below) / 2 / 1000 * H + 1))
    if y1 - y0 < 16:
        return None
    crop = image.crop((0, y0, W, y1))
    ch = y1 - y0

    def ny(v):
        return int(round(min(1000, max(0, (v / 1000 * H - y0) / ch * 1000))))

    new_blocks = []
    remap = {}
    for k, i in enumerate(inside):
        x = blocks[i]
        new_blocks.append(Block([x.bbox[0], ny(x.bbox[1]), x.bbox[2], ny(x.bbox[3])], x.category, x.text))
        remap[i] = k
    lines = []
    for l in rec.get("lines", []):
        if l["block"] in remap:
            x1, ly0, x2, ly1 = l["box"]
            lines.append(dict(l, box=[x1, max(0, ly0 - y0), x2, max(0, ly1 - y0)], block=remap[l["block"]]))
    tests = make_tests(rid, new_blocks, lines, rng, fields=fields)
    return {"image": crop, "blocks": [x.to_dict() for x in new_blocks], "tests": tests,
            "band": [a, b], "crop_px": [0, y0, W, y1]}


def items_from_record(rec: dict, rng: random.Random, per_page: int = 2, max_chars: int = 1500,
                      min_blocks: int = 2, prefer=("table",)) -> List[dict]:
    """Up to ``per_page`` items: the whole page when it is short enough, else bands
    (bands holding a ``prefer`` category are picked first half of the time)."""
    blocks = [Block.from_dict(x) for x in rec["blocks"]]
    image = Image.open(rec["_image"]).convert("RGB")
    fields = rec.get("kind") == "resume"
    out = []
    if _chars(blocks) <= max_chars:
        tests = make_tests(f"{rec['id']}_p", blocks, rec.get("lines", []), rng, fields=fields)
        out.append({"suffix": "p", "image": image, "blocks": rec["blocks"], "tests": tests, "band": [0, 1000],
                    "crop_px": [0, 0, image.width, image.height]})
        return out
    cands = bands(blocks, min_blocks=min_blocks, max_chars=max_chars)
    rng.shuffle(cands)
    if prefer and rng.random() < 0.5:
        cands.sort(key=lambda ab: -any(x.category in prefer for x in blocks
                                        if x.bbox[1] >= ab[0] and x.bbox[3] <= ab[1]))
    used: List[Tuple[int, int]] = []
    for a, b in cands:
        if len(out) >= per_page:
            break
        if any(min(b, d) - max(a, c) > 0.3 * (b - a) for c, d in used):   # keep items mostly disjoint
            continue
        k = len(out)
        item = crop_band(rec, image, a, b, f"{rec['id']}_r{k}", rng, fields)
        if item is None or not item["tests"]:
            continue
        item["suffix"] = f"r{k}"
        out.append(item)
        used.append((a, b))
    return out


def build(label_files: Iterable[str], out: str, per_page: int = 2, max_chars: int = 1500, limit: Optional[int] = None,
          dev_frac: float = 0.03, seed: int = 0, source: str = "folio-render") -> Dict:
    from ..data.dataset import read_labels
    from ..data.sources.common import LabelWriter
    rng = random.Random(seed)
    n_pages = 0
    with LabelWriter(out, source, longest_side=None, dev_frac=dev_frac,
                     args={"labels": list(label_files), "per_page": per_page, "max_chars": max_chars}) as w:
        for path in label_files:
            for rec in read_labels(path):
                if limit and w.n_new >= limit:
                    break
                if "blocks" not in rec:
                    w.skip("no layout label")
                    continue
                n_pages += 1
                items = items_from_record(rec, rng, per_page=per_page, max_chars=max_chars)
                if not items:
                    w.skip("no band fits")
                for it in items:
                    rid = f"{rec['id']}_{it['suffix']}"
                    # the page's own split decides, so dev pages never leak into training items
                    w.add(rid, it["image"], blocks=it["blocks"], split=rec.get("split", "train"),
                          extra={"tests": it["tests"], "kind": rec.get("kind", ""),
                                 "condition": rec.get("condition", ""), "fields": rec.get("kind") == "resume"},
                          meta={"page": rec["id"], "band": it["band"], "crop_px": it["crop_px"]})
        return {"pages": n_pages, "items": w.n_new, "total": len(w.done), "skipped": dict(w.skipped), "out": out}
