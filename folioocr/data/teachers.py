"""Labels for real pages that have no source: two teachers must agree.

The plan's rule: only teachers whose licences allow training on their output
(Infinity-Parser2-Pro and olmOCR 2 are Apache-2.0; never Chandra or Surya).
A page is kept when the two transcriptions agree on at least 97% of
characters after normalisation; the rest go to a review queue.

Each teacher runs on its own (``run_teacher``), one at a time, so a free GPU
holds only one model and a session can end between them; outputs are cached
per page. ``merge`` then compares the caches and writes Folio labels: the
layout blocks of a teacher that produced valid layout JSON when one did,
otherwise Markdown (doc2md).
"""
from __future__ import annotations

import glob
import json
import os
from typing import Iterable, List, Optional, Tuple

from PIL import Image

from ..bench.engines import PageItem, safe_key
from ..textnorm import levenshtein, markdown_to_text, normalize_for_eval
from .sources.common import LabelWriter

FORBIDDEN = {"chandra", "surya"}       # licences forbid training on their output for competing use


def agreement(md_a: str, md_b: str) -> float:
    a = normalize_for_eval(markdown_to_text(md_a or ""))
    b = normalize_for_eval(markdown_to_text(md_b or ""))
    if not a and not b:
        return 1.0
    return 1.0 - levenshtein(a, b) / max(len(a), len(b))


def stage_pages(inputs: Iterable[str], stage: str, longest_side: int = 1600, dpi: float = 200) -> List[Tuple[str, str]]:
    """Folders/files of images and PDFs, or label files -> [(key, image path)], rendered once
    into ``stage`` so every teacher reads the identical image."""
    from ..imaging import IMAGE_EXTS, load_pages
    os.makedirs(stage, exist_ok=True)
    out = []
    files = []
    for x in inputs:
        if x.endswith(".jsonl"):
            from .dataset import read_labels
            for r in read_labels(x):
                out.append((safe_key(r["id"]), r["_image"]))
            continue
        if os.path.isdir(x):
            files += sorted(p for p in glob.glob(os.path.join(x, "**", "*"), recursive=True)
                            if os.path.splitext(p)[1].lower() in IMAGE_EXTS | {".pdf"})
        else:
            files.append(x)
    for f in files:
        base = safe_key(os.path.splitext(os.path.basename(f))[0])
        if f.lower().endswith(".pdf"):
            import pypdfium2 as pdfium
            n = len(pdfium.PdfDocument(f))
            for i in range(1, n + 1):
                key = f"{base}_p{i}"
                dst = os.path.join(stage, key + ".png")
                if not os.path.exists(dst):
                    load_pages(f, dpi=dpi, pages=[i])[0].save(dst)
                out.append((key, dst))
        else:
            dst = os.path.join(stage, base + ".png")
            if not os.path.exists(dst):
                im = Image.open(f).convert("RGB")
                if max(im.size) > longest_side:
                    s = longest_side / max(im.size)
                    im = im.resize((round(im.width * s), round(im.height * s)), Image.LANCZOS)
                im.save(dst)
            out.append((base, dst))
    return out


def run_teacher(engine, name: str, pages: List[Tuple[str, str]], cache: str, licence: str, chunk: int = 8,
                engine_kind: str = "") -> dict:
    if engine_kind in FORBIDDEN or name.lower() in FORBIDDEN:
        raise SystemExit(f"{engine_kind or name}: its licence does not allow training on its output (see the plan)")
    os.makedirs(cache, exist_ok=True)
    meta_path = os.path.join(cache, "teacher.json")
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump({"name": name, "engine": engine_kind, "licence": licence}, f, indent=1)
    todo = [(k, p) for k, p in pages if not os.path.exists(os.path.join(cache, f"{k}.json"))]
    done = 0
    for s in range(0, len(todo), chunk):
        part = todo[s:s + chunk]
        res = engine.run_full([PageItem(key=k, image=p) for k, p in part])
        for k, _ in part:
            r = res.get(k) or {"markdown": "", "blocks": None, "raw": "", "error": "no output"}
            with open(os.path.join(cache, f"{k}.json"), "w", encoding="utf-8") as f:
                json.dump(r, f, ensure_ascii=False)
        done += len(part)
        print(f"  {name}: {done}/{len(todo)} pages", flush=True)
    return {"teacher": name, "pages": len(pages), "ran": len(todo)}


def _load(cache: str, key: str) -> Optional[dict]:
    p = os.path.join(cache, f"{key}.json")
    if not os.path.exists(p):
        return None
    with open(p, encoding="utf-8") as f:
        return json.load(f)


def merge(pages: List[Tuple[str, str]], cache_a: str, cache_b: str, out: str, threshold: float = 0.97,
          dev_frac: float = 0.0) -> dict:
    ta = json.load(open(os.path.join(cache_a, "teacher.json")))
    tb = json.load(open(os.path.join(cache_b, "teacher.json")))
    for t in (ta, tb):
        if t.get("engine") in FORBIDDEN or t["name"].lower() in FORBIDDEN:
            raise SystemExit(f"teacher {t['name']} is not allowed (licence)")
    review_path = os.path.join(out, "review.jsonl")
    os.makedirs(out, exist_ok=True)
    stats = {"pages": len(pages), "kept": 0, "review": 0, "missing": 0, "blank": 0, "errors": 0,
             "label_blocks": 0, "label_markdown": 0}
    sims = []
    args = {"teachers": [ta, tb], "threshold": threshold}
    with LabelWriter(out, "teacher-consensus", longest_side=None, dev_frac=dev_frac, args=args) as w, \
            open(review_path, "a", encoding="utf-8") as rv:
        for key, img_path in pages:
            if w.has(f"consensus/{key}"):
                continue
            a, b = _load(cache_a, key), _load(cache_b, key)
            if a is None or b is None:
                stats["missing"] += 1
                continue
            if a.get("error") or b.get("error"):
                stats["errors"] += 1
                continue
            if not (a["markdown"] or "").strip() and not (b["markdown"] or "").strip():
                stats["blank"] += 1
                continue
            sim = agreement(a["markdown"], b["markdown"])
            sims.append(sim)
            rid = f"consensus/{key}"
            if sim < threshold:
                stats["review"] += 1
                rv.write(json.dumps({"id": rid, "image": img_path, "agreement": round(sim, 4),
                                     ta["name"]: a["markdown"], tb["name"]: b["markdown"]}, ensure_ascii=False) + "\n")
                continue
            meta = {"agreement": round(sim, 4), "teachers": [ta["name"], tb["name"]]}
            img = Image.open(img_path)
            if a.get("blocks"):
                w.add(rid, img, blocks=a["blocks"], meta={**meta, "label_from": ta["name"]})
                stats["label_blocks"] += 1
            elif b.get("blocks"):
                w.add(rid, img, blocks=b["blocks"], meta={**meta, "label_from": tb["name"]})
                stats["label_blocks"] += 1
            else:
                w.add(rid, img, markdown=a["markdown"], meta={**meta, "label_from": ta["name"]})
                stats["label_markdown"] += 1
            stats["kept"] += 1
    if sims:
        import numpy as np
        stats["agreement_median"] = float(np.median(sims))
    stats["review_file"] = review_path
    return stats
