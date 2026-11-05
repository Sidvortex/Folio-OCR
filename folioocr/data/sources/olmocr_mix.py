"""olmOCR-mix-1025 -> Folio labels (``markdown`` records: they train ``doc2md``).

Layout on the Hub (allenai/olmOCR-mix-1025):

    <subset>_<split>.parquet                      one row per page: id, url, page_number,
                                                  natural_text, is_rotation_valid, rotation_correction, ...
    pdf_tarballs/<subset>_<split>_NNNNN.tar.gz    single-page PDFs, member "<id>.pdf"

Subsets: 00_documents, 01_books, 02_loc_transcripts, 03_national_archives
(the last two are historical scans with human transcriptions -- the closest
public data to olmOCR-Bench's old-scans category). Download one parquet and a
tarball or two; each tarball holds a few thousand pages:

    hf download allenai/olmOCR-mix-1025 --repo-type dataset --local-dir olmocr-mix \\
        --include "02_loc_transcripts_train.parquet" "pdf_tarballs/02_loc_transcripts_train_00000.tar.gz"

Pages are rendered from their PDF; a page olmOCR marked as rotated is turned
upright the way olmOCR's pipeline does it (``rotation_correction`` degrees,
PIL's counter-clockwise transpose). Pages without text are skipped unless
``keep_blank`` > 0.
"""
from __future__ import annotations

import hashlib
import os
import random
from typing import Dict, Iterable, Optional

from PIL import Image

from ...imaging import render_pdf
from .common import LabelWriter, iter_tar_members, normalize_markdown

_TRANSPOSE = {90: Image.Transpose.ROTATE_90, 180: Image.Transpose.ROTATE_180, 270: Image.Transpose.ROTATE_270}


def _rows(parquet: str) -> Dict[str, dict]:
    import pandas as pd
    df = pd.read_parquet(parquet)
    ids = df["id"].astype(str).tolist() if "id" in df.columns else [str(i) for i in df.index]
    cols = [c for c in ("url", "page_number", "natural_text", "primary_language", "is_rotation_valid",
                        "rotation_correction", "is_table", "is_diagram") if c in df.columns]
    out = {}
    for rid, rec in zip(ids, df[cols].to_dict("records")):
        out[rid.lstrip("./")] = rec
    return out


def _is_null(v) -> bool:
    return v is None or (isinstance(v, float) and v != v)


def convert(parquet: str, tarballs: Iterable[str], out: str, subset: Optional[str] = None,
            limit: Optional[int] = None, longest_side: int = 1600, render_side: int = 2048,
            keep_blank: float = 0.0, rotated: str = "fix", dev_frac: float = 0.01, seed: int = 0) -> dict:
    subset = subset or os.path.basename(parquet).rsplit("_", 1)[0]
    rows = _rows(parquet)
    rng = random.Random(seed)
    args = {"parquet": os.path.abspath(parquet), "subset": subset, "longest_side": longest_side}
    with LabelWriter(out, "olmocr-mix-1025", longest_side=longest_side, dev_frac=dev_frac, args=args) as w:
        for name, data in iter_tar_members(tarballs, ".pdf"):
            if limit and w.n_new >= limit:
                break
            key = name[:-4].lstrip("./")
            row = rows.get(key)
            rid = f"olmocrmix/{subset}/{key}"
            if row is None:
                w.skip("no parquet row")
                continue
            if w.has(rid):
                continue
            text = row.get("natural_text")
            if _is_null(text) or not str(text).strip():
                if rng.random() >= keep_blank:
                    w.skip("no text")
                    continue
                text = ""
            upright = True if _is_null(row.get("is_rotation_valid")) else bool(row["is_rotation_valid"])
            rot = 0 if _is_null(row.get("rotation_correction")) else int(row["rotation_correction"]) % 360
            if not upright and rot and rotated == "skip":
                w.skip("rotated")
                continue
            try:
                img = render_pdf(data, longest_side=render_side, pages=[1])[0]
            except Exception:
                w.skip("render failed")
                continue
            if not upright and rot in _TRANSPOSE:
                img = img.transpose(_TRANSPOSE[rot])
            md = normalize_markdown(str(text))
            meta = {k: (None if _is_null(row.get(k)) else row.get(k))
                    for k in ("url", "page_number", "primary_language", "is_table", "is_diagram")}
            meta["subset"] = subset
            meta["sha256"] = hashlib.sha256(data).hexdigest()
            meta = {k: (v.item() if hasattr(v, "item") else v) for k, v in meta.items()}
            w.add(rid, img, markdown=md, meta=meta)
        return {"written": w.n_new, "total": len(w.done), "skipped": dict(w.skipped), "out": out}
