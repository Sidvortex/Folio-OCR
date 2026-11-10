"""LightOnOCR-mix-0126 -> Folio labels (``markdown`` records, doc2md).

LightOnOCR-mix ships transcriptions only (``key``, ``page_idx``, ``content``:
Markdown with LaTeX math and HTML tables). The pages themselves are PDFs in
``pixparse/pdfa-eng-wds`` (WebDataset shards, ``<key>.pdf`` members, ~1,200
documents and ~0.8 GB per shard). This adapter joins the two: it reads the
keys of the shards you downloaded, scans the transcription parquet files row
group by row group (only groups holding one of those keys are read in full),
and renders the matching pages.

    hf download pixparse/pdfa-eng-wds --repo-type dataset --local-dir pdfa --include "pdfa-eng-train-0000.tar"
    python scripts/prepare_data.py lighton-mix --shards pdfa/pdfa-eng-train-0000.tar \\
        --parquet "hf://datasets/lightonai/LightOnOCR-mix-0126/<path>.parquet" ... --out data/lighton

``--parquet`` takes local files or ``hf://`` paths (read remotely, column by
column, through huggingface_hub's filesystem). Licence: "other"; the PDFA pages
fall under Common Crawl's and Digital Corpora's terms.
"""
from __future__ import annotations

import hashlib
import os
import tarfile
from typing import Dict, Iterable, List, Optional, Tuple

from ...imaging import render_pdf
from .common import LabelWriter, normalize_markdown


def shard_index(shards: Iterable[str]) -> Dict[str, Tuple[str, str]]:
    """key -> (shard path, member name) for every ``<key>.pdf`` in the shards."""
    idx = {}
    for p in shards:
        with tarfile.open(p, "r:*") as tar:
            for m in tar.getmembers():
                if m.isfile() and m.name.endswith(".pdf"):
                    key = os.path.basename(m.name)[:-4]
                    idx[key] = (p, m.name)
    return idx


def _open(path: str):
    if path.startswith("hf://"):
        from huggingface_hub import HfFileSystem
        return HfFileSystem().open(path, "rb")
    return open(path, "rb")


def matching_rows(parquets: Iterable[str], keys) -> List[dict]:
    import pyarrow as pa
    import pyarrow.compute as pc
    import pyarrow.parquet as pq
    want = pa.array(sorted(keys))
    rows = []
    for path in parquets:
        with _open(path) as fh:
            pf = pq.ParquetFile(fh)
            for g in range(pf.num_row_groups):
                k = pf.read_row_group(g, columns=["key"]).column("key")
                mask = pc.is_in(k, value_set=want)
                if not pc.any(mask).as_py():
                    continue
                t = pf.read_row_group(g, columns=["key", "page_idx", "content"]).filter(mask)
                rows.extend(t.to_pylist())
    return rows


def convert(shards: Iterable[str], parquets: Iterable[str], out: str, limit: Optional[int] = None,
            longest_side: int = 1600, render_side: int = 2048, dev_frac: float = 0.01) -> dict:
    shards, parquets = list(shards), list(parquets)
    idx = shard_index(shards)
    rows = matching_rows(parquets, idx.keys())
    with LabelWriter(out, "lightonocr-mix-0126", longest_side=longest_side, dev_frac=dev_frac,
                     args={"shards": shards, "parquets": parquets}) as w:
        if not rows:
            w.skip("no transcription for these shards' keys (add more parquet files)")
        rows.sort(key=lambda r: (idx[r["key"]][0], r["key"], r["page_idx"]))
        open_tars: Dict[str, tarfile.TarFile] = {}
        try:
            for r in rows:
                if limit and w.n_new >= limit:
                    break
                rid = f"lighton/{r['key']}/p{int(r['page_idx'])}"
                if w.has(rid):
                    continue
                md = normalize_markdown(r.get("content") or "")
                if not md:
                    w.skip("empty transcription")
                    continue
                shard, member = idx[r["key"]]
                tar = open_tars.get(shard) or open_tars.setdefault(shard, tarfile.open(shard, "r:*"))
                data = tar.extractfile(member).read()
                try:
                    img = render_pdf(data, longest_side=render_side, pages=[int(r["page_idx"]) + 1])[0]
                except Exception:
                    w.skip("render failed")
                    continue
                w.add(rid, img, markdown=md, meta={"key": r["key"], "page_idx": int(r["page_idx"]),
                                                   "sha256": hashlib.sha256(data).hexdigest()})
        finally:
            for t in open_tars.values():
                t.close()
        return {"written": w.n_new, "total": len(w.done), "matched_rows": len(rows), "shard_docs": len(idx),
                "skipped": dict(w.skipped), "out": out}
