"""olmOCR-synthmix-1025 -> SFT labels and RL pages.

Layout on the Hub (allenai/olmOCR-synthmix-1025):

    bench_data/<subset>.jsonl          olmOCR-Bench-format tests (present/absent/order/table/math)
    bench_data/pdfs/<subset>/*.pdf     the pages those tests are about
    training/<subset>/*.md, *.pdf      Markdown + PDF pairs for supervised training

Subsets: arxiv_cs, arxiv_math, arxiv_physics, general, tables.

``convert_sft`` turns the training pairs into ``markdown`` records (doc2md).
``convert_rl`` turns the benchmark pages into RL records: the page image, its
tests and, when a training Markdown with the same name exists, that Markdown
as the reference for the text-accuracy reward.

    hf download allenai/olmOCR-synthmix-1025 --repo-type dataset --local-dir synthmix
"""
from __future__ import annotations

import glob
import json
import os
from collections import defaultdict
from typing import Dict, List, Optional

from ...imaging import render_pdf
from ..dedup import file_sha256
from .common import LabelWriter, normalize_markdown


def _training_pairs(root: str) -> Dict[str, Dict[str, str]]:
    pairs: Dict[str, Dict[str, str]] = defaultdict(dict)
    for p in glob.glob(os.path.join(root, "training", "**", "*"), recursive=True):
        stem, ext = os.path.splitext(p)
        if ext.lower() in (".md", ".pdf"):
            rel = os.path.relpath(stem, os.path.join(root, "training"))
            pairs[rel][ext.lower()] = p
    return pairs


def convert_sft(root: str, out: str, limit: Optional[int] = None, longest_side: int = 1600, render_side: int = 2048,
                dev_frac: float = 0.01) -> dict:
    pairs = _training_pairs(root)
    with LabelWriter(out, "olmocr-synthmix-1025", longest_side=longest_side, dev_frac=dev_frac,
                     args={"root": os.path.abspath(root), "part": "training"}) as w:
        for rel in sorted(pairs):
            if limit and w.n_new >= limit:
                break
            pr = pairs[rel]
            rid = f"synthmix/{rel}"
            if w.has(rid):
                continue
            if ".md" not in pr or ".pdf" not in pr:
                w.skip("unpaired")
                continue
            with open(pr[".md"], encoding="utf-8") as f:
                md = normalize_markdown(f.read())
            if not md:
                w.skip("empty markdown")
                continue
            try:
                img = render_pdf(pr[".pdf"], longest_side=render_side, pages=[1])[0]
            except Exception:
                w.skip("render failed")
                continue
            w.add(rid, img, markdown=md, meta={"subset": rel.split(os.sep)[0], "sha256": file_sha256(pr[".pdf"])})
        return {"written": w.n_new, "total": len(w.done), "skipped": dict(w.skipped), "out": out}


def load_bench(bench_data: str) -> Dict[str, List[dict]]:
    """pdf path (relative to bench_data/pdfs) -> its tests."""
    by_pdf: Dict[str, List[dict]] = defaultdict(list)
    for jl in sorted(glob.glob(os.path.join(bench_data, "*.jsonl"))):
        cat = os.path.splitext(os.path.basename(jl))[0]
        with open(jl, encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    t = json.loads(line)
                    t["_category"] = cat
                    by_pdf[t["pdf"]].append(t)
    return dict(by_pdf)


def convert_rl(root: str, out: str, limit: Optional[int] = None, longest_side: int = 1600, render_side: int = 2048,
               dev_frac: float = 0.03) -> dict:
    bench = os.path.join(root, "bench_data")
    by_pdf = load_bench(bench)
    pairs = _training_pairs(root)
    with LabelWriter(out, "olmocr-synthmix-1025", longest_side=longest_side, dev_frac=dev_frac,
                     args={"root": os.path.abspath(root), "part": "bench_data"}) as w:
        for rel_pdf in sorted(by_pdf):
            if limit and w.n_new >= limit:
                break
            rid = f"synthmix-rl/{rel_pdf[:-4]}"
            if w.has(rid):
                continue
            path = os.path.join(bench, "pdfs", rel_pdf)
            if not os.path.exists(path):
                w.skip("missing pdf")
                continue
            try:
                img = render_pdf(path, longest_side=render_side, pages=[int(by_pdf[rel_pdf][0].get("page", 1))])[0]
            except Exception:
                w.skip("render failed")
                continue
            tests = []
            for i, t in enumerate(by_pdf[rel_pdf]):
                t = {k: v for k, v in t.items() if not k.startswith("_") and v is not None}
                t["id"] = f"{rid}::{t.get('id', i)}"            # unique across RL sets
                tests.append(t)
            ref = pairs.get(rel_pdf[:-4], {}).get(".md")
            md = None
            if ref:
                with open(ref, encoding="utf-8") as f:
                    md = normalize_markdown(f.read()) or None
            w.add(rid, img, markdown=md, extra={"tests": tests, "kind": "page"},
                  meta={"subset": rel_pdf.split("/")[0], "has_reference": md is not None, "sha256": file_sha256(path)})
        return {"written": w.n_new, "total": len(w.done), "skipped": dict(w.skipped), "out": out}
