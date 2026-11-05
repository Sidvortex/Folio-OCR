"""olmOCR-Bench: render pages, run an engine, write candidate Markdown, score.

Get the benchmark once (about 1,400 one-page PDFs and 7,010 tests):

    hf download allenai/olmOCR-bench --repo-type dataset --local-dir olmOCR-bench

and point ``--bench`` at ``olmOCR-bench/bench_data`` (it holds ``pdfs/`` and
the category ``*.jsonl`` files). Candidate Markdown is written where olmOCR's
own scorer expects it (``<bench>/<name>/<category>/<doc>_pg1_repeat1.md``), so
``python -m olmocr.bench.benchmark --dir <bench> --candidate <name>`` gives the
official number; our scorer (same test classes, same averaging) also writes
per-test results for paired bootstrap comparisons.

Settings:
    common  every engine sees the same PNG, longest side 1,288 px (the rendering
            LlamaIndex used to compare systems)
    native  each engine renders the PDF its own way (Chandra, Surya: their CLI;
            Folio/Infinity-Parser2: 300 DPI, like infinity_parser2)
"""
from __future__ import annotations

import glob
import json
import os
import time
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from .engines import PageItem, safe_key
from .stats import bootstrap_ci
from .unittests import clean_test, summarize


def load_bench_tests(bench: str) -> Tuple[List[dict], Dict[str, str]]:
    """All tests (with ``category`` = jsonl name) and pdf -> category."""
    tests, pdf_cat = [], {}
    for path in sorted(glob.glob(os.path.join(bench, "*.jsonl"))):
        cat = os.path.splitext(os.path.basename(path))[0]
        with open(path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    t = json.loads(line)
                    if t.get("checked") == "rejected":       # a reviewer threw this test out (holdout review)
                        continue
                    t["_category"] = cat
                    tests.append(t)
                    pdf_cat.setdefault(t["pdf"], cat)
    return tests, pdf_cat


def list_items(bench: str, tests: List[dict], categories: Optional[List[str]] = None) -> List[PageItem]:
    pages = {(t["pdf"], int(t.get("page", 1))) for t in tests if not categories or t["_category"] in categories}
    pdf_root = Path(bench) / "pdfs"
    if not categories:
        for p in pdf_root.rglob("*.pdf"):
            rel = str(p.relative_to(pdf_root))
            if not any(rel == q for q, _ in pages):
                pages.add((rel, 1))
    items = [PageItem(key=safe_key(f"{pdf[:-4]}_pg{page}"), pdf=str(pdf_root / pdf), page=page)
             for pdf, page in sorted(pages)]
    return items


def render_items(items: List[PageItem], cache: str, longest_side: Optional[int], dpi: float) -> None:
    from ..imaging import render_pdf
    os.makedirs(cache, exist_ok=True)
    for it in items:
        png = os.path.join(cache, f"{it.key}.png")
        if not os.path.exists(png):
            img = render_pdf(it.pdf, dpi=dpi, longest_side=longest_side, pages=[it.page])[0]
            img.save(png)
        it.image = png


def stratified_sample(items: List[PageItem], bench: str, n: int, seed: int = 0) -> List[PageItem]:
    """About ``n`` pages, the same share from every category folder (deterministic)."""
    import random
    groups: Dict[str, List[PageItem]] = defaultdict(list)
    for it in items:
        groups[os.path.relpath(it.pdf, os.path.join(bench, "pdfs")).split(os.sep)[0]].append(it)
    frac = min(1.0, n / max(1, len(items)))
    out = []
    for g in sorted(groups):
        lst = sorted(groups[g], key=lambda x: x.key)
        # one fixed order per category: a bigger sample, or a sample of fewer categories,
        # always contains the pages of the smaller one
        random.Random(f"{seed}:{g}").shuffle(lst)
        out += lst[:max(1, round(frac * len(lst)))]
    return sorted(out, key=lambda x: x.key)


def candidate_path(bench: str, name: str, item: PageItem) -> str:
    rel = os.path.relpath(item.pdf, os.path.join(bench, "pdfs"))
    return os.path.join(bench, name, f"{rel[:-4]}_pg{item.page}_repeat1.md")


def write_candidate(bench: str, name: str, items: List[PageItem], md: Dict[str, str]) -> None:
    for it in items:
        if it.key in md:
            p = candidate_path(bench, name, it)
            os.makedirs(os.path.dirname(p), exist_ok=True)
            with open(p, "w", encoding="utf-8") as f:
                f.write(md[it.key] or "")


def score_candidate(bench: str, name: str, tests: List[dict], categories: Optional[List[str]] = None,
                    only_pdfs: Optional[set] = None) -> Dict:
    """Same semantics as olmocr.bench.benchmark: per-jsonl pass rates, averaged;
    a baseline test per PDF; a test passes if most repeats pass. ``only_pdfs``
    restricts scoring to a page sample (partial runs)."""
    from olmocr.bench.tests import BaselineTest, load_single_test
    results = []
    pdfs_seen = set()
    cand = os.path.join(bench, name)
    md_cache: Dict[Tuple[str, int], List[str]] = {}

    def mds(pdf: str, page: int) -> List[str]:
        key = (pdf, page)
        if key not in md_cache:
            stem = os.path.join(cand, pdf[:-4])
            files = sorted(glob.glob(f"{glob.escape(stem)}_pg{page}_repeat*.md"))
            md_cache[key] = [open(f, encoding="utf-8").read() for f in files]
        return md_cache[key]

    for t in tests:
        if categories and t["_category"] not in categories:
            continue
        if only_pdfs is not None and t["pdf"] not in only_pdfs:
            continue
        pdfs_seen.add(t["pdf"])
        obj = load_single_test(clean_test(t))
        texts = mds(t["pdf"], int(t.get("page", 1)))
        if not texts:
            passed, why = False, "missing output"
        else:
            votes = [obj.run(x)[0] for x in texts]
            passed, why = sum(votes) > len(votes) / 2, ""
        results.append({"id": t["id"], "page": t["pdf"], "category": t["_category"], "type": t["type"], "passed": passed,
                        "why": why})
    for pdf in sorted(pdfs_seen):
        b = BaselineTest(id=f"{pdf}_baseline", pdf=pdf, page=1, type="baseline")
        texts = mds(pdf, 1)
        passed = bool(texts) and sum(b.run(x)[0] for x in texts) > len(texts) / 2
        results.append({"id": b.id, "page": pdf, "category": "baseline", "type": "baseline", "passed": passed, "why": ""})
    s = summarize(results)
    s["ci95"] = bootstrap_ci(results)
    s["results"] = results
    return s


def run(args, engine) -> Dict:
    tests, _ = load_bench_tests(args.bench)
    cats = args.categories.split(",") if args.categories else None
    if cats:
        known = sorted({t["_category"] for t in tests})
        unknown = [c for c in cats if c not in known]
        if unknown:
            raise SystemExit(f"unknown categories {unknown}; this benchmark has: {', '.join(known)}")
    items = list_items(args.bench, tests, cats)
    # sample first, then shard: the shards of a sample are exactly that sample, split
    if getattr(args, "sample", None):
        items = stratified_sample(items, args.bench, args.sample, args.seed)
    if args.shard:
        i, n = (int(x) for x in args.shard.split("/"))
        items = items[i::n]
    if args.limit:
        items = items[:args.limit]
    todo = [it for it in items if not os.path.exists(candidate_path(args.bench, args.name, it))]
    print(f"{len(items)} pages selected, {len(items) - len(todo)} already done, {len(todo)} to run")
    if todo and engine is not None:
        use_pdf = args.render == "native" and engine.native_pdf
        if not use_pdf:
            mode = f"long{args.longest_side}" if args.render == "common" else f"dpi{int(args.dpi)}"
            render_items(todo, os.path.join(args.cache, mode), args.longest_side if args.render == "common" else None,
                         args.dpi)
        t0 = time.time()
        peak = None
        try:
            import torch
            if torch.cuda.is_available():
                torch.cuda.reset_peak_memory_stats()
        except ImportError:
            torch = None
        for s in range(0, len(todo), args.chunk):
            chunk = todo[s:s + args.chunk]
            md = engine.run(chunk)
            write_candidate(args.bench, args.name, chunk, md)
            done = s + len(chunk)
            rate = done / (time.time() - t0)
            print(f"  {done}/{len(todo)} pages, {rate:.2f} pages/s", flush=True)
        if torch is not None and torch.cuda.is_available():
            peak = torch.cuda.max_memory_allocated() / 2**30
        timing = {"pages": len(todo), "seconds": round(time.time() - t0, 1), "peak_gpu_gb": peak}
    else:
        timing = {}
    partial = bool(args.limit or args.shard or getattr(args, "sample", None))
    only = {os.path.relpath(it.pdf, os.path.join(args.bench, "pdfs")) for it in items} if partial else None
    summary = score_candidate(args.bench, args.name, tests, cats, only_pdfs=only)
    if partial:
        print(f"note: scored on the {len(only)} selected PDFs only (a sample: wider confidence interval)")
    out = Path(args.results) / f"{args.name}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    meta = {"name": args.name, "engine": args.engine, "render": args.render, "longest_side": args.longest_side,
            "dpi": args.dpi, "model": getattr(args, "model", None), "adapter": getattr(args, "adapter", None),
            "max_pixels": getattr(args, "max_pixels", None), "timing": timing, "categories": cats,
            "limit": args.limit, "shard": args.shard, "sample": getattr(args, "sample", None),
            "seed": getattr(args, "seed", None), "pages_scored": len(only) if only is not None else None}
    out.write_text(json.dumps({"meta": meta, **summary}, indent=1))
    print(format_bench(summary, args.name))
    print(f"per-test results -> {out}")
    return summary


def format_bench(s: Dict, name: str) -> str:
    lo, hi = s.get("ci95", (float("nan"), float("nan")))
    lines = [f"{name}: {100 * s['overall']:.1f}  (95% CI {100 * lo:.1f}-{100 * hi:.1f}, {s['n_tests']} tests)"]
    for c, v in s["categories"].items():
        lines.append(f"    {c:22s} {100 * v['rate']:5.1f}  ({v['passed']}/{v['total']})")
    return "\n".join(lines)
