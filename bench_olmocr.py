#!/usr/bin/env python3
"""Run one engine on olmOCR-Bench and score it (Phase 0 baselines, gate checks).

    # Infinity-Parser2-Flash, common setting (1,288 px), free T4: cap the visual budget
    python bench_olmocr.py --bench olmOCR-bench/bench_data --name flash_common \\
        --engine folio --model infly/Infinity-Parser2-Flash --render common --max-pixels 2000000

    # our fine-tune
    python bench_olmocr.py --bench olmOCR-bench/bench_data --name folio_v1 \\
        --engine folio --adapter runs/resume-t4/best --render common

    # Chandra 2 through its own CLI, its own PDF rendering
    python bench_olmocr.py --bench olmOCR-bench/bench_data --name chandra2_native --engine chandra --render native

    # split a long run over Kaggle sessions (re-run the same command to resume)
    python bench_olmocr.py ... --shard 0/3

    # free GPU budget: a stratified 300-page sample (same pages for every engine with the same --seed)
    python bench_olmocr.py ... --sample 300

    # score existing Markdown only
    python bench_olmocr.py --bench olmOCR-bench/bench_data --name flash_common --score-only
"""
import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from folioocr.bench.engines import add_engine_args, make_engine  # noqa: E402
from folioocr.bench.olmocr_bench import run  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--bench", required=True, help="olmOCR-bench/bench_data")
    ap.add_argument("--name", required=True, help="candidate name (output folder and results file)")
    ap.add_argument("--render", default="common", choices=["common", "native"])
    ap.add_argument("--longest-side", type=int, default=1288)
    ap.add_argument("--dpi", type=float, default=300)
    ap.add_argument("--cache", default="bench_cache", help="rendered page images")
    ap.add_argument("--results", default="results/olmocr", help="where <name>.json goes")
    ap.add_argument("--categories", default=None, help="comma-separated jsonl names, e.g. old_scans,multi_column")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--sample", type=int, default=None,
                    help="run a stratified sample of about N pages (same share per category) -- for free GPUs")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--shard", default=None, help="i/n: run every n-th page starting at i")
    ap.add_argument("--chunk", type=int, default=16, help="pages per engine call (and per save)")
    ap.add_argument("--score-only", action="store_true")
    add_engine_args(ap)
    args = ap.parse_args()
    engine = None if args.score_only else make_engine(args)
    run(args, engine)


if __name__ == "__main__":
    main()
