#!/usr/bin/env python3
"""Write OmniDocBench predictions (one Markdown file per page image) for its official scorer.

    hf download opendatalab/OmniDocBench --repo-type dataset --local-dir OmniDocBench
    python bench_omnidoc.py --images OmniDocBench/images --name folio_v1 --adapter runs/resume-t4/best

Then score with the official code (github.com/opendatalab/OmniDocBench): set
the ground-truth JSON and ``results/omnidoc/<name>`` as the prediction folder
in its end-to-end config, and run its evaluation script as its README says.
We do not re-implement OmniDocBench's metrics (edit distance, CDM, TEDS,
reading order); the plan's rule is to report the official numbers only.
"""
import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from folioocr.bench.engines import PageItem, add_engine_args, make_engine  # noqa: E402
from folioocr.imaging import IMAGE_EXTS  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--images", required=True)
    ap.add_argument("--name", required=True)
    ap.add_argument("--results", default="results/omnidoc")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--shard", default=None, help="i/n")
    ap.add_argument("--render", default="common", help=argparse.SUPPRESS)
    add_engine_args(ap)
    args = ap.parse_args()
    files = sorted(f for f in os.listdir(args.images) if os.path.splitext(f)[1].lower() in IMAGE_EXTS)
    if args.shard:
        i, n = (int(x) for x in args.shard.split("/"))
        files = files[i::n]
    if args.limit:
        files = files[:args.limit]
    out = os.path.join(args.results, args.name)
    os.makedirs(out, exist_ok=True)
    todo = [f for f in files if not os.path.exists(os.path.join(out, os.path.splitext(f)[0] + ".md"))]
    print(f"{len(files)} images, {len(todo)} to run -> {out}")
    engine = make_engine(args)
    for s in range(0, len(todo), 16):
        chunk = [PageItem(key=os.path.splitext(f)[0], image=os.path.join(args.images, f)) for f in todo[s:s + 16]]
        for key, md in engine.run(chunk).items():
            with open(os.path.join(out, f"{key}.md"), "w", encoding="utf-8") as fh:
                fh.write(md or "")
        print(f"  {min(s + 16, len(todo))}/{len(todo)}", flush=True)


if __name__ == "__main__":
    main()
