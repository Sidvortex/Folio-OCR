#!/usr/bin/env python3
"""Contamination check: index the evaluation pages, then check training data against them.

    # 1) index every evaluation set once (pages, text, test strings, file hashes)
    python scripts/dedup.py index --out eval_index.npz --olmocr olmOCR-bench/bench_data \\
        --omnidoc OmniDocBench --resume resume_bench --holdout holdout/bench_data

    # 2) check each training source; --write-clean writes labels.clean.jsonl next to it
    python scripts/dedup.py check --index eval_index.npz --labels data/resumes/labels.jsonl data/olmocr_loc/labels.jsonl \\
        --report results/contamination.json --write-clean

    # 3) (optional) see how far apart pages are on your data before trusting the image threshold
    python scripts/dedup.py calibrate --images data/general/images --n 60

Train on the ``labels.clean.jsonl`` files and publish the report's counts with
any claim (the plan's rule). See folioocr/data/dedup.py for the four signals.
"""
import argparse
import glob
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    i = sub.add_parser("index")
    i.add_argument("--out", required=True)
    i.add_argument("--olmocr", default=None, help="olmOCR-bench/bench_data")
    i.add_argument("--omnidoc", default=None, help="OmniDocBench folder (OmniDocBench.json + images/)")
    i.add_argument("--resume", default=None, help="a frozen Resume-Bench folder")
    i.add_argument("--holdout", default=None, help="private holdout in olmOCR-Bench layout")
    i.add_argument("--labels", nargs="*", default=[], help="other evaluation label files (name=path)")
    i.add_argument("--phash-bits", type=int, default=24)
    c = sub.add_parser("check")
    c.add_argument("--index", required=True)
    c.add_argument("--labels", nargs="+", required=True)
    c.add_argument("--report", default="results/contamination.json")
    c.add_argument("--write-clean", action="store_true")
    c.add_argument("--phash-bits", type=int, default=None, help="override the index's image threshold")
    k = sub.add_parser("calibrate")
    k.add_argument("--images", required=True)
    k.add_argument("--n", type=int, default=60)
    args = ap.parse_args()

    from folioocr.data import dedup as D
    if args.cmd == "index":
        idx = D.EvalIndex(phash_bits=args.phash_bits)
        sets = []
        if args.olmocr:
            sets.append(D.pages_olmocr_bench(args.olmocr, "olmocr-bench"))
        if args.omnidoc:
            sets.append(D.pages_omnidoc(args.omnidoc))
        if args.resume:
            sets.append(D.pages_resume_bench(args.resume))
        if args.holdout:
            sets.append(D.pages_olmocr_bench(args.holdout, "holdout"))
        for spec in args.labels:
            name, _, path = spec.partition("=")
            sets.append(D.pages_labels(path or name, name if path else os.path.basename(os.path.dirname(name))))
        if not sets:
            ap.error("give at least one evaluation set")
        for pages in sets:
            for p in pages:
                idx.add(p)
        os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
        idx.save(args.out)
        print(json.dumps(idx.summary(), indent=1))
    elif args.cmd == "check":
        idx = D.EvalIndex.load(args.index, phash_bits=args.phash_bits)
        reports = [D.check_labels(idx, p, write_clean=args.write_clean) for p in args.labels]
        total = {"records": sum(r["records"] for r in reports), "dropped": sum(r["dropped"] for r in reports)}
        os.makedirs(os.path.dirname(os.path.abspath(args.report)), exist_ok=True)
        with open(args.report, "w", encoding="utf-8") as f:
            json.dump({"index": os.path.abspath(args.index), "index_summary": idx.summary(), "total": total,
                       "sources": reports}, f, indent=1, ensure_ascii=False)
        for r in reports:
            print(f"{r['labels']}: {r['dropped']}/{r['records']} match an evaluation page "
                  f"(by signal {r['by_signal']}, by set {r['by_eval_set']})")
        print(f"-> {args.report}")
    else:
        files = sorted(glob.glob(os.path.join(args.images, "*")))
        print(json.dumps(D.calibrate(files, n=args.n), indent=1))


if __name__ == "__main__":
    main()
