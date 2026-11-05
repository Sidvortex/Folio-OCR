#!/usr/bin/env python3
"""Evaluate a model on a synthetic (or any labelled) dev set: unit tests, CER, fields, loops, speed.

    python bench_dev.py --labels data/resumes/labels.jsonl --tests data/resumes/tests.jsonl --name flash
    python bench_dev.py --labels data/resumes/labels.jsonl --tests data/resumes/tests.jsonl --name v1 --adapter runs/resume-t4/best

Uses pages with split=dev by default (``--split all`` for everything). Writes
results/dev/<name>.json with per-test results, so ``compare.py`` can test
whether one model beats another on the same pages.
"""
import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--labels", required=True)
    ap.add_argument("--tests", default=None)
    ap.add_argument("--name", required=True)
    ap.add_argument("--split", default="dev")
    ap.add_argument("--where", nargs="*", default=[],
                    help="keep records whose field has this value, e.g. condition=old columns=2 source=olmocr-mix-1025")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--results", default="results/dev")
    ap.add_argument("--model", default="infly/Infinity-Parser2-Flash")
    ap.add_argument("--adapter", default=None)
    ap.add_argument("--task", default="doc2json", choices=["doc2json", "doc2md"])
    ap.add_argument("--max-pixels", type=int, default=2_000_000)
    ap.add_argument("--dtype", default="auto")
    ap.add_argument("--load-4bit", action="store_true")
    ap.add_argument("--batch-size", type=int, default=1)
    ap.add_argument("--max-new-tokens", type=int, default=8192)
    args = ap.parse_args()

    from folioocr.bench.devset import evaluate_records, headline
    from folioocr.bench.unittests import load_tests_jsonl, score
    from folioocr.data.dataset import read_labels
    from folioocr.model.generate import PageParser
    from folioocr.model.load import ModelConfig, load_model

    recs = read_labels(args.labels, None if args.split == "all" else args.split)
    for cond in args.where:
        k, _, v = cond.partition("=")
        recs = [r for r in recs if str(r.get(k, (r.get("meta") or {}).get(k))) == v]
    if not recs:
        ap.error(f"no pages with split={args.split} {' '.join(args.where)} in {args.labels}")
    tests = load_tests_jsonl(args.tests) if args.tests else None
    model, proc = load_model(ModelConfig(model=args.model, adapter=args.adapter, dtype=args.dtype, load_4bit=args.load_4bit))
    parser = PageParser(model, proc, task=args.task, max_pixels=args.max_pixels, batch_size=args.batch_size,
                        max_new_tokens=args.max_new_tokens)
    m = evaluate_records(parser, recs, tests, limit=args.limit)
    outputs = m.pop("_outputs")
    if tests:
        md = {pid: o["markdown"] for pid, o in outputs.items()}
        m["results"] = score({p: t for p, t in tests.items() if p in md}, md)["results"]
    os.makedirs(args.results, exist_ok=True)
    out = os.path.join(args.results, f"{args.name}.json")
    with open(out, "w", encoding="utf-8") as f:
        json.dump({"name": args.name, "model": args.model, "adapter": args.adapter, **m, "outputs": outputs}, f,
                  ensure_ascii=False, indent=1)
    print(f"{args.name}: {headline(m)}")
    if m.get("tests_by_category"):
        print("    " + ", ".join(f"{c} {100 * v:.1f}" for c, v in m["tests_by_category"].items()))
    print(f"-> {out}")


if __name__ == "__main__":
    main()
