#!/usr/bin/env python3
"""Resume-Bench: build it once from ResumeOCR-Lite ground truth, then score engines.

    # 1) freeze the benchmark (tests never change after this)
    python scripts/bench_resume.py build --gt ../ResumeOCR-Lite/data/realtest/gt.jsonl --out resume_bench

    # 2) run engines on it
    python scripts/bench_resume.py run --bench resume_bench --name flash --engine folio --model infly/Infinity-Parser2-Flash
    python scripts/bench_resume.py run --bench resume_bench --name folio_v1 --engine folio --adapter runs/resume-t4/best
    python scripts/bench_resume.py run --bench resume_bench --name chandra2 --engine chandra

Synthetic pages (make_dataset output) work too: ``build --synthetic data/synth/labels.jsonl``
turns their line labels into the same format (useful before real resumes exist).
"""
import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from folioocr.bench import resume_bench as rb  # noqa: E402
from folioocr.bench.engines import PageItem, add_engine_args, make_engine, safe_key  # noqa: E402
from folioocr.textnorm import markdown_to_text  # noqa: E402


def synthetic_to_gt(labels: str, out_gt: str, split: str) -> str:
    root = os.path.dirname(os.path.abspath(labels))
    with open(labels, encoding="utf-8") as f, open(out_gt, "w", encoding="utf-8") as g:
        for line in f:
            r = json.loads(line)
            if split and split != "all" and r.get("split") != split:
                continue
            tier = {"clean": "digital", "scan": "scan", "old": "scan", "photo": "photo"}[r["condition"]]
            lines = [{"box": l["box"], "text": l["text"], "ignore": bool(l.get("math"))} for l in r["lines"]]
            # tables carry no line labels: count their rows for CER and coverage, without tests
            # (a real Resume-Bench page has every table row as a GT line)
            for b in r["blocks"]:
                if b["category"] != "table":
                    continue
                x1, y1, x2, y2 = b["bbox"]
                box = [x1 * r["width"] / 1000, y1 * r["height"] / 1000, x2 * r["width"] / 1000, y2 * r["height"] / 1000]
                for row in markdown_to_text(b["text"]).split("\n"):
                    if row.strip():
                        lines.append({"box": box, "text": row.strip(), "ignore": False, "notest": True})
            g.write(json.dumps({"image": os.path.join(root, r["image"]), "id": r["id"], "tier": tier, "split": "test",
                                "lines": lines}) + "\n")
    return out_gt


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    b = sub.add_parser("build")
    b.add_argument("--gt", default=None, help="ResumeOCR-Lite gt.jsonl")
    b.add_argument("--synthetic", default=None, help="make_dataset labels.jsonl instead of real GT")
    b.add_argument("--synthetic-split", default="dev", help="dev, train or all")
    b.add_argument("--split", default="test")
    b.add_argument("--out", required=True)
    r = sub.add_parser("run")
    r.add_argument("--bench", required=True)
    r.add_argument("--name", required=True)
    r.add_argument("--results", default="results/resume")
    r.add_argument("--render", default="common", help=argparse.SUPPRESS)
    add_engine_args(r)
    # a resume's footer can hold its email or phone number: every engine transcribes headers and footers here
    r.set_defaults(keep_headers=True)
    r.add_argument("--drop-headers", dest="keep_headers", action="store_false",
                   help="drop page headers/footers like olmOCR-Bench does (not the Resume-Bench setting)")
    s = sub.add_parser("score")
    s.add_argument("--bench", required=True)
    s.add_argument("--name", required=True)
    s.add_argument("--results", default="results/resume")
    args = ap.parse_args()

    if args.cmd == "build":
        gt = args.gt
        split = args.split
        if args.synthetic:
            os.makedirs(args.out, exist_ok=True)
            gt = synthetic_to_gt(args.synthetic, os.path.join(args.out, "synthetic_gt.jsonl"), args.synthetic_split)
            split = "test"
        if not gt:
            ap.error("--gt or --synthetic is required")
        print(json.dumps(rb.build(gt, args.out, split), indent=1))
        return

    pages = rb.load(args.bench)
    md_dir = os.path.join(args.results, args.name)
    os.makedirs(md_dir, exist_ok=True)
    if args.cmd == "run":
        engine = make_engine(args)
        todo = [p for p in pages if not os.path.exists(os.path.join(md_dir, f"{safe_key(p['id'])}.md"))]
        for i in range(0, len(todo), 8):
            chunk = [PageItem(key=safe_key(p["id"]), image=p["image"]) for p in todo[i:i + 8]]
            for k, md in engine.run(chunk).items():
                with open(os.path.join(md_dir, f"{k}.md"), "w", encoding="utf-8") as f:
                    f.write(md or "")
            print(f"  {min(i + 8, len(todo))}/{len(todo)} pages", flush=True)
    md = {}
    for p in pages:
        path = os.path.join(md_dir, f"{safe_key(p['id'])}.md")
        md[p["id"]] = open(path, encoding="utf-8").read() if os.path.exists(path) else ""
    res = rb.score(pages, md, os.path.join(args.bench, "tests.jsonl"))
    out = os.path.join(args.results, f"{args.name}.json")
    with open(out, "w", encoding="utf-8") as f:
        json.dump({"name": args.name, **res}, f, indent=1)
    print(rb.format_score(res, args.name))
    print(f"-> {out}")


if __name__ == "__main__":
    main()
