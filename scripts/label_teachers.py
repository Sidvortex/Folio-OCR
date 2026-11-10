#!/usr/bin/env python3
"""Teacher-consensus labels for real pages without a source (Phase 2).

    # 1) render the pages once, so both teachers read identical images
    python scripts/label_teachers.py stage --inputs scans/ more.pdf --stage teach/pages

    # 2) each teacher on its own (one model on the GPU at a time; re-run to resume)
    vllm serve infly/Infinity-Parser2-Pro ...            # on a box that fits it
    python scripts/label_teachers.py run --stage teach/pages --name ip2pro --licence Apache-2.0 --cache teach/ip2pro \\
        --engine openai --api-base http://localhost:8000/v1 --api-model infly/Infinity-Parser2-Pro --api-prompt doc2json
    python scripts/label_teachers.py run --stage teach/pages --name olmocr2 --licence Apache-2.0 --cache teach/olmocr2 \\
        --engine openai --api-base http://localhost:8001/v1 --api-model allenai/olmOCR-2-7B-1025 --api-prompt olmocr

    # 3) keep the pages the two agree on (>= 97% of characters); the rest go to review.jsonl
    python scripts/label_teachers.py merge --stage teach/pages --a teach/ip2pro --b teach/olmocr2 --out data/consensus

Teachers must allow training on their output: Chandra and Surya are refused.
On free GPUs, ``--engine folio`` runs Flash itself (or a fine-tune) as one of
the two teachers; record each teacher's licence with ``--licence``.
"""
import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("stage")
    s.add_argument("--inputs", nargs="+", required=True, help="image/PDF files or folders, or label .jsonl files")
    s.add_argument("--stage", required=True)
    s.add_argument("--dpi", type=float, default=200)
    r = sub.add_parser("run")
    r.add_argument("--stage", required=True)
    r.add_argument("--name", required=True)
    r.add_argument("--licence", required=True, help="the teacher's licence, recorded in the data card")
    r.add_argument("--cache", required=True)
    r.add_argument("--render", default="common", help=argparse.SUPPRESS)
    from folioocr.bench.engines import add_engine_args
    add_engine_args(r)
    m = sub.add_parser("merge")
    m.add_argument("--stage", required=True)
    m.add_argument("--a", required=True)
    m.add_argument("--b", required=True)
    m.add_argument("--out", required=True)
    m.add_argument("--threshold", type=float, default=0.97)
    m.add_argument("--dev-frac", type=float, default=0.0)
    args = ap.parse_args()

    from folioocr.data import teachers as T
    if args.cmd == "stage":
        pages = T.stage_pages(args.inputs, args.stage, dpi=args.dpi)
        with open(os.path.join(args.stage, "pages.json"), "w", encoding="utf-8") as f:
            json.dump(pages, f)
        print(f"{len(pages)} pages staged -> {args.stage}/pages.json")
        return
    with open(os.path.join(args.stage, "pages.json"), encoding="utf-8") as f:
        pages = [tuple(x) for x in json.load(f)]
    if args.cmd == "run":
        if args.engine in T.FORBIDDEN:
            ap.error(f"--engine {args.engine}: its licence does not allow training on its output")
        from folioocr.bench.engines import make_engine
        print(json.dumps(T.run_teacher(make_engine(args), args.name, pages, args.cache, args.licence,
                                       engine_kind=args.engine)))
    else:
        print(json.dumps(T.merge(pages, args.a, args.b, args.out, threshold=args.threshold, dev_frac=args.dev_frac),
                         indent=1))


if __name__ == "__main__":
    main()
