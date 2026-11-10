#!/usr/bin/env python3
"""Parse PDFs or page images into Markdown (and layout JSON).

    python scripts/infer.py resume.pdf                                   # base Infinity-Parser2-Flash
    python scripts/infer.py resume.pdf --adapter runs/resume-t4/best     # our fine-tune
    python scripts/infer.py scans/ --out out/ --json --second-look
    python scripts/infer.py resume.jpg --adapter runs/resume-t4/best --crosscheck ../ResumeOCR-Lite
    python scripts/infer.py page.png --raw                               # print raw model output (Phase 0: check its format)

Writes <out>/<name>.md (pages separated by blank lines) and, with --json,
<out>/<name>.json with blocks, boxes (0-1000) and per-page flags.
"""
import argparse
import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from folioocr.imaging import IMAGE_EXTS, load_pages  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("inputs", nargs="+", help="PDF/image files or folders")
    ap.add_argument("--out", default="out")
    ap.add_argument("--model", default="infly/Infinity-Parser2-Flash")
    ap.add_argument("--adapter", default=None)
    ap.add_argument("--task", default="doc2json", choices=["doc2json", "doc2md"])
    ap.add_argument("--max-pixels", type=int, default=2_000_000)
    ap.add_argument("--dpi", type=float, default=200, help="PDF render DPI")
    ap.add_argument("--dtype", default="auto")
    ap.add_argument("--load-4bit", action="store_true")
    ap.add_argument("--max-new-tokens", type=int, default=8192)
    ap.add_argument("--keep-headers", action="store_true", help="keep running headers/footers in the Markdown")
    ap.add_argument("--second-look", action="store_true")
    ap.add_argument("--crosscheck", default=None, help="path to ResumeOCR-Lite (flags field disagreements)")
    ap.add_argument("--json", action="store_true")
    ap.add_argument("--raw", action="store_true", help="print the raw model output of each page")
    args = ap.parse_args()

    files = []
    for p in args.inputs:
        if os.path.isdir(p):
            files += sorted(os.path.join(p, f) for f in os.listdir(p)
                            if os.path.splitext(f)[1].lower() in IMAGE_EXTS | {".pdf"})
        else:
            files.append(p)
    if not files:
        ap.error("no input files")

    from folioocr.infer import CrossChecker, parse_document
    from folioocr.model.generate import PageParser
    from folioocr.model.load import ModelConfig, load_model

    model, proc = load_model(ModelConfig(model=args.model, adapter=args.adapter, dtype=args.dtype, load_4bit=args.load_4bit))
    parser = PageParser(model, proc, task=args.task, max_pixels=args.max_pixels, max_new_tokens=args.max_new_tokens,
                        keep_non_body=args.keep_headers)
    checker = CrossChecker(args.crosscheck) if args.crosscheck else None
    os.makedirs(args.out, exist_ok=True)
    for path in files:
        t0 = time.time()
        pages = load_pages(path, dpi=args.dpi)
        res = parse_document(parser, pages, second=args.second_look, checker=checker, keep_raw=args.raw)
        stem = os.path.splitext(os.path.basename(path))[0]
        with open(os.path.join(args.out, f"{stem}.md"), "w", encoding="utf-8") as f:
            f.write("\n\n".join(r["markdown"] for r in res))
        if args.json:
            with open(os.path.join(args.out, f"{stem}.json"), "w", encoding="utf-8") as f:
                json.dump({"file": path, "pages": res}, f, ensure_ascii=False, indent=1)
        flags = sum(len(r.get("field_flags", [])) for r in res)
        print(f"{path}: {len(pages)} page(s) in {time.time() - t0:.1f}s"
              + (f", {flags} field flag(s)" if checker else "")
              + (", looped" if any(r["looped"] for r in res) else ""))
        if args.raw:
            for r in res:
                print(f"----- raw output, page {r['page']} ({r['tokens']} tokens) -----\n{r.pop('raw')}")


if __name__ == "__main__":
    main()
