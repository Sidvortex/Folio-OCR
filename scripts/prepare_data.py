#!/usr/bin/env python3
"""Convert public OCR datasets to Folio labels (Phase 2 data engine).

    # real pages with GPT-4.1 / human transcriptions (doc2md records)
    python scripts/prepare_data.py olmocr-mix --parquet olmocr-mix/02_loc_transcripts_train.parquet \\
        --tarballs olmocr-mix/pdf_tarballs/02_loc_transcripts_train_00000.tar.gz --out data/olmocr_loc

    # synthetic arXiv/table pages: training pairs (SFT) and benchmark-style tests (RL)
    python scripts/prepare_data.py synthmix --root synthmix --out-sft data/synthmix --out-rl data/rl_synthmix

    # Infinity-Doc2-5M (CC BY-NC-SA, gated, academic use only)
    python scripts/prepare_data.py infinity-doc2 --root Infinity-Doc2-5M/Document_Parsing/doc2json --out data/infdoc2

    # LightOnOCR-mix transcriptions joined with pdfa-eng-wds shards
    python scripts/prepare_data.py lighton-mix --shards pdfa/pdfa-eng-train-0000.tar --parquet <files> --out data/lighton

    # look at what came out
    python scripts/prepare_data.py preview --labels data/olmocr_loc/labels.jsonl --n 8

Every converter appends and skips pages already written, so re-running after
a session ends continues where it stopped. Each output folder gets a
``source.json`` with the source's licence line for the data card.
"""
import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)

    def common(p, dev=0.01):
        p.add_argument("--limit", type=int, default=None, help="stop after this many new pages")
        p.add_argument("--longest-side", type=int, default=1600, help="stored image size")
        p.add_argument("--dev-frac", type=float, default=dev)

    p = sub.add_parser("olmocr-mix")
    p.add_argument("--parquet", required=True)
    p.add_argument("--tarballs", nargs="+", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--subset", default=None)
    p.add_argument("--keep-blank", type=float, default=0.0, help="share of empty pages to keep (empty target)")
    p.add_argument("--rotated", default="fix", choices=["fix", "skip"])
    common(p)

    p = sub.add_parser("synthmix")
    p.add_argument("--root", required=True)
    p.add_argument("--out-sft", default=None)
    p.add_argument("--out-rl", default=None)
    common(p, dev=0.03)

    p = sub.add_parser("infinity-doc2")
    p.add_argument("--root", nargs="+", required=True, help="extracted sub-task folders (images/, labels/)")
    p.add_argument("--out", required=True)
    p.add_argument("--bbox-scale", default="auto", choices=["auto", "pixels", "1000"])
    p.add_argument("--subtasks", default="doc2json,doc2md")
    common(p)

    p = sub.add_parser("lighton-mix")
    p.add_argument("--shards", nargs="+", required=True)
    p.add_argument("--parquet", nargs="+", required=True)
    p.add_argument("--out", required=True)
    common(p)

    p = sub.add_parser("preview")
    p.add_argument("--labels", required=True)
    p.add_argument("--n", type=int, default=8)
    p.add_argument("--out", default=None, help="default: <labels dir>/preview")

    args = ap.parse_args()
    if args.cmd == "olmocr-mix":
        from folioocr.data.sources import olmocr_mix
        r = olmocr_mix.convert(args.parquet, args.tarballs, args.out, subset=args.subset, limit=args.limit,
                               longest_side=args.longest_side, keep_blank=args.keep_blank, rotated=args.rotated,
                               dev_frac=args.dev_frac)
    elif args.cmd == "synthmix":
        from folioocr.data.sources import synthmix
        if not (args.out_sft or args.out_rl):
            ap.error("give --out-sft and/or --out-rl")
        r = {}
        if args.out_sft:
            r["sft"] = synthmix.convert_sft(args.root, args.out_sft, limit=args.limit, longest_side=args.longest_side,
                                            dev_frac=args.dev_frac)
        if args.out_rl:
            r["rl"] = synthmix.convert_rl(args.root, args.out_rl, limit=args.limit, longest_side=args.longest_side,
                                          dev_frac=args.dev_frac)
    elif args.cmd == "infinity-doc2":
        from folioocr.data.sources import infinity_doc2
        print("note: Infinity-Doc2-5M is CC BY-NC-SA 4.0, academic use only; models trained on it inherit that.",
              file=sys.stderr)
        r = infinity_doc2.convert(args.root, args.out, limit=args.limit, longest_side=args.longest_side,
                                  bbox_scale=args.bbox_scale, dev_frac=args.dev_frac,
                                  subtasks=tuple(args.subtasks.split(",")))
    elif args.cmd == "lighton-mix":
        from folioocr.data.sources import lighton_mix
        r = lighton_mix.convert(args.shards, args.parquet, args.out, limit=args.limit, longest_side=args.longest_side,
                                dev_frac=args.dev_frac)
    else:
        r = preview(args.labels, args.n, args.out)
    print(json.dumps(r, indent=1, ensure_ascii=False))


def preview(labels: str, n: int, out=None) -> dict:
    from PIL import Image
    from folioocr.render.make_dataset import draw_preview
    root = os.path.dirname(os.path.abspath(labels))
    out = out or os.path.join(root, "preview")
    os.makedirs(out, exist_ok=True)
    done = 0
    with open(labels, encoding="utf-8") as f:
        for line in f:
            if done >= n:
                break
            if not line.strip():
                continue
            r = json.loads(line)
            stem = os.path.splitext(os.path.basename(r["image"]))[0]
            if "blocks" in r:
                draw_preview(r, root, os.path.join(out, f"{stem}.png"))
                text = "\n\n".join(f"[{b['category']}] {b['text']}" for b in r["blocks"])
            else:
                Image.open(os.path.join(root, r["image"])).convert("RGB").save(os.path.join(out, f"{stem}.jpg"))
                text = r.get("markdown") or ""
            if r.get("tests"):
                text += "\n\n---- tests ----\n" + "\n".join(json.dumps(t, ensure_ascii=False) for t in r["tests"][:40])
            with open(os.path.join(out, f"{stem}.txt"), "w", encoding="utf-8") as g:
                g.write(text)
            done += 1
    return {"previews": done, "out": out}


if __name__ == "__main__":
    main()
