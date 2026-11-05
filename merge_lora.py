#!/usr/bin/env python3
"""Merge LoRA adapters into the base model (Phase 4), optionally averaging several fine-tunes.

    # one adapter -> a standalone model folder (loads with --model <out> everywhere, no --adapter)
    python merge_lora.py --adapter runs/rl-t4/best --out models/folio-v1

    # average the best SFT and RL adapters in weight space; keep it only if dev scores improve
    python merge_lora.py --adapter runs/general-t4/best runs/rl-t4/best --weights 0.5 0.5 --out models/folio-avg
    python bench_dev.py --labels data/dev_old/labels.jsonl --name avg --model models/folio-avg

    # then a 4-bit copy for consumer GPUs (needs CUDA), and re-run every benchmark on it
    python merge_lora.py quantize --model models/folio-v1 --out models/folio-v1-nf4 --bits 4

Merging runs on the CPU (a 2B model needs about 5 GB of RAM in bfloat16).
"""
import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


def main() -> None:
    if len(sys.argv) > 1 and sys.argv[1] == "quantize":
        ap = argparse.ArgumentParser(description="4- or 8-bit copy of a merged model (bitsandbytes, CUDA)")
        ap.add_argument("cmd")
        ap.add_argument("--model", required=True)
        ap.add_argument("--out", required=True)
        ap.add_argument("--bits", type=int, default=4, choices=[4, 8])
        args = ap.parse_args()
        from folioocr.model.merge import quantize
        print(json.dumps(quantize(args.model, args.out, args.bits), indent=1))
        return
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--base", default="infly/Infinity-Parser2-Flash")
    ap.add_argument("--adapter", nargs="+", required=True)
    ap.add_argument("--weights", nargs="+", type=float, default=None, help="default: equal weights")
    ap.add_argument("--out", required=True)
    ap.add_argument("--dtype", default="auto", choices=["auto", "float32", "bfloat16", "float16"])
    args = ap.parse_args()
    from folioocr.model.merge import merge
    print(json.dumps(merge(args.base, args.adapter, args.out, args.weights, args.dtype), indent=1))


if __name__ == "__main__":
    main()
