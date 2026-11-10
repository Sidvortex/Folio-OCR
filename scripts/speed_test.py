#!/usr/bin/env python3
"""Pages per second, peak GPU memory and cost per 1,000 pages for one engine.

    python scripts/speed_test.py --images data/resumes/images --pages 30 --engine folio --adapter runs/resume-t4/best
    python scripts/speed_test.py --images data/resumes/images --pages 30 --engine folio --load-4bit --gpu-price 0.0

Warm-up pages are excluded. Report the GPU name with every number: speed
claims from different hardware are not comparable (the plan's rule).
"""
import argparse
import json
import os
import resource
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from folioocr.bench.engines import PageItem, add_engine_args, make_engine  # noqa: E402
from folioocr.imaging import IMAGE_EXTS  # noqa: E402


def gpu_name():
    try:
        import torch
        return torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu"
    except ImportError:
        return "cpu"


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--images", required=True)
    ap.add_argument("--pages", type=int, default=20)
    ap.add_argument("--warmup", type=int, default=2)
    ap.add_argument("--gpu-price", type=float, default=None, help="USD per GPU-hour, for cost per 1,000 pages")
    ap.add_argument("--out", default="results/speed")
    ap.add_argument("--name", default=None)
    ap.add_argument("--render", default="common", help=argparse.SUPPRESS)
    add_engine_args(ap)
    args = ap.parse_args()
    files = sorted(f for f in os.listdir(args.images) if os.path.splitext(f)[1].lower() in IMAGE_EXTS)
    files = files[: args.pages + args.warmup]
    items = [PageItem(key=os.path.splitext(f)[0], image=os.path.join(args.images, f)) for f in files]
    engine = make_engine(args)
    engine.run(items[: args.warmup])
    try:
        import torch
        if torch.cuda.is_available():
            torch.cuda.synchronize()
            torch.cuda.reset_peak_memory_stats()
    except ImportError:
        torch = None
    t0 = time.time()
    engine.run(items[args.warmup:])
    if torch is not None and torch.cuda.is_available():
        torch.cuda.synchronize()
    dt = time.time() - t0
    n = len(items) - args.warmup
    res = {"engine": args.engine, "model": args.model, "adapter": args.adapter, "gpu": gpu_name(), "pages": n,
           "seconds": round(dt, 2), "pages_per_second": round(n / dt, 3), "seconds_per_page": round(dt / n, 2),
           "peak_gpu_gb": round(torch.cuda.max_memory_allocated() / 2**30, 2) if torch is not None and torch.cuda.is_available() else None,
           "peak_rss_gb": round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 2**20, 2),
           "max_pixels": args.max_pixels, "load_4bit": args.load_4bit}
    if args.gpu_price is not None:
        res["usd_per_1000_pages"] = round(args.gpu_price * (dt / n) * 1000 / 3600, 3)
    os.makedirs(args.out, exist_ok=True)
    name = args.name or f"{args.engine}_{res['gpu'].replace(' ', '_')}"
    with open(os.path.join(args.out, f"{name}.json"), "w") as f:
        json.dump(res, f, indent=1)
    print(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
