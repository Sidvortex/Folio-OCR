#!/usr/bin/env python3
"""Build RL items (Phase 3): page bands with their own unit tests, from rendered label files.

    python make_rl_set.py --labels data/resumes/labels.jsonl data/render_old/labels.jsonl \\
        data/render_layout/labels.jsonl --out data/rl_regions --per-page 2 --max-chars 1200

Each item is a band of a page cut where no block crosses (or the whole page when
it is short), its blocks re-normalised to the crop, and tests generated from
them. A page's dev/train split carries over to its items. Weight the item mix
toward what the plan targets (old scans, multi-column, tables, resumes) by
the label files you pass and by ``--data a:w`` weights in train_rl.py.
olmOCR-synthmix pages come from ``prepare_data.py synthmix --out-rl``.
"""
import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--labels", nargs="+", required=True, help="make_dataset label files (they have line boxes)")
    ap.add_argument("--out", required=True)
    ap.add_argument("--per-page", type=int, default=2)
    ap.add_argument("--max-chars", type=int, default=1200, help="label characters per item (~tokens x 3.5)")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    from folioocr.rl.regions import build
    print(json.dumps(build(args.labels, args.out, per_page=args.per_page, max_chars=args.max_chars,
                           limit=args.limit, seed=args.seed), indent=1))


if __name__ == "__main__":
    main()
