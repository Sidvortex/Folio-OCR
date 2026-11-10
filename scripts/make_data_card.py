#!/usr/bin/env python3
"""Write the data card for a training mixture (one licence line per source, plus contamination counts).

    python scripts/make_data_card.py --mixture mixtures/general-free-v1.json --contamination results/contamination.json \\
        --out docs/DATA_CARD.md
    python scripts/make_data_card.py --run runs/general-t4          # the sources a run actually trained on

Each label folder's ``source.json`` (written by make_dataset / prepare_data /
label_teachers) supplies its licence; a missing one shows up as "unknown".
"""
import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from folioocr.data.mixture import data_card, load_mixture  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--mixture")
    g.add_argument("--run", help="a scripts/train_sft.py output folder (reads its train_config.json)")
    g.add_argument("--data", nargs="+", help="label files, optionally path:weight")
    ap.add_argument("--contamination", default=None, help="report from scripts/dedup.py check")
    ap.add_argument("--out", default=None, help="default: print")
    args = ap.parse_args()
    roles, name, task_mix = {}, "", ""
    if args.mixture:
        mix = load_mixture(args.mixture)
        specs, name, task_mix = mix.specs(), mix.name, mix.task_mix or ""
        roles = {s.labels: s.role for s in mix.sources}
    elif args.run:
        with open(os.path.join(args.run, "train_config.json"), encoding="utf-8") as f:
            cfg = json.load(f)
        specs, name, task_mix = cfg["data"], os.path.basename(os.path.normpath(args.run)), cfg.get("task_mix", "")
    else:
        specs = args.data
    card = data_card(specs, name=name, contamination=args.contamination, task_mix=task_mix, roles=roles)
    if args.out:
        os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
        with open(args.out, "w", encoding="utf-8") as f:
            f.write(card)
        print(f"-> {args.out}")
    else:
        print(card)


if __name__ == "__main__":
    main()
