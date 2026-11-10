#!/usr/bin/env python3
"""Does A beat B? Paired bootstrap on two result files scored on the same pages.

    python scripts/compare.py results/olmocr/folio_v1.json results/olmocr/flash_common.json
    python scripts/compare.py results/resume/folio_v1.json results/resume/chandra2.json

Prints both scores, the difference, its 95% interval and whether it excludes
zero (the plan's rule for claiming a win). Works on olmOCR-Bench, Resume-Bench
and dev-set result files: unit tests where there are per-test ``results``, and
page CER and field accuracy where there are per-page rows (Resume-Bench and
dev sets; gate G1 needs both).
"""
import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from folioocr.bench.stats import paired_bootstrap, paired_bootstrap_pages  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("a")
    ap.add_argument("b")
    ap.add_argument("--iters", type=int, default=10000)
    ap.add_argument("--json", action="store_true")
    args = ap.parse_args()
    da, db = (json.load(open(p, encoding="utf-8")) for p in (args.a, args.b))
    na, nb = os.path.basename(args.a), os.path.basename(args.b)
    out = {}
    ra = da.get("results") if isinstance(da, dict) else da
    rb = db.get("results") if isinstance(db, dict) else db
    if ra and rb:
        out["tests"] = paired_bootstrap(ra, rb, iters=args.iters)

    def page_rows(d):
        if not isinstance(d, dict):
            return None
        rows = d.get("page_rows") or d.get("pages")          # dev results / Resume-Bench results
        return rows if isinstance(rows, list) and rows else None

    pa, pb = page_rows(da), page_rows(db)
    if pa and pb:
        for metric in ("line_cer", "fields"):
            if metric == "fields" and not any(r.get("fields", [0, 0])[1] for r in pa):
                continue
            try:
                out[metric] = paired_bootstrap_pages(pa, pb, metric, iters=args.iters)
            except (ValueError, KeyError) as e:
                out[metric] = {"error": str(e)}
    if args.json:
        print(json.dumps(out, indent=1))
        return
    print(f"A = {na}\nB = {nb}")
    if "tests" in out:
        r = out["tests"]
        lo, hi = r["ci95"]
        print(f"\nunit tests: A {100 * r['a']:.2f}, B {100 * r['b']:.2f}, A - B = {100 * r['diff']:+.2f} "
              f"(95% CI {100 * lo:+.2f} to {100 * hi:+.2f}; {r['n_tests']} common tests, {r['n_categories']} categories)")
        print("    " + (("significant: " + ("A beats B" if r["diff"] > 0 else "B beats A")) if r["significant"]
                        else "not significant: the interval includes zero, no claim either way"))
    for metric, label in (("line_cer", "page CER (lower is better)"), ("fields", "field accuracy")):
        if metric not in out:
            continue
        r = out[metric]
        if "error" in r:
            print(f"\n{label}: {r['error']}")
            continue
        lo, hi = r["ci95"]
        print(f"\n{label}: A {100 * r['a']:.2f}%, B {100 * r['b']:.2f}%, A - B = {100 * r['diff']:+.2f} "
              f"(95% CI {100 * lo:+.2f} to {100 * hi:+.2f}; {r['n_pages']} pages)")
        if r["significant"]:
            print("    significant: " + ("A beats B" if r["a_better"] else "B beats A"))
        else:
            print("    not significant: the interval includes zero, no claim either way")


if __name__ == "__main__":
    main()
