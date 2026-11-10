#!/usr/bin/env python3
"""Draw the README charts from benchmark result files (every bar is a run you made).

    # after the gate runs: your model next to the others, same pages, same harness
    python scripts/make_charts.py --olmocr results/olmocr/rl_s300.json results/olmocr/flash_s300.json \\
        results/olmocr/chandra2_s300.json results/olmocr/surya2_s300.json \\
        --resume results/resume/rl_v1.json results/resume/flash.json results/resume/chandra2.json \\
        --ours rl_s300 rl_v1 --names rl_s300=Folio-OCR rl_v1=Folio-OCR flash_s300=Flash ...

    # only the published numbers (before you have runs of your own)
    python scripts/make_charts.py --published docs/published_scores.json

Writes SVGs to docs/charts/ and prints Markdown to paste into the README.
Bars carry 95% bootstrap intervals; published scores are drawn hatched and
labelled as such, never mixed into your own runs.
"""
import argparse
import json
import os

ACCENT, OTHER, PUB, INK, GRID = "#2563eb", "#9ca3af", "#d1d5db", "#111827", "#e5e7eb"


def _load(p):
    with open(p, encoding="utf-8") as f:
        return json.load(f)


def _style(plt):
    plt.rcParams.update({"font.size": 10, "axes.edgecolor": GRID, "axes.labelcolor": INK, "xtick.color": INK,
                         "ytick.color": INK, "text.color": INK, "axes.spines.top": False,
                         "axes.spines.right": False, "svg.fonttype": "none"})


def _name(path, data, names):
    key = data.get("meta", {}).get("name") or data.get("name") or os.path.splitext(os.path.basename(path))[0]
    return key, names.get(key, key)


def olmocr_overall(plt, runs, published, ours, out):
    rows = []
    for key, label, d in runs:
        lo, hi = d.get("ci95", (None, None))
        rows.append((label, 100 * d["overall"], lo, hi, key in ours, False))
    for m in published:
        rows.append((f"{m['name']} ({m['size']}), published", m["score"], None, None, False, True))
    rows.sort(key=lambda r: r[1])
    fig, ax = plt.subplots(figsize=(7.5, 0.42 * len(rows) + 1.2), facecolor="white")
    for i, (label, v, lo, hi, mine, pub) in enumerate(rows):
        ax.barh(i, v, color=PUB if pub else (ACCENT if mine else OTHER), hatch="///" if pub else None,
                edgecolor="white", height=0.62)
        if lo is not None:
            ax.errorbar(v, i, xerr=[[v - 100 * lo], [100 * hi - v]], fmt="none", ecolor=INK, capsize=3, lw=1)
        ax.text((100 * hi if hi else v) + 0.3, i, f"{v:.1f}", va="center", fontsize=9,
                fontweight="bold" if mine else "normal")
    ax.set_yticks(range(len(rows)), [r[0] for r in rows])
    lo_x = min(r[1] for r in rows) - 8
    ax.set_xlim(max(0, lo_x), 100)
    ax.set_xlabel("olmOCR-Bench overall score")
    ax.xaxis.grid(True, color=GRID)
    ax.set_axisbelow(True)
    scope = ""
    if runs:
        meta = runs[0][2].get("meta", {})
        scope = (f"Our runs: {meta.get('pages_scored')}-page sample, seed {meta.get('seed')}"
                 if meta.get("sample") else "Our runs: full benchmark")
        scope += ", 95% intervals."
    if published:
        scope += ("\n" if scope else "") + "Hatched: published by each team on their own harness, not our runs."
    ax.set_title(scope, fontsize=8.5, loc="left", color="#4b5563")
    fig.tight_layout()
    fig.savefig(out)
    plt.close(fig)


def olmocr_categories(plt, runs, ours, out):
    cats = sorted({c for _, _, d in runs for c in d.get("categories", {}) if c != "baseline"})
    if not cats:
        return False
    n = len(runs)
    w = 0.8 / n
    fig, ax = plt.subplots(figsize=(max(7, 1.1 * len(cats) + 2), 4), facecolor="white")
    greys = ["#6b7280", "#9ca3af", "#cbd5e1", "#4b5563", "#d1d5db"]
    gi = 0
    for j, (key, label, d) in enumerate(runs):
        col = ACCENT if key in ours else greys[gi % len(greys)]
        gi += key not in ours
        vals = [100 * d["categories"].get(c, {}).get("rate", 0) for c in cats]
        ax.bar([i + (j - (n - 1) / 2) * w for i in range(len(cats))], vals, width=w, color=col, label=label)
    ax.set_xticks(range(len(cats)), [c.replace("_", " ") for c in cats], rotation=20, ha="right")
    ax.set_ylim(0, 100)
    ax.set_ylabel("pass rate (%)")
    ax.yaxis.grid(True, color=GRID)
    ax.set_axisbelow(True)
    ax.legend(frameon=False, fontsize=8, ncol=min(4, n), loc="lower left", bbox_to_anchor=(0, 1.0))
    fig.tight_layout()
    fig.savefig(out)
    plt.close(fig)
    return True


def resume(plt, runs, ours, out):
    labels = [label for _, label, _ in runs]
    cer = [100 * d["tiers"]["all"]["line_cer"] for _, _, d in runs]
    fld = [100 * (d["tiers"]["all"].get("fields") or 0) for _, _, d in runs]
    cols = [ACCENT if key in ours else OTHER for key, _, _ in runs]
    fig, (a, b) = plt.subplots(1, 2, figsize=(9, 0.45 * len(runs) + 1.6), facecolor="white")
    for ax, vals, title, fmt in ((a, cer, "Page CER (lower is better)", "{:.2f}%"),
                                 (b, fld, "Field accuracy (higher is better)", "{:.1f}%")):
        ax.barh(range(len(vals)), vals, color=cols, height=0.62)
        for i, v in enumerate(vals):
            ax.text(v, i, " " + fmt.format(v), va="center", fontsize=9)
        ax.set_yticks(range(len(vals)), labels)
        ax.set_title(title, fontsize=10, loc="left")
        ax.xaxis.grid(True, color=GRID)
        ax.set_axisbelow(True)
        ax.invert_yaxis()
    b.set_xlim(0, 108)
    a.set_xlim(0, max(cer) * 1.3 if cer and max(cer) > 0 else 1)
    fig.tight_layout()
    fig.savefig(out)
    plt.close(fig)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--olmocr", nargs="*", default=[], help="results/olmocr/*.json, same sample for all")
    ap.add_argument("--resume", nargs="*", default=[], help="results/resume/*.json")
    ap.add_argument("--published", default=None, help="docs/published_scores.json, drawn hatched")
    ap.add_argument("--ours", nargs="*", default=[], help="run names to highlight")
    ap.add_argument("--names", nargs="*", default=[], help="run=Display Name")
    ap.add_argument("--out", default="docs/charts")
    args = ap.parse_args()
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    _style(plt)
    names = dict(n.split("=", 1) for n in args.names)
    ours = set(args.ours)
    os.makedirs(args.out, exist_ok=True)
    md = []
    olm = [(*_name(p, d, names), d) for p in args.olmocr for d in [_load(p)]]
    pub = _load(args.published)["models"] if args.published else []
    if olm or pub:
        path = os.path.join(args.out, "olmocr_overall.svg" if olm else "olmocr_published.svg")
        olmocr_overall(plt, olm, pub, ours, path)
        md.append(f"![olmOCR-Bench overall]({path})")
    if olm:
        path = os.path.join(args.out, "olmocr_categories.svg")
        if olmocr_categories(plt, olm, ours, path):
            md.append(f"![olmOCR-Bench by category]({path})")
    res = [(*_name(p, d, names), d) for p in args.resume for d in [_load(p)]]
    if res:
        path = os.path.join(args.out, "resume_bench.svg")
        resume(plt, res, ours, path)
        md.append(f"![Resume-Bench]({path})")
    if not md:
        ap.error("give --olmocr, --resume and/or --published")
    print("Paste into README.md:\n")
    print("\n\n".join(md))


if __name__ == "__main__":
    main()
