"""Resume-Bench: real resume pages with line-level ground truth.

Ground truth comes from ResumeOCR-Lite's tools (``pdf_to_gt.py`` for the
born-digital page, ``project_gt.py`` to carry it onto scans and phone photos,
``review_gt.py`` to check it), in its ``gt.jsonl`` format:

    {"image": "pages/r07_scan.jpg", "id": "r07_scan", "source": "r07", "tier": "scan",
     "split": "test", "lines": [{"box": [x1, y1, x2, y2], "text": "...", "ignore": false}, ...]}

``build`` freezes a benchmark folder (tests are generated once and never
change), ``score`` compares any engine's Markdown with it:

    fields      share of GT emails, phone numbers, URLs and dates found exactly
                (own normaliser: survives underscores, case, spacing)
    line_cer    each GT line aligned to its best match anywhere in the output
                (order-free), errors / GT characters
    extra       share of output characters (spaces aside) not matched to any
                GT line (hallucinated or duplicated text)
    tests       olmOCR-style unit tests: line presence, reading order of
                adjacent lines in one column, field presence
All are reported overall and per tier (digital / scan / photo).
"""
from __future__ import annotations

import json
import os
from collections import defaultdict
from typing import Dict, List

import numpy as np

from ..textnorm import field_set, levenshtein, markdown_to_text, normalize_for_eval, normalize_field, extract_fields, strip_leading_bullet


def _xyxy(box) -> List[float]:
    b = np.asarray(box, float)
    if b.ndim == 2:
        return [b[:, 0].min(), b[:, 1].min(), b[:, 0].max(), b[:, 1].max()]
    return list(b[:4])


def _safe(t: str) -> bool:
    return "_" not in t and "*" not in t


def build(gt_jsonl: str, out_dir: str, split: str = "test") -> Dict:
    root = os.path.dirname(os.path.abspath(gt_jsonl))
    os.makedirs(out_dir, exist_ok=True)
    pages, tests = [], []
    with open(gt_jsonl, encoding="utf-8") as f:
        recs = [json.loads(l) for l in f if l.strip()]
    for r in recs:
        if split and r.get("split", "test") != split:
            continue
        lines = [dict(l, box=_xyxy(l["box"])) for l in r["lines"] if not l.get("ignore") and l.get("text", "").strip()]
        pid = r["id"]
        img = r["image"] if os.path.isabs(r["image"]) else os.path.join(root, r["image"])
        pages.append({"id": pid, "image": img, "tier": r.get("tier", "digital"), "source": r.get("source", pid),
                      "lines": lines})
        pdf = f"{pid}.pdf"
        k = 0
        for l in lines:
            t = strip_leading_bullet(l["text"].strip())
            if len(t) >= 8 and _safe(t) and not l.get("notest"):
                tests.append({"pdf": pdf, "page": 1, "id": f"{pid}_present_{k}", "type": "present", "text": t,
                              "max_diffs": len(t) // 25})
                k += 1
        # reading order: a line and the next line stacked under it in the same column
        k = 0
        for i, a in enumerate(lines):
            if a.get("notest"):
                continue
            ax1, ay1, ax2, ay2 = a["box"]
            h = max(1.0, ay2 - ay1)
            below = [b for b in lines if b is not a and not b.get("notest") and b["box"][1] >= ay2 - 0.3 * h and b["box"][1] - ay2 < 1.8 * h
                     and min(ax2, b["box"][2]) - max(ax1, b["box"][0]) > 0.3 * min(ax2 - ax1, b["box"][2] - b["box"][0])]
            if not below:
                continue
            b = min(below, key=lambda z: z["box"][1])
            ta, tb = strip_leading_bullet(a["text"].strip()), strip_leading_bullet(b["text"].strip())
            if len(ta) >= 10 and len(tb) >= 10 and ta != tb and _safe(ta + tb):
                tests.append({"pdf": pdf, "page": 1, "id": f"{pid}_order_{k}", "type": "order", "before": ta[:60],
                              "after": tb[:60], "max_diffs": min(2, len(ta[:60]) // 10, len(tb[:60]) // 10)})
                k += 1
        seen = set()
        for kind, raw in extract_fields("\n".join(l["text"] for l in lines)):
            key = (kind, normalize_field(kind, raw))
            if key in seen or not _safe(raw):
                continue
            seen.add(key)
            tests.append({"pdf": pdf, "page": 1, "id": f"{pid}_field_{kind}_{len(seen) - 1}", "type": "present",
                          "text": raw, "max_diffs": 0, "case_sensitive": False})
    with open(os.path.join(out_dir, "pages.jsonl"), "w", encoding="utf-8") as f:
        for p in pages:
            f.write(json.dumps(p, ensure_ascii=False) + "\n")
    with open(os.path.join(out_dir, "tests.jsonl"), "w", encoding="utf-8") as f:
        for t in tests:
            f.write(json.dumps(t, ensure_ascii=False) + "\n")
    info = {"pages": len(pages), "tests": len(tests), "tiers": dict(_count(p["tier"] for p in pages)), "split": split,
            "source": os.path.abspath(gt_jsonl)}
    with open(os.path.join(out_dir, "info.json"), "w") as f:
        json.dump(info, f, indent=1)
    return info


def _count(xs):
    d = defaultdict(int)
    for x in xs:
        d[x] += 1
    return d


def load(bench_dir: str) -> List[dict]:
    with open(os.path.join(bench_dir, "pages.jsonl"), encoding="utf-8") as f:
        return [json.loads(l) for l in f if l.strip()]


def line_cer(lines: List[dict], hyp_md: str, mode: str = "normalized"):
    """Order-free CER: each GT line vs its best-matching span of the output."""
    from rapidfuzz.fuzz import partial_ratio_alignment
    hyp = normalize_for_eval(markdown_to_text(hyp_md).replace("\n", " "), mode)
    covered = np.zeros(len(hyp) + 1, bool)
    err = tot = 0
    for l in lines:
        ref = normalize_for_eval(strip_leading_bullet(l["text"]), mode)
        if not ref:
            continue
        tot += len(ref)
        if not hyp:
            err += len(ref)
            continue
        al = partial_ratio_alignment(ref, hyp)
        span = hyp[al.dest_start:al.dest_end]
        err += min(len(ref), levenshtein(ref, span))
        covered[al.dest_start:al.dest_end] = True
    # extra text: output characters (spaces aside) that no GT line accounts for
    ink = np.array([c != " " for c in hyp], bool)
    extra = float((ink & ~covered[:len(hyp)]).sum() / ink.sum()) if hyp and ink.any() else 0.0
    return err, tot, extra


def score(pages: List[dict], md_by_id: Dict[str, str], tests_path: str = None) -> Dict:
    per_tier = defaultdict(lambda: {"err": 0, "tot": 0, "f_ok": 0, "f_tot": 0, "extra": [], "n": 0})
    by_kind = defaultdict(lambda: [0, 0])
    page_rows = []
    for p in pages:
        md = md_by_id.get(p["id"], "")
        e, t, extra = line_cer(p["lines"], md)
        ref_fields = field_set("\n".join(l["text"] for l in p["lines"]))
        hyp_fields = field_set(markdown_to_text(md))
        f_ok = f_tot = 0
        for kind, vals in ref_fields.items():
            got = hyp_fields.get(kind, set())
            ok = sum(1 for v in vals if v in got)
            by_kind[kind][0] += ok
            by_kind[kind][1] += len(vals)
            f_ok += ok
            f_tot += len(vals)
        for tier in (p["tier"], "all"):
            d = per_tier[tier]
            d["err"] += e
            d["tot"] += t
            d["f_ok"] += f_ok
            d["f_tot"] += f_tot
            d["extra"].append(extra)
            d["n"] += 1
        page_rows.append({"id": p["id"], "tier": p["tier"], "line_cer": e / max(1, t), "err": e, "chars": t,
                          "fields": [f_ok, f_tot], "extra": extra})
    tiers = {k: {"pages": v["n"], "line_cer": v["err"] / max(1, v["tot"]),
                 "fields": v["f_ok"] / v["f_tot"] if v["f_tot"] else None,
                 "extra": float(np.mean(v["extra"])) if v["extra"] else 0.0} for k, v in per_tier.items()}
    out = {"tiers": tiers, "fields_by_kind": {k: a / b for k, (a, b) in by_kind.items() if b}, "pages": page_rows}
    if tests_path and os.path.exists(tests_path):
        from .unittests import load_tests_jsonl, score as score_tests
        s = score_tests(load_tests_jsonl(tests_path), md_by_id, missing_ok=True)
        out["tests"] = s["overall"]
        out["tests_by_category"] = {c: v["rate"] for c, v in s["categories"].items()}
        out["results"] = s["results"]
    return out


def format_score(s: Dict, name: str = "") -> str:
    a = s["tiers"].get("all", {})
    head = f"{name}: line CER {100 * a.get('line_cer', float('nan')):.2f}%"
    if a.get("fields") is not None:
        head += f" | fields {100 * a['fields']:.1f}%"
    head += f" | extra text {100 * a.get('extra', 0):.1f}%"
    if s.get("tests") is not None:
        head += f" | tests {100 * s['tests']:.1f}"
    lines = [head]
    for tier, v in sorted(s["tiers"].items()):
        if tier == "all":
            continue
        f = f"{100 * v['fields']:.1f}%" if v["fields"] is not None else "-"
        lines.append(f"    {tier:8s} pages {v['pages']:4d}  line CER {100 * v['line_cer']:6.2f}%  fields {f:>6s}  extra {100 * v['extra']:.1f}%")
    if s.get("fields_by_kind"):
        lines.append("    fields: " + ", ".join(f"{k} {100 * v:.1f}%" for k, v in sorted(s["fields_by_kind"].items())))
    return "\n".join(lines)
