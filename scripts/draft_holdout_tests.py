#!/usr/bin/env python3
"""Draft unit tests for the private holdout (gate G4), for a person to check.

The holdout is ~300 fresh pages in olmOCR-Bench's categories, collected after
the training data is frozen, with hand-checked tests. This drafts the tests so
checking is faster than writing:

    holdout_src/<category>/<name>.pdf          one page per PDF, sorted into categories by you

    python scripts/draft_holdout_tests.py --src holdout_src --out holdout \\
        --engine openai --api-base http://localhost:8000/v1 --api-model infly/Infinity-Parser2-Pro --api-prompt doc2json

    # then open holdout/review.html next to holdout/bench_data/<category>.jsonl and, test by test,
    # set "checked" to "verified" or "rejected" (fix a test's text if it is almost right)
    python scripts/draft_holdout_tests.py --out holdout --status

    # score any model on it exactly like olmOCR-Bench ("rejected" tests are skipped)
    python scripts/bench_olmocr.py --bench holdout/bench_data --name folio_v1 --adapter runs/rl-t4/best

Tests come from a reference parse (any engine, ideally a strong teacher with
layout output): present (sentences), absent (headers/footers), order
(consecutive blocks), table cells and formulas. A drafted test is only a
suggestion until a person marks it verified.
"""
import argparse
import glob
import html
import json
import os
import random
import re
import shutil
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

_SENT = re.compile(r"(?<=[.!?;:])\s+|\n+")


def sentence_tests(pid: str, md_text: str, rng: random.Random, n: int = 8):
    from folioocr.render.tests_gen import _olmocr_safe
    cands = []
    for s in _SENT.split(md_text):
        s = " ".join(s.split())
        if 30 <= len(s) <= 100 and len(s.split()) >= 5 and _olmocr_safe(s):
            cands.append(s)
    cands = sorted(set(cands))
    rng.shuffle(cands)
    return [{"pdf": f"{pid}.pdf", "page": 1, "id": f"{pid}_present_s{i}", "type": "present", "text": s,
             "max_diffs": len(s) // 25} for i, s in enumerate(cands[:n])]


def draft(src: str, out: str, engine, seed: int = 0) -> dict:
    from folioocr.bench.engines import PageItem, safe_key
    from folioocr.imaging import render_pdf
    from folioocr.render.tests_gen import make_tests
    from folioocr.schema import NON_BODY, Block
    from folioocr.textnorm import markdown_to_text
    rng = random.Random(seed)
    pdfs = sorted(glob.glob(os.path.join(src, "*", "*.pdf")))
    if not pdfs:
        raise SystemExit(f"no <category>/<name>.pdf under {src}")
    work = os.path.join(out, "work")
    os.makedirs(os.path.join(work, "pages"), exist_ok=True)
    os.makedirs(os.path.join(work, "ref"), exist_ok=True)
    pages = []
    for p in pdfs:
        cat = os.path.basename(os.path.dirname(p))
        name = os.path.splitext(os.path.basename(p))[0]
        dst = os.path.join(out, "bench_data", "pdfs", cat, f"{name}.pdf")
        os.makedirs(os.path.dirname(dst), exist_ok=True)
        if not os.path.exists(dst):
            shutil.copy(p, dst)
        key = safe_key(f"{cat}__{name}")
        img = os.path.join(work, "pages", f"{key}.png")
        if not os.path.exists(img):
            render_pdf(p, longest_side=1600, pages=[1])[0].save(img)
        pages.append((cat, name, key, img))
    todo = [x for x in pages if not os.path.exists(os.path.join(work, "ref", f"{x[2]}.json"))]
    for s in range(0, len(todo), 8):
        chunk = todo[s:s + 8]
        res = engine.run_full([PageItem(key=k, image=img) for _, _, k, img in chunk])
        for _, _, k, _ in chunk:
            with open(os.path.join(work, "ref", f"{k}.json"), "w", encoding="utf-8") as f:
                json.dump(res.get(k, {"markdown": "", "blocks": None}), f, ensure_ascii=False)
        print(f"  reference parses: {min(s + 8, len(todo))}/{len(todo)}", flush=True)
    by_cat = {}
    review = []
    for cat, name, key, img in pages:
        with open(os.path.join(work, "ref", f"{key}.json"), encoding="utf-8") as f:
            ref = json.load(f)
        pid = f"{cat}/{name}"
        tests = []
        body_md = ref.get("markdown") or ""
        if ref.get("blocks"):
            blocks = [Block.from_dict(b) for b in ref["blocks"]]
            tests += [t for t in make_tests(pid, blocks, [], rng, fields=False) if t["type"] != "present"]
            body_md = "\n\n".join(b.text for b in blocks if b.category not in NON_BODY and b.category != "table")
        tests += sentence_tests(pid, markdown_to_text(body_md), rng)
        for t in tests:
            t["id"] = t["id"].replace("/", "_")
            t["checked"] = None
        by_cat.setdefault(cat, []).extend(tests)
        review.append((cat, name, img, ref.get("markdown") or "", tests))
    for cat, tests in by_cat.items():
        path = os.path.join(out, "bench_data", f"{cat}.jsonl")
        keep = []
        if os.path.exists(path):                         # never overwrite a reviewer's work
            with open(path, encoding="utf-8") as f:
                keep = [json.loads(l) for l in f if l.strip()]
        have = {t["id"] for t in keep}
        with open(path, "w", encoding="utf-8") as f:
            for t in keep + [t for t in tests if t["id"] not in have]:
                f.write(json.dumps(t, ensure_ascii=False) + "\n")
    write_review(out, review)
    return status(out)


def write_review(out: str, review) -> None:
    rows = []
    for cat, name, img, md, tests in review:
        rel = os.path.relpath(img, out)
        ts = "".join(f"<li><code>{html.escape(t['id'])}</code> <b>{t['type']}</b> "
                     f"{html.escape(json.dumps({k: v for k, v in t.items() if k in ('text', 'before', 'after', 'cell', 'math', 'up', 'down', 'left', 'right', 'top_heading')}, ensure_ascii=False))}</li>"
                     for t in tests)
        rows.append(f"<section><h2>{html.escape(cat)}/{html.escape(name)}</h2><div class='row'>"
                    f"<img src='{html.escape(rel)}'><div><pre>{html.escape(md[:6000])}</pre><ol>{ts}</ol></div></div></section>")
    page = ("<!doctype html><meta charset='utf-8'><title>Holdout review</title><style>body{font:14px system-ui;"
            "margin:16px}.row{display:flex;gap:16px;align-items:flex-start}img{width:48%;border:1px solid #ccc}"
            "pre{white-space:pre-wrap;background:#f6f6f6;padding:8px;max-height:420px;overflow:auto}"
            "section{border-bottom:2px solid #ddd;padding:12px 0}</style>"
            "<h1>Holdout review</h1><p>For every test, edit <code>bench_data/&lt;category&gt;.jsonl</code>: set "
            "<code>\"checked\": \"verified\"</code> when the test is true of the page image, "
            "<code>\"rejected\"</code> when it is not. Judge against the image, not the reference text.</p>"
            + "".join(rows))
    with open(os.path.join(out, "review.html"), "w", encoding="utf-8") as f:
        f.write(page)


def status(out: str) -> dict:
    counts = {"verified": 0, "rejected": 0, "unchecked": 0}
    per_cat = {}
    for path in sorted(glob.glob(os.path.join(out, "bench_data", "*.jsonl"))):
        cat = os.path.splitext(os.path.basename(path))[0]
        c = {"verified": 0, "rejected": 0, "unchecked": 0}
        with open(path, encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    k = json.loads(line).get("checked") or "unchecked"
                    c[k] = c.get(k, 0) + 1
        per_cat[cat] = c
        for k, v in c.items():
            counts[k] = counts.get(k, 0) + v
    return {"total": counts, "by_category": per_cat,
            "ready": counts["unchecked"] == 0 and counts["verified"] > 0}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--src", default=None, help="<category>/<name>.pdf pages")
    ap.add_argument("--out", required=True)
    ap.add_argument("--status", action="store_true")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--render", default="common", help=argparse.SUPPRESS)
    from folioocr.bench.engines import add_engine_args, make_engine
    add_engine_args(ap)
    args = ap.parse_args()
    if args.status:
        print(json.dumps(status(args.out), indent=1))
        return
    if not args.src:
        ap.error("--src is required to draft")
    print(json.dumps(draft(args.src, args.out, make_engine(args), args.seed), indent=1))


if __name__ == "__main__":
    main()
