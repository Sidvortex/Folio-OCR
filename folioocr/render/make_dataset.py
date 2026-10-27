"""Build a synthetic page dataset: images + layout-JSON labels + unit tests.

    python -m folioocr.render.make_dataset --out data/synth --n 2000 --workers 4

Writes
    <out>/images/<id>.jpg
    <out>/labels.jsonl   one record per page (blocks in reading order, line boxes)
    <out>/tests.jsonl    olmOCR-Bench-format unit tests (pdf = "<id>.pdf")
    <out>/meta.json      the settings and counts
    <out>/preview/       a few pages with their boxes drawn (``--preview``)
"""
from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import os
import random
import re
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
from PIL import Image, ImageDraw

from ..schema import Block, normalize_bbox, validate_blocks
from .assets import Family, find_assets_dir, font_face_css, load_families, uncovered
from .content import GeneralGenerator, ResumeGenerator
from .degrade import degrade
from .layouts import build_general_page, build_resume_page, html_document
from .tests_gen import make_tests

DEFAULT_MIX = "resume=0.7,article=0.1,table=0.08,math=0.05,letter=0.04,indic=0.03"
DEFAULT_CONDITIONS = "clean=0.3,scan=0.4,photo=0.2,old=0.1"
SAFE_REPLACE = {"➢": "•", "➤": "•", "✓": "•", "✔": "•", "▪": "•", "◦": "•", "►": "•", "▸": "•", "◆": "•",
                "₹": "Rs.", "−": "-", "…": "...", "→": "->", "◾": "•", "■": "•", "●": "•", "○": "•"}
_TAG = re.compile(r"<[^>]+>")


def parse_mix(spec: str) -> Dict[str, float]:
    out = {}
    for part in spec.split(","):
        k, v = part.split("=")
        out[k.strip()] = float(v)
    s = sum(out.values())
    return {k: v / s for k, v in out.items()}


def _alias_css(fam: Optional[Family], alias: str) -> str:
    if fam is None:
        return ""
    return font_face_css(fam).replace(f"'{fam.css_name}'", f"'{alias}'")


# ------------------------------------------------------------------ worker
_R = None
_CFG: Dict = {}


def _init(cfg: Dict) -> None:
    global _R, _CFG
    from .browser import Renderer
    _CFG = cfg
    _R = Renderer()


def build_page(seed: int, cfg: Dict, renderer) -> Optional[Tuple[dict, List[dict], Image.Image, int]]:
    rng = random.Random(seed)
    nrng = np.random.default_rng(seed)
    fams = load_families(cfg["fonts_dir"])
    mix, conds = cfg["mix"], cfg["conditions"]
    kind = rng.choices(list(mix), weights=list(mix.values()))[0]
    cond = rng.choices(list(conds), weights=list(conds.values()))[0]
    if kind == "resume":
        doc = ResumeGenerator(rng).generate()
        css, st, body, layout = build_resume_page(rng, doc, fams)
        layout = "resume/" + layout
    else:
        doc = GeneralGenerator(rng).generate(kind)
        css, st, body, layout = build_general_page(rng, doc, fams, typewriter=cond == "old" and rng.random() < 0.6)
        layout = "general/" + layout
    by_name = {f.name: f for f in fams}
    fallback = by_name.get("notosans") or by_name.get("arimo")
    symbols = by_name.get("notosanssymbols2")
    font_css = "\n".join(x for x in {font_face_css(st.body), font_face_css(st.head)}) + "\n" + \
        _alias_css(fallback, "F_fallback") + "\n" + _alias_css(symbols, "F_symbols")
    html = html_document(css, body, font_css, cfg.get("katex_dir"))
    # characters no loaded font covers would render as boxes while the label says otherwise
    text = _TAG.sub(" ", body)
    covering = [f for f in (st.body, st.head, fallback, symbols) if f is not None]
    bad = uncovered(text, covering)
    if bad:
        for ch in bad:
            if ch in SAFE_REPLACE:
                html = html.replace(ch, SAFE_REPLACE[ch])
        bad = uncovered(_TAG.sub(" ", html.split("<body>", 1)[-1]), covering)
        if bad and cfg.get("strict_coverage", True):
            return None
    dpi = rng.uniform(*cfg["dpi"])
    t, r, b, l = st.margins
    sidebar = layout.endswith("sidebar")
    lim_bottom = st.height - b
    lim_right = st.width - (0 if sidebar else r) + 2
    rend = renderer.render(html, st.width, st.height, dpi, lim_bottom, lim_right)
    if rend.math_failed:
        return None
    raw_blocks = [x for x in rend.blocks if x["category"] == "figure" or x["text"].strip()]
    if len(raw_blocks) < 3:
        return None
    keep_idx = {id(x): i for i, x in enumerate(raw_blocks)}
    old_to_new = {}
    for i, x in enumerate(rend.blocks):
        if id(x) in keep_idx:
            old_to_new[i] = keep_idx[id(x)]
    lines = [dict(l, block=old_to_new[l["block"]]) for l in rend.lines if l["block"] in old_to_new]
    boxes = {"blocks": np.array([x["box"] for x in raw_blocks], np.float32).reshape(-1, 4),
             "lines": np.array([x["box"] for x in lines], np.float32).reshape(-1, 4)}
    img, boxes, quality, log = degrade(np.asarray(rend.image), boxes, cond, nrng, dpi)
    H, W = img.shape[:2]
    blocks = [Block(normalize_bbox(bx, W, H), x["category"], "" if x["category"] == "figure" else x["text"].strip())
              for x, bx in zip(raw_blocks, boxes["blocks"])]
    errs = validate_blocks(blocks)
    if errs:
        return None
    lines_out = [{"box": [int(round(v)) for v in bx], "text": l["text"], "block": l["block"], "math": bool(l.get("math")),
                  **({"broken": True} if l.get("broken") else {})} for l, bx in zip(lines, boxes["lines"])]
    pid = f"{cfg['prefix']}{seed:07d}"
    tests = make_tests(pid, blocks, lines_out, rng, fields=(kind == "resume"))
    rec = {"id": pid, "image": f"images/{pid}.jpg", "width": W, "height": H, "kind": kind, "layout": layout,
           "condition": cond, "dpi": round(dpi, 1), "blocks": [b.to_dict() for b in blocks], "lines": lines_out,
           "split": "dev" if (seed * 2654435761 % 1000) < cfg["dev_frac"] * 1000 else "train",
           "meta": {"removed": rend.removed, "degrade": log, "render_s": round(rend.seconds, 2)}}
    return rec, tests, Image.fromarray(img), quality


def _work(seed: int):
    try:
        out = build_page(seed, _CFG, _R)
    except Exception as e:  # keep the pool alive; report and skip
        return seed, None, f"{type(e).__name__}: {e}"
    if out is None:
        return seed, None, "skipped"
    rec, tests, img, q = out
    img.save(os.path.join(_CFG["out"], rec["image"]), quality=q)
    return seed, (rec, tests), ""


# ----------------------------------------------------------------- preview
COLORS = {"title": (220, 50, 50), "text": (30, 120, 220), "table": (230, 160, 0), "figure": (40, 170, 90),
          "formula": (150, 60, 200), "header": (120, 120, 120), "footer": (120, 120, 120)}


def draw_preview(rec: dict, root: str, dst: str, max_side: int = 1100) -> None:
    img = Image.open(os.path.join(root, rec["image"])).convert("RGB")
    W, H = img.size
    s = min(1.0, max_side / max(W, H))
    img = img.resize((int(W * s), int(H * s)))
    d = ImageDraw.Draw(img)
    for i, b in enumerate(rec["blocks"]):
        x1, y1, x2, y2 = [v / 1000 for v in b["bbox"]]
        col = COLORS.get(b["category"], (200, 0, 200))
        d.rectangle([x1 * img.width, y1 * img.height, x2 * img.width, y2 * img.height], outline=col, width=2)
        d.text((x1 * img.width + 2, y1 * img.height - 10), f"{i}:{b['category']}", fill=col)
    for l in rec["lines"]:
        x1, y1, x2, y2 = [v * s for v in l["box"]]
        d.rectangle([x1, y1, x2, y2], outline=(0, 200, 200), width=1)
    img.save(dst)


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", required=True)
    ap.add_argument("--n", type=int, default=100, help="pages to attempt")
    ap.add_argument("--start-seed", type=int, default=0)
    ap.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2) - 1))
    ap.add_argument("--mix", default=DEFAULT_MIX, help=f"page kinds and weights (default {DEFAULT_MIX})")
    ap.add_argument("--conditions", default=DEFAULT_CONDITIONS, help=f"degradations (default {DEFAULT_CONDITIONS})")
    ap.add_argument("--dpi", default="140-200", help="render DPI range")
    ap.add_argument("--dev-frac", type=float, default=0.03)
    ap.add_argument("--assets", default=None, help="folder with fonts/ and katex/ (default: ./assets or $FOLIO_ASSETS)")
    ap.add_argument("--prefix", default="s", help="id prefix (use different prefixes for different datasets)")
    ap.add_argument("--preview", type=int, default=0, help="draw boxes on the first N pages")
    args = ap.parse_args(argv)

    assets = find_assets_dir(args.assets)
    katex = assets / "katex"
    lo, hi = (float(x) for x in args.dpi.split("-"))
    out = Path(args.out)
    (out / "images").mkdir(parents=True, exist_ok=True)
    cfg = {"out": str(out), "fonts_dir": str(assets / "fonts"), "katex_dir": str(katex) if (katex / "katex.min.js").exists() else None,
           "mix": parse_mix(args.mix), "conditions": parse_mix(args.conditions), "dpi": (lo, hi),
           "dev_frac": args.dev_frac, "prefix": args.prefix}
    if cfg["katex_dir"] is None:
        print("warning: KaTeX not found under assets/katex; math pages will be skipped", file=sys.stderr)
        cfg["mix"] = parse_mix(",".join(f"{k}={v}" for k, v in cfg["mix"].items() if k != "math"))
    seeds = list(range(args.start_seed, args.start_seed + args.n))
    t0 = time.time()
    n_ok, reasons = 0, {}
    with open(out / "labels.jsonl", "a", encoding="utf-8") as fl, open(out / "tests.jsonl", "a", encoding="utf-8") as ft:
        ctx = mp.get_context("spawn")
        with ctx.Pool(args.workers, initializer=_init, initargs=(cfg,)) as pool:
            for i, (seed, res, why) in enumerate(pool.imap_unordered(_work, seeds, chunksize=2), 1):
                if res is None:
                    key = why.split(":")[0]
                    reasons[key] = reasons.get(key, 0) + 1
                    if why != "skipped" and reasons[key] <= 3:
                        print(f"  seed {seed}: {why}", file=sys.stderr)
                    continue
                rec, tests = res
                fl.write(json.dumps(rec, ensure_ascii=False) + "\n")
                for t in tests:
                    ft.write(json.dumps(t, ensure_ascii=False) + "\n")
                n_ok += 1
                if i % 50 == 0:
                    rate = i / (time.time() - t0)
                    print(f"  {i}/{len(seeds)} pages ({n_ok} kept, {rate:.2f}/s)", flush=True)
    meta = {"n_attempted": len(seeds), "n_kept": n_ok, "skipped": reasons, "seconds": round(time.time() - t0, 1),
            **{k: v for k, v in cfg.items() if k not in ("out",)}, "args": vars(args)}
    (out / "meta.json").write_text(json.dumps(meta, indent=2, default=str))
    print(f"kept {n_ok}/{len(seeds)} pages in {time.time() - t0:.0f}s -> {out}  (skipped: {reasons})")
    if args.preview:
        (out / "preview").mkdir(exist_ok=True)
        with open(out / "labels.jsonl", encoding="utf-8") as f:
            for k, line in enumerate(f):
                if k >= args.preview:
                    break
                rec = json.loads(line)
                draw_preview(rec, str(out), str(out / "preview" / f"{rec['id']}.png"))


if __name__ == "__main__":
    main()
