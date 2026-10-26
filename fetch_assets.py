#!/usr/bin/env python3
"""Download what the page renderer needs into ./assets (about 120 MB).

    python fetch_assets.py                       # fonts + KaTeX into ./assets
    python fetch_assets.py --fonts-from ../ResumeOCR-Lite/fonts   # reuse fonts you already have
    python fetch_assets.py --skip-indic          # Latin fonts only

Fonts: open-licensed families from github.com/google/fonts (OFL/Apache/UFL),
the resume fonts ResumeOCR-Lite uses plus symbol, typewriter and Indic-script
faces. KaTeX (MIT) comes from the npm registry tarball.
"""
from __future__ import annotations

import argparse
import concurrent.futures as cf
import os
import shutil
import sys
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from folioocr.render.assets import (EXTRA_FAMILIES, INDIC, KATEX_VERSION, RESUME_FAMILIES,  # noqa: E402
                                    fetch_font_family, fetch_katex)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dest", default="assets")
    ap.add_argument("--fonts-from", default=None, help="copy font families from this folder first")
    ap.add_argument("--skip-indic", action="store_true")
    ap.add_argument("--katex-version", default=KATEX_VERSION)
    ap.add_argument("--workers", type=int, default=8)
    args = ap.parse_args()

    dest = Path(args.dest)
    fonts = dest / "fonts"
    fonts.mkdir(parents=True, exist_ok=True)
    if args.fonts_from:
        for d in Path(args.fonts_from).iterdir():
            if d.is_dir() and not (fonts / d.name).exists():
                shutil.copytree(d, fonts / d.name)
        print(f"copied fonts from {args.fonts_from}")
    families = RESUME_FAMILIES + [f for f in EXTRA_FAMILIES if not (args.skip_indic and f in INDIC)]
    todo = [f for f in families if not any((fonts / f).glob("*.[ot]tf"))]
    print(f"fetching {len(todo)} font families ...")
    failed = []
    with cf.ThreadPoolExecutor(args.workers) as ex:
        futs = {ex.submit(fetch_font_family, f, str(fonts)): f for f in todo}
        for fut in cf.as_completed(futs):
            fam = futs[fut]
            try:
                n = fut.result()
                print(f"  {fam}: {n} files")
            except Exception as e:  # keep going; the renderer uses what is there
                failed.append(fam)
                print(f"  {fam}: FAILED ({e})")
    k = fetch_katex(str(dest), args.katex_version)
    print(f"KaTeX {args.katex_version} -> {k}")
    n_fam = sum(1 for d in fonts.iterdir() if d.is_dir() and any(d.glob('*.[ot]tf')))
    print(f"done: {n_fam} font families in {fonts}" + (f"; failed: {failed}" if failed else ""))


if __name__ == "__main__":
    main()
