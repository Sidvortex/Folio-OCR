"""Render an HTML page with headless Chromium and measure it.

One ``Renderer`` per process (Playwright's sync API is not shareable across
processes). Each call: write the HTML to a temp file (so file:// fonts and
KaTeX load from the same origin), open it at the requested DPI, render math,
wait for fonts, wrap words, trim overflowing blocks, measure, screenshot.
"""
from __future__ import annotations

import io
import os
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path
from typing import List

from PIL import Image

_JS = (Path(__file__).with_name("page.js")).read_text(encoding="utf-8")


@dataclass
class Rendered:
    image: Image.Image
    blocks: List[dict]            # {category, box (image px), text, mathFailed}
    lines: List[dict]             # {box (image px), text, block, math, broken}
    scale: float
    removed: int = 0
    math_failed: int = 0
    seconds: float = 0.0


class Renderer:
    def __init__(self, timeout_ms: int = 30000):
        from playwright.sync_api import sync_playwright
        self._pw = sync_playwright().start()
        self.browser = self._pw.chromium.launch(args=["--font-render-hinting=none", "--disable-lcd-text",
                                                      "--allow-file-access-from-files"])
        self.timeout_ms = timeout_ms
        self.tmp = Path(tempfile.mkdtemp(prefix="folio_render_"))
        self.n = 0

    def close(self) -> None:
        try:
            self.browser.close()
        finally:
            self._pw.stop()

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()

    def render(self, html: str, width: int, height: int, dpi: float, lim_bottom: float, lim_right: float) -> Rendered:
        t0 = time.time()
        scale = dpi / 96.0
        self.n += 1
        path = self.tmp / f"p{os.getpid()}_{self.n % 4}.html"
        path.write_text(html, encoding="utf-8")
        ctx = self.browser.new_context(viewport={"width": width, "height": height}, device_scale_factor=scale)
        try:
            ctx.add_init_script(_JS)
            page = ctx.new_page()
            page.set_default_timeout(self.timeout_ms)
            page.goto(path.as_uri(), wait_until="load")
            page.evaluate("document.fonts.ready.then(() => true)")
            failed = page.evaluate("window.__folio.renderMath()")
            page.evaluate("document.fonts.ready.then(() => true)")
            page.evaluate("window.__folio.wrapWords()")
            removed = page.evaluate(f"window.__folio.trim({lim_bottom}, {lim_right})")
            page.evaluate("document.fonts.ready.then(() => true)")
            data = page.evaluate("window.__folio.extract()")
            png = page.screenshot(clip={"x": 0, "y": 0, "width": width, "height": height}, type="png")
        finally:
            ctx.close()
        img = Image.open(io.BytesIO(png)).convert("RGB")
        sx = img.width / width
        sy = img.height / height

        def scale_box(b):
            return [b[0] * sx, b[1] * sy, b[2] * sx, b[3] * sy]

        blocks = [{**b, "box": scale_box(b["box"])} for b in data["blocks"]]
        lines = [{**l, "box": scale_box(l["box"])} for l in data["lines"]]
        return Rendered(img, blocks, lines, sx, removed=int(removed), math_failed=int(failed),
                        seconds=time.time() - t0)
