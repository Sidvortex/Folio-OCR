"""The page renderer: labels and unit tests must agree with each other exactly.

Needs Chromium (``python -m playwright install chromium``) and ./assets
(``python fetch_assets.py``); skipped otherwise.
"""
import numpy as np
import pytest

from conftest import assets_dir, chromium_ok

pytestmark = pytest.mark.skipif(assets_dir() is None or not chromium_ok(), reason="needs assets/ and Chromium")


@pytest.fixture(scope="module")
def pages():
    from folioocr.render.browser import Renderer
    from folioocr.render.make_dataset import build_page, parse_mix
    a = assets_dir()
    cfg = {"fonts_dir": str(a / "fonts"), "katex_dir": str(a / "katex"), "prefix": "t", "dev_frac": 0.0,
           "mix": parse_mix("resume=0.4,article=0.2,table=0.2,math=0.2"), "conditions": parse_mix("clean=0.4,scan=0.3,photo=0.3"),
           "dpi": (120, 160)}
    out = []
    with Renderer() as r:
        for seed in range(8):
            res = build_page(seed, cfg, r)
            if res is not None:
                out.append(res)
    assert len(out) >= 6
    return out


def test_labels_are_valid(pages):
    from folioocr.schema import Block, validate_blocks
    for rec, tests, img, _ in pages:
        blocks = [Block.from_dict(b) for b in rec["blocks"]]
        assert validate_blocks(blocks) == []
        assert img.size == (rec["width"], rec["height"])
        for l in rec["lines"]:
            x1, y1, x2, y2 = l["box"]
            assert 0 <= x1 <= x2 <= rec["width"] and 0 <= y1 <= y2 <= rec["height"]


def test_every_generated_test_passes_on_the_ground_truth(pages):
    pytest.importorskip("olmocr.bench.tests")
    from folioocr.bench.unittests import run_page
    from folioocr.schema import to_markdown
    n = 0
    for rec, tests, _, _ in pages:
        md = to_markdown(rec["blocks"])
        for r in run_page(tests, md, rec["id"]):
            assert r["passed"], (r["id"], r["why"])
            n += 1
    assert n > 50


def test_boxes_cover_ink(pages):
    """Text blocks sit on dark pixels: the box is where the text is."""
    for rec, _, img, _ in pages:
        if rec["condition"] != "clean":
            continue
        g = np.asarray(img.convert("L"), float)
        W, H = img.size
        for b in rec["blocks"]:
            if b["category"] not in ("text", "title"):
                continue
            x1, y1, x2, y2 = [int(v / 1000 * s) for v, s in zip(b["bbox"], (W, H, W, H))]
            if x2 - x1 < 4 or y2 - y1 < 4:
                continue
            assert g[y1:y2, x1:x2].min() < 140, (rec["id"], b)
