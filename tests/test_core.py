"""Format, text and image rules that silently corrupt training or scoring when wrong.

    python -m pytest -q tests
"""
import json
import random

import pytest

from folioocr import prompts
from folioocr.convert import html_to_markdown, surya_page_to_markdown
from folioocr.imaging import fit_image, smart_resize, visual_tokens
from folioocr.loops import is_looping, strip_loop, tail_period
from folioocr.schema import (Block, denormalize_bbox, dump_target, normalize_bbox, parse_output, to_markdown,
                             validate_blocks)
from folioocr.textnorm import (extract_fields, field_set, markdown_to_text, normalize_field, normalize_for_eval,
                               strip_leading_bullet)


# ------------------------------------------------------------------ schema
def _blocks():
    return [Block([100, 20, 900, 60], "header", "Journal of Things, Vol. 3"),
            Block([100, 80, 600, 120], "title", "# Ravada Siddharth"),
            Block([100, 130, 900, 300], "text", "- Built **REST** APIs\n- Cut latency by 30%"),
            Block([100, 320, 900, 500], "table", "<table><tr><th>A</th><th>B</th></tr><tr><td>1</td><td>2</td></tr></table>"),
            Block([400, 520, 600, 560], "formula", "$$E = mc^2$$"),
            Block([100, 600, 300, 700], "figure", ""),
            Block([100, 950, 900, 980], "footer", "Page 1 of 2")]


def test_dump_parse_roundtrip():
    bl = _blocks()
    for fence in (True, False):
        r = parse_output(dump_target(bl, fence=fence))
        assert r.ok and not r.repaired and not r.truncated
        assert [b.to_dict() for b in r.blocks] == [b.to_dict() for b in bl]


def test_dump_starts_like_infinity_parser_output():
    # infinity_parser2's post-processor searches for '{"bbox":' to cut truncated output
    assert '{"bbox": [' in dump_target(_blocks())


def test_parse_repairs_truncated_and_bad_items():
    t = dump_target(_blocks())
    r = parse_output(t[: len(t) // 2])
    assert not r.ok and r.repaired and r.truncated and 1 <= len(r.blocks) < 7
    r = parse_output('[{"bbox": [1, 2, 3, 4], "category": "text", "text": "a"}, {"bbox": [1], "category": "x"}]')
    assert len(r.blocks) == 1 and r.dropped == 1
    r = parse_output("no json here")
    assert r.blocks == [] and not r.ok


def test_markdown_drops_headers_footers_like_infinity():
    md = to_markdown(_blocks())
    assert "Journal of Things" not in md and "Page 1 of 2" not in md
    assert md.startswith("# Ravada Siddharth") and "$$E = mc^2$$" in md
    assert "Page 1 of 2" in to_markdown(_blocks(), keep_non_body=True)


def test_bbox_normalisation():
    assert normalize_bbox([0, 0, 800, 1000], 800, 1000) == [0, 0, 1000, 1000]
    assert normalize_bbox([900, -5, 100, 50], 1000, 1000) == [100, 0, 900, 50]
    assert denormalize_bbox([500, 250, 1000, 1000], 2000, 4000) == [1000, 1000, 2000, 4000]


def test_validate_blocks():
    assert validate_blocks(_blocks()) == []
    assert validate_blocks([Block([0, 0, 10, 10], "figure", "x")])
    assert validate_blocks([Block([0, 0, 10, 10], "paragraph", "x")])


def test_prompts_are_infinity_parser_prompts():
    assert "Allowed layout categories: ['header', 'title', 'text', 'figure', 'table', 'formula'" in prompts.DOC2JSON
    assert prompts.DOC2MD.strip().startswith("You are an AI assistant specialized in converting PDF images to Markdown")
    with pytest.raises(ValueError):
        prompts.prompt_for("ocr")


# ------------------------------------------------------------------- image
def test_smart_resize_matches_qwen_vl_utils_rules():
    for (h, w), mx in [((3300, 2550), 2_000_000), ((1056, 816), 16777216), ((50, 40), 16777216), ((7000, 300), 4_000_000)]:
        rh, rw = smart_resize(h, w, max_pixels=mx)
        assert rh % 32 == 0 and rw % 32 == 0
        assert 2048 <= rh * rw <= mx
    assert smart_resize(1056, 816, max_pixels=16777216) == (1056, 832)    # rounds to the nearest multiple
    assert smart_resize(3300, 2550, max_pixels=2_000_000) == (1600, 1216)  # floors when shrinking


def test_fit_image_and_tokens():
    from PIL import Image
    img = Image.new("RGB", (2550, 3300), "white")
    out = fit_image(img, 2_000_000)
    assert out.size == (1216, 1600)
    assert visual_tokens(img, 2_000_000) == 1216 * 1600 // 1024


# -------------------------------------------------------------------- text
def test_fields_extraction_and_normalisation():
    s = ("Email: Ravada.S@gmail.com | +91 98765 43210 | linkedin.com/in/ravada-s | github.com/Sidvortex "
         "| Jan 2021 – Present, 06/2019 - 08/2020, CGPA 8.72/10, 2019 - 2023, first_last@x.co.in")
    kinds = {}
    for k, v in extract_fields(s):
        kinds.setdefault(k, []).append(v)
    assert kinds["email"] == ["Ravada.S@gmail.com", "first_last@x.co.in"]
    assert kinds["phone"] == ["+91 98765 43210"]               # years / CGPA are not phone numbers
    assert set(kinds["url"]) == {"linkedin.com/in/ravada-s", "github.com/Sidvortex"}
    assert {"jan 2021", "06/2019", "08/2020"} <= field_set(s)["date"]
    assert normalize_field("phone", "+91-98765-43210") == normalize_field("phone", "+91 98765 43210")
    assert normalize_field("url", "https://www.GitHub.com/x/") == "github.com/x"


def test_phone_numbers_are_not_table_numbers():
    from folioocr.textnorm import plausible_phone
    for ok in ("+91 98765 43210", "+91-9876543210", "(555) 123-4567", "555.123.4567", "+44 20 7946 0958",
               "080 2345 6789"):
        assert plausible_phone(ok), ok
    for bad in ("60.17 58.92 88", "2012 2016 2020", "9.40 8.72 7.91 66", "12345"):
        assert not plausible_phone(bad), bad
    assert [v for k, v in extract_fields("| 60.17 | 58.92 | 88 |") if k == "phone"] == []


def test_markdown_to_text():
    md = "# Title\n\n- **Built** *fast* APIs\n• Item two\n<table><tr><th>A</th><td>B</td></tr></table>\n[site](http://x.com)"
    t = markdown_to_text(md)
    assert "Built fast APIs" in t and "Item two" in t and "A B" in t and "site" in t and "#" not in t and "**" not in t
    assert strip_leading_bullet("•  Built APIs") == "Built APIs"
    assert strip_leading_bullet("-3 dB gain") == "-3 dB gain"


def test_normalize_for_eval():
    assert normalize_for_eval("“Café” – résumé • 2021—22") == '"Cafe" - resume 2021-22'
    assert normalize_for_eval("a   b ", "strict") == "a b"


# ------------------------------------------------------------------ loops
def test_loop_detection():
    x = "intro " + "the same line again\n" * 30
    assert tail_period(x) == len("the same line again\n") and is_looping(x)
    assert strip_loop(x).count("the same line again") == 1
    assert not is_looping("an ordinary ending, with varied words 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15")
    assert is_looping("x" * 300)


# ---------------------------------------------------------------- convert
def test_html_to_markdown_like_chandra_output():
    html = ('<div data-label="Page-Header">Journal X</div><div data-label="Section-Header"><h2>Intro</h2></div>'
            '<div data-label="Text"><p>Energy <math>E=mc^2</math> holds.</p></div>'
            '<div data-label="Equation-Block"><math display="block">a^2+b^2=c^2</math></div>'
            '<div data-label="Table"><table border="1"><tr><th colspan="2">H</th></tr><tr><td>1</td><td>2</td></tr></table></div>'
            '<div data-label="List-Group"><ul><li>one</li><li><b>two</b></li></ul></div>'
            '<div data-label="Page-Footer">7</div>')
    md = html_to_markdown(html)
    assert "Journal X" not in md and "\n7" not in md
    assert "## Intro" in md and "$E=mc^2$" in md and "$$a^2+b^2=c^2$$" in md
    assert '<table><tr><th colspan="2">H</th></tr>' in md and "- **two**" in md


def test_surya_page():
    page = {"blocks": [{"label": "PageHeader", "html": "<p>Header</p>"}, {"label": "Text", "html": "<p>Body text</p>"}]}
    assert surya_page_to_markdown(page) == "Body text"
