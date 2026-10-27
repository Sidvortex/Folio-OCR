"""HTML layouts for synthetic pages.

Every element that becomes a label carries ``data-cat`` (a schema category);
titles carry ``data-level``. ``page.js`` turns each such element into one
block: its box, and its text serialised as Markdown (``<strong>`` -> ``**``,
list items -> ``- ``, ``data-latex`` -> ``$...$``). So the rules here are:

* put the *visible* text in the DOM (no CSS text-transform, no CSS ``content``
  that shows text), and mark bold/italic with ``<strong>``/``<em>`` tags;
* never nest one ``data-cat`` element inside another;
* elements in a flex row are read left to right, joined by a space;
* ``data-fixed="1"`` marks headers/footers that sit outside the text flow.
"""
from __future__ import annotations

import html
import random
import re
from dataclasses import dataclass
from typing import List, Optional, Sequence, Tuple

from .assets import Family
from .content import Cell, Entry, GeneralDoc, GenBlock, ResumeDoc, Section, Table, pick

PAGE_SIZES = {"letter": (816, 1056), "a4": (794, 1123)}   # CSS px at 96/in


def esc(s: str) -> str:
    return html.escape(s, quote=False)


def attr(s: str) -> str:
    return html.escape(s, quote=True)


_BOLD = re.compile(r"\*\*(.+?)\*\*")
_ITAL = re.compile(r"(?<![\w*])\*(?!\s)(.+?)(?<!\s)\*(?![\w*])")


def inline_html(text: str, inline_math: Sequence[str] = ()) -> str:
    """Plain text with **bold**/*italic* markers and \\x00 math slots -> HTML."""
    out = esc(text)
    out = _BOLD.sub(r"<strong>\1</strong>", out)
    out = _ITAL.sub(r"<em>\1</em>", out)
    for m in inline_math:
        out = out.replace("\u0000", f'<span class="im" data-latex="{attr(m)}"></span>', 1)
    return out.replace("\u0000", "")


@dataclass
class Style:
    page: str
    width: int
    height: int
    margins: Tuple[int, int, int, int]       # top right bottom left (px)
    body: Family
    head: Family
    base_pt: float
    line_height: float
    ink: str
    accent: str
    muted: str
    justify: bool
    dense: bool

    def font_stack(self, fam: Family, extra: Sequence[str] = ()) -> str:
        names = [f"'{fam.css_name}'"] + [f"'{x}'" for x in extra] + ["'F_fallback'", "'F_symbols'", "sans-serif"]
        return ", ".join(names)


def choose_style(rng: random.Random, families: Sequence[Family], kind: str, dense: bool = False,
                 serif_bias: float = 0.4) -> Style:
    latin = [f for f in families if f.kind in ("serif", "sans", "mono", "display")]
    serif = [f for f in latin if f.kind == "serif"]
    sans = [f for f in latin if f.kind == "sans"]
    body = pick(rng, serif if (serif and rng.random() < serif_bias) else (sans or latin))
    head = body if rng.random() < 0.55 else pick(rng, [f for f in latin if f.kind != "mono"] or latin)
    page = "a4" if rng.random() < 0.5 else "letter"
    w, h = PAGE_SIZES[page]
    m = rng.randint(28, 52) if dense else rng.randint(40, 86)
    margins = (max(24, m + rng.randint(-10, 10)), m, max(24, m + rng.randint(-8, 14)), m)
    base = rng.uniform(8.0, 9.6) if dense else rng.uniform(9.2, 11.8)
    if kind != "resume":
        base = rng.uniform(9.0, 11.5)
    ink = pick(rng, ["#000", "#111", "#1a1a1a", "#222", "#262626", "#1f2933", "#0f172a", "#2b2b2b"])
    accent = pick(rng, ["#1f4e79", "#0b6e4f", "#7a1f1f", "#2f3e9e", "#444", "#000", "#00599c", "#5b2c6f", "#8a4b08", "#205e78"])
    muted = pick(rng, ["#444", "#555", "#5f6b7a", "#666", "#3d4852"])
    return Style(page, w, h, margins, body, head, round(base, 2), round(rng.uniform(1.12, 1.42), 3),
                 ink, accent, muted, rng.random() < 0.25, dense)


def base_css(st: Style) -> str:
    t, r, b, l = st.margins
    return f"""
html, body {{ margin:0; padding:0; background:#fff; }}
* {{ box-sizing:border-box; }}
#page {{ position:relative; width:{st.width}px; height:{st.height}px; overflow:hidden; background:#fff;
  padding:{t}px {r}px {b}px {l}px; font-family:{st.font_stack(st.body)}; font-size:{st.base_pt}pt;
  line-height:{st.line_height}; color:{st.ink}; hyphens:none; overflow-wrap:break-word; font-kerning:normal;
  text-rendering:geometricPrecision; }}
#page [data-cat], #page [data-group] {{ break-inside:avoid; }}
#page p {{ margin:0; }}
#page ul, #page ol {{ margin:0; }}
.fixed {{ position:absolute; left:{l}px; right:{r}px; }}
.row {{ display:flex; justify-content:space-between; align-items:baseline; gap:12px; }}
.row > :last-child {{ text-align:right; white-space:nowrap; }}
.im {{ white-space:nowrap; }}
table.t {{ border-collapse:collapse; width:100%; }}
table.t td, table.t th {{ padding:2px 6px; vertical-align:top; text-align:left; overflow-wrap:normal; word-break:normal; }}
"""


# ------------------------------------------------------------------ helpers
def svg_photo(rng: random.Random, size: int) -> str:
    bg = pick(rng, ["#c9d3dd", "#d8d2c8", "#cfcfcf", "#bcc8b8", "#d6c7d9"])
    fg = pick(rng, ["#7d8a96", "#8b7f70", "#888", "#6e7f69", "#8d7393"])
    shape = '<circle cx="50" cy="50" r="50"/>' if rng.random() < 0.5 else '<rect width="100" height="100" rx="8"/>'
    return (f'<svg width="{size}" height="{size}" viewBox="0 0 100 100" aria-hidden="true">'
            f'<g fill="{bg}">{shape}</g><circle cx="50" cy="38" r="17" fill="{fg}"/>'
            f'<path d="M18 92 C22 66 78 66 82 92 Z" fill="{fg}"/></svg>')


def svg_icon(rng: random.Random, kind: str, color: str, size: float) -> str:
    paths = {
        "email": '<rect x="2" y="5" width="20" height="14" rx="2" fill="none" stroke="{c}" stroke-width="2"/><path d="M3 6l9 7 9-7" fill="none" stroke="{c}" stroke-width="2"/>',
        "phone": '<path d="M6 3h4l2 5-3 2a12 12 0 006 6l2-3 5 2v4a2 2 0 01-2 2A17 17 0 014 5a2 2 0 012-2z" fill="{c}"/>',
        "location": '<path d="M12 2a7 7 0 00-7 7c0 5 7 13 7 13s7-8 7-13a7 7 0 00-7-7z" fill="{c}"/><circle cx="12" cy="9" r="2.5" fill="#fff"/>',
        "url": '<circle cx="12" cy="12" r="9" fill="none" stroke="{c}" stroke-width="2"/><path d="M3 12h18M12 3c3 3 3 15 0 18M12 3c-3 3-3 15 0 18" fill="none" stroke="{c}" stroke-width="1.5"/>',
    }
    p = paths.get(kind, paths["url"]).format(c=color)
    return (f'<svg aria-hidden="true" width="{size:.1f}" height="{size:.1f}" viewBox="0 0 24 24" '
            f'style="vertical-align:-0.15em;margin-right:4px">{p}</svg>')


def svg_figure(rng: random.Random, w: int, h: int) -> str:
    """A plot-like drawing (axes, bars or a curve, tick labels). Figures are labelled ''."""
    pts = []
    n = rng.randint(5, 12)
    if rng.random() < 0.5:
        bw = (w - 40) / n
        bars = "".join(f'<rect x="{30 + i * bw + 2:.1f}" y="{h - 20 - v:.1f}" width="{bw - 4:.1f}" height="{v:.1f}" fill="#8aa1b5"/>'
                       for i, v in enumerate(rng.uniform(10, h - 40) for _ in range(n)))
        body = bars
    else:
        for i in range(n):
            pts.append(f"{30 + i * (w - 40) / (n - 1):.1f},{h - 20 - rng.uniform(10, h - 40):.1f}")
        body = f'<polyline points="{" ".join(pts)}" fill="none" stroke="#2f5d8a" stroke-width="2"/>'
    ticks = "".join(f'<text x="{30 + i * (w - 40) / 4:.0f}" y="{h - 5}" font-size="9" fill="#555">{i * 5}</text>' for i in range(5))
    return (f'<svg width="{w}" height="{h}" viewBox="0 0 {w} {h}" aria-hidden="true">'
            f'<line x1="28" y1="{h - 20}" x2="{w - 5}" y2="{h - 20}" stroke="#333"/><line x1="28" y1="5" x2="28" y2="{h - 20}" stroke="#333"/>'
            f'{body}{ticks}</svg>')


def table_html(tbl: Table, cls: str = "t", style: str = "") -> str:
    rows = []
    for r in tbl.rows:
        cells = []
        for c in r:
            tag = "th" if c.header else "td"
            span = (f' colspan="{c.colspan}"' if c.colspan > 1 else "") + (f' rowspan="{c.rowspan}"' if c.rowspan > 1 else "")
            txt = esc(c.text)
            if c.bold and txt:
                txt = f"<strong>{txt}</strong>"
            cells.append(f"<{tag}{span}>{txt}</{tag}>")
        rows.append("<tr>" + "".join(cells) + "</tr>")
    return f'<table data-cat="table" class="{cls}" style="{style}">' + "".join(rows) + "</table>"


def heading_css(rng: random.Random, st: Style, size_pt: float) -> Tuple[str, str]:
    """CSS for section headings + a variant name (rule, bar, plain, shaded, side-rule)."""
    v = rng.choice(["rule", "rule", "plain", "bar", "shaded", "underline", "spaced"])
    color = st.accent if rng.random() < 0.6 else st.ink
    common = (f"font-family:{st.font_stack(st.head)}; font-size:{size_pt:.1f}pt; font-weight:{rng.choice([600, 700, 700])};"
              f"color:{color}; margin:{rng.randint(8, 16)}px 0 {rng.randint(3, 7)}px 0;")
    if v == "rule":
        css = common + f"border-bottom:{rng.choice([1, 1, 1.5, 2])}px solid {color}; padding-bottom:2px;"
    elif v == "bar":
        css = common + f"border-left:4px solid {st.accent}; padding-left:8px;"
    elif v == "shaded":
        css = common + f"background:{pick(rng, ['#e8e8e8', '#dfe6ee', '#ececec', '#e4ebe4'])}; padding:2px 6px;"
    elif v == "underline":
        css = common + "text-decoration:underline; text-underline-offset:3px;"
    elif v == "spaced":
        css = common + "letter-spacing:0.12em;"
    else:
        css = common
    return css, v


# ------------------------------------------------------------- resume parts
def contact_html(rng: random.Random, doc: ResumeDoc, st: Style, mode: str, align: str) -> str:
    icons = rng.random() < 0.35
    items = []
    for kind, val in doc.contacts:
        lab = ""
        if mode == "labelled":
            lab = {"email": "Email: ", "phone": pick(rng, ["Phone: ", "Mobile: ", "Ph: ", "Contact: "]), "location": "Address: ",
                   "url": ""}[kind]
        ic = svg_icon(rng, kind, st.accent, st.base_pt * 1.2) if icons else ""
        items.append((kind, f"{ic}{esc(lab + val)}"))
    if mode == "lines" or mode == "labelled":
        return "".join(f'<div data-cat="text" class="ct" style="text-align:{align}">{h}</div>' for _, h in items)
    sep = pick(rng, [" | ", " | ", " • ", " · ", " ◦ ", "  ", " / "])
    if icons:
        sep = "&nbsp;&nbsp;&nbsp;"
    k = rng.randint(2, 3) if len(items) > 3 and rng.random() < 0.5 else len(items)
    line1 = esc(sep).join(h for _, h in items[:k]) if not icons else sep.join(h for _, h in items[:k])
    out = f'<div data-cat="text" class="ct" style="text-align:{align}">{line1}</div>'
    if k < len(items):
        line2 = esc(sep).join(h for _, h in items[k:]) if not icons else sep.join(h for _, h in items[k:])
        out += f'<div data-cat="text" class="ct" style="text-align:{align}">{line2}</div>'
    return out


def entry_html(rng: random.Random, e: Entry, st: Style, variant: int, date_col: bool) -> str:
    out = []
    right = f'<span class="dt">{esc(e.right)}</span>' if e.right else ""
    if date_col and e.right:
        main = f"<strong>{esc(e.left)}</strong>" + (f", {esc(e.sub_left)}" if e.sub_left else "")
        out.append(f'<div data-cat="text" class="row dc" style="justify-content:flex-start">'
                   f'<span class="dt" style="min-width:24%;text-align:left">{esc(e.right)}</span><span>{main}</span></div>')
    elif variant == 0:      # Role | Company ......... dates
        sep = pick(rng, [" | ", ", ", " — ", " at "])
        left = f"<strong>{esc(e.left)}</strong>" + (f"{esc(sep)}{esc(e.sub_left)}" if e.sub_left else "")
        out.append(f'<div data-cat="text" class="row"><span>{left}</span>{right}</div>')
        if e.sub_right and rng.random() < 0.5:
            out.append(f'<div data-cat="text" class="sub"><em>{esc(e.sub_right)}</em></div>')
    elif variant == 1:      # Company ... Location / Role ... dates
        if e.sub_left:
            loc = f'<span>{esc(e.sub_right)}</span>' if e.sub_right else ""
            out.append(f'<div data-cat="text" class="row"><span><strong>{esc(e.sub_left)}</strong></span>{loc}</div>')
        out.append(f'<div data-cat="text" class="row"><span><em>{esc(e.left)}</em></span>{right}</div>')
    else:                   # one line: Role, Company (dates)
        txt = f"<strong>{esc(e.left)}</strong>"
        if e.sub_left:
            txt += f", {esc(e.sub_left)}"
        if e.right:
            txt += f" ({esc(e.right)})"
        out.append(f'<div data-cat="text">{txt}</div>')
    if e.note:
        out.append(f'<div data-cat="text" class="note">{esc(e.note)}</div>')
    if e.bullets:
        tag = "ul"
        out.append(f'<{tag} data-cat="text" class="bl">' + "".join(f"<li>{esc(b)}</li>" for b in e.bullets) + f"</{tag}>")
    return "".join(out)


def section_html(rng: random.Random, sec: Section, st: Style, hcss_class: str, entry_variant: int,
                 date_col: bool, skills_mode: str) -> str:
    parts = [f'<section><div data-cat="title" data-level="2" class="{hcss_class}">{esc(sec.title)}</div>']
    if sec.kind == "paragraph":
        parts.append(f'<p data-cat="text">{esc(sec.paragraph)}</p>')
    elif sec.kind == "entries":
        for e in sec.entries:
            parts.append(f'<div class="ent">{entry_html(rng, e, st, entry_variant, date_col)}</div>')
    elif sec.kind == "bullets":
        parts.append('<ul data-cat="text" class="bl">' + "".join(f"<li>{esc(i)}</li>" for i in sec.items) + "</ul>")
    elif sec.kind == "skills":
        if skills_mode == "table":
            rows = [[Cell(k, bold=True), Cell(v)] for k, v in sec.pairs]
            parts.append(table_html(Table(rows), cls="t sk"))
        elif skills_mode == "list":
            parts.append('<ul data-cat="text" class="bl">' + "".join(
                f"<li><strong>{esc(k)}:</strong> {esc(v)}</li>" for k, v in sec.pairs) + "</ul>")
        else:
            parts.append('<div data-cat="text">' + "<br>".join(
                f"<strong>{esc(k)}:</strong> {esc(v)}" for k, v in sec.pairs) + "</div>")
    elif sec.kind == "inline":
        if rng.random() < 0.5:
            parts.append(f'<div data-cat="text">{esc(pick(rng, [", ", " | ", " • ", " · "]).join(sec.items))}</div>')
        else:
            parts.append('<ul data-cat="text" class="bl">' + "".join(f"<li>{esc(i)}</li>" for i in sec.items) + "</ul>")
    elif sec.kind == "table" and sec.table:
        border = pick(rng, ["1px solid #555", "1px solid #888", "1px solid #000"])
        parts.append(table_html(sec.table, cls="t bord", style=f"--b:{border}"))
    parts.append("</section>")
    return "".join(parts)


def resume_css(rng: random.Random, st: Style) -> Tuple[str, str]:
    hsize = st.base_pt * rng.uniform(1.05, 1.35)
    hcss, hvar = heading_css(rng, st, hsize)
    name_pt = st.base_pt * rng.uniform(1.7, 2.9)
    bullet = pick(rng, ["disc", "disc", "circle", "square", "'▪ '", "'➢ '", "'– '", "'• '", "'◦ '", "'✓ '"])
    indent = rng.randint(14, 24)
    gap = rng.randint(2, 6) if not st.dense else rng.randint(0, 3)
    css = f"""
.name {{ font-family:{st.font_stack(st.head)}; font-size:{name_pt:.1f}pt; font-weight:{rng.choice([600, 700, 700, 800])};
  color:{st.accent if rng.random() < 0.4 else st.ink}; line-height:1.1; letter-spacing:{rng.choice([0, 0, 0.02, 0.05])}em; }}
.headline {{ font-size:{st.base_pt * rng.uniform(1.05, 1.3):.1f}pt; color:{st.muted}; margin-top:2px; }}
.ct {{ font-size:{st.base_pt * rng.uniform(0.88, 1.0):.1f}pt; margin-top:2px; }}
.sec {{ {hcss} }}
.ent {{ margin-top:{gap + 2}px; }}
.sub {{ color:{st.muted}; }}
.note {{ color:{st.muted}; font-size:{st.base_pt * 0.95:.1f}pt; }}
.dt {{ color:{st.muted if rng.random() < 0.5 else st.ink}; font-size:{st.base_pt * rng.uniform(0.9, 1.0):.1f}pt; }}
ul.bl {{ padding-left:{indent}px; margin:{gap}px 0 0 0; list-style-type:{bullet}; }}
ul.bl li {{ margin:{rng.choice([0, 1, 2])}px 0; {'text-align:justify;' if st.justify else ''} }}
p[data-cat] {{ {'text-align:justify;' if st.justify else ''} }}
table.bord, table.bord td, table.bord th {{ border:var(--b); }}
table.bord th {{ background:{pick(rng, ['#eee', '#e3e8ef', '#f2f2f2', 'transparent'])}; }}
table.sk td:first-child {{ white-space:nowrap; width:1%; }}
"""
    return css, hvar


def resume_classic(rng: random.Random, doc: ResumeDoc, st: Style) -> Tuple[str, str]:
    css, _ = resume_css(rng, st)
    align = "center" if rng.random() < 0.5 else "left"
    body = []
    if doc.title_line:
        body.append(f'<div data-cat="title" data-level="1" style="text-align:center;font-weight:700;'
                    f'font-size:{st.base_pt * 1.5:.1f}pt;margin-bottom:8px;text-decoration:underline">{esc(doc.title_line)}</div>')
    photo = ""
    if doc.photo:
        sz = rng.randint(80, 110)
        photo = (f'<div data-cat="figure" style="float:right;margin-left:12px;width:{sz}px;height:{sz}px">'
                 f'{svg_photo(rng, sz)}</div>')
        align = "left"
    body.append(f'<header style="text-align:{align};margin-bottom:6px">{photo}')
    body.append(f'<div data-cat="title" data-level="1" class="name">{esc(doc.name)}</div>')
    if doc.headline:
        body.append(f'<div data-cat="text" class="headline">{esc(doc.headline)}</div>')
    body.append(contact_html(rng, doc, st, rng.choice(["inline", "inline", "lines", "labelled"]), align))
    body.append('<div style="clear:both"></div></header>')
    variant = rng.randrange(3)
    date_col = rng.random() < 0.12
    skills_mode = rng.choice(["lines", "lines", "list", "table"])
    for sec in doc.sections:
        body.append(section_html(rng, sec, st, "sec", variant, date_col, skills_mode))
    body.append(personal_and_declaration(rng, doc, st))
    return css, "".join(body)


def personal_and_declaration(rng: random.Random, doc: ResumeDoc, st: Style) -> str:
    out = []
    if doc.personal:
        title = "PERSONAL DETAILS" if rng.random() < 0.6 else "Personal Details"
        out.append(f'<section><div data-cat="title" data-level="2" class="sec">{title}</div>')
        if rng.random() < 0.5:
            rows = [[Cell(k), Cell(v)] for k, v in doc.personal]
            out.append(table_html(Table(rows), cls="t sk"))
        else:
            sep = pick(rng, [": ", " : ", " – "])
            out.append('<div data-cat="text">' + "<br>".join(f"{esc(k)}{esc(sep)}{esc(v)}" for k, v in doc.personal) + "</div>")
        out.append("</section>")
    if doc.declaration:
        out.append(f'<section><div data-cat="title" data-level="2" class="sec">{"DECLARATION" if rng.random() < 0.6 else "Declaration"}</div>')
        out.append(f'<p data-cat="text">{esc(doc.declaration)}</p>')
        # two blocks, as a reader sees them: place and date on the left, the signature name on the right
        out.append(f'<div class="row" style="margin-top:14px;align-items:flex-end">'
                   f'<div data-cat="text">Place: {esc(doc.place)}<br>Date: {esc(doc.date)}</div>'
                   f'<div data-cat="text">({esc(doc.name.title())})</div></div></section>')
    if doc.footer:
        out.append(f'<div data-cat="footer" data-fixed="1" class="fixed" style="bottom:{max(10, st.margins[2] // 2 - 6)}px;'
                   f'text-align:center;font-size:{st.base_pt * 0.8:.1f}pt;color:{st.muted}">{esc(doc.footer)}</div>')
    return "".join(out)


def resume_sidebar(rng: random.Random, doc: ResumeDoc, st: Style) -> Tuple[str, str]:
    css, _ = resume_css(rng, st)
    side_left = rng.random() < 0.7
    side_w = rng.randint(28, 36)
    dark = rng.random() < 0.35
    side_bg = pick(rng, ["#1f2d3d", "#2d3a4a", "#243b55", "#3b3b3b", "#1e3a34"]) if dark else pick(rng, ["#eef2f6", "#f3efe8", "#eef3ee", "#f0f0f0", "#e9eef5"])
    side_ink = "#f2f2f2" if dark else st.ink
    t, r, b, l = st.margins
    css += f"""
#page {{ padding:0; }}
.wrap {{ display:flex; flex-direction:{'row' if side_left else 'row-reverse'}; height:100%; }}
.side {{ width:{side_w}%; background:{side_bg}; color:{side_ink}; padding:{t}px {max(14, l // 2)}px {b}px {max(16, l // 2)}px; }}
.main {{ flex:1; padding:{t}px {r}px {b}px {max(18, l // 2)}px; }}
.side .sec {{ color:{side_ink}; border-color:{side_ink}; background:transparent; }}
.side .ct, .side .dt, .side .sub, .side .note {{ color:{side_ink}; }}
"""
    side, main = [], []
    if doc.photo:
        sz = rng.randint(90, 130)
        side.append(f'<div data-cat="figure" style="width:{sz}px;height:{sz}px;margin:0 auto 12px auto">{svg_photo(rng, sz)}</div>')
    name_in_side = rng.random() < 0.35
    tgt = side if name_in_side else main
    tgt.append(f'<div data-cat="title" data-level="1" class="name">{esc(doc.name)}</div>')
    if doc.headline:
        tgt.append(f'<div data-cat="text" class="headline">{esc(doc.headline)}</div>')
    ct_title = pick(rng, ["CONTACT", "Contact", "Contact Details"])
    side.append(f'<section><div data-cat="title" data-level="2" class="sec">{ct_title}</div>')
    side.append(contact_html(rng, doc, st, "lines", "left") + "</section>")
    variant = rng.randrange(3)
    for sec in doc.sections:
        if sec.side and rng.random() < 0.85:
            side.append(section_html(rng, sec, st, "sec", variant, False, "lines"))
        else:
            main.append(section_html(rng, sec, st, "sec", variant, rng.random() < 0.1, "lines"))
    main.append(personal_and_declaration(rng, doc, st))
    side_html = f'<aside class="side">{"".join(side)}</aside>'
    main_html = f'<div class="main">{"".join(main)}</div>'
    # DOM order = reading order: the left column first
    return css, f'<div class="wrap">{side_html}{main_html}</div>' if side_left else f'<div class="wrap">{main_html}{side_html}</div>'


def resume_split(rng: random.Random, doc: ResumeDoc, st: Style) -> Tuple[str, str]:
    css, _ = resume_css(rng, st)
    left_w = rng.randint(55, 66)
    css += f"""
.cols {{ display:flex; gap:{rng.randint(18, 30)}px; }}
.cols > .c1 {{ width:{left_w}%; }}
.cols > .c2 {{ flex:1; {'border-left:1px solid #bbb; padding-left:14px;' if rng.random() < 0.5 else ''} }}
"""
    head = [f'<header style="margin-bottom:8px">']
    if doc.photo:
        sz = rng.randint(76, 100)
        head.append(f'<div data-cat="figure" style="float:left;margin-right:14px;width:{sz}px;height:{sz}px">{svg_photo(rng, sz)}</div>')
    head.append(f'<div data-cat="title" data-level="1" class="name">{esc(doc.name)}</div>')
    if doc.headline:
        head.append(f'<div data-cat="text" class="headline">{esc(doc.headline)}</div>')
    head.append(contact_html(rng, doc, st, "inline", "left"))
    head.append('<div style="clear:both"></div></header>')
    c1, c2 = [], []
    variant = rng.randrange(3)
    for sec in doc.sections:
        (c2 if sec.side or sec.title.lower().startswith(("education", "academic")) and rng.random() < 0.5 else c1).append(
            section_html(rng, sec, st, "sec", variant, False, rng.choice(["lines", "list"])))
    c1.append(personal_and_declaration(rng, doc, st))
    return css, "".join(head) + f'<div class="cols"><div class="c1">{"".join(c1)}</div><div class="c2">{"".join(c2)}</div></div>'


RESUME_LAYOUTS = {"classic": (resume_classic, 0.45), "sidebar": (resume_sidebar, 0.3), "split": (resume_split, 0.25)}


def build_resume_page(rng: random.Random, doc: ResumeDoc, families: Sequence[Family]) -> Tuple[str, Style, str]:
    names = list(RESUME_LAYOUTS)
    layout = rng.choices(names, weights=[RESUME_LAYOUTS[n][1] for n in names])[0]
    if doc.personal and layout == "sidebar" and rng.random() < 0.5:
        layout = "classic"
    st = choose_style(rng, families, "resume", dense=rng.random() < 0.25)
    css, body = RESUME_LAYOUTS[layout][0](rng, doc, st)
    return base_css(st) + css, st, body if body else "", layout


# ------------------------------------------------------------ general pages
def general_block_html(rng: random.Random, b: GenBlock, st: Style, fig_w: int, eq_no: List[int], number_eqs: bool) -> str:
    if b.cat == "title":
        return f'<div data-cat="title" data-level="{b.level}" class="h{b.level}">{inline_html(b.text)}</div>'
    if b.cat == "formula":
        eq = f'<div data-cat="formula" class="eq" data-latex="{attr(b.latex)}" data-display="1"></div>'
        if number_eqs:
            eq_no[0] += 1
            return (f'<div class="eqrow" data-group="1">{eq}<div data-cat="formula_caption" class="eqn">({eq_no[0]})</div></div>')
        return eq
    if b.cat == "table" and b.table:
        out = '<div data-group="1">'
        if b.table.caption:
            out += f'<div data-cat="table_caption" class="cap">{inline_html(b.table.caption)}</div>'
        border = pick(rng, ["1px solid #444", "1px solid #999", "0"])
        style = f"--b:{border}"
        cls = "t num" + (" bord" if border != "0" else " booktabs")
        out += table_html(b.table, cls=cls, style=style)
        if b.table.footnote:
            out += f'<div data-cat="table_footnote" class="tfn">{inline_html(b.table.footnote)}</div>'
        return out + "</div>"
    if b.items:
        tag = "ol" if b.ordered else "ul"
        return f'<{tag} data-cat="text" class="bl">' + "".join(f"<li>{inline_html(i)}</li>" for i in b.items) + f"</{tag}>"
    return f'<p data-cat="text">{inline_html(b.text, b.inline_math)}</p>'


def build_general_page(rng: random.Random, doc: GeneralDoc, families: Sequence[Family],
                       typewriter: bool = False) -> Tuple[str, Style, str, str]:
    st = choose_style(rng, families, "general", serif_bias=0.6 if doc.kind in ("article", "math") else 0.35)
    if doc.script != "latn":
        indic = [f for f in families if f.kind == "indic" and f.script == doc.script]
        if indic:
            st.body = st.head = pick(rng, indic)
    if (typewriter or doc.kind == "letter" and rng.random() < 0.5) and doc.script == "latn":
        typew = [f for f in families if f.name in ("courierprime", "specialelite", "cousine")]
        if typew:
            st.body = pick(rng, typew)
    t, r, b, l = st.margins
    cols = doc.columns
    css = f"""
.h1 {{ font-family:{st.font_stack(st.head)}; font-size:{st.base_pt * rng.uniform(1.5, 2.1):.1f}pt; font-weight:700; text-align:{pick(rng, ['center', 'left'])}; margin:0 0 6px 0; line-height:1.2; }}
.h2 {{ font-family:{st.font_stack(st.head)}; font-size:{st.base_pt * rng.uniform(1.08, 1.3):.1f}pt; font-weight:700; margin:10px 0 4px 0; }}
.authors {{ text-align:center; color:{st.muted}; margin-bottom:10px; }}
.body p {{ margin:0 0 {rng.randint(4, 9)}px 0; text-indent:{rng.choice([0, 0, 14])}px; {'text-align:justify;' if rng.random() < 0.6 else ''} }}
.body {{ column-count:{cols}; column-gap:{rng.randint(18, 30)}px; column-fill:auto; height:100%; }}
.eq {{ text-align:center; margin:8px 0; }}
.eqrow {{ display:flex; align-items:center; }} .eqrow .eq {{ flex:1; }} .eqn {{ width:40px; text-align:right; }}
.cap {{ font-size:{st.base_pt * 0.92:.1f}pt; margin:8px 0 3px 0; {'font-weight:600;' if rng.random() < 0.4 else ''} }}
.tfn {{ font-size:{st.base_pt * 0.82:.1f}pt; color:{st.muted}; margin:2px 0 8px 0; }}
table.num td:not(:first-child) {{ text-align:right; }}
table.bord, table.bord td, table.bord th {{ border:var(--b); }}
table.booktabs {{ border-top:1.5px solid #000; border-bottom:1.5px solid #000; }}
table.booktabs tr:first-child th {{ border-bottom:1px solid #000; }}
ul.bl, ol.bl {{ padding-left:20px; margin:0 0 6px 0; }}
.fn {{ font-size:{st.base_pt * 0.8:.1f}pt; }}
"""
    fixed = []
    if doc.header:
        fixed.append(f'<div data-cat="header" data-fixed="1" class="fixed" style="top:{max(8, t // 2 - 8)}px;font-size:{st.base_pt * 0.82:.1f}pt;'
                     f'color:{st.muted};display:flex;justify-content:space-between"><span>{esc(doc.header)}</span>'
                     f'{"<span>" + esc(doc.page_no) + "</span>" if doc.page_no and rng.random() < 0.5 else ""}</div>')
    elif doc.page_no and rng.random() < 0.3:
        fixed.append(f'<div data-cat="header" data-fixed="1" class="fixed" style="top:{max(8, t // 2 - 8)}px;text-align:right;'
                     f'font-size:{st.base_pt * 0.82:.1f}pt">{esc(doc.page_no)}</div>')
    footer_bits = []
    if doc.footer:
        footer_bits.append(esc(doc.footer))
    if doc.page_no and not any(doc.page_no in f for f in fixed):
        footer_bits.append(esc(doc.page_no))
    if footer_bits:
        fixed.append(f'<div data-cat="footer" data-fixed="1" class="fixed" style="bottom:{max(8, b // 2 - 10)}px;text-align:center;'
                     f'font-size:{st.base_pt * 0.82:.1f}pt;color:{st.muted}">{" · ".join(footer_bits)}</div>')
    fn_h = 0
    if doc.footnotes:
        fn_lines = "".join(f'<div data-cat="page_footnote" class="fn">{esc(x)}</div>' for x in doc.footnotes)
        fn_h = int(len(doc.footnotes) * st.base_pt * 1.9) + 10
        fixed.append(f'<div data-fixed="1" class="fixed" style="bottom:{b - 2}px;border-top:0.8px solid #888;padding-top:3px">{fn_lines}</div>')
    head = []
    if doc.title:
        head.append(f'<div data-cat="title" data-level="1" class="h1">{inline_html(doc.title)}</div>')
    if doc.subtitle:
        head.append(f'<div data-cat="text" class="authors">{esc(doc.subtitle)}</div>')
    eq_no = [0]
    number_eqs = rng.random() < 0.5
    blocks = []
    figure_done = False
    for blk in doc.blocks:
        blocks.append(general_block_html(rng, blk, st, 300, eq_no, number_eqs))
        if doc.kind == "article" and not figure_done and blk.cat == "text" and rng.random() < 0.15:
            fw = (st.width - l - r) // (cols if cols else 1) - 10
            fh = rng.randint(120, 200)
            blocks.append(f'<div data-group="1"><div data-cat="figure">{svg_figure(rng, fw, fh)}</div>'
                          f'<div data-cat="figure_caption" class="cap">Figure 1: {esc(GeneralCaption.make(rng))}</div></div>')
            figure_done = True
    reserve = fn_h + 4
    body = (f'<div class="headblock">{"".join(head)}</div>'
            f'<div class="body" style="height:calc(100% - var(--hb, 0px) - {reserve}px)">{"".join(blocks)}</div>'
            + "".join(fixed))
    return base_css(st) + css, st, body, doc.kind


class GeneralCaption:
    @staticmethod
    def make(rng: random.Random) -> str:
        return pick(rng, ["Accuracy versus training steps", "Distribution of response times", "Comparison of methods on the test split",
                          "Monthly revenue by region", "Loss curves for three learning rates", "Ablation over input resolution"])


def html_document(css: str, body: str, font_css: str, katex_dir: Optional[str]) -> str:
    katex = ""
    if katex_dir:
        from pathlib import Path
        k = Path(katex_dir).resolve()
        katex = (f'<link rel="stylesheet" href="{(k / "katex.min.css").as_uri()}">'
                 f'<script src="{(k / "katex.min.js").as_uri()}"></script>')
    return (f'<!doctype html><html><head><meta charset="utf-8">{katex}<style>{font_css}\n{css}</style></head>'
            f'<body><div id="page">{body}</div></body></html>')
