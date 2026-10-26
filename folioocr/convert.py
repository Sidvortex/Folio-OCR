"""Other engines' outputs -> Markdown in the same conventions as ours.

Chandra 2 and Surya OCR 2 emit HTML layout blocks (``<div data-label=...>``)
with ``<math>`` for LaTeX. For olmOCR-Bench every engine must hand in Markdown
with page headers/footers removed, tables as HTML and math in ``$``/``$$``.
"""
from __future__ import annotations

import re
from typing import Iterable, Optional

from bs4 import BeautifulSoup, NavigableString, Tag

NON_BODY_LABELS = {"page-header", "page-footer", "pageheader", "pagefooter", "header", "footer",
                   "page_header", "page_footer"}
_BLOCK_TAGS = {"p", "div", "h1", "h2", "h3", "h4", "h5", "h6", "ul", "ol", "li", "pre", "table",
               "caption", "blockquote", "hr"}


def _clean_table(tbl: Tag) -> str:
    keep = {"colspan", "rowspan"}
    for t in tbl.find_all(True):
        t.attrs = {k: v for k, v in t.attrs.items() if k in keep}
    tbl.attrs = {}
    html = str(tbl)
    html = re.sub(r"\s*\n\s*", "", html)
    return html


def _inline(node, out: list) -> None:
    if isinstance(node, NavigableString):
        out.append(re.sub(r"\s+", " ", str(node)))
        return
    if not isinstance(node, Tag):
        return
    name = node.name.lower()
    if name == "br":
        out.append("\n")
    elif name == "math":
        tex = node.get_text().strip()
        if tex:
            display = node.get("display") in ("block", "true", "display") or "display" in node.attrs
            out.append(f"$${tex}$$" if display else f"${tex}$")
    elif name in ("b", "strong"):
        inner = _render_children(node).strip()
        if inner:
            out.append(f"**{inner}**")
    elif name in ("i", "em"):
        inner = _render_children(node).strip()
        if inner:
            out.append(f"*{inner}*")
    elif name == "img":
        return
    elif name == "input":
        if node.get("type") in ("checkbox", "radio"):
            out.append("[x] " if node.has_attr("checked") else "[ ] ")
    else:
        out.append(_render_children(node))


def _render_children(node: Tag) -> str:
    out: list = []
    for ch in node.children:
        _inline(ch, out)
    return "".join(out)


def _block(node, lines: list) -> None:
    if isinstance(node, NavigableString):
        s = re.sub(r"\s+", " ", str(node)).strip()
        if s:
            lines.append(s)
        return
    if not isinstance(node, Tag):
        return
    name = node.name.lower()
    if name == "table":
        lines.append(_clean_table(node))
    elif name in ("h1", "h2", "h3", "h4", "h5", "h6"):
        txt = _render_children(node).strip()
        if txt:
            lines.append("#" * int(name[1]) + " " + txt)
    elif name in ("ul", "ol"):
        items = []
        for i, li in enumerate(node.find_all("li", recursive=False), 1):
            txt = _render_children(li).strip()
            if txt:
                items.append((f"{i}. " if name == "ol" else "- ") + txt)
        if items:
            lines.append("\n".join(items))
    elif name == "hr":
        return
    elif name == "math" and (node.get("display") or "display" in node.attrs):
        tex = node.get_text().strip()
        if tex:
            lines.append(f"$${tex}$$")
    elif name in ("div", "section", "body", "html", "blockquote") or any(
            isinstance(c, Tag) and c.name in _BLOCK_TAGS for c in node.children):
        for ch in node.children:
            _block(ch, lines)
    else:
        txt = _render_children(node).strip()
        if txt:
            lines.append(txt)


def html_to_markdown(html: str, drop_non_body: bool = True) -> str:
    """HTML (optionally made of ``data-label`` layout divs) -> Markdown."""
    soup = BeautifulSoup(html or "", "html.parser")
    if drop_non_body:
        for div in soup.find_all(attrs={"data-label": True}):
            label = str(div.get("data-label", "")).strip().lower()
            if label in NON_BODY_LABELS:
                div.decompose()
    lines: list = []
    for ch in soup.contents:
        _block(ch, lines)
    md = "\n\n".join(l.strip() for l in lines if l and l.strip())
    return re.sub(r"\n{3,}", "\n\n", md).strip()


def surya_page_to_markdown(page: dict, drop_non_body: bool = True) -> str:
    """One page of ``surya_ocr`` results.json (blocks with label + html)."""
    parts = []
    for b in page.get("blocks", []):
        label = str(b.get("label", "")).replace(" ", "").lower()
        if drop_non_body and label in NON_BODY_LABELS:
            continue
        md = html_to_markdown(b.get("html", "") or "", drop_non_body=drop_non_body)
        if md:
            parts.append(md)
    return "\n\n".join(parts)


def join_pages(pages: Iterable[str]) -> str:
    return "\n\n".join(p for p in pages if p)
