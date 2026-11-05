"""TEDS: tree-edit-distance similarity between two HTML tables (Zhong et al., PubTabNet).

TEDS = 1 - EditDistance(pred, ref) / max(|pred|, |ref|) over the table trees
(table -> tr -> td). A node rename costs 1 when the tag, colspan or rowspan
differ, and the normalised character edit distance of the contents when two
cells are compared. Like many TEDS variants we read ``th`` as ``td`` and drop
``thead``/``tbody``/``tfoot`` wrappers, so the score is about structure and
content, not about which wrapper a model chose.
"""
from __future__ import annotations

import re
from typing import List, Optional

from apted import APTED, Config

from ..textnorm import levenshtein

_TABLE = re.compile(r"<table\b.*?</table>", re.I | re.S)


class Node:
    def __init__(self, tag: str, colspan: int = 1, rowspan: int = 1, content: str = "", children=None):
        self.tag, self.colspan, self.rowspan, self.content = tag, colspan, rowspan, content
        self.children = list(children or [])


class _Cost(Config):
    valuecls = float

    def rename(self, a: Node, b: Node) -> float:
        if a.tag != b.tag or a.colspan != b.colspan or a.rowspan != b.rowspan:
            return 1.0
        if a.tag == "td" and (a.content or b.content):
            return levenshtein(a.content, b.content) / max(len(a.content), len(b.content))
        return 0.0

    def children(self, node: Node):
        return node.children


def _span(el, name: str) -> int:
    try:
        return max(1, int(el.get(name, 1)))
    except (TypeError, ValueError):
        return 1


def parse_table(html: str) -> Optional[Node]:
    """HTML (one table) -> tree, or None when it does not parse as a table with rows."""
    from bs4 import BeautifulSoup
    soup = BeautifulSoup(html or "", "html.parser")
    t = soup.find("table")
    if t is None:
        return None
    rows = []
    for tr in t.find_all("tr"):
        if tr.find_parent("table") is not t:          # a nested table's rows belong to that table
            continue
        cells = []
        for c in tr.find_all(["td", "th"], recursive=False):
            text = " ".join(c.get_text(" ").split())
            cells.append(Node("td", _span(c, "colspan"), _span(c, "rowspan"), text))
        rows.append(Node("tr", children=cells))
    if not rows:
        return None
    return Node("table", children=rows)


def _size(n: Node) -> int:
    return 1 + sum(_size(c) for c in n.children)


def teds(pred_html: str, ref_html: str) -> float:
    ref = parse_table(ref_html)
    if ref is None:
        raise ValueError("reference is not a table")
    pred = parse_table(pred_html)
    if pred is None:
        return 0.0
    dist = APTED(pred, ref, _Cost()).compute_edit_distance()
    return max(0.0, 1.0 - float(dist) / max(_size(pred), _size(ref)))


def find_tables(text: str) -> List[str]:
    return _TABLE.findall(text or "")


def teds_many(pred_tables: List[str], ref_tables: List[str]) -> float:
    """Tables paired in reading order; missing or extra tables score 0."""
    if not ref_tables:
        raise ValueError("no reference tables")
    n = max(len(pred_tables), len(ref_tables))
    return sum(teds(p, r) for p, r in zip(pred_tables, ref_tables)) / n
