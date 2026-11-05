"""Rewards a program can check (Phase 3), with the plan's starting weights.

    tests    0.5  share of the item's unit tests passed (present / absent /
                  order / table / math / fields), olmOCR's own test classes
    text     0.2  1 - normalised edit distance to the exact label
    table    0.1  TEDS on items whose label has tables
    fields   0.1  exact match on emails, phone numbers, dates and URLs
    format   0.1  valid JSON (doc2json), valid HTML tables, a clean stop, and a
                  length within 0.5-1.5x the label's

A detected loop scores 0 overall. Weights are renormalised over the rewards
that apply to an item (an item without a label text gets no ``text`` reward,
a page without tables no ``table`` reward, ...).

An RL item is a Folio label record plus ``tests``: ``blocks`` or
``markdown`` give the reference (optional for items with tests).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional

from ..loops import is_looping
from ..schema import Block, parse_output, to_markdown
from ..textnorm import field_set, levenshtein, markdown_to_text, normalize_for_eval
from .teds import find_tables, parse_table, teds_many

WEIGHTS = {"tests": 0.5, "text": 0.2, "table": 0.1, "fields": 0.1, "format": 0.1}


@dataclass
class RewardConfig:
    weights: Dict[str, float] = field(default_factory=lambda: dict(WEIGHTS))
    length_lo: float = 0.5
    length_hi: float = 1.5
    task: str = "doc2json"


def reference_markdown(item: dict) -> Optional[str]:
    if item.get("blocks"):
        return to_markdown([Block.from_dict(b) for b in item["blocks"]])
    return item.get("markdown")


def reference_tables(item: dict) -> List[str]:
    if item.get("blocks"):
        return [b["text"] for b in item["blocks"] if b["category"] == "table" and b.get("text")]
    return find_tables(item.get("markdown") or "")


def text_accuracy(ref_md: str, hyp_md: str) -> float:
    r = normalize_for_eval(markdown_to_text(ref_md))
    h = normalize_for_eval(markdown_to_text(hyp_md))
    if not r and not h:
        return 1.0
    return max(0.0, 1.0 - levenshtein(r, h) / max(len(r), len(h)))


def field_score(ref_md: str, hyp_md: str) -> Optional[float]:
    ref, hyp = field_set(markdown_to_text(ref_md)), field_set(markdown_to_text(hyp_md))
    tot = sum(len(v) for v in ref.values())
    if not tot:
        return None
    return sum(1 for k, vals in ref.items() for v in vals if v in hyp.get(k, set())) / tot


def score(item: dict, output: str, cfg: Optional[RewardConfig] = None, stopped: bool = True,
          looped: bool = False) -> Dict:
    """Reward of one sampled ``output`` for ``item``. ``stopped``: the generation
    ended with an end token (not the token cap); ``looped``: a loop was detected
    while generating."""
    cfg = cfg or RewardConfig()
    w = cfg.weights
    out: Dict = {"components": {}}
    if looped or is_looping(output):
        out.update(reward=0.0, loop=True)
        return out
    if cfg.task == "doc2json":
        pr = parse_output(output)
        hyp_md = to_markdown(pr.blocks)
        json_ok = pr.ok and not pr.repaired
        hyp_tables = [b.text for b in pr.blocks if b.category == "table" and b.text]
    else:
        from ..model.generate import strip_md_fences
        hyp_md = strip_md_fences(output)
        json_ok = None
        hyp_tables = find_tables(hyp_md)
    comp = out["components"]

    tests = item.get("tests") or []
    if tests:
        from ..bench.unittests import run_page
        res = run_page(tests, hyp_md, item.get("id", ""), baseline=False)
        comp["tests"] = sum(r["passed"] for r in res) / len(res)
        out["failed_tests"] = [r["id"] for r in res if not r["passed"]][:20]

    ref_md = reference_markdown(item)
    if ref_md is not None:
        comp["text"] = text_accuracy(ref_md, hyp_md)
        ref_tables = reference_tables(item)
        if ref_tables:
            comp["table"] = teds_many(hyp_tables, ref_tables)
        if item.get("fields", True):
            fs = field_score(ref_md, hyp_md)
            if fs is not None:
                comp["fields"] = fs

    checks = [bool(stopped)]
    if json_ok is not None:
        checks.append(bool(json_ok))
    if hyp_tables:
        checks.append(all(parse_table(t) is not None for t in hyp_tables))
    if ref_md:
        ratio = len(markdown_to_text(hyp_md)) / max(1, len(markdown_to_text(ref_md)))
        out["length_ratio"] = round(ratio, 3)
        checks.append(cfg.length_lo <= ratio <= cfg.length_hi)
    comp["format"] = sum(checks) / len(checks)

    tot = sum(w[k] for k in comp)
    out["reward"] = sum(w[k] * v for k, v in comp.items()) / tot if tot else 0.0
    out["loop"] = False
    return out
