"""Evaluate a parser on a labelled dev set (our label format + optional unit tests).

Metrics per run:
    tests      olmOCR-style unit-test score (mean of category pass rates)
    cer        page CER of the Markdown reading text (normalised), against the label
    fields     field accuracy: share of label emails/phones/URLs/dates found exactly
    parse_ok   share of doc2json outputs that parsed without repair
    loops      share of pages that looped; tokens and seconds per page
"""
from __future__ import annotations

import json
import os
import time
from typing import Dict, List, Optional

from PIL import Image

from ..schema import Block, to_markdown
from ..textnorm import cer, field_set, markdown_to_text


def field_accuracy(ref_md: str, hyp_md: str) -> Dict[str, List[int]]:
    """{kind: [found, total]} for every field in the reference body text."""
    ref, hyp = field_set(markdown_to_text(ref_md)), field_set(markdown_to_text(hyp_md))
    out: Dict[str, List[int]] = {}
    for kind, vals in ref.items():
        got = hyp.get(kind, set())
        out[kind] = [sum(1 for v in vals if v in got), len(vals)]
    return out


def evaluate_records(parser, records: List[dict], tests_by_page: Optional[Dict[str, List[dict]]] = None,
                     limit: Optional[int] = None, batch_size: Optional[int] = None) -> Dict:
    recs = records[:limit] if limit else records
    bs = batch_size or parser.batch_size
    outputs = {}
    t0 = time.time()
    for s in range(0, len(recs), bs):
        chunk = recs[s:s + bs]
        imgs = [Image.open(r["_image"]).convert("RGB") for r in chunk]
        for r, res in zip(chunk, parser.parse(imgs)):
            outputs[r["id"]] = res
    secs = time.time() - t0
    err = tot = 0
    fields: Dict[str, List[int]] = {}
    rows = []
    for r in recs:
        ref_md = to_markdown([Block.from_dict(b) for b in r["blocks"]]) if "blocks" in r else r.get("markdown", "")
        hyp_md = outputs[r["id"]].markdown
        e, n = cer(markdown_to_text(ref_md), markdown_to_text(hyp_md))
        err += e
        tot += n
        f_ok = f_tot = 0
        for k, (f, t) in field_accuracy(ref_md, hyp_md).items():
            fields.setdefault(k, [0, 0])
            fields[k][0] += f
            fields[k][1] += t
            f_ok, f_tot = f_ok + f, f_tot + t
        # per-page rows: compare.py bootstraps CER and fields over pages with them
        rows.append({"id": r["id"], "tier": r.get("condition") or r.get("source") or "all", "err": e, "chars": n,
                     "fields": [f_ok, f_tot]})
    n = max(1, len(recs))
    m = {
        "pages": len(recs),
        "cer": err / max(1, tot),
        "fields": sum(f for f, _ in fields.values()) / max(1, sum(t for _, t in fields.values())) if fields else None,
        "fields_by_kind": {k: f / t for k, (f, t) in fields.items() if t},
        "parse_ok": sum(o.parse_ok for o in outputs.values()) / n,
        "loops": sum(o.looped for o in outputs.values()) / n,
        "truncated": sum(o.truncated for o in outputs.values()) / n,
        "tokens_per_page": sum(o.tokens for o in outputs.values()) / n,
        "seconds_per_page": secs / n,
        "page_rows": rows,
    }
    if tests_by_page:
        from .unittests import score
        md = {pid: o.markdown for pid, o in outputs.items()}
        sub = {pid: t for pid, t in tests_by_page.items() if pid in md}
        if sub:
            s = score(sub, md)
            m["tests"] = s["overall"]
            m["tests_by_category"] = {c: v["rate"] for c, v in s["categories"].items()}
    m["_outputs"] = {pid: {"markdown": o.markdown, "raw": o.raw[:20000], "looped": o.looped} for pid, o in outputs.items()}
    return m


def headline(m: Dict) -> str:
    parts = [f"pages {m['pages']}", f"CER {100 * m['cer']:.2f}%"]
    if m.get("tests") is not None:
        parts.append(f"tests {100 * m['tests']:.1f}")
    if m.get("fields") is not None:
        parts.append(f"fields {100 * m['fields']:.1f}%")
    parts += [f"parse_ok {100 * m['parse_ok']:.0f}%", f"loops {100 * m['loops']:.0f}%",
              f"{m['seconds_per_page']:.1f}s/page"]
    return " | ".join(parts)


def save_metrics(m: Dict, path: str) -> None:
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(m, f, indent=1, ensure_ascii=False)
