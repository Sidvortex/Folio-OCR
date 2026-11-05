"""Training mixtures: which label files, in what shares, under which licences.

A mixture file (``mixtures/*.json``):

    {"name": "free-permissive-v1",
     "task_mix": "doc2json=0.6,doc2md=0.4",
     "sources": [{"labels": "data/general/labels.clean.jsonl", "weight": 0.2, "role": "our renders"}, ...]}

Weights are shares of training samples (they are normalised). Every label
file's folder should hold the ``source.json`` its adapter wrote; ``licences``
reads them, and ``train_sft.py`` refuses a mixture with a non-commercial
source unless told otherwise. ``data_card`` writes the card the plan asks for:
one line per source with its licence, plus the contamination counts.
"""
from __future__ import annotations

import datetime as _dt
import json
import os
from dataclasses import dataclass
from typing import Dict, List, Optional


@dataclass
class Source:
    labels: str
    weight: float = 1.0
    role: str = ""


@dataclass
class Mixture:
    name: str
    sources: List[Source]
    task_mix: Optional[str] = None
    description: str = ""
    path: Optional[str] = None

    def specs(self) -> List[str]:
        return [f"{s.labels}:{s.weight}" for s in self.sources]


def load_mixture(path: str) -> Mixture:
    with open(path, encoding="utf-8") as f:
        d = json.load(f)
    srcs = [Source(labels=s["labels"], weight=float(s.get("weight", 1.0)), role=s.get("role", ""))
            for s in d["sources"]]
    if not srcs:
        raise ValueError(f"{path}: no sources")
    return Mixture(name=d.get("name", os.path.splitext(os.path.basename(path))[0]), sources=srcs,
                   task_mix=d.get("task_mix"), description=d.get("description", ""), path=path)


def split_spec(spec: str):
    if os.path.exists(spec):
        return spec, 1.0
    path, _, w = spec.rpartition(":")
    return (path, float(w)) if path and w else (spec, 1.0)


def source_info(labels: str) -> dict:
    d = os.path.dirname(os.path.abspath(labels))
    p = os.path.join(d, "source.json")
    if os.path.exists(p):
        with open(p, encoding="utf-8") as f:
            return json.load(f)
    return {"source": "unknown", "licence": "unknown", "commercial": "check",
            "terms": "no source.json next to this label file: write its licence line before training on it"}


def count_split(labels: str) -> Dict[str, int]:
    n = {"train": 0, "dev": 0}
    if not os.path.exists(labels):
        return n
    with open(labels, encoding="utf-8") as f:
        for line in f:
            if line.strip():
                s = json.loads(line).get("split", "train")
                n[s] = n.get(s, 0) + 1
    return n


def licences(specs: List[str]) -> dict:
    rows = []
    for spec in specs:
        path, w = split_spec(spec)
        info = source_info(path)
        rows.append({"labels": path, "weight": w, "source": info.get("source"), "licence": info.get("licence"),
                     "commercial": info.get("commercial", "check"), "terms": info.get("terms", ""),
                     "notes": info.get("notes", ""), "url": info.get("url", "")})
    return {"sources": rows,
            "noncommercial": [r["labels"] for r in rows if r["commercial"] == "no"],
            "unknown": [r["labels"] for r in rows if r["source"] == "unknown"]}


def data_card(specs: List[str], name: str = "", contamination: Optional[str] = None, task_mix: str = "",
              roles: Optional[Dict[str, str]] = None) -> str:
    lic = licences(specs)
    tot = sum(r["weight"] for r in lic["sources"]) or 1.0
    cont = {}
    if contamination and os.path.exists(contamination):
        with open(contamination, encoding="utf-8") as f:
            rep = json.load(f)
        for s in rep.get("sources", []):
            cont[os.path.abspath(s["labels"])] = s
    lines = [f"# Data card{': ' + name if name else ''}", "",
             f"Written {_dt.date.today().isoformat()} from the label files' `source.json` lines."
             + (f" Task mix: `{task_mix}`." if task_mix else ""), ""]
    if lic["noncommercial"]:
        lines += ["**Non-commercial:** this mixture includes a source whose licence forbids commercial use. "
                  "A model trained on it must be treated as non-commercial (and share-alike where the licence says so).",
                  ""]
    if lic["unknown"]:
        lines += ["**Unknown licence:** " + ", ".join(f"`{u}`" for u in lic["unknown"])
                  + " (no `source.json`).", ""]
    lines += ["| Source | Share | Pages (train / dev) | Licence | Commercial use | Contamination dropped |",
              "|---|---|---|---|---|---|"]
    for r in lic["sources"]:
        n = count_split(r["labels"])
        c = cont.get(os.path.abspath(r["labels"]))
        cd = f"{c['dropped']} of {c['records']}" if c else "not checked"
        role = (roles or {}).get(r["labels"], "")
        label = f"`{r['labels']}`" + (f" ({role})" if role else "")
        lines.append(f"| {label} | {100 * r['weight'] / tot:.0f}% | {n.get('train', 0)} / {n.get('dev', 0)} | "
                     f"{r['licence']} | {r['commercial']} | {cd} |")
    lines += ["", "Terms and provenance:", ""]
    seen = set()
    for r in lic["sources"]:
        if r["source"] in seen:
            continue
        seen.add(r["source"])
        extra = " ".join(x for x in (r["terms"], r["notes"]) if x)
        lines.append(f"- **{r['source']}** ({r['url'] or 'no url'}): {extra}")
    if cont:
        lines += ["", "Contamination check: perceptual hash, MinHash over word 5-grams (Jaccard >= 0.8), "
                      "benchmark test strings, file hashes and source URLs against every evaluation page "
                      f"(`{contamination}`)."]
    lines += ["", "\"Commercial use\" is a reading of each licence, not legal advice: \"check\" means a lawyer "
                  "should look before any commercial use."]
    return "\n".join(lines) + "\n"
