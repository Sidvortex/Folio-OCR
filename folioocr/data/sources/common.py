"""Shared pieces of the public-dataset adapters.

Every adapter writes the same thing: a folder with ``images/``, a
``labels.jsonl`` in Folio's label format (``blocks`` for pages with layout,
``markdown`` for pages with text only, which train ``doc2md``) and a
``source.json`` holding the source's licence line for the data card. Writers
append: re-running an adapter skips pages it already wrote, so long
conversions survive a 12-hour session limit.
"""
from __future__ import annotations

import datetime as _dt
import hashlib
import json
import os
import re
from typing import Dict, Iterable, List, Optional

from PIL import Image

# One line per source for the data card. "commercial" is a reading of the
# licence, not legal advice: "no" means the licence forbids it, "check" means
# a lawyer should look before any commercial use (the plan's rule).
LICENCES: Dict[str, Dict[str, str]] = {
    "folio-render": {
        "licence": "ours (generated)", "commercial": "yes",
        "terms": "pages generated from HTML we write; fonts OFL/Apache/UFL, KaTeX MIT",
        "url": "folioocr/render"},
    "olmocr-mix-1025": {
        "licence": "ODC-BY 1.0", "commercial": "check",
        "terms": "Ai2 Responsible Use Guidelines; the card says research and educational use",
        "notes": "00_documents and 01_books were transcribed by GPT-4.1; 02_loc_transcripts and "
                 "03_national_archives are human transcriptions cleaned up with ChatGPT",
        "url": "https://huggingface.co/datasets/allenai/olmOCR-mix-1025"},
    "olmocr-synthmix-1025": {
        "licence": "ODC-BY 1.0", "commercial": "check",
        "terms": "Ai2 Responsible Use Guidelines; the card says research and educational use",
        "notes": "pages re-rendered from HTML written by claude-sonnet-4; tests in olmOCR-Bench format",
        "url": "https://huggingface.co/datasets/allenai/olmOCR-synthmix-1025"},
    "infinity-doc2-5m": {
        "licence": "CC BY-NC-SA 4.0", "commercial": "no",
        "terms": "gated: academic research and educational purposes only; a model trained on it should be "
                 "treated as non-commercial and share-alike",
        "url": "https://huggingface.co/datasets/infly/Infinity-Doc2-5M"},
    "lightonocr-mix-0126": {
        "licence": "other", "commercial": "check",
        "terms": "PDFA pages: Common Crawl terms of use and the Digital Corpora terms; transcriptions distilled "
                 "from a teacher model",
        "url": "https://huggingface.co/datasets/lightonai/LightOnOCR-mix-0126"},
    "teacher-consensus": {
        "licence": "per page source", "commercial": "check",
        "terms": "labels agreed by two permissively licensed teachers; the page images keep their source's terms",
        "url": "label_teachers.py"},
}


def split_for(key: str, dev_frac: float) -> str:
    """Stable train/dev assignment from the record id."""
    h = int(hashlib.sha1(key.encode("utf-8")).hexdigest()[:8], 16)
    return "dev" if (h % 10000) < dev_frac * 10000 else "train"


def safe_name(key: str) -> str:
    return re.sub(r"[^A-Za-z0-9._-]+", "_", key).strip("_")[:180] or "page"


# ----------------------------------------------------------------- markdown
_FRONT = re.compile(r"\A\s*---\s*\n.*?\n---\s*\n", re.S)
_DISPLAY = re.compile(r"\\\[(.+?)\\\]", re.S)
_INLINE = re.compile(r"\\\((.+?)\\\)", re.S)
_MD_IMG_LINE = re.compile(r"^[ \t]*!\[[^\]]*\]\([^)]*\)[ \t]*$\n?", re.M)
_MD_IMG = re.compile(r"!\[[^\]]*\]\([^)]*\)")
_SEP_ROW = re.compile(r"^\s*\|?\s*:?-{2,}:?\s*(\|\s*:?-{2,}:?\s*)*\|?\s*$")


def _cells(line: str) -> List[str]:
    s = line.strip()
    if s.startswith("|"):
        s = s[1:]
    if s.endswith("|"):
        s = s[:-1]
    return [c.strip() for c in s.split("|")]


def pipe_tables_to_html(md: str) -> str:
    """Markdown pipe tables -> plain HTML tables (the format Infinity-Parser2 writes)."""
    from html import escape
    lines = md.split("\n")
    out, i = [], 0
    while i < len(lines):
        if (i + 1 < len(lines) and lines[i].strip().startswith("|") and _SEP_ROW.match(lines[i + 1])
                and "|" in lines[i + 1]):
            head = _cells(lines[i])
            rows = []
            j = i + 2
            while j < len(lines) and lines[j].strip().startswith("|"):
                rows.append(_cells(lines[j]))
                j += 1
            html = "<table><tr>" + "".join(f"<th>{escape(c, quote=False)}</th>" for c in head) + "</tr>"
            for r in rows:
                html += "<tr>" + "".join(f"<td>{escape(c, quote=False)}</td>" for c in r) + "</tr>"
            out.append(html + "</table>")
            i = j
        else:
            out.append(lines[i])
            i += 1
    return "\n".join(out)


def normalize_markdown(md: str) -> str:
    """Bring a dataset's Markdown to Infinity-Parser2's doc2md conventions:
    no front matter, ``$..$`` / ``$$..$$`` math, HTML tables, no figure links."""
    if not md:
        return ""
    t = _FRONT.sub("", md.replace("\r\n", "\n"))
    t = _DISPLAY.sub(lambda m: "$$" + m.group(1).strip() + "$$", t)
    t = _INLINE.sub(lambda m: "$" + m.group(1).strip() + "$", t)
    t = _MD_IMG_LINE.sub("", t)
    t = _MD_IMG.sub("", t)
    t = pipe_tables_to_html(t)
    t = re.sub(r"[ \t]+\n", "\n", t)
    t = re.sub(r"\n{3,}", "\n\n", t)
    return t.strip()


# ------------------------------------------------------------------- writer
class LabelWriter:
    """Appends records + images to ``out/labels.jsonl`` and ``out/images/``."""

    def __init__(self, out: str, source: str, longest_side: Optional[int] = 1600, quality: int = 90,
                 dev_frac: float = 0.01, args: Optional[dict] = None):
        if source not in LICENCES:
            raise ValueError(f"unknown source {source!r}: add its licence line to LICENCES first")
        self.out, self.source = out, source
        self.longest_side, self.quality, self.dev_frac = longest_side, quality, dev_frac
        self.args = args or {}
        os.makedirs(os.path.join(out, "images"), exist_ok=True)
        self.labels = os.path.join(out, "labels.jsonl")
        self.done = set()
        if os.path.exists(self.labels):
            with open(self.labels, encoding="utf-8") as f:
                for line in f:
                    if line.strip():
                        self.done.add(json.loads(line)["id"])
        self.f = open(self.labels, "a", encoding="utf-8")
        self.n_new = 0
        self.skipped: Dict[str, int] = {}

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()

    def skip(self, why: str) -> None:
        self.skipped[why] = self.skipped.get(why, 0) + 1

    def has(self, rid: str) -> bool:
        return rid in self.done

    def add(self, rid: str, image: Image.Image, markdown: Optional[str] = None, blocks: Optional[List[dict]] = None,
            meta: Optional[dict] = None, split: Optional[str] = None, extra: Optional[dict] = None) -> Optional[dict]:
        if rid in self.done:
            return None
        if markdown is not None and blocks is not None:
            raise ValueError("give markdown or blocks, not both")
        if markdown is None and blocks is None and not (extra and extra.get("tests")):
            raise ValueError("a record needs a label (markdown or blocks) or, for RL, tests")
        img = image.convert("RGB")
        if self.longest_side and max(img.size) > self.longest_side:
            s = self.longest_side / max(img.size)
            img = img.resize((max(1, round(img.width * s)), max(1, round(img.height * s))), Image.LANCZOS)
        rel = f"images/{safe_name(rid)}.jpg"
        img.save(os.path.join(self.out, rel), quality=self.quality)
        rec = {"id": rid, "image": rel, "width": img.width, "height": img.height, "source": self.source,
               "split": split or split_for(rid, self.dev_frac)}
        if blocks is not None:
            rec["blocks"] = blocks
        elif markdown is not None:
            rec["markdown"] = markdown
        if extra:
            rec.update(extra)
        if meta:
            rec["meta"] = meta
        self.f.write(json.dumps(rec, ensure_ascii=False) + "\n")
        self.done.add(rid)
        self.n_new += 1
        return rec

    def close(self) -> None:
        if self.f.closed:
            return
        self.f.close()
        info_path = os.path.join(self.out, "source.json")
        prev = {}
        if os.path.exists(info_path):
            with open(info_path, encoding="utf-8") as f:
                prev = json.load(f)
        skipped = dict(prev.get("skipped", {}))
        for k, v in self.skipped.items():
            skipped[k] = skipped.get(k, 0) + v
        info = {"source": self.source, **LICENCES[self.source], "records": len(self.done), "skipped": skipped,
                "updated": _dt.date.today().isoformat(), "args": self.args}
        with open(info_path, "w", encoding="utf-8") as f:
            json.dump(info, f, indent=1, ensure_ascii=False)


def iter_tar_members(paths: Iterable[str], suffix: str = ".pdf"):
    """(member name, bytes) for files in .tar / .tar.gz archives, streamed."""
    import tarfile
    for p in paths:
        with tarfile.open(p, "r:*") as tar:
            for m in tar:
                if m.isfile() and m.name.endswith(suffix):
                    f = tar.extractfile(m)
                    if f is not None:
                        yield m.name, f.read()
