"""Contamination control: an index of every evaluation page, and a check of training data against it.

A training page is dropped when it matches an evaluation page by any of:

    image   perceptual hash of the page (256-bit DCT pHash), Hamming distance
            <= ``phash_bits`` (default 24 of 256; see ``calibrate``)
    text    MinHash over word 5-grams, estimated Jaccard >= 0.8 (the plan's rule),
            found through LSH (16 bands x 8 rows) and confirmed on the signatures
    tests   for benchmarks that only publish test strings (olmOCR-Bench, the
            holdout): the training text contains >= half (and >= 3) of the
            page's long test strings
    file    the same source file (SHA-256 of the PDF/image bytes) or source URL

Index once, check every source (ours, olmOCR-mix, Infinity-Doc2, ...), and
publish the counts with any claim.
"""
from __future__ import annotations

import glob
import hashlib
import json
import os
import re
import unicodedata
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Set, Tuple

import numpy as np
from PIL import Image

from ..textnorm import markdown_to_text

HASH_SIZE = 16                    # 16x16 low-frequency DCT -> 256 bits
NUM_PERM = 128
BANDS, ROWS = 16, 8
_MERSENNE = np.uint64((1 << 31) - 1)
_WORD = re.compile(r"\w+", re.U)


# ------------------------------------------------------------------- image
def phash(img: Image.Image, hash_size: int = HASH_SIZE) -> np.ndarray:
    """DCT perceptual hash as ``hash_size**2 / 64`` uint64 words."""
    import cv2
    g = np.asarray(img.convert("L").resize((hash_size * 4, hash_size * 4), Image.LANCZOS), np.float32)
    d = cv2.dct(g)[:hash_size, :hash_size].flatten()
    bits = d > np.median(d[1:])
    return np.packbits(bits).view(">u8").astype(np.uint64)


_POP8 = np.array([bin(i).count("1") for i in range(256)], np.uint8)


def hamming(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Bit distance between hash ``a`` (words,) and each row of ``b`` (n, words)."""
    x = np.bitwise_xor(b, a[None, :])
    if hasattr(np, "bitwise_count"):
        return np.bitwise_count(x).sum(1).astype(np.int32)
    return _POP8[x.view(np.uint8)].reshape(len(b), -1).sum(1).astype(np.int32)


def file_sha256(path_or_bytes) -> str:
    h = hashlib.sha256()
    if isinstance(path_or_bytes, (bytes, bytearray)):
        h.update(path_or_bytes)
    else:
        with open(path_or_bytes, "rb") as f:
            for chunk in iter(lambda: f.read(1 << 20), b""):
                h.update(chunk)
    return h.hexdigest()


# -------------------------------------------------------------------- text
def norm_words(text: str) -> List[str]:
    t = unicodedata.normalize("NFKC", text or "").lower()
    return _WORD.findall(t)


def shingles(text: str, n: int = 5) -> Set[int]:
    w = norm_words(text)
    if len(w) < n:
        return {_h32(" ".join(w))} if w else set()
    return {_h32(" ".join(w[i:i + n])) for i in range(len(w) - n + 1)}


def _h32(s: str) -> int:
    return int.from_bytes(hashlib.blake2b(s.encode("utf-8"), digest_size=4).digest(), "little")


class MinHasher:
    def __init__(self, num_perm: int = NUM_PERM, seed: int = 1):
        rng = np.random.default_rng(seed)
        self.a = rng.integers(1, int(_MERSENNE), num_perm, dtype=np.uint64)
        self.b = rng.integers(0, int(_MERSENNE), num_perm, dtype=np.uint64)

    def signature(self, sh: Set[int]) -> Optional[np.ndarray]:
        if not sh:
            return None
        x = np.fromiter(sh, np.uint64, len(sh)) % _MERSENNE
        v = (x[:, None] * self.a[None, :] + self.b[None, :]) % _MERSENNE
        return v.min(0).astype(np.uint32)


def jaccard_est(a: np.ndarray, b: np.ndarray) -> float:
    return float((a == b).mean())


def _bands(sig: np.ndarray) -> List[bytes]:
    return [sig[i * ROWS:(i + 1) * ROWS].tobytes() for i in range(BANDS)]


# ------------------------------------------------------------------- index
@dataclass
class EvalPage:
    key: str                      # "<set>:<page id>"
    image: Optional[str] = None
    text: str = ""
    tests: List[str] = field(default_factory=list)
    sha256: List[str] = field(default_factory=list)
    url: Optional[str] = None


def _test_strings(t: dict) -> List[str]:
    out = []
    for k in ("text", "before", "after", "cell"):
        v = t.get(k)
        if isinstance(v, str) and len(v) >= 20 and t.get("type") != "absent":
            out.append(v)
    return out


def pages_olmocr_bench(bench_data: str, set_name: str = "olmocr-bench", render_side: int = 1288) -> List[EvalPage]:
    """olmOCR-Bench layout (also the holdout): ``<cat>.jsonl`` tests + ``pdfs/<cat>/*.pdf``."""
    by_pdf: Dict[str, List[dict]] = defaultdict(list)
    for jl in sorted(glob.glob(os.path.join(bench_data, "*.jsonl"))):
        with open(jl, encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    t = json.loads(line)
                    by_pdf[t["pdf"]].append(t)
    pages = []
    for rel, tests in sorted(by_pdf.items()):
        path = os.path.join(bench_data, "pdfs", rel)
        text = ""
        if os.path.exists(path):
            try:
                import pypdfium2 as pdfium
                pdf = pdfium.PdfDocument(path)
                pg = pdf[int(tests[0].get("page", 1)) - 1]
                text = pg.get_textpage().get_text_range()
                pdf.close()
            except Exception:
                text = ""
        strs = sorted({s for t in tests for s in _test_strings(t)})
        pages.append(EvalPage(key=f"{set_name}:{rel}", image=path if os.path.exists(path) else None, text=text,
                              tests=strs, sha256=[file_sha256(path)] if os.path.exists(path) else []))
    return pages


def pages_omnidoc(root: str, set_name: str = "omnidocbench") -> List[EvalPage]:
    js = [p for p in glob.glob(os.path.join(root, "*.json")) if "omnidoc" in os.path.basename(p).lower()]
    if not js:
        raise FileNotFoundError(f"no OmniDocBench*.json in {root}")
    with open(js[0], encoding="utf-8") as f:
        data = json.load(f)
    img_dir = os.path.join(root, "images")
    pages = []
    for p in data:
        info = p.get("page_info", {})
        name = os.path.basename(info.get("image_path", ""))
        parts = []
        for d in p.get("layout_dets", []):
            for k in ("text", "html", "latex"):
                if isinstance(d.get(k), str):
                    parts.append(d[k])
        path = os.path.join(img_dir, name)
        pages.append(EvalPage(key=f"{set_name}:{name}", image=path if os.path.exists(path) else None,
                              text=markdown_to_text("\n".join(parts)),
                              sha256=[file_sha256(path)] if os.path.exists(path) else []))
    return pages


def pages_resume_bench(bench_dir: str, set_name: str = "resume-bench") -> List[EvalPage]:
    out = []
    with open(os.path.join(bench_dir, "pages.jsonl"), encoding="utf-8") as f:
        for line in f:
            if line.strip():
                p = json.loads(line)
                img = p["image"] if os.path.exists(p["image"]) else None
                out.append(EvalPage(key=f"{set_name}:{p['id']}", image=img,
                                    text="\n".join(l["text"] for l in p["lines"]),
                                    sha256=[file_sha256(img)] if img else []))
    return out


def pages_labels(labels: str, set_name: str) -> List[EvalPage]:
    """Any Folio label file used for evaluation (e.g. a frozen dev set)."""
    from .dataset import read_labels
    return [EvalPage(key=f"{set_name}:{r['id']}", image=r["_image"], text=record_text(r),
                     tests=[s for t in r.get("tests", []) for s in _test_strings(t)])
            for r in read_labels(labels)]


def record_text(r: dict) -> str:
    if "blocks" in r:
        return markdown_to_text("\n\n".join(b.get("text", "") for b in r["blocks"]))
    return markdown_to_text(r.get("markdown") or "")


class EvalIndex:
    def __init__(self, phash_bits: int = 24, jaccard: float = 0.8, test_share: float = 0.5, min_tests: int = 3):
        self.keys: List[str] = []
        self.hashes: List[np.ndarray] = []
        self.hash_keys: List[str] = []
        self.sigs: Dict[str, np.ndarray] = {}
        self.lsh: List[Dict[bytes, List[str]]] = [defaultdict(list) for _ in range(BANDS)]
        self.test_shingles: Dict[str, List[Set[int]]] = {}
        self.sha: Dict[str, str] = {}
        self.urls: Dict[str, str] = {}
        self.mh = MinHasher()
        self.phash_bits, self.jaccard, self.test_share, self.min_tests = phash_bits, jaccard, test_share, min_tests
        self._sh_index: Dict[int, List[Tuple[str, int]]] = defaultdict(list)
        self._hash_mat: Optional[np.ndarray] = None

    # ------------------------------------------------------------ build
    def add(self, p: EvalPage) -> None:
        self.keys.append(p.key)
        if p.image:
            try:
                self.hashes.append(phash(_load_image(p.image)))
                self.hash_keys.append(p.key)
            except Exception:
                pass
        sig = self.mh.signature(shingles(p.text)) if len(norm_words(p.text)) >= 20 else None
        if sig is not None:
            self.sigs[p.key] = sig
            for i, b in enumerate(_bands(sig)):
                self.lsh[i][b].append(p.key)
        if p.tests:
            # whole 5-grams only: a shorter string could never be found among a page's 5-grams
            sh = [shingles(s) for s in p.tests if len(norm_words(s)) >= 5]
            self.test_shingles[p.key] = sh
            for si, s in enumerate(sh):
                for h in s:
                    self._sh_index[h].append((p.key, si))
        for h in p.sha256:
            self.sha[h] = p.key
        if p.url:
            self.urls[p.url] = p.key
        self._hash_mat = None

    def summary(self) -> dict:
        sets = defaultdict(int)
        for k in self.keys:
            sets[k.split(":", 1)[0]] += 1
        return {"pages": len(self.keys), "by_set": dict(sets), "image_hashes": len(self.hashes),
                "text_signatures": len(self.sigs), "pages_with_test_strings": len(self.test_shingles),
                "file_hashes": len(self.sha), "urls": len(self.urls)}

    # ------------------------------------------------------------ query
    def match(self, image: Optional[Image.Image] = None, text: str = "", sha256: Optional[str] = None,
              url: Optional[str] = None) -> List[dict]:
        hits = []
        if sha256 and sha256 in self.sha:
            hits.append({"by": "file", "eval": self.sha[sha256]})
        if url and url in self.urls:
            hits.append({"by": "url", "eval": self.urls[url]})
        if image is not None and self.hashes:
            if self._hash_mat is None:
                self._hash_mat = np.stack(self.hashes)
            d = hamming(phash(image), self._hash_mat)
            j = int(d.argmin())
            if d[j] <= self.phash_bits:
                hits.append({"by": "image", "eval": self.hash_keys[j], "bits": int(d[j])})
        sh = shingles(text) if text else set()
        if sh:
            sig = self.mh.signature(sh)
            cands = set()
            for i, b in enumerate(_bands(sig)):
                cands.update(self.lsh[i].get(b, ()))
            for k in cands:
                jac = jaccard_est(sig, self.sigs[k])
                if jac >= self.jaccard:
                    hits.append({"by": "text", "eval": k, "jaccard": round(jac, 3)})
            counts: Dict[Tuple[str, int], int] = defaultdict(int)
            for h in sh:
                for ks in self._sh_index.get(h, ()):
                    counts[ks] += 1
            found: Dict[str, int] = defaultdict(int)
            for (k, si), c in counts.items():
                if c >= len(self.test_shingles[k][si]):
                    found[k] += 1
            for k, n in found.items():
                tot = len(self.test_shingles[k])
                if n >= self.min_tests and n >= self.test_share * tot:
                    hits.append({"by": "tests", "eval": k, "strings": f"{n}/{tot}"})
        return hits

    # ------------------------------------------------------------- save
    def save(self, path: str) -> None:
        test_lists = {k: [sorted(s) for s in v] for k, v in self.test_shingles.items()}
        meta = {"keys": self.keys, "hash_keys": self.hash_keys, "sha": self.sha, "urls": self.urls,
                "tests": test_lists, "params": {"phash_bits": self.phash_bits, "jaccard": self.jaccard,
                                                "test_share": self.test_share, "min_tests": self.min_tests}}
        sig_keys = sorted(self.sigs)
        np.savez_compressed(path, hashes=np.stack(self.hashes) if self.hashes else np.zeros((0, HASH_SIZE ** 2 // 64), np.uint64),
                            sigs=np.stack([self.sigs[k] for k in sig_keys]) if sig_keys else np.zeros((0, NUM_PERM), np.uint32),
                            meta=np.frombuffer(json.dumps({**meta, "sig_keys": sig_keys}).encode("utf-8"), np.uint8))

    @classmethod
    def load(cls, path: str, **overrides) -> "EvalIndex":
        z = np.load(path)
        meta = json.loads(bytes(z["meta"]).decode("utf-8"))
        params = {**meta["params"], **{k: v for k, v in overrides.items() if v is not None}}
        idx = cls(**params)
        idx.keys, idx.hash_keys, idx.sha, idx.urls = meta["keys"], meta["hash_keys"], meta["sha"], meta["urls"]
        idx.hashes = list(z["hashes"])
        for k, s in zip(meta["sig_keys"], z["sigs"]):
            idx.sigs[k] = s
            for i, b in enumerate(_bands(s)):
                idx.lsh[i][b].append(k)
        for k, lists in meta["tests"].items():
            idx.test_shingles[k] = [set(x) for x in lists]
            for si, s in enumerate(idx.test_shingles[k]):
                for h in s:
                    idx._sh_index[h].append((k, si))
        return idx


def _load_image(path: str) -> Image.Image:
    if path.lower().endswith(".pdf"):
        from ..imaging import render_pdf
        return render_pdf(path, longest_side=1024, pages=[1])[0]
    return Image.open(path).convert("RGB")


# ------------------------------------------------------------------- check
def check_labels(index: EvalIndex, labels: str, write_clean: bool = False, max_listed: int = 200) -> dict:
    """Match every record of a label file; optionally write ``labels.clean.jsonl`` beside it."""
    from .dataset import read_labels
    recs = read_labels(labels)
    by_signal: Dict[str, int] = defaultdict(int)
    by_set: Dict[str, int] = defaultdict(int)
    listed, dropped = [], set()
    for r in recs:
        meta = r.get("meta") or {}
        try:
            img = Image.open(r["_image"]).convert("RGB")
        except Exception:
            img = None
        hits = index.match(img, record_text(r), meta.get("sha256"), meta.get("url"))
        if hits:
            dropped.add(r["id"])
            for h in hits:
                by_signal[h["by"]] += 1
            for s in {h["eval"].split(":", 1)[0] for h in hits}:
                by_set[s] += 1
            if len(listed) < max_listed:
                listed.append({"id": r["id"], "hits": hits})
    report = {"labels": os.path.abspath(labels), "records": len(recs), "dropped": len(dropped),
              "by_signal": dict(by_signal), "by_eval_set": dict(by_set), "examples": listed,
              "params": {"phash_bits": index.phash_bits, "jaccard": index.jaccard,
                         "test_share": index.test_share, "min_tests": index.min_tests}}
    if write_clean:
        src = os.path.abspath(labels)
        dst = os.path.join(os.path.dirname(src), "labels.clean.jsonl")
        with open(src, encoding="utf-8") as f, open(dst, "w", encoding="utf-8") as g:
            for line in f:
                if line.strip() and json.loads(line)["id"] not in dropped:
                    g.write(line if line.endswith("\n") else line + "\n")
        report["clean"] = dst
    return report


def calibrate(images: Sequence[str], n: int = 60, seed: int = 0) -> dict:
    """pHash distances: the same page after scan/photo degradation vs different pages.

    Use it to set ``phash_bits`` for your data: matches should sit well below
    the nearest *different* page."""
    from ..render.degrade import degrade
    rng = np.random.default_rng(seed)
    paths = list(images)[:n]
    ims = [Image.open(p).convert("RGB") for p in paths]
    hs = np.stack([phash(im) for im in ims])
    same, other = [], []
    for i, im in enumerate(ims):
        for cond in ("scan", "photo", "old"):
            arr, _, q, _ = degrade(np.asarray(im), {"blocks": np.zeros((0, 4), np.float32),
                                                    "lines": np.zeros((0, 4), np.float32)}, cond, rng, 150)
            import io
            buf = io.BytesIO()
            Image.fromarray(arr).save(buf, "JPEG", quality=q)
            same.append(int(hamming(phash(Image.open(buf)), hs[i:i + 1])[0]))
        d = hamming(hs[i], np.delete(hs, i, 0))
        other.append(int(d.min()))
    return {"same_page_degraded": {"median": float(np.median(same)), "p95": float(np.percentile(same, 95)),
                                   "max": int(max(same))},
            "nearest_other_page": {"min": int(min(other)), "p5": float(np.percentile(other, 5)),
                                   "median": float(np.median(other))}, "pages": len(ims)}
