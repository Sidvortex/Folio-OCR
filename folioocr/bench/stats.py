"""Paired bootstrap for "does A beat B?" on unit-test benchmarks.

Both systems are scored on the same tests. The statistic is olmOCR-Bench's:
the mean over categories of per-category pass rates. Documents are resampled
with replacement *within each category* (all tests of a document move
together), the statistic is recomputed for A and B on the same resample, and
the 95% interval of A - B is read off the bootstrap distribution. A claim of
"beats" needs that interval to exclude zero.
"""
from __future__ import annotations

import json
from collections import defaultdict
from typing import Dict, List, Tuple

import numpy as np


def _by_cat_doc(results: List[dict]) -> Dict[str, Dict[str, List[Tuple[str, bool]]]]:
    out: Dict[str, Dict[str, List[Tuple[str, bool]]]] = defaultdict(lambda: defaultdict(list))
    for r in results:
        out[r["category"]][r["page"]].append((r["id"], bool(r["passed"])))
    return out


def category_mean(results: List[dict]) -> float:
    per = defaultdict(list)
    for r in results:
        per[r["category"]].append(bool(r["passed"]))
    return float(np.mean([np.mean(v) for v in per.values()])) if per else float("nan")


def paired_bootstrap(a: List[dict], b: List[dict], iters: int = 10000, seed: int = 0) -> Dict:
    """``a``/``b``: per-test results ({id, page, category, passed}) on the same tests."""
    bi = {r["id"]: bool(r["passed"]) for r in b}
    common = [r for r in a if r["id"] in bi]
    if not common:
        raise ValueError("no common tests")
    grid = _by_cat_doc(common)
    rng = np.random.default_rng(seed)
    cats = sorted(grid)
    # per category: arrays of (passed_a, passed_b, n) per document
    arrays = {}
    for c in cats:
        docs = sorted(grid[c])
        pa = np.array([sum(p for _, p in grid[c][d]) for d in docs], float)
        pb = np.array([sum(bi[t] for t, _ in grid[c][d]) for d in docs], float)
        n = np.array([len(grid[c][d]) for d in docs], float)
        arrays[c] = (pa, pb, n)
    point_a = np.mean([arrays[c][0].sum() / arrays[c][2].sum() for c in cats])
    point_b = np.mean([arrays[c][1].sum() / arrays[c][2].sum() for c in cats])
    diffs = np.empty(iters)
    for k in range(iters):
        ma, mb = [], []
        for c in cats:
            pa, pb, n = arrays[c]
            idx = rng.integers(0, len(n), len(n))
            ma.append(pa[idx].sum() / max(1.0, n[idx].sum()))
            mb.append(pb[idx].sum() / max(1.0, n[idx].sum()))
        diffs[k] = np.mean(ma) - np.mean(mb)
    lo, hi = np.percentile(diffs, [2.5, 97.5])
    return {"a": float(point_a), "b": float(point_b), "diff": float(point_a - point_b), "ci95": [float(lo), float(hi)],
            "p_a_not_better": float((diffs <= 0).mean()), "significant": bool(lo > 0 or hi < 0),
            "n_tests": len(common), "n_categories": len(cats), "iters": iters}


def paired_bootstrap_pages(rows_a: List[dict], rows_b: List[dict], metric: str, iters: int = 10000,
                           seed: int = 0, stratum: str = "tier") -> Dict:
    """Paired bootstrap on page-level ratio metrics (Resume-Bench page rows).

    ``metric``: ``"line_cer"`` (sum of errors / sum of GT characters; lower is
    better) or ``"fields"`` (fields found / fields in GT; higher is better).
    Pages are resampled within each ``stratum`` (tier), the same pages for A and
    B, and the ratio of sums is recomputed for both. ``a_better`` says whether A
    wins in that metric's direction; ``significant`` whether the 95% interval of
    A - B excludes zero.
    """
    def num_den(r):
        if metric == "line_cer":
            return float(r["err"]), float(r["chars"])
        if metric == "fields":
            return float(r["fields"][0]), float(r["fields"][1])
        raise ValueError(metric)

    if metric == "line_cer" and not all("err" in r for r in rows_a + rows_b):
        raise ValueError("result files predate per-page error counts; re-run `scripts/bench_resume.py score`")
    bmap = {r["id"]: r for r in rows_b}
    ids = [r["id"] for r in rows_a if r["id"] in bmap]
    if not ids:
        raise ValueError("no common pages")
    amap = {r["id"]: r for r in rows_a}
    strata = defaultdict(list)
    for i in ids:
        strata[amap[i].get(stratum, "all")].append(i)
    arrs = []
    for s in sorted(strata):
        na, da = zip(*(num_den(amap[i]) for i in strata[s]))
        nb, db = zip(*(num_den(bmap[i]) for i in strata[s]))
        arrs.append((np.array(na), np.array(da), np.array(nb), np.array(db)))

    def stat(idx_list):
        na = sum(a[0][ix].sum() for a, ix in zip(arrs, idx_list))
        da = sum(a[1][ix].sum() for a, ix in zip(arrs, idx_list))
        nb = sum(a[2][ix].sum() for a, ix in zip(arrs, idx_list))
        db = sum(a[3][ix].sum() for a, ix in zip(arrs, idx_list))
        return na / max(da, 1e-9), nb / max(db, 1e-9)

    full = [np.arange(len(a[0])) for a in arrs]
    pa, pb = stat(full)
    rng = np.random.default_rng(seed)
    diffs = np.empty(iters)
    for k in range(iters):
        sa, sb = stat([rng.integers(0, len(a[0]), len(a[0])) for a in arrs])
        diffs[k] = sa - sb
    lo, hi = np.percentile(diffs, [2.5, 97.5])
    lower_better = metric == "line_cer"
    return {"metric": metric, "a": float(pa), "b": float(pb), "diff": float(pa - pb), "ci95": [float(lo), float(hi)],
            "lower_is_better": lower_better, "a_better": bool(pa < pb if lower_better else pa > pb),
            "significant": bool(lo > 0 or hi < 0), "n_pages": len(ids), "iters": iters}


def bootstrap_ci(results: List[dict], iters: int = 2000, seed: int = 0) -> Tuple[float, float]:
    """95% interval of one system's category-mean score."""
    grid = _by_cat_doc(results)
    rng = np.random.default_rng(seed)
    cats = sorted(grid)
    arr = {c: (np.array([sum(p for _, p in grid[c][d]) for d in grid[c]], float),
               np.array([len(grid[c][d]) for d in grid[c]], float)) for c in cats}
    vals = []
    for _ in range(iters):
        m = []
        for c in cats:
            p, n = arr[c]
            idx = rng.integers(0, len(n), len(n))
            m.append(p[idx].sum() / max(1.0, n[idx].sum()))
        vals.append(np.mean(m))
    lo, hi = np.percentile(vals, [2.5, 97.5])
    return float(lo), float(hi)


def load_results(path: str) -> List[dict]:
    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    return data["results"] if isinstance(data, dict) else data
