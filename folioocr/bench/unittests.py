"""Score Markdown against olmOCR-Bench-format unit tests.

Uses olmOCR's own test classes (``olmocr.bench.tests``), so a pass here means
the same as a pass in olmOCR-Bench. Like the benchmark, a ``baseline`` test
(non-empty, not looping, no CJK/emoji) is added for every page, and the
overall score is the mean of the per-category pass rates.
"""
from __future__ import annotations

import json
from collections import defaultdict
from typing import Dict, Iterable, List

_CACHE: Dict[str, object] = {}


def _olmocr():
    try:
        from olmocr.bench import tests as T
    except ImportError as e:  # pragma: no cover
        raise ImportError("olmocr is required for unit-test scoring: pip install olmocr==0.4.27 fuzzysearch") from e
    return T


def category_of(test: dict) -> str:
    return "field" if "_field_" in test.get("id", "") else test["type"]


def load_tests_jsonl(path: str) -> Dict[str, List[dict]]:
    """page id (pdf name without .pdf) -> tests."""
    by_page: Dict[str, List[dict]] = defaultdict(list)
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                t = json.loads(line)
                by_page[t["pdf"][:-4] if t["pdf"].endswith(".pdf") else t["pdf"]].append(t)
    return dict(by_page)


_FIELDS: Dict[str, set] = {}


def clean_test(t: dict) -> dict:
    """Only the fields olmOCR's test class for this type accepts (its dataclasses reject
    unknown keys such as our ``_category`` or another dataset's extra columns)."""
    import dataclasses
    T = _olmocr()
    cls = {"present": T.TextPresenceTest, "absent": T.TextPresenceTest, "order": T.TextOrderTest,
           "table": T.TableTest, "math": T.MathTest, "baseline": T.BaselineTest}.get(t.get("type"))
    if cls is None:
        return dict(t)
    names = _FIELDS.get(cls.__name__)
    if names is None:
        names = _FIELDS[cls.__name__] = {f.name for f in dataclasses.fields(cls)}
    return {k: v for k, v in t.items() if k in names}


def _test_obj(t: dict):
    obj = _CACHE.get(t["id"])
    if obj is None:
        obj = _CACHE[t["id"]] = _olmocr().load_single_test(clean_test(t))
    return obj


def run_page(tests: Iterable[dict], md: str, page_id: str = "", baseline: bool = True) -> List[dict]:
    T = _olmocr()
    out = []
    for t in tests:
        try:
            ok, why = _test_obj(t).run(md or "")
        except Exception as e:  # a crashing test counts as a failure, like the benchmark
            ok, why = False, f"error: {e}"
        out.append({"id": t["id"], "page": page_id, "category": category_of(t), "passed": bool(ok), "why": why})
    if baseline:
        b = T.BaselineTest(id=f"{page_id}_baseline", pdf=f"{page_id}.pdf", page=1, type="baseline")
        ok, why = b.run(md or "")
        out.append({"id": b.id, "page": page_id, "category": "baseline", "passed": bool(ok), "why": why})
    return out


def summarize(results: List[dict]) -> Dict:
    per: Dict[str, List[bool]] = defaultdict(list)
    for r in results:
        per[r["category"]].append(r["passed"])
    cats = {c: {"passed": sum(v), "total": len(v), "rate": sum(v) / len(v)} for c, v in sorted(per.items())}
    overall = sum(c["rate"] for c in cats.values()) / len(cats) if cats else float("nan")
    return {"overall": overall, "n_tests": len(results), "categories": cats}


def score(tests_by_page: Dict[str, List[dict]], md_by_page: Dict[str, str], baseline: bool = True,
          missing_ok: bool = False) -> Dict:
    results = []
    for pid, tests in tests_by_page.items():
        if pid not in md_by_page and not missing_ok:
            raise KeyError(f"no output for page {pid}")
        results.extend(run_page(tests, md_by_page.get(pid, ""), pid, baseline))
    s = summarize(results)
    s["results"] = results
    return s


def format_summary(s: Dict, name: str = "") -> str:
    lines = [f"{name + ': ' if name else ''}overall {100 * s['overall']:.1f} over {s['n_tests']} tests"]
    for c, v in s["categories"].items():
        lines.append(f"    {c:9s} {100 * v['rate']:5.1f}%  ({v['passed']}/{v['total']})")
    return "\n".join(lines)
