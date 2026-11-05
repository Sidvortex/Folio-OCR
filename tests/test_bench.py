"""Scoring must agree with olmOCR-Bench and separate good from bad output."""
import json
import os
import random

import pytest

olmocr = pytest.importorskip("olmocr.bench.tests")

from folioocr.bench.stats import category_mean, paired_bootstrap, paired_bootstrap_pages  # noqa: E402
from folioocr.bench.unittests import run_page, score, summarize  # noqa: E402

PAGE_MD = ("# Ravada Siddharth\n\nravada.s@gmail.com | +91 98765 43210\n\n## Experience\n\n"
           "**Software Engineer**, Acme Corp (Jan 2021 – Present)\n\n- Built REST APIs with FastAPI\n- Cut latency by 30%\n\n"
           "<table><tr><th>Skill</th><th>Level</th></tr><tr><td>Python</td><td>Expert</td></tr></table>")
TESTS = [
    {"pdf": "p.pdf", "page": 1, "id": "p_present_0", "type": "present", "text": "Built REST APIs with FastAPI", "max_diffs": 1},
    {"pdf": "p.pdf", "page": 1, "id": "p_absent_0", "type": "absent", "text": "Confidential draft", "max_diffs": 0},
    {"pdf": "p.pdf", "page": 1, "id": "p_order_0", "type": "order", "before": "Software Engineer", "after": "Cut latency by 30%", "max_diffs": 0},
    {"pdf": "p.pdf", "page": 1, "id": "p_table_0", "type": "table", "cell": "Python", "right": "Expert", "max_diffs": 0},
    {"pdf": "p.pdf", "page": 1, "id": "p_field_email_0", "type": "present", "text": "ravada.s@gmail.com", "max_diffs": 0, "case_sensitive": False},
]


def test_unit_tests_pass_on_correct_markdown_and_fail_on_wrong():
    res = run_page(TESTS, PAGE_MD, "p")
    assert all(r["passed"] for r in res)
    bad = PAGE_MD.replace("FastAPI", "FlaskAPI").replace("Expert", "Novice").replace("ravada.s", "ravda.s")
    bad = "- Cut latency by 30%\n\n" + bad.replace("- Cut latency by 30%", "") + "\n\nConfidential draft"
    res = {r["id"]: r["passed"] for r in run_page(TESTS, bad, "p")}
    assert not any(res[t["id"]] for t in TESTS[1:])          # absent, order, table, field all fail
    assert res["p_baseline"]


def test_field_tests_have_their_own_category():
    s = summarize(run_page(TESTS, PAGE_MD, "p"))
    assert set(s["categories"]) == {"present", "absent", "order", "table", "field", "baseline"}
    assert s["overall"] == 1.0


def _fake_results(p_pass: float, n_docs: int = 60, seed: int = 0):
    rng = random.Random(seed)
    out = []
    for d in range(n_docs):
        for c in ("tables", "old_scans"):
            for k in range(5):
                out.append({"id": f"{c}_{d}_{k}", "page": f"{c}/{d}.pdf", "category": c, "passed": rng.random() < p_pass})
    return out


def test_paired_bootstrap_detects_a_real_gap_and_not_noise():
    a, b = _fake_results(0.85, seed=1), _fake_results(0.65, seed=2)
    r = paired_bootstrap(a, b, iters=2000)
    assert r["significant"] and r["diff"] > 0.1 and r["ci95"][0] > 0
    c = _fake_results(0.75, seed=3)
    d = [dict(x, passed=x["passed"] if random.Random(i).random() > 0.02 else not x["passed"]) for i, x in enumerate(c)]
    r = paired_bootstrap(c, d, iters=2000)
    assert not r["significant"]
    assert abs(category_mean(c) - r["a"]) < 1e-9


def test_page_metric_bootstrap_knows_lower_cer_is_better():
    rng = random.Random(0)
    good, bad = [], []
    for i in range(80):
        tier = ("digital", "scan", "photo")[i % 3]
        chars = rng.randint(800, 1500)
        good.append({"id": f"p{i}", "tier": tier, "err": int(chars * 0.01), "chars": chars, "fields": [9, 10]})
        bad.append({"id": f"p{i}", "tier": tier, "err": int(chars * 0.04), "chars": chars, "fields": [6, 10]})
    r = paired_bootstrap_pages(good, bad, "line_cer", iters=1000)
    assert r["significant"] and r["a_better"] and r["diff"] < 0 and r["lower_is_better"]
    r = paired_bootstrap_pages(good, bad, "fields", iters=1000)
    assert r["significant"] and r["a_better"] and abs(r["a"] - 0.9) < 1e-9
    r = paired_bootstrap_pages(good, good, "line_cer", iters=200)
    assert not r["significant"]


def test_stratified_sample_is_nested_and_category_stable(tmp_path):
    from folioocr.bench.engines import PageItem
    from folioocr.bench.olmocr_bench import stratified_sample
    bench = str(tmp_path)
    items = [PageItem(key=f"{c}_{i}", pdf=os.path.join(bench, "pdfs", c, f"{i}.pdf"))
             for c in ("old_scans", "tables", "multi_column") for i in range(100)]
    small = {it.key for it in stratified_sample(items, bench, 30, seed=0)}
    big = {it.key for it in stratified_sample(items, bench, 90, seed=0)}
    assert len(small) == 30 and len(big) == 90 and small <= big
    only = {it.key for it in stratified_sample([x for x in items if x.key.startswith("tables")], bench, 10, seed=0)}
    assert only == {k for k in small if k.startswith("tables")}
