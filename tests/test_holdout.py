"""Holdout drafting: tests in olmOCR-Bench layout, review status, rejected tests skipped when scoring."""
import json

import pytest
from PIL import Image

pytest.importorskip("olmocr.bench.tests")

import draft_holdout_tests as H  # noqa: E402
from folioocr.bench.engines import Engine  # noqa: E402

TEXT = ("The committee met on Tuesday to review the annual budget. It approved the new library wing. "
        "Members asked for a report on maintenance costs before the next session.")


class Teacher(Engine):
    def run_full(self, items):
        return {it.key: {"markdown": TEXT, "raw": "",
                         "blocks": [{"bbox": [50, 20, 950, 60], "category": "header", "text": "City Council Minutes"},
                                    {"bbox": [50, 100, 950, 300], "category": "text", "text": TEXT},
                                    {"bbox": [50, 320, 950, 500], "category": "text",
                                     "text": "Attendance was recorded by the clerk for all members present."}]}
                for it in items}


def test_draft_review_and_score(tmp_path):
    src = tmp_path / "src" / "old_scans"
    src.mkdir(parents=True)
    Image.new("RGB", (600, 800), "white").save(src / "minutes1.pdf", "PDF", resolution=72)
    out = tmp_path / "holdout"
    st = H.draft(str(tmp_path / "src"), str(out), Teacher())
    assert st["total"]["unchecked"] > 3 and not st["ready"]
    path = out / "bench_data" / "old_scans.jsonl"
    tests = [json.loads(l) for l in open(path)]
    kinds = {t["type"] for t in tests}
    assert {"present", "absent", "order"} <= kinds and all(t["pdf"] == "old_scans/minutes1.pdf" for t in tests)
    assert (out / "bench_data" / "pdfs" / "old_scans" / "minutes1.pdf").exists() and (out / "review.html").exists()
    # the reviewer verifies all but one
    for i, t in enumerate(tests):
        t["checked"] = "rejected" if i == 0 else "verified"
    path.write_text("\n".join(json.dumps(t) for t in tests) + "\n")
    st = H.status(str(out))
    assert st["ready"] and st["total"]["rejected"] == 1
    # re-drafting never overwrites the review
    H.draft(str(tmp_path / "src"), str(out), Teacher())
    assert [json.loads(l)["checked"] for l in open(path)] == [t["checked"] for t in tests]
    # score a candidate that wrote exactly the body text: every verified test passes
    from folioocr.bench.olmocr_bench import load_bench_tests, score_candidate
    cand = out / "bench_data" / "perfect" / "old_scans"
    cand.mkdir(parents=True)
    (cand / "minutes1_pg1_repeat1.md").write_text(TEXT + "\n\nAttendance was recorded by the clerk for all members present.")
    loaded, _ = load_bench_tests(str(out / "bench_data"))
    assert len(loaded) == len(tests) - 1
    s = score_candidate(str(out / "bench_data"), "perfect", loaded)
    assert s["overall"] == pytest.approx(1.0)
