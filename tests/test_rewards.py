"""RL rewards: a perfect answer scores 1, every kind of damage costs what it should."""
import random

import pytest

pytest.importorskip("olmocr.bench.tests")

from folioocr.render.tests_gen import make_tests  # noqa: E402
from folioocr.rl.rewards import RewardConfig, score, text_accuracy  # noqa: E402
from folioocr.rl.teds import teds  # noqa: E402
from folioocr.schema import Block, dump_target  # noqa: E402

TABLE = ("<table><tr><th>Degree</th><th>Institute</th><th>Year</th></tr><tr><td>B.Tech CSE</td>"
         "<td>MUIT Noida</td><td>2027</td></tr><tr><td>Class XII</td><td>DPS Ranchi</td><td>2023</td></tr></table>")
BLOCKS = [
    Block([100, 40, 900, 80], "title", "# Ravi Kumar"),
    Block([100, 90, 900, 110], "text", "ravi.kumar@example.com | +91 98765 43210 | github.com/ravik"),
    Block([100, 130, 900, 150], "title", "## Experience"),
    Block([100, 160, 900, 230], "text", "**Data Intern**, Acme Analytics (May 2025 - Jul 2025)\n"
                                        "- Built a churn model that cut false alarms by a third\n"
                                        "- Wrote the weekly reporting pipeline in Python and SQL"),
    Block([100, 260, 900, 400], "table", TABLE),
    Block([100, 960, 900, 980], "footer", "Page 1 of 1"),
]
LINES = [{"box": [100, 160, 900, 175], "text": "Data Intern, Acme Analytics (May 2025 - Jul 2025)", "block": 3},
         {"box": [100, 180, 900, 195], "text": "Built a churn model that cut false alarms by a third", "block": 3},
         {"box": [100, 200, 900, 215], "text": "Wrote the weekly reporting pipeline in Python and SQL", "block": 3}]


@pytest.fixture
def item():
    tests = make_tests("r1", BLOCKS, LINES, random.Random(0))
    return {"id": "r1", "blocks": [b.to_dict() for b in BLOCKS], "tests": tests}


def test_perfect_answer_scores_one(item):
    r = score(item, dump_target(BLOCKS))
    assert r["reward"] == pytest.approx(1.0) and set(r["components"]) == {"tests", "text", "table", "fields", "format"}


def test_damage_costs(item):
    perfect = score(item, dump_target(BLOCKS))["reward"]
    half = score(item, dump_target(BLOCKS[:3]))
    assert half["reward"] < perfect - 0.3 and half["components"]["format"] < 1 and half["components"]["table"] == 0
    typo = [Block(b.bbox, b.category, b.text.replace("98765", "98766")) for b in BLOCKS]
    t = score(item, dump_target(typo))
    assert t["components"]["fields"] < 1 and t["components"]["text"] > 0.99 and t["reward"] < perfect
    tab = [Block(b.bbox, b.category, b.text.replace("<td>2027</td>", "<td>2021</td>")) for b in BLOCKS]
    assert 0.9 < score(item, dump_target(tab))["components"]["table"] < 1
    cut = dump_target(BLOCKS)[:-40]
    c = score(item, cut, stopped=False)
    assert c["components"]["format"] < 1 and c["reward"] < perfect
    looped = dump_target(BLOCKS[:2])[:-6] + ", " + ", ".join(['{"bbox": [1, 2, 3, 4], "category": "text", "text": "a b c"}'] * 60)
    assert score(item, looped)["reward"] == 0.0
    assert score(item, dump_target(BLOCKS), looped=True)["reward"] == 0.0


def test_dropping_text_does_not_pay(item):
    # passing the "absent" (footer) tests by writing nothing must not beat reading the page
    empty = score(item, "```json\n[]\n```")
    assert empty["reward"] < 0.35
    with_footer = score(item, dump_target(BLOCKS[:-1] + [Block([100, 960, 900, 980], "text", "Page 1 of 1")]))
    assert with_footer["reward"] < score(item, dump_target(BLOCKS))["reward"]


def test_weights_renormalise_over_what_applies():
    md_item = {"id": "m", "markdown": "Plain page with some words on it and nothing else at all."}
    r = score(md_item, "Plain page with some words on it and nothing else at all.", RewardConfig(task="doc2md"))
    assert set(r["components"]) == {"text", "format"} and r["reward"] == pytest.approx(1.0)
    tests_only = {"id": "t", "tests": [{"pdf": "t.pdf", "page": 1, "id": "t_present_0", "type": "present",
                                        "text": "Plain page with some words", "max_diffs": 0}]}
    r = score(tests_only, "```json\n[{\"bbox\": [0, 0, 10, 10], \"category\": \"text\", \"text\": \"Plain page with some words\"}]\n```")
    assert set(r["components"]) == {"tests", "format"} and r["reward"] == pytest.approx(1.0)


def test_text_accuracy_and_teds_bounds():
    assert text_accuracy("abc", "abc") == 1.0 and text_accuracy("", "") == 1.0 and text_accuracy("abc", "") == 0.0
    assert teds(TABLE, TABLE) == 1.0 and teds("<p>no</p>", TABLE) == 0.0
