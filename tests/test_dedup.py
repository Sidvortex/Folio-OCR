"""Contamination index: each signal finds its copy and leaves unrelated pages alone."""
import io
import json
import random

import numpy as np
from PIL import Image, ImageDraw

from folioocr.data import dedup as D
from folioocr.data.sources.common import LabelWriter

WORDS = ("alpha beta gamma delta epsilon zeta theta kappa lambda sigma omega river stone cloud field "
         "market signal vector matrix kernel token layer model paper table figure column result method").split()


def _text(seed, n=120):
    r = random.Random(seed)
    return " ".join(r.choice(WORDS) + str(r.randint(0, 99)) for _ in range(n))


def _page(seed, text):
    r = random.Random(seed)
    im = Image.new("RGB", (850, 1100), "white")
    d = ImageDraw.Draw(im)
    y = 60
    for i in range(0, len(text), 90):
        x0 = 60 + r.randint(0, 200)
        d.rectangle([x0, y, x0 + r.randint(200, 600), y + 12], fill=(40, 40, 40))
        y += r.randint(18, 40)
        if y > 1040:
            break
    return im


def test_minhash_estimates_jaccard():
    mh = D.MinHasher()
    a = D.shingles(_text(1, 400))
    words = _text(1, 400).split()
    b = D.shingles(" ".join(words[:300] + _text(2, 100).split()))
    true = len(a & b) / len(a | b)
    est = D.jaccard_est(mh.signature(a), mh.signature(b))
    assert abs(true - est) < 0.12


def test_index_finds_copies_by_each_signal(tmp_path):
    # an olmOCR-Bench-style evaluation set: tests + one PDF per page
    bench = tmp_path / "bench_data"
    (bench / "pdfs" / "old_scans").mkdir(parents=True)
    eval_text = [_text(10 + i) for i in range(4)]
    tests = []
    for i in range(4):
        _page(10 + i, eval_text[i]).save(bench / "pdfs" / "old_scans" / f"p{i}.pdf", "PDF", resolution=100)
        words = eval_text[i].split()
        for k in range(4):
            tests.append({"pdf": f"old_scans/p{i}.pdf", "page": 1, "id": f"p{i}_{k}", "type": "present",
                          "text": " ".join(words[k * 10:k * 10 + 8])})
    (bench / "old_scans.jsonl").write_text("\n".join(json.dumps(t) for t in tests))
    # a Resume-Bench-style set with full text
    rb = tmp_path / "rb"
    rb.mkdir()
    res_img = tmp_path / "res.jpg"
    _page(99, _text(99)).save(res_img)
    with open(rb / "pages.jsonl", "w") as f:
        f.write(json.dumps({"id": "r1", "image": str(res_img), "lines": [{"text": _text(99)}]}) + "\n")

    idx = D.EvalIndex()
    for p in D.pages_olmocr_bench(str(bench)) + D.pages_resume_bench(str(rb)):
        idx.add(p)
    s = idx.summary()
    assert s["pages"] == 5 and s["image_hashes"] == 5 and s["pages_with_test_strings"] == 4
    path = str(tmp_path / "idx.npz")
    idx.save(path)
    idx = D.EvalIndex.load(path)

    train = tmp_path / "train"
    with LabelWriter(str(train), "folio-render", longest_side=1000) as w:
        # 1) the same page image, re-encoded and resized (no text overlap)
        buf = io.BytesIO()
        D._load_image(str(bench / "pdfs" / "old_scans" / "p0.pdf")).resize((700, 906)).save(buf, "JPEG", quality=60)
        w.add("img_copy", Image.open(buf), markdown=_text(500))
        # 2) the resume's text on a different-looking page
        w.add("text_copy", _page(7, "x" * 900).rotate(3, fillcolor="white"), markdown=_text(99))
        # 3) a page that quotes three of p2's four test strings inside other text
        words = eval_text[2].split()
        quoted = " ".join(" ".join(words[k * 10:k * 10 + 8]) for k in range(3))
        w.add("tests_copy", _page(8, "y" * 900).rotate(-4, fillcolor="white"),
              markdown=_text(501, 60) + " " + quoted + " " + _text(502, 60))
        # 4) unrelated
        w.add("clean", _page(9, "z" * 900).rotate(5, fillcolor="white"), markdown=_text(503))
    rep = D.check_labels(idx, str(train / "labels.jsonl"), write_clean=True)
    hits = {e["id"]: {h["by"] for h in e["hits"]} for e in rep["examples"]}
    assert "image" in hits["img_copy"]
    assert hits["text_copy"] == {"text"}
    assert hits["tests_copy"] == {"tests"}
    assert "clean" not in hits and rep["dropped"] == 3
    kept = [json.loads(l)["id"] for l in open(train / "labels.clean.jsonl")]
    assert kept == ["clean"]


def test_hamming_without_bitwise_count():
    a = np.array([0b1011, 0], np.uint64)
    b = np.array([[0b0001, 1], [0b1011, 0]], np.uint64)
    assert D.hamming(a, b).tolist() == [3, 0]
    x = np.bitwise_xor(b, a[None, :])
    assert (D._POP8[x.view(np.uint8)].reshape(2, -1).sum(1)).tolist() == [3, 0]
