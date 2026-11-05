"""Teacher consensus: agreeing pages become labels, the rest go to review; banned teachers are refused."""
import json
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer

import pytest
from PIL import Image

from folioocr.bench.engines import Engine, MarkdownDirEngine, OpenAIEngine, PageItem
from folioocr.data import teachers as T
from folioocr.data.dataset import read_labels


class FakeLayoutTeacher(Engine):
    name = "fake"

    def __init__(self, texts):
        self.texts = texts

    def run_full(self, items):
        out = {}
        for it in items:
            t = self.texts[it.key]
            out[it.key] = {"markdown": t, "raw": t,
                           "blocks": [{"bbox": [10, 10, 990, 200], "category": "text", "text": t}]}
        return out


def test_agreement():
    assert T.agreement("Hello **world**", "Hello world") == 1.0
    assert T.agreement("", "") == 1.0
    assert 0.8 < T.agreement("The quick brown fox jumps", "The quick brown fax jumps") < 1.0


def test_stage_run_merge(tmp_path):
    src = tmp_path / "scans"
    src.mkdir()
    for i in range(4):
        Image.new("RGB", (400, 500), "white").save(src / f"p{i}.png")
    Image.new("RGB", (300, 400), "white").save(tmp_path / "doc.pdf", "PDF", save_all=True,
                                               append_images=[Image.new("RGB", (300, 400), "white")])
    pages = T.stage_pages([str(src), str(tmp_path / "doc.pdf")], str(tmp_path / "stage"))
    assert [k for k, _ in pages] == ["p0", "p1", "p2", "p3", "doc_p1", "doc_p2"]
    texts_a = {"p0": "Invoice number 12345 due 2026", "p1": "Totally different words here", "p2": "",
               "p3": "Same page text", "doc_p1": "Doc page one", "doc_p2": "Doc page two"}
    md_b = tmp_path / "b_md"
    md_b.mkdir()
    for k, t in {"p0": "Invoice number 12345 due 2026", "p1": "Nothing alike at all, sorry", "p2": "",
                 "p3": "Same page text", "doc_p1": "Doc page one"}.items():
        (md_b / f"{k}.md").write_text(t)
    T.run_teacher(FakeLayoutTeacher(texts_a), "ip2pro", pages, str(tmp_path / "A"), "Apache-2.0", engine_kind="openai")
    T.run_teacher(MarkdownDirEngine(str(md_b)), "olmocr2", pages, str(tmp_path / "B"), "Apache-2.0",
                  engine_kind="mddir")
    st = T.merge(pages, str(tmp_path / "A"), str(tmp_path / "B"), str(tmp_path / "out"))
    # doc_p2: B wrote nothing -> its cached output is empty and differs from A
    assert st["kept"] == 3 and st["blank"] == 1 and st["review"] == 2 and st["label_blocks"] == 3
    recs = read_labels(str(tmp_path / "out" / "labels.jsonl"))
    assert sorted(r["id"] for r in recs) == ["consensus/doc_p1", "consensus/p0", "consensus/p3"]
    assert recs[0]["meta"]["label_from"] == "ip2pro" and recs[0]["blocks"][0]["category"] == "text"
    review = [json.loads(l) for l in open(tmp_path / "out" / "review.jsonl")]
    assert {r["id"] for r in review} == {"consensus/p1", "consensus/doc_p2"}
    info = json.load(open(tmp_path / "out" / "source.json"))
    assert info["args"]["teachers"][0]["licence"] == "Apache-2.0"
    with pytest.raises(SystemExit):
        T.run_teacher(MarkdownDirEngine(str(md_b)), "chandra2", pages, str(tmp_path / "C"), "OpenRAIL-M",
                      engine_kind="chandra")


def test_openai_engine_request_and_parsing(tmp_path):
    seen = {}

    class H(BaseHTTPRequestHandler):
        def do_POST(self):
            body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
            seen.update(body)
            ans = '```json\n[{"bbox": [0, 0, 500, 100], "category": "title", "text": "# Hi"}]\n```'
            data = json.dumps({"choices": [{"message": {"content": ans}}]}).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)

        def log_message(self, *a):
            pass

    srv = HTTPServer(("127.0.0.1", 0), H)
    th = threading.Thread(target=srv.serve_forever, daemon=True)
    th.start()
    try:
        img = tmp_path / "x.png"
        Image.new("RGB", (2000, 1000), "white").save(img)
        eng = OpenAIEngine(f"http://127.0.0.1:{srv.server_port}/v1", "infly/Infinity-Parser2-Pro", prompt="doc2json",
                           longest_side=1000)
        out = eng.run_full([PageItem(key="x", image=str(img))])["x"]
    finally:
        srv.shutdown()
    assert out["markdown"] == "# Hi" and out["blocks"][0]["category"] == "title"
    assert seen["model"] == "infly/Infinity-Parser2-Pro" and seen["chat_template_kwargs"] == {"enable_thinking": False}
    content = seen["messages"][0]["content"]
    assert content[0]["image_url"]["url"].startswith("data:image/png;base64,") and "layout" in content[1]["text"]
