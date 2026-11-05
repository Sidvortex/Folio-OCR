"""Public-dataset adapters on small fixtures built in the layouts their cards describe."""
import io
import json
import tarfile

import pytest
from PIL import Image, ImageDraw

from folioocr.data.dataset import read_labels
from folioocr.data.sources.common import LabelWriter, normalize_markdown, pipe_tables_to_html, split_for


def _page_pdf(text: str, size=(600, 800), pages: int = 1) -> bytes:
    ims = []
    for p in range(pages):
        im = Image.new("RGB", size, "white")
        d = ImageDraw.Draw(im)
        d.rectangle([20, 20, 120, 60], fill="black")          # a mark at the top-left, to check orientation
        d.text((40, 120), f"{text} p{p}", fill="black")
        ims.append(im)
    buf = io.BytesIO()
    ims[0].save(buf, "PDF", save_all=True, append_images=ims[1:], resolution=72)
    return buf.getvalue()


def _tar(path, members, gz=True):
    with tarfile.open(path, "w:gz" if gz else "w") as t:
        for name, data in members.items():
            info = tarfile.TarInfo(name)
            info.size = len(data)
            t.addfile(info, io.BytesIO(data))


def test_markdown_normalisation_matches_doc2md_conventions():
    md = ("---\nprimary_language: en\nis_rotation_valid: true\n---\n# Title\n\nInline \\(a+b\\) and\n\\[x^2\\]\n\n"
          "![a chart](page_1_2_3_4.png)\n\n| A | B |\n|---|---|\n| 1 | 2 |\n")
    out = normalize_markdown(md)
    assert out.startswith("# Title") and "$a+b$" in out and "$$x^2$$" in out and "![" not in out
    assert "<table><tr><th>A</th><th>B</th></tr><tr><td>1</td><td>2</td></tr></table>" in out
    assert pipe_tables_to_html("a | b") == "a | b"
    assert split_for("x", 0.0) == "train" and split_for("x", 1.0) == "dev"


def test_label_writer_appends_and_resumes(tmp_path):
    img = Image.new("RGB", (3000, 1000), "white")
    with LabelWriter(str(tmp_path), "olmocr-mix-1025", longest_side=1600) as w:
        w.add("a/1", img, markdown="hello")
        with pytest.raises(ValueError):
            w.add("a/2", img)
    with LabelWriter(str(tmp_path), "olmocr-mix-1025") as w:
        assert w.has("a/1") and w.add("a/1", img, markdown="again") is None
        w.add("a/2", img, markdown="two")
    recs = read_labels(str(tmp_path / "labels.jsonl"))
    assert [r["id"] for r in recs] == ["a/1", "a/2"] and recs[0]["width"] == 1600
    info = json.load(open(tmp_path / "source.json"))
    assert info["licence"] == "ODC-BY 1.0" and info["records"] == 2


def test_olmocr_mix_adapter(tmp_path):
    pd = pytest.importorskip("pandas")
    from folioocr.data.sources import olmocr_mix
    up = _page_pdf("upright")
    # a page stored turned 90 degrees clockwise; olmOCR says: rotate by 90 (PIL ROTATE_90) to fix
    sideways = Image.new("RGB", (800, 600), "white")
    ImageDraw.Draw(sideways).rectangle([740, 20, 780, 120], fill="black")
    buf = io.BytesIO()
    sideways.save(buf, "PDF", resolution=72)
    _tar(tmp_path / "02_loc_transcripts_train_00000.tar.gz",
         {"0000/aaa-1.pdf": up, "0000/bbb-1.pdf": buf.getvalue(), "0000/ccc-1.pdf": up, "0000/ddd-1.pdf": up})
    df = pd.DataFrame({
        "url": ["http://x/a.pdf", "http://x/b.pdf", "http://x/c.pdf"],
        "page_number": [1, 3, 1],
        "natural_text": ["A \\(x\\) page", "Sideways page", None],
        "is_rotation_valid": [True, False, True],
        "rotation_correction": [0, 90, 0],
    }, index=["0000/aaa-1", "0000/bbb-1", "0000/ccc-1"])
    df.to_parquet(tmp_path / "02_loc_transcripts_train.parquet")
    out = tmp_path / "out"
    r = olmocr_mix.convert(str(tmp_path / "02_loc_transcripts_train.parquet"),
                           [str(tmp_path / "02_loc_transcripts_train_00000.tar.gz")], str(out), render_side=800)
    assert r["written"] == 2 and r["skipped"] == {"no text": 1, "no parquet row": 1}
    recs = {x["id"]: x for x in read_labels(str(out / "labels.jsonl"))}
    a = recs["olmocrmix/02_loc_transcripts/0000/aaa-1"]
    assert a["markdown"] == "A $x$ page" and a["meta"]["url"] == "http://x/a.pdf" and a["meta"]["subset"] == "02_loc_transcripts"
    b = Image.open(recs["olmocrmix/02_loc_transcripts/0000/bbb-1"]["_image"])
    assert b.height > b.width                                   # turned upright
    # the mark that sat at the right edge of the sideways page is now at the top
    px = b.convert("L").load()
    assert min(px[x, y] for x in range(b.width) for y in range(0, 60)) < 80
    # re-running adds nothing
    r2 = olmocr_mix.convert(str(tmp_path / "02_loc_transcripts_train.parquet"),
                            [str(tmp_path / "02_loc_transcripts_train_00000.tar.gz")], str(out), render_side=800)
    assert r2["written"] == 0 and r2["total"] == 2


def test_synthmix_adapter(tmp_path):
    from folioocr.data.sources import synthmix
    root = tmp_path / "synthmix"
    (root / "bench_data" / "pdfs" / "general").mkdir(parents=True)
    (root / "training" / "general").mkdir(parents=True)
    (root / "bench_data" / "pdfs" / "general" / "p1.pdf").write_bytes(_page_pdf("bench"))
    tests = [{"pdf": "general/p1.pdf", "page": 1, "id": "p1_present_0", "type": "present", "text": "Hello world",
              "max_diffs": 0, "before": None},
             {"pdf": "general/p1.pdf", "page": 1, "id": "p1_absent_0", "type": "absent", "text": "Page 3"}]
    (root / "bench_data" / "general.jsonl").write_text("\n".join(json.dumps(t) for t in tests))
    (root / "training" / "general" / "p1.md").write_text("---\nprimary_language: en\n---\nHello world \\(x\\)")
    (root / "training" / "general" / "p1.pdf").write_bytes(_page_pdf("train"))
    (root / "training" / "general" / "lonely.md").write_text("no pdf")
    r = synthmix.convert_sft(str(root), str(tmp_path / "sft"), render_side=800)
    assert r["written"] == 1 and r["skipped"] == {"unpaired": 1}
    assert read_labels(str(tmp_path / "sft" / "labels.jsonl"))[0]["markdown"] == "Hello world $x$"
    r = synthmix.convert_rl(str(root), str(tmp_path / "rl"), render_side=800)
    rec = read_labels(str(tmp_path / "rl" / "labels.jsonl"))[0]
    assert r["written"] == 1 and rec["markdown"] == "Hello world $x$" and len(rec["tests"]) == 2
    assert rec["tests"][0]["id"] == "synthmix-rl/general/p1::p1_present_0" and "before" not in rec["tests"][0]


def test_infinity_doc2_adapter_fills_placeholders_and_normalises_boxes(tmp_path):
    from folioocr.data.sources import infinity_doc2
    root = tmp_path / "doc2json"
    (root / "images" / "part_0001").mkdir(parents=True)
    (root / "labels").mkdir()
    Image.new("RGB", (2000, 2600), "white").save(root / "images" / "part_0001" / "a.jpg")
    Image.new("RGB", (900, 900), "white").save(root / "images" / "part_0001" / "b.jpg")
    answer = ('```json\n[{"bbox": <bbox>, "category": <ref-object>, "text": "# Heading"},\n'
              '{"bbox": <bbox>, "category": <ref-object>, "text": "\\\\[E=mc^2\\\\]"},\n'
              '{"bbox": <bbox>, "category": <ref-object>, "text": ""}]\n```')
    recs = [
        {"images": ["images/part_0001/a.jpg"],
         "conversations": [{"from": "human", "value": "<image>\nExtract layout"}, {"from": "gpt", "value": answer}],
         "objects": {"ref": ["title", "formula", "figure"], "bbox": [[200, 260, 1800, 520], [400, 1300, 1600, 1560],
                                                                    [100, 2000, 1900, 2500]]},
         "attributes": {"task": "Document Parsing", "subtask": "doc2json"}},
        {"images": ["images/part_0001/a.jpg"], "conversations": [{"from": "gpt", "value": "x"}],
         "attributes": {"subtask": "docvqa"}},
        {"images": ["images/part_0001/b.jpg"],
         "conversations": [{"from": "gpt", "value": '[{"bbox": <bbox>, "category": <ref-object>, "text": "x"}]'}],
         "objects": {"ref": ["text", "text"], "bbox": [[0, 0, 10, 10], [0, 0, 5, 5]]},
         "attributes": {"subtask": "doc2json"}},
    ]
    (root / "labels" / "label.jsonl").write_text("\n".join(json.dumps(r) for r in recs))
    r = infinity_doc2.convert([str(root)], str(tmp_path / "out"))
    assert r["written"] == 1 and r["skipped"] == {"subtask docvqa": 1, "placeholder count mismatch": 1}
    assert r["bbox_modes"] == {"pixels": 1}
    rec = read_labels(str(tmp_path / "out" / "labels.jsonl"))[0]
    assert [b["category"] for b in rec["blocks"]] == ["title", "formula", "figure"]
    assert rec["blocks"][0]["bbox"] == [100, 100, 900, 200]
    assert rec["blocks"][1]["text"] == "$$E=mc^2$$" and rec["blocks"][2]["text"] == ""
    assert json.load(open(tmp_path / "out" / "source.json"))["commercial"] == "no"


def test_lighton_mix_joins_transcriptions_with_pdfa_shards(tmp_path):
    pa = pytest.importorskip("pyarrow")
    import pyarrow.parquet as pq
    from folioocr.data.sources import lighton_mix
    _tar(tmp_path / "pdfa-eng-train-0000.tar", {"0000001.pdf": _page_pdf("doc one", pages=2),
                                                "0000001.json": b"{}", "0000002.pdf": _page_pdf("doc two")}, gz=False)
    t = pa.table({"key": ["0000001", "0000001", "0000009", "0000002"], "page_idx": [0, 1, 0, 0],
                  "content": ["First \\(a\\)", "Second", "not in shard", ""]})
    pq.write_table(t, tmp_path / "part.parquet", row_group_size=2)
    r = lighton_mix.convert([str(tmp_path / "pdfa-eng-train-0000.tar")], [str(tmp_path / "part.parquet")],
                            str(tmp_path / "out"), render_side=600)
    assert r["matched_rows"] == 3 and r["written"] == 2 and r["skipped"] == {"empty transcription": 1}
    recs = {x["id"]: x for x in read_labels(str(tmp_path / "out" / "labels.jsonl"))}
    assert recs["lighton/0000001/p0"]["markdown"] == "First $a$" and "lighton/0000001/p1" in recs
