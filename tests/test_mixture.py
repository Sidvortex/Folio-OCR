"""Mixtures: shares, licence lines, the non-commercial guard and the data card."""
import json

import pytest
from PIL import Image

from folioocr.data.mixture import data_card, licences, load_mixture
from folioocr.data.sources.common import LabelWriter


def _source(tmp_path, name, source, n=3):
    d = tmp_path / name
    with LabelWriter(str(d), source, dev_frac=0.0) as w:
        for i in range(n):
            w.add(f"{name}/{i}", Image.new("RGB", (100, 100), "white"), markdown=f"page {i}")
    return str(d / "labels.jsonl")


def test_mixture_licences_and_card(tmp_path):
    a = _source(tmp_path, "olm", "olmocr-mix-1025")
    b = _source(tmp_path, "inf", "infinity-doc2-5m", n=2)
    c = str(tmp_path / "nolicence" / "labels.jsonl")
    (tmp_path / "nolicence").mkdir()
    open(c, "w").write(json.dumps({"id": "x", "image": "x.jpg", "markdown": "m"}) + "\n")
    mix_path = tmp_path / "m.json"
    mix_path.write_text(json.dumps({"name": "t", "task_mix": "doc2json=0.5,doc2md=0.5", "sources": [
        {"labels": a, "weight": 0.6, "role": "real"}, {"labels": b, "weight": 0.3}, {"labels": c, "weight": 0.1}]}))
    mix = load_mixture(str(mix_path))
    assert mix.specs() == [f"{a}:0.6", f"{b}:0.3", f"{c}:0.1"]
    lic = licences(mix.specs())
    assert lic["noncommercial"] == [b] and lic["unknown"] == [c]
    rep = tmp_path / "cont.json"
    rep.write_text(json.dumps({"sources": [{"labels": a, "records": 3, "dropped": 1}]}))
    card = data_card(mix.specs(), name="t", contamination=str(rep), roles={a: "real"})
    assert "**Non-commercial:**" in card and "CC BY-NC-SA 4.0" in card and "| 60% | 3 / 0 |" in card
    assert "1 of 3" in card and "not checked" in card and "**Unknown licence:**" in card


def test_train_sft_refuses_noncommercial_without_flag(tmp_path):
    from folioocr.train.sft import build_args
    b = _source(tmp_path, "inf", "infinity-doc2-5m")
    with pytest.raises(SystemExit):
        build_args(["--profile", "cpu-test", "--data", b, "--out", str(tmp_path / "run")])
    args = build_args(["--profile", "cpu-test", "--data", f"{b}:2", "--out", str(tmp_path / "run"),
                       "--allow-noncommercial"])
    assert args.licences["noncommercial"] == [b] and args.task_mix == "doc2json=0.8,doc2md=0.2"
    m = tmp_path / "m.json"
    m.write_text(json.dumps({"sources": [{"labels": b, "weight": 1}], "task_mix": "doc2md=1"}))
    args = build_args(["--profile", "cpu-test", "--mixture", str(m), "--out", str(tmp_path / "r2"),
                       "--allow-noncommercial"])
    assert args.data == [f"{b}:1.0"] and args.task_mix == "doc2md=1"
