"""Merging: one adapter reproduces the PEFT model; several are averaged as deltas, not as A/B matrices."""
import json
import os

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("peft")

from PIL import Image  # noqa: E402

from folioocr.model.load import ModelConfig, add_lora, load_model  # noqa: E402
from folioocr.model.merge import merge  # noqa: E402


def _random_adapter(tiny_model, path, seed):
    torch.manual_seed(seed)
    model, proc = load_model(ModelConfig(model=tiny_model, dtype="float32"), for_training=True)
    model = add_lora(model, r=4, alpha=8, dropout=0.0, gradient_checkpointing=False)
    with torch.no_grad():
        for n, p in model.named_parameters():
            if "lora_B" in n:
                p.normal_(0, 0.05)        # peft starts B at zero; give the adapter a real delta
    model.save_pretrained(path)
    return path


def _inputs(proc):
    img = Image.new("RGB", (64, 64), "white")
    msgs = [{"role": "user", "content": [{"type": "image", "image": img}, {"type": "text", "text": "read this"}]}]
    text = proc.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True, enable_thinking=False)
    enc = proc(text=[text], images=[img], return_tensors="pt")
    enc.pop("token_type_ids", None)
    return enc


def test_merge_matches_peft_and_averages_deltas(tiny_model, tmp_path):
    from transformers import AutoModelForImageTextToText
    a = _random_adapter(tiny_model, str(tmp_path / "a"), 1)
    b = _random_adapter(tiny_model, str(tmp_path / "b"), 2)
    info = merge(tiny_model, [a], str(tmp_path / "m1"), [1.0], dtype="float32")
    assert info["layers_merged"] > 0
    peft_model, proc = load_model(ModelConfig(model=tiny_model, adapter=a, dtype="float32"))
    merged = AutoModelForImageTextToText.from_pretrained(str(tmp_path / "m1"), dtype=torch.float32).eval()
    enc = _inputs(proc)
    with torch.no_grad():
        want = peft_model(**enc).logits
        got = merged(**enc).logits
    assert torch.allclose(want, got, atol=1e-4)
    assert not any("lora" in n for n, _ in merged.named_parameters())

    merge(tiny_model, [a, b], str(tmp_path / "avg"), [0.5, 0.5], dtype="float32")
    base = AutoModelForImageTextToText.from_pretrained(tiny_model, dtype=torch.float32)
    avg = AutoModelForImageTextToText.from_pretrained(str(tmp_path / "avg"), dtype=torch.float32)
    from safetensors.torch import load_file
    sa, sb = load_file(os.path.join(a, "adapter_model.safetensors")), load_file(os.path.join(b, "adapter_model.safetensors"))
    key = next(k for k in sa if k.endswith("lora_A.weight") and "q_proj" in k)
    mod = key[len("base_model.model."):-len(".lora_A.weight")]
    cfg = json.load(open(os.path.join(a, "adapter_config.json")))
    scale = cfg["lora_alpha"] / cfg["r"]

    def delta(s):
        return s[key.replace("lora_A", "lora_B")] @ s[key] * scale

    expect = dict(base.named_parameters())[mod + ".weight"] + 0.5 * delta(sa) + 0.5 * delta(sb)
    assert torch.allclose(dict(avg.named_parameters())[mod + ".weight"], expect, atol=1e-6)
