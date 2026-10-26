"""The training and inference path on a tiny random Qwen3.5 model (CPU, ~1 minute).

The overfit test is the important one: if the prompt/target layout, label
masking, chunked loss, LoRA wiring, generation stop tokens or output parsing
were wrong, a model could not learn to reproduce one page's labels exactly.
"""
import json
import os

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("transformers")
pytest.importorskip("peft")

from PIL import Image, ImageDraw  # noqa: E402

from folioocr.data.dataset import Collator, PageDataset  # noqa: E402
from folioocr.model.generate import PageParser  # noqa: E402
from folioocr.model.load import ModelConfig, add_lora, load_model, special_ids  # noqa: E402
from folioocr.schema import Block, dump_target  # noqa: E402
from folioocr.train.losses import chunked_causal_lm_loss  # noqa: E402

BLOCKS = [{"bbox": [62, 50, 600, 120], "category": "title", "text": "# Asha"},
          {"bbox": [62, 160, 940, 300], "category": "text", "text": "- SQL\n- Go"}]


@pytest.fixture()
def page_record(tmp_path):
    img = Image.new("RGB", (160, 200), "white")
    d = ImageDraw.Draw(img)
    d.text((10, 10), "Asha", fill="black")
    d.text((10, 40), "- SQL", fill="black")
    p = tmp_path / "p.png"
    img.save(p)
    return {"id": "p0", "image": str(p), "_image": str(p), "blocks": BLOCKS, "split": "train"}


def test_special_ids_include_both_stop_tokens(tiny_model):
    _, proc = load_model(ModelConfig(model=tiny_model, dtype="float32"))
    ids = special_ids(proc)
    tok = proc.tokenizer
    assert tok.convert_tokens_to_ids("<|im_end|>") in ids["eos"]
    assert tok.convert_tokens_to_ids("<|endoftext|>") in ids["eos"]


def test_dataset_masks_prompt_and_supervises_target(tiny_model, page_record):
    model, proc = load_model(ModelConfig(model=tiny_model, dtype="float32"), for_training=True)
    ds = PageDataset([([page_record], 1.0)], proc, {"doc2json": 1.0}, max_pixels=40_000, max_length=4096)
    ex = ds[0]
    sup = [t for t, l in zip(ex["input_ids"], ex["labels"]) if l != -100]
    text = proc.tokenizer.decode(sup)
    assert text.startswith("```json\n[\n{\"bbox\": [62, 50, 600, 120]") and text.endswith("<|im_end|>")
    assert ex["labels"][: len(ex["labels"]) - len(sup)] == [-100] * (len(ex["labels"]) - len(sup))
    batch = Collator(proc.tokenizer.pad_token_id)([ex, ds[0]])
    ref = model(**batch).loss
    mine, n = chunked_causal_lm_loss(model, batch, chunk=37)
    assert n == 2 * (len(sup) - 0) and abs(float(ref) - float(mine)) < 1e-4


def test_long_targets_lose_whole_blocks(tiny_model, page_record):
    _, proc = load_model(ModelConfig(model=tiny_model, dtype="float32"))
    full = PageDataset([([page_record], 1.0)], proc, {"doc2json": 1.0}, max_pixels=40_000, max_length=8192)[0]
    n_prompt = sum(1 for l in full["labels"] if l == -100)
    one = len(proc.tokenizer(dump_target([Block.from_dict(BLOCKS[0])]), add_special_tokens=False)["input_ids"])
    ds = PageDataset([([page_record], 1.0)], proc, {"doc2json": 1.0}, max_pixels=40_000, max_length=n_prompt + one + 3)
    ex = ds[0]
    sup = proc.tokenizer.decode([t for t, l in zip(ex["input_ids"], ex["labels"]) if l != -100])
    assert sup == dump_target([Block.from_dict(BLOCKS[0])]) + "<|im_end|>"   # still valid JSON, one block dropped
    assert ds.dropped_blocks == 1


def test_overfit_one_page_then_parse_it_back(tiny_model, page_record):
    torch.manual_seed(0)
    model, proc = load_model(ModelConfig(model=tiny_model, dtype="float32"), for_training=True)
    model = add_lora(model, r=16, alpha=32, dropout=0.0, gradient_checkpointing=False)
    for name, p in model.named_parameters():       # tiny random base: let it learn everything
        if "language_model" in name or "lm_head" in name or "embed_tokens" in name:
            p.requires_grad_(True)
    ds = PageDataset([([page_record], 1.0)], proc, {"doc2json": 1.0}, max_pixels=40_000, max_length=4096)
    batch = Collator(proc.tokenizer.pad_token_id)([ds[0]])
    opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=3e-3)
    model.train()
    for step in range(400):
        loss, _ = chunked_causal_lm_loss(model, batch)
        loss.backward()
        opt.step()
        opt.zero_grad()
        if float(loss) < 0.002:
            break
    assert float(loss) < 0.01, float(loss)
    model.eval()
    parser = PageParser(model, proc, max_pixels=40_000, max_new_tokens=400, loop_retry=False)
    res = parser.parse([Image.open(page_record["image"])])[0]
    assert res.parse_ok and not res.looped and not res.truncated
    assert [b.to_dict() for b in res.blocks] == BLOCKS
    assert res.markdown == "# Asha\n\n- SQL\n- Go"
