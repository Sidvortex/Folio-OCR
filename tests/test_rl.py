"""GRPO on the tiny model: the update moves the policy toward reward; the CLI saves, resumes and logs."""
import json
import os

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("transformers")
pytest.importorskip("peft")

from PIL import Image, ImageDraw  # noqa: E402

from folioocr.model.load import ModelConfig, add_lora, load_model  # noqa: E402
from folioocr.rl.grpo import GRPOConfig, GRPOTrainer, completion_logprobs, group_advantages  # noqa: E402


def test_group_advantages():
    assert group_advantages([1.0, 1.0, 1.0]) is None
    a = group_advantages([0.0, 1.0])
    assert a[0] < 0 < a[1] and abs(a[0] + a[1]) < 1e-9 and abs(a[1] - 1.0) < 1e-3
    assert group_advantages([0.2, 0.6], scale="none") == pytest.approx([-0.2, 0.2])


def test_completion_logprobs_match_full_logits(tiny_model, tmp_path):
    model, proc = load_model(ModelConfig(model=tiny_model, dtype="float32"))
    img = Image.new("RGB", (64, 64), "white")
    msgs = [{"role": "user", "content": [{"type": "image", "image": img}, {"type": "text", "text": "read"}]}]
    text = proc.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True, enable_thinking=False)
    enc = proc(text=[text], images=[img], return_tensors="pt")
    enc.pop("token_type_ids", None)
    n = enc["input_ids"].shape[1]
    ans = proc.tokenizer("hello", add_special_tokens=False, return_tensors="pt")["input_ids"]
    ids = torch.cat([enc["input_ids"], ans], 1)
    inp = {"input_ids": ids, "attention_mask": torch.ones_like(ids), "pixel_values": enc["pixel_values"],
           "image_grid_thw": enc["image_grid_thw"]}
    if "mm_token_type_ids" in enc:
        inp["mm_token_type_ids"] = torch.cat([enc["mm_token_type_ids"], torch.zeros_like(ans)], 1)
    with torch.no_grad():
        mine = completion_logprobs(model, inp, n, chunk=2)
        full = model(**inp).logits[0, n - 1:-1].float().log_softmax(-1).gather(-1, ans[0][:, None])[:, 0]
    assert mine.shape == full.shape and torch.allclose(mine, full, atol=1e-5)


def test_grpo_moves_the_policy_toward_reward(tiny_model, tmp_path):
    torch.manual_seed(0)
    img = tmp_path / "toy.png"
    Image.new("RGB", (64, 64), "white").save(img)
    model, proc = load_model(ModelConfig(model=tiny_model, dtype="float32"), for_training=True)
    model = add_lora(model, r=16, alpha=32, dropout=0.0, gradient_checkpointing=False)
    for name, p in model.named_parameters():
        if "language_model" in name or "lm_head" in name:
            p.requires_grad_(True)

    def reward(item, text, stopped, looped):          # toy task: write the letter "a"
        r = text.count("a") / 8.0
        return {"reward": r, "components": {"a": r}}

    cfg = GRPOConfig(group=8, prompts_per_step=1, steps=25, lr=1e-2, max_new_tokens=8, max_pixels=4096,
                     eval_every=0, dump_every=0, save_every=0, task="doc2md")
    tr = GRPOTrainer(model, proc, [([{"id": "toy", "_image": str(img)}], 1.0)], cfg, str(tmp_path / "run"),
                     reward_fn=reward)
    hist = [tr.train_step()["reward"] for _ in range(cfg.steps)]
    assert sum(hist[-5:]) / 5 > sum(hist[:5]) / 5 + 0.5, hist


@pytest.fixture()
def rl_items(tmp_path):
    from folioocr.rl.regions import build
    img = Image.new("RGB", (400, 520), "white")
    d = ImageDraw.Draw(img)
    for y in (30, 120, 260, 400):
        d.rectangle([30, y, 360, y + 40], fill=(30, 30, 30))
    img.save(tmp_path / "page.png")
    blocks = [{"bbox": [60, 50, 900, 140], "category": "title", "text": "# Asha Rao"},
              {"bbox": [60, 220, 900, 320], "category": "text", "text": "asha.rao@example.com"},
              {"bbox": [60, 490, 900, 590], "category": "text", "text": "Built dashboards for retail sales teams"},
              {"bbox": [60, 760, 900, 860], "category": "text", "text": "Python, SQL, Power BI and Excel"}]
    lines = [{"box": [30, 120, 360, 160], "text": "asha.rao@example.com", "block": 1, "math": False},
             {"box": [30, 260, 360, 300], "text": "Built dashboards for retail sales teams", "block": 2, "math": False},
             {"box": [30, 400, 360, 440], "text": "Python, SQL, Power BI and Excel", "block": 3, "math": False}]
    with open(tmp_path / "labels.jsonl", "w") as f:
        for i, split in enumerate(("train", "train", "dev")):
            f.write(json.dumps({"id": f"pg{i}", "image": "page.png", "kind": "resume", "blocks": blocks,
                                "lines": lines, "split": split}) + "\n")
    r = build([str(tmp_path / "labels.jsonl")], str(tmp_path / "rl"), per_page=2, max_chars=60)
    assert r["items"] >= 3
    return str(tmp_path / "rl" / "labels.jsonl")


def test_train_rl_cli_saves_resumes_and_uses_kl(tiny_model, rl_items, tmp_path):
    import train_rl
    model, proc = load_model(ModelConfig(model=tiny_model, dtype="float32"), for_training=True)
    model = add_lora(model, r=4, alpha=8, dropout=0.0, gradient_checkpointing=False)
    model.save_pretrained(str(tmp_path / "sft"))
    proc.save_pretrained(str(tmp_path / "sft"))
    out = str(tmp_path / "run")
    args = ["--profile", "cpu-test", "--model", tiny_model, "--adapter", str(tmp_path / "sft"), "--data", rl_items,
            "--out", out, "--steps", "2", "--beta", "0.05", "--eval-every", "2", "--dump-every", "1",
            "--save-every", "1", "--max-new-tokens", "24"]
    res = train_rl.main(args)
    assert res["steps"] == 2
    logs = [json.loads(l) for l in open(os.path.join(out, "train_log.jsonl"))]
    assert len(logs) == 2 and {"tests", "format"} <= set(logs[0]["components"]) and "kl" in logs[0]
    dev = [json.loads(l) for l in open(os.path.join(out, "dev_log.jsonl"))]
    assert dev[0]["step"] == 2 and dev[0]["items"] >= 1
    assert os.path.exists(os.path.join(out, "samples", "step_000001.jsonl"))
    assert os.path.exists(os.path.join(out, "best", "adapter_model.safetensors"))
    assert os.path.exists(os.path.join(out, "adapter", "adapter_model.safetensors"))
    assert not os.path.exists(os.path.join(out, "adapter", "ref"))            # only the policy is saved
    res = train_rl.main(args[:args.index("--steps") + 1] + ["3"] + args[args.index("--steps") + 2:] + ["--resume"])
    assert res["steps"] == 3
    assert len(open(os.path.join(out, "train_log.jsonl")).readlines()) == 3
