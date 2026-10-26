"""A tiny, randomly initialised Qwen3.5 vision-language model for CPU tests.

Same architecture class (Qwen3_5ForConditionalGeneration: Gated-DeltaNet and
full-attention layers, patch-16 vision tower with 2x2 merge), same processor
class (Qwen3VLProcessor) and a Qwen-style chat template, saved like a real
checkpoint. Every code path (load, LoRA, SFT, generation, GRPO) can then be
exercised without downloading 4 GB of weights.

    python -m folioocr.model.tiny --out tests/tiny-model
"""
from __future__ import annotations

import argparse
from pathlib import Path

SPECIALS = ["<|endoftext|>", "<|im_start|>", "<|im_end|>", "<|vision_start|>", "<|vision_end|>",
            "<|image_pad|>", "<|video_pad|>", "<think>", "</think>"]

# Mirrors the structure of Qwen's template: system-less user turn with an image
# and text, and the empty think block that enable_thinking=False adds.
CHAT_TEMPLATE = (
    "{% for message in messages %}<|im_start|>{{ message['role'] }}\n"
    "{% if message['content'] is string %}{{ message['content'] }}{% else %}"
    "{% for item in message['content'] %}{% if item['type'] == 'image' %}<|vision_start|><|image_pad|><|vision_end|>"
    "{% elif item['type'] == 'text' %}{{ item['text'] }}{% endif %}{% endfor %}{% endif %}<|im_end|>\n{% endfor %}"
    "{% if add_generation_prompt %}<|im_start|>assistant\n"
    "{% if enable_thinking is defined and enable_thinking is false %}<think>\n\n</think>\n\n{% endif %}{% endif %}"
)


def _byte_vocab():
    bs = list(range(ord("!"), ord("~") + 1)) + list(range(ord("¡"), ord("¬") + 1)) + list(range(ord("®"), ord("ÿ") + 1))
    cs = bs[:]
    n = 0
    for b in range(256):
        if b not in bs:
            bs.append(b)
            cs.append(256 + n)
            n += 1
    return {chr(c): i for i, c in enumerate(cs)}


def build_tiny(out: str, hidden: int = 64, layers: int = 4, seed: int = 0) -> str:
    import torch
    from transformers import (Qwen2VLImageProcessor, Qwen3_5Config, Qwen3_5ForConditionalGeneration,
                              Qwen3VLProcessor, Qwen3VLVideoProcessor)
    from transformers.models.qwen3_5.tokenization_qwen3_5 import Qwen3_5Tokenizer

    torch.manual_seed(seed)
    tok = Qwen3_5Tokenizer(vocab=_byte_vocab(), merges=[])
    tok.add_special_tokens({"additional_special_tokens": SPECIALS[1:]})
    tok.pad_token = "<|endoftext|>"
    ids = {t: tok.convert_tokens_to_ids(t) for t in SPECIALS}
    ip = Qwen2VLImageProcessor(patch_size=16, temporal_patch_size=2, merge_size=2, min_pixels=2048, max_pixels=1 << 22)
    proc = Qwen3VLProcessor(image_processor=ip, tokenizer=tok, video_processor=Qwen3VLVideoProcessor(),
                            chat_template=CHAT_TEMPLATE)
    cfg = Qwen3_5Config(
        text_config=dict(vocab_size=len(tok), hidden_size=hidden, intermediate_size=hidden * 2, num_hidden_layers=layers,
                         num_attention_heads=2, num_key_value_heads=1, head_dim=32, linear_key_head_dim=16,
                         linear_value_head_dim=16, linear_num_key_heads=2, linear_num_value_heads=4,
                         max_position_embeddings=32768, eos_token_id=ids["<|im_end|>"], pad_token_id=ids["<|endoftext|>"],
                         rope_parameters={"rope_type": "default", "rope_theta": 10000.0, "partial_rotary_factor": 0.25,
                                          "mrope_section": [2, 1, 1], "mrope_interleaved": True}),
        vision_config=dict(depth=2, hidden_size=hidden, intermediate_size=hidden * 2, num_heads=2, patch_size=16,
                           spatial_merge_size=2, temporal_patch_size=2, out_hidden_size=hidden, num_position_embeddings=2304),
        image_token_id=ids["<|image_pad|>"], video_token_id=ids["<|video_pad|>"],
        vision_start_token_id=ids["<|vision_start|>"], vision_end_token_id=ids["<|vision_end|>"],
    )
    model = Qwen3_5ForConditionalGeneration(cfg)
    model.generation_config.eos_token_id = [ids["<|im_end|>"], ids["<|endoftext|>"]]
    model.generation_config.pad_token_id = ids["<|endoftext|>"]
    Path(out).mkdir(parents=True, exist_ok=True)
    model.save_pretrained(out)
    proc.save_pretrained(out)
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="tests/tiny-model")
    ap.add_argument("--hidden", type=int, default=64)
    ap.add_argument("--layers", type=int, default=4)
    args = ap.parse_args()
    print(build_tiny(args.out, args.hidden, args.layers))


if __name__ == "__main__":
    main()
