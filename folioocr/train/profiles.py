"""Hardware profiles: defaults for each kind of GPU (any value can be overridden on the command line).

t4        free Kaggle / Colab T4 (16 GB, no bfloat16): 4-bit base + LoRA, ~2k visual tokens
a100      Colab Pro / rented A100 40-80 GB: bf16 base + LoRA, ~3k visual tokens
h100-full rented H100 80 GB: full fine-tune of the language model, vision tower frozen
cpu-test  tiny model on CPU for smoke tests
"""

PROFILES = {
    "t4": dict(load_4bit=True, dtype="float16", full=False, lora_r=32, lora_alpha=64, batch=1, grad_accum=8,
               max_pixels=2_000_000, max_length=6144, lr=1e-4, warmup=0.03, eval_pages=24, max_new_tokens=4096),
    "a100": dict(load_4bit=False, dtype="bfloat16", full=False, lora_r=64, lora_alpha=128, batch=2, grad_accum=8,
                 max_pixels=3_200_000, max_length=12288, lr=1e-4, warmup=0.03, eval_pages=48, max_new_tokens=8192),
    "h100-full": dict(load_4bit=False, dtype="bfloat16", full=True, lora_r=0, lora_alpha=0, batch=4, grad_accum=4,
                      max_pixels=4_000_000, max_length=16384, lr=1e-5, warmup=0.03, eval_pages=64, max_new_tokens=12288),
    "cpu-test": dict(load_4bit=False, dtype="float32", full=False, lora_r=8, lora_alpha=16, batch=1, grad_accum=1,
                     max_pixels=200_000, max_length=3000, lr=2e-3, warmup=0.0, eval_pages=2, max_new_tokens=64),
}
