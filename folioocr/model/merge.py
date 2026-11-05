"""Phase 4: fold LoRA adapters into the base weights, and quantise the result.

``merge`` adds ``sum_i w_i * delta_i`` to every adapted weight, where delta_i is
adapter i's exact ``scale * B @ A`` (peft's own ``get_delta_weight``). One
adapter with weight 1 is an ordinary merge; several adapters with weights
summing to 1 average the fine-tunes in weight space (the plan: "average the
best SFT and RL checkpoints if dev scores improve"; LightOnOCR-2 merged
checkpoints this way). Averaging the adapters' A and B matrices instead would
not average the fine-tunes: (B1 + B2)(A1 + A2) != B1 A1 + B2 A2.

The merge runs on the CPU in the base checkpoint's precision (``--dtype
float32`` merges without rounding the deltas, if RAM allows). A QLoRA adapter
trained on the 4-bit base is merged into the 16-bit base, as usual.
"""
from __future__ import annotations

import json
import os
from typing import Optional, Sequence


def merge(base: str, adapters: Sequence[str], out: str, weights: Optional[Sequence[float]] = None,
          dtype: str = "auto") -> dict:
    import torch
    from peft import PeftModel
    from peft.tuners.lora import LoraLayer
    from transformers import AutoModelForImageTextToText

    from .load import load_processor

    adapters = list(adapters)
    weights = list(weights) if weights else [1.0 / len(adapters)] * len(adapters)
    if len(weights) != len(adapters):
        raise ValueError("one weight per adapter")
    for a in adapters:
        with open(os.path.join(a, "adapter_config.json")) as f:
            cfg = json.load(f)
        if cfg.get("use_dora"):
            raise ValueError(f"{a}: DoRA adapters are not supported by this merge")
    td = None if dtype == "auto" else {"float32": torch.float32, "bfloat16": torch.bfloat16,
                                       "float16": torch.float16}[dtype]
    model = AutoModelForImageTextToText.from_pretrained(base, dtype=td or "auto")
    pm = PeftModel.from_pretrained(model, adapters[0], adapter_name="a0")
    for i, a in enumerate(adapters[1:], 1):
        pm.load_adapter(a, adapter_name=f"a{i}")
    names = [f"a{i}" for i in range(len(adapters))]
    n_layers = 0
    with torch.no_grad():
        for module in pm.modules():
            if not isinstance(module, LoraLayer):
                continue
            base_w = module.get_base_layer().weight
            delta = None
            for nm, w in zip(names, weights):
                if nm not in module.lora_A:
                    continue
                d = module.get_delta_weight(nm).float() * w
                delta = d if delta is None else delta + d
            if delta is not None:
                base_w.data = (base_w.data.float() + delta.to(base_w.device)).to(base_w.dtype)
                n_layers += 1
    merged = pm.unload()                      # drop the LoRA layers; their deltas are already in the weights
    os.makedirs(out, exist_ok=True)
    merged.save_pretrained(out)
    load_processor(base).save_pretrained(out)
    info = {"base": base, "adapters": adapters, "weights": weights, "layers_merged": n_layers,
            "dtype": str(next(merged.parameters()).dtype)}
    with open(os.path.join(out, "merge_info.json"), "w") as f:
        json.dump(info, f, indent=1)
    return info


def quantize(model_path: str, out: str, bits: int = 4) -> dict:
    """bitsandbytes NF4 (``bits=4``) or LLM.int8 (``bits=8``) copy of a merged model, with the
    vision tower and lm_head left in 16-bit. Needs a CUDA GPU (bitsandbytes quantises on it).
    The saved folder loads quantised by itself: ``--model <out>`` in every bench script."""
    import torch
    from transformers import AutoModelForImageTextToText, BitsAndBytesConfig

    from .load import load_processor, resolve_dtype

    if not torch.cuda.is_available():
        raise SystemExit("quantize needs a CUDA GPU (bitsandbytes); run it on Kaggle/Colab")
    dt = resolve_dtype("auto")
    if bits == 4:
        q = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_compute_dtype=dt,
                               bnb_4bit_use_double_quant=True, llm_int8_skip_modules=["visual", "lm_head"])
    elif bits == 8:
        q = BitsAndBytesConfig(load_in_8bit=True, llm_int8_skip_modules=["visual", "lm_head"])
    else:
        raise ValueError("bits must be 4 or 8")
    model = AutoModelForImageTextToText.from_pretrained(model_path, quantization_config=q, dtype=dt,
                                                        device_map={"": 0})
    model.save_pretrained(out)
    load_processor(model_path).save_pretrained(out)
    size = sum(os.path.getsize(os.path.join(out, f)) for f in os.listdir(out) if f.endswith(".safetensors"))
    info = {"source": model_path, "bits": bits, "gb_on_disk": round(size / 2**30, 2),
            "gpu_gb_loaded": round(torch.cuda.memory_allocated() / 2**30, 2)}
    with open(os.path.join(out, "quant_info.json"), "w") as f:
        json.dump(info, f, indent=1)
    return info
