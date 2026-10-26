"""Load the base model (Infinity-Parser2-Flash by default) with an optional LoRA adapter.

T4 GPUs (free Kaggle/Colab) have no bfloat16: ``dtype="auto"`` picks float16
there, bfloat16 on Ampere and newer, float32 on CPU. ``load_4bit`` quantises the
language model with bitsandbytes NF4 (the vision tower and lm_head stay in
16-bit), which is how a 2B model trains on a 16 GB T4.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

DEFAULT_BASE = "infly/Infinity-Parser2-Flash"

# LoRA targets: every linear layer of the language model (full-attention,
# Gated-DeltaNet linear-attention and MLP projections), never the vision tower.
LORA_TARGET_REGEX = (r".*language_model.*\.(q_proj|k_proj|v_proj|o_proj|in_proj_qkv|in_proj_z|out_proj|"
                     r"gate_proj|up_proj|down_proj)$")


@dataclass
class ModelConfig:
    model: str = DEFAULT_BASE
    adapter: Optional[str] = None
    dtype: str = "auto"                  # auto | float16 | bfloat16 | float32
    load_4bit: bool = False
    device_map: Optional[str] = "auto"   # "auto", "cpu", "cuda:0", or None (training on one GPU)
    attn: Optional[str] = "sdpa"


def resolve_dtype(name: str):
    import torch
    if name == "auto":
        if torch.cuda.is_available():
            return torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
        return torch.float32
    return {"float16": torch.float16, "fp16": torch.float16, "bfloat16": torch.bfloat16, "bf16": torch.bfloat16,
            "float32": torch.float32, "fp32": torch.float32}[name]


def load_processor(path: str):
    from transformers import AutoProcessor
    proc = AutoProcessor.from_pretrained(path)
    tok = proc.tokenizer
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    return proc


def load_model(cfg: ModelConfig, for_training: bool = False) -> Tuple[object, object]:
    import torch
    from transformers import AutoModelForImageTextToText

    dtype = resolve_dtype(cfg.dtype)
    kwargs = {"dtype": dtype}
    if cfg.attn:
        kwargs["attn_implementation"] = cfg.attn
    if cfg.load_4bit:
        from transformers import BitsAndBytesConfig
        kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_compute_dtype=dtype,
            bnb_4bit_use_double_quant=True, llm_int8_skip_modules=["visual", "lm_head"])
    if cfg.device_map and torch.cuda.is_available():
        kwargs["device_map"] = cfg.device_map
    model = AutoModelForImageTextToText.from_pretrained(cfg.model, **kwargs)
    processor = load_processor(cfg.model)
    if cfg.adapter:
        from peft import PeftModel
        model = PeftModel.from_pretrained(model, cfg.adapter, is_trainable=for_training)
    if not for_training:
        model.eval()
    return model, processor


def add_lora(model, r: int = 64, alpha: int = 128, dropout: float = 0.05, target_regex: str = LORA_TARGET_REGEX,
             quantized: bool = False, gradient_checkpointing: bool = True):
    from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
    if quantized:
        model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=gradient_checkpointing,
                                                gradient_checkpointing_kwargs={"use_reentrant": False})
    elif gradient_checkpointing:
        model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
        model.enable_input_require_grads()
    cfg = LoraConfig(r=r, lora_alpha=alpha, lora_dropout=dropout, target_modules=target_regex, bias="none",
                     task_type="CAUSAL_LM")
    return get_peft_model(model, cfg)


def freeze_vision(model) -> int:
    """Freeze the vision tower (full fine-tuning keeps it frozen by default). Returns params frozen."""
    n = 0
    for name, p in model.named_parameters():
        if ".visual." in name or name.startswith("visual.") or "model.visual" in name:
            if p.requires_grad:
                p.requires_grad_(False)
                n += p.numel()
    return n


def special_ids(processor) -> dict:
    """Stop on both <|im_end|> (end of the assistant turn) and <|endoftext|>,
    like Chandra's and Infinity-Parser2's generation code."""
    tok = processor.tokenizer

    def tid(t):
        i = tok.convert_tokens_to_ids(t)
        return i if isinstance(i, int) and i >= 0 and tok.convert_ids_to_tokens(i) == t else None

    im_end, eot = tid("<|im_end|>"), tid("<|endoftext|>")
    eos = {i for i in (im_end, eot, tok.eos_token_id) if isinstance(i, int) and i >= 0}
    return {"im_end": im_end, "eos": sorted(eos), "pad": tok.pad_token_id if tok.pad_token_id is not None else eot}
