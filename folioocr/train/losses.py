"""Memory-lean causal-LM loss for long OCR targets.

A page target is thousands of tokens and the vocabulary has ~248k entries, so
full logits for one sequence are several GB (more than a T4 has to spare).
This loss runs the backbone, keeps only the hidden states at supervised
positions, and applies ``lm_head`` + cross-entropy chunk by chunk under
activation checkpointing: at most one chunk of logits exists at a time, and
prompt/image positions never get logits at all.
"""
from __future__ import annotations

import torch
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint


def unwrap(model):
    """DDP / PeftModel wrappers -> the *ForConditionalGeneration model (LoRA layers stay injected)."""
    m = model
    if hasattr(m, "module") and not hasattr(m, "lm_head"):
        m = m.module
    if hasattr(m, "get_base_model"):
        m = m.get_base_model()
    return m


def backbone_and_head(model):
    base = unwrap(model)
    return base.model, base.lm_head


def chunked_causal_lm_loss(model, inputs: dict, chunk: int = 1024, num_items: int | None = None):
    """Returns (loss, n_supervised). ``inputs`` holds labels plus the model inputs."""
    labels = inputs["labels"]
    fwd = {k: v for k, v in inputs.items() if k != "labels"}
    backbone, head = backbone_and_head(model)
    out = backbone(**fwd, use_cache=False)
    hidden = out.last_hidden_state if hasattr(out, "last_hidden_state") else out[0]
    h = hidden[:, :-1]
    y = labels[:, 1:].to(h.device)
    mask = y != -100
    n = int(mask.sum())
    if n == 0:
        return hidden.sum() * 0.0, 0
    hs, ys = h[mask], y[mask]

    def piece(hh, yy):
        return F.cross_entropy(head(hh).float(), yy, reduction="sum")

    total = hidden.new_zeros((), dtype=torch.float32)
    for i in range(0, hs.shape[0], chunk):
        hh, yy = hs[i:i + chunk], ys[i:i + chunk]
        if torch.is_grad_enabled():
            total = total + checkpoint(piece, hh, yy, use_reentrant=False)
        else:
            total = total + piece(hh, yy)
    denom = num_items if num_items else n
    return total / denom, n
