"""Detect generation loops (the model repeating the same span until the token cap)."""
from __future__ import annotations

import math


def tail_period(text: str, max_period: int = 300, min_span: int = 240, min_repeats: int = 4) -> int:
    """Smallest period p such that the tail of ``text`` is the same p characters
    repeated (at least ``min_repeats`` times and ``min_span`` characters); 0 if none."""
    t = text[-max(min_span, max_period * min_repeats) * 2:]
    n = len(t)
    for p in range(1, min(max_period, n // 2) + 1):
        reps = max(min_repeats, math.ceil(min_span / p))
        if p * reps > n:
            break
        unit = t[n - p:]
        if t[n - p * reps:] == unit * reps:
            return p
    return 0


def is_looping(text: str, **kw) -> bool:
    return tail_period(text, **kw) > 0


def strip_loop(text: str, **kw) -> str:
    """Drop the repeated tail, keeping one copy of the repeated unit."""
    p = tail_period(text, **kw)
    if not p:
        return text
    unit = text[-p:]
    end = len(text)
    while end - p >= 0 and text[end - p:end] == unit:
        end -= p
    return text[:end + p]


def make_stopping_criteria(tokenizer, check_every: int = 48, window_tokens: int = 1200, **kw):
    """A ``StoppingCriteria`` that ends a sequence once its tail is looping.
    Returns a per-sequence bool tensor (transformers >= 4.39)."""
    import torch
    from transformers import StoppingCriteria

    class LoopStop(StoppingCriteria):
        def __init__(self):
            self.prompt_len = None
            self.flagged = set()

        def __call__(self, input_ids, scores, **unused):
            if self.prompt_len is None:
                self.prompt_len = input_ids.shape[1] - 1
            done = torch.zeros(input_ids.shape[0], dtype=torch.bool, device=input_ids.device)
            gen = input_ids.shape[1] - self.prompt_len
            if gen < check_every or gen % check_every:
                for i in self.flagged:
                    done[i] = True
                return done
            for i in range(input_ids.shape[0]):
                if i in self.flagged:
                    done[i] = True
                    continue
                tail = tokenizer.decode(input_ids[i, max(self.prompt_len, input_ids.shape[1] - window_tokens):],
                                        skip_special_tokens=True)
                if is_looping(tail, **kw):
                    self.flagged.add(i)
                    done[i] = True
            return done

    return LoopStop()
