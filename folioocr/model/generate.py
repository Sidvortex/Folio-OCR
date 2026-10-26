"""Batched page parsing with a loaded model.

Inputs are built exactly like Infinity-Parser2's transformers backend: resize
with qwen-vl-utils' smart_resize (``imaging.fit_image``), chat template with
``enable_thinking=False``, processor called with ``do_resize=False``. Outputs
are parsed into layout blocks; a generation that starts looping is stopped
early and retried once with mild sampling.
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import List, Optional, Sequence

from PIL import Image

from .. import prompts
from ..imaging import MIN_PIXELS, fit_image
from ..loops import is_looping, make_stopping_criteria, strip_loop
from ..schema import Block, parse_output, to_markdown
from .load import special_ids


@dataclass
class PageResult:
    raw: str
    blocks: List[Block] = field(default_factory=list)
    markdown: str = ""
    task: str = "doc2json"
    tokens: int = 0
    looped: bool = False
    truncated: bool = False
    parse_ok: bool = True
    retried: bool = False
    seconds: float = 0.0


def strip_md_fences(text: str) -> str:
    t = text.strip()
    for fence in ("```markdown", "```md", "```"):
        if t.startswith(fence):
            t = t[len(fence):].lstrip("\n")
            break
    if t.endswith("```"):
        t = t[:-3]
    return t.strip()


class PageParser:
    def __init__(self, model, processor, task: str = "doc2json", max_pixels: int = 2_000_000,
                 min_pixels: int = MIN_PIXELS, max_new_tokens: int = 8192, batch_size: int = 1,
                 loop_retry: bool = True, keep_non_body: bool = False):
        self.model, self.processor = model, processor
        self.task = task
        self.max_pixels, self.min_pixels = max_pixels, min_pixels
        self.max_new_tokens = max_new_tokens
        self.batch_size = batch_size
        self.loop_retry = loop_retry
        self.keep_non_body = keep_non_body
        self.ids = special_ids(processor)
        processor.tokenizer.padding_side = "left"

    # ------------------------------------------------------------- inputs
    def build_inputs(self, images: Sequence[Image.Image], prompt: str):
        imgs = [fit_image(im, self.max_pixels, self.min_pixels) for im in images]
        texts = []
        for im in imgs:
            msgs = [{"role": "user", "content": [{"type": "image", "image": im}, {"type": "text", "text": prompt}]}]
            texts.append(self.processor.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True,
                                                            enable_thinking=False))
        enc = self.processor(text=texts, images=imgs, do_resize=False, padding=True, return_tensors="pt")
        enc.pop("token_type_ids", None)
        return enc

    def _device(self):
        try:
            return next(p for p in self.model.parameters()).device
        except StopIteration:  # pragma: no cover
            import torch
            return torch.device("cpu")

    # ----------------------------------------------------------- generate
    def generate_raw(self, images: Sequence[Image.Image], prompt: str, sample: bool = False,
                     max_new_tokens: Optional[int] = None) -> List[dict]:
        import torch
        enc = self.build_inputs(images, prompt)
        dev = self._device()
        enc = {k: (v.to(dev) if hasattr(v, "to") else v) for k, v in enc.items()}
        stop = make_stopping_criteria(self.processor.tokenizer)
        gen_kw = dict(max_new_tokens=max_new_tokens or self.max_new_tokens, eos_token_id=self.ids["eos"],
                      pad_token_id=self.ids["pad"], stopping_criteria=[stop])
        if sample:
            gen_kw.update(do_sample=True, temperature=0.3, top_p=0.9, repetition_penalty=1.05)
        else:
            gen_kw.update(do_sample=False, temperature=None, top_p=None, top_k=None)
        t0 = time.time()
        with torch.inference_mode():
            out = self.model.generate(**enc, **gen_kw)
        dt = (time.time() - t0) / max(1, len(images))
        n_in = enc["input_ids"].shape[1]
        res = []
        for i in range(len(images)):
            new = out[i, n_in:]
            eos_hit = any(int(t) in self.ids["eos"] for t in new)
            n_tok = int((new != self.ids["pad"]).sum()) if self.ids["pad"] is not None else len(new)
            text = self.processor.tokenizer.decode(new, skip_special_tokens=True)
            res.append({"text": text, "tokens": n_tok, "looped": i in stop.flagged or is_looping(text),
                        "truncated": not eos_hit and i not in stop.flagged, "seconds": dt})
        return res

    # -------------------------------------------------------------- parse
    def _to_result(self, g: dict, task: str) -> PageResult:
        text = strip_loop(g["text"]) if g["looped"] else g["text"]
        r = PageResult(raw=g["text"], task=task, tokens=g["tokens"], looped=g["looped"], truncated=g["truncated"],
                       seconds=g["seconds"])
        if task == "doc2json":
            pr = parse_output(text)
            r.blocks, r.parse_ok = pr.blocks, pr.ok
            r.markdown = to_markdown(pr.blocks, keep_non_body=self.keep_non_body)
        else:
            r.markdown = strip_md_fences(text)
        return r

    def parse(self, images: Sequence[Image.Image], task: Optional[str] = None) -> List[PageResult]:
        task = task or self.task
        prompt = prompts.prompt_for(task)
        results: List[PageResult] = []
        for s in range(0, len(images), self.batch_size):
            batch = list(images[s:s + self.batch_size])
            for img, g in zip(batch, self.generate_raw(batch, prompt)):
                r = self._to_result(g, task)
                if self.loop_retry and (r.looped or (task == "doc2json" and not r.parse_ok and r.truncated)):
                    g2 = self.generate_raw([img], prompt, sample=True)[0]
                    r2 = self._to_result(g2, task)
                    if not r2.looped and len(r2.markdown) >= 0.5 * len(r.markdown):
                        r2.retried = True
                        r = r2
                    else:
                        r.retried = True
                results.append(r)
        return results
