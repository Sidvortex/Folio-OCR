"""Label files -> model inputs for supervised fine-tuning.

A label file is JSONL with one page per line:

    {"id": ..., "image": "images/x.jpg", "blocks": [{"bbox": [...], "category": ..., "text": ...}, ...]}

(``make_dataset`` writes these; ``adapters`` convert public datasets to it).
Records may carry ``"markdown"`` instead of ``"blocks"``; they can only train
the ``doc2md`` task. Image paths are relative to the label file.

Each example is the user turn (image + task prompt, rendered with the model's
own chat template exactly as at inference) followed by the target and
``<|im_end|>``; only the target tokens are supervised. A target that does not
fit ``max_length`` loses blocks from the end (whole blocks, never half a JSON);
the shortened JSON's closing and the end-of-turn token then get no loss, so the
model never learns to stop before the page ends.
"""
from __future__ import annotations

import json
import os
import random
from typing import Dict, List, Optional, Sequence, Tuple

from PIL import Image

from .. import prompts
from ..imaging import MIN_PIXELS, PIXELS_PER_TOKEN, fit_image
from ..schema import Block, dump_target, to_markdown


def read_labels(path: str, split: Optional[str] = None) -> List[dict]:
    root = os.path.dirname(os.path.abspath(path))
    out = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            r = json.loads(line)
            if split and r.get("split", "train") != split:
                continue
            r["_image"] = r["image"] if os.path.isabs(r["image"]) else os.path.join(root, r["image"])
            out.append(r)
    return out


def parse_sources(specs: Sequence[str], split: Optional[str] = "train") -> List[Tuple[List[dict], float]]:
    """``path.jsonl[:weight]`` -> [(records, weight)]."""
    out = []
    for spec in specs:
        path, _, w = spec.rpartition(":") if not os.path.exists(spec) else (spec, "", "")
        if not path:
            path, w = spec, ""
        recs = read_labels(path, split)
        if not recs:
            raise ValueError(f"no records with split={split!r} in {path}")
        out.append((recs, float(w) if w else 1.0))
    return out


class PageDataset:
    """Map-style dataset over a weighted mixture of label files."""

    def __init__(self, sources: List[Tuple[List[dict], float]], processor, task_mix: Dict[str, float],
                 max_pixels: int, min_pixels: int = MIN_PIXELS, max_length: int = 8192, fence: bool = True,
                 length: Optional[int] = None, seed: int = 0):
        self.sources = [s for s in sources if s[0]]
        self.processor = processor
        self.tok = processor.tokenizer
        tot = sum(w for _, w in self.sources)
        self.weights = [w / tot for _, w in self.sources]
        self.task_mix = task_mix
        self.max_pixels, self.min_pixels = max_pixels, min_pixels
        self.max_length = max_length
        self.fence = fence
        self.length = length or sum(len(r) for r, _ in self.sources)
        self.seed = seed
        self.im_end = self.tok.convert_tokens_to_ids("<|im_end|>")
        self.dropped_blocks = 0

    def __len__(self) -> int:
        return self.length

    def pick(self, idx: int) -> Tuple[dict, str]:
        rng = random.Random(self.seed * 1_000_003 + idx)
        k = rng.choices(range(len(self.sources)), weights=self.weights)[0]
        recs = self.sources[k][0]
        rec = recs[rng.randrange(len(recs))] if len(self.sources) > 1 or self.length != len(recs) else recs[idx % len(recs)]
        tasks = list(self.task_mix)
        task = rng.choices(tasks, weights=[self.task_mix[t] for t in tasks])[0]
        if "blocks" not in rec:
            task = "doc2md"
        return rec, task

    def target(self, rec: dict, task: str, n_blocks: Optional[int] = None) -> str:
        if "blocks" not in rec:
            return rec["markdown"]
        blocks = [Block.from_dict(b) for b in rec["blocks"]][:n_blocks]
        return dump_target(blocks, fence=self.fence) if task == "doc2json" else to_markdown(blocks)

    def _prompt(self, rec: dict, task: str, max_pixels: int):
        img = fit_image(Image.open(rec["_image"]), max_pixels, self.min_pixels)
        msgs = [{"role": "user", "content": [{"type": "image", "image": img},
                                             {"type": "text", "text": prompts.prompt_for(task)}]}]
        text = self.processor.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True, enable_thinking=False)
        return self.processor(text=[text], images=[img], do_resize=False, return_tensors="pt")

    def encode(self, rec: dict, task: str) -> Dict:
        # the prompt (mostly image tokens) must leave room for a real target:
        # shrink the image rather than train on an empty answer
        mp = self.max_pixels
        for _ in range(3):
            enc = self._prompt(rec, task, mp)
            prompt_ids = enc["input_ids"][0].tolist()
            budget = self.max_length - len(prompt_ids) - 1
            if budget >= min(256, self.max_length // 4):
                break
            mp = max(self.min_pixels, mp // 2)
        n = len(rec.get("blocks", [])) or None
        text = self.target(rec, task, n)
        tgt = self.tok(text, add_special_tokens=False)["input_ids"]
        while len(tgt) > budget and n and n > 1:          # drop whole blocks from the end
            n = max(1, int(n * budget / max(1, len(tgt))) if len(tgt) > 1.5 * budget else n - 1)
            text = self.target(rec, task, n)
            tgt = self.tok(text, add_special_tokens=False)["input_ids"]
        supervise = True
        cut = False                                        # the target stops before the page does
        if len(tgt) > budget:
            if "blocks" in rec:                            # even one block does not fit: learn nothing from it
                supervise = False
            tgt = tgt[:max(0, budget)]                     # markdown-only record: plain cut
            cut = True
        if n is not None and n < len(rec.get("blocks", [])):
            self.dropped_blocks += len(rec["blocks"]) - n
            cut = True
        ids = prompt_ids + tgt + [self.im_end]
        tail = tgt + [self.im_end]
        if cut:
            # never teach "stop here" on a page that goes on: the closing of a shortened JSON
            # and the end-of-turn token get no loss
            keep = len(tgt)
            if task == "doc2json" and "blocks" in rec and len(tgt) <= budget:
                close = text.rfind("\n]")
                if close > 0:
                    keep = max(0, len(self.tok(text[:close], add_special_tokens=False)["input_ids"]) - 1)
            tail = tgt[:keep] + [-100] * (len(tgt) - keep) + [-100]
        labels = [-100] * len(prompt_ids) + (tail if supervise else [-100] * (len(tgt) + 1))
        out = {"input_ids": ids, "labels": labels, "pixel_values": enc["pixel_values"],
               "image_grid_thw": enc["image_grid_thw"], "id": rec.get("id", ""), "task": task}
        if "mm_token_type_ids" in enc:
            out["mm_token_type_ids"] = enc["mm_token_type_ids"][0].tolist() + [0] * (len(tgt) + 1)
        return out

    def __getitem__(self, idx: int) -> Dict:
        rec, task = self.pick(idx)
        return self.encode(rec, task)


class Collator:
    """Right-pads token sequences; concatenates image patches in batch order."""

    def __init__(self, pad_id: int):
        self.pad_id = pad_id

    def __call__(self, batch: List[Dict]) -> Dict:
        import torch
        n = max(len(b["input_ids"]) for b in batch)
        def pad(seq, v):
            return seq + [v] * (n - len(seq))
        out = {
            "input_ids": torch.tensor([pad(b["input_ids"], self.pad_id) for b in batch], dtype=torch.long),
            "labels": torch.tensor([pad(b["labels"], -100) for b in batch], dtype=torch.long),
            "attention_mask": torch.tensor([[1] * len(b["input_ids"]) + [0] * (n - len(b["input_ids"])) for b in batch],
                                           dtype=torch.long),
            "pixel_values": torch.cat([b["pixel_values"] for b in batch], 0),
            "image_grid_thw": torch.cat([b["image_grid_thw"] for b in batch], 0),
        }
        if "mm_token_type_ids" in batch[0]:
            out["mm_token_type_ids"] = torch.tensor([pad(b["mm_token_type_ids"], 0) for b in batch], dtype=torch.long)
        return out


def visual_token_budget(max_pixels: int) -> int:
    return max_pixels // PIXELS_PER_TOKEN
