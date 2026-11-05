"""GRPO for the page parser, small enough for a free T4 (Phase 3).

Each step, for every sampled item (a page or a band of one, with unit tests):

1. sample ``group`` answers from the current policy (temperature 1);
2. score them with ``rewards.score`` (tests, text, TEDS, fields, format; a loop scores 0);
3. advantage = (reward - group mean) / group std (``scale="group"``) or
   reward - group mean (``scale="none"``, Dr. GRPO); a group whose rewards are
   all equal teaches nothing and is skipped (the plan: "drop pages where all
   samples score the same");
4. loss = -sum(advantage * log p(answer token)) / (answer tokens in the step),
   plus ``beta`` * KL to the starting adapter when ``beta`` > 0 (k3 estimator).

One gradient step per batch of samples, so the policy that sampled is the
policy being updated: the importance ratio is 1 and PPO clipping has nothing
to clip. Log-probabilities are computed only at answer positions, chunk by
chunk under activation checkpointing (no full-vocabulary logits for the
prompt). On two GPUs every rank samples its own items and gradients are
averaged before the step.

Guards from the plan: dev reward / tests / CER every ``eval_every`` steps with
early stop on a plateau, and ``dump_n`` sampled answers written every
``dump_every`` steps to read for reward hacking.
"""
from __future__ import annotations

import json
import math
import os
import random
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Dict, List, Optional, Sequence, Tuple

from .. import prompts
from ..schema import Block, dump_target
from .rewards import RewardConfig, score as reward_score


@dataclass
class GRPOConfig:
    group: int = 4
    prompts_per_step: int = 2
    steps: int = 200
    lr: float = 1e-5
    temperature: float = 1.0
    top_p: float = 1.0
    beta: float = 0.0
    scale: str = "group"                 # group | none
    max_new_tokens: int = 1024
    cap_factor: float = 1.6              # per-item token cap: factor x reference tokens + 64
    max_pixels: int = 1_000_000
    grad_clip: float = 1.0
    eval_every: int = 50
    eval_items: int = 24
    dump_every: int = 100
    dump_n: int = 50
    save_every: int = 25
    patience: int = 4
    seed: int = 0
    task: str = "doc2json"
    chunk: int = 512


# ------------------------------------------------------------------ helpers
def multimodal_token_ids(tokenizer) -> List[int]:
    """Image/video/vision special tokens: an answer must never contain them (a sampled
    ``<|image_pad|>`` would no longer match the image features in the training pass)."""
    toks = set(getattr(tokenizer, "all_special_tokens", [])) | set(getattr(tokenizer, "added_tokens_encoder", {}))
    out = set()
    for t in toks:
        if any(k in t for k in ("image", "video", "vision")):
            i = tokenizer.convert_tokens_to_ids(t)
            if isinstance(i, int) and i >= 0:
                out.add(i)
    return sorted(out)


def completion_logprobs(model, inputs: dict, n_prompt: int, temperature: float = 1.0, chunk: int = 512,
                        suppress: Optional[Sequence[int]] = None):
    """log p(token) for every answer token (positions >= n_prompt), with grad. ``suppress``:
    token ids masked out exactly as during sampling, so both use the same distribution."""
    import torch
    from torch.utils.checkpoint import checkpoint
    from ..train.losses import backbone_and_head
    backbone, head = backbone_and_head(model)
    out = backbone(**inputs, use_cache=False)
    hidden = out.last_hidden_state if hasattr(out, "last_hidden_state") else out[0]
    ids = inputs["input_ids"]
    h = hidden[0, n_prompt - 1:-1]
    y = ids[0, n_prompt:].to(h.device)

    sup = torch.tensor(list(suppress), dtype=torch.long, device=h.device) if suppress else None

    def piece(hh, yy):
        logits = head(hh).float() / temperature
        if sup is not None:
            logits = logits.index_fill(-1, sup, float("-inf"))
        return logits.log_softmax(-1).gather(-1, yy[:, None])[:, 0]

    parts = []
    for i in range(0, h.shape[0], chunk):
        if torch.is_grad_enabled():
            parts.append(checkpoint(piece, h[i:i + chunk], y[i:i + chunk], use_reentrant=False))
        else:
            parts.append(piece(h[i:i + chunk], y[i:i + chunk]))
    return torch.cat(parts) if parts else hidden.new_zeros(0)


def group_advantages(rewards: Sequence[float], scale: str = "group", eps: float = 1e-4) -> Optional[List[float]]:
    """None when the group's rewards are all equal (no signal)."""
    n = len(rewards)
    mean = sum(rewards) / n
    var = sum((r - mean) ** 2 for r in rewards) / n
    if var < 1e-12:
        return None
    if scale == "group":
        sd = math.sqrt(var)
        return [(r - mean) / (sd + eps) for r in rewards]
    return [r - mean for r in rewards]


def reference_tokens(item: dict, tokenizer, fence: bool = True) -> Optional[int]:
    if item.get("blocks"):
        text = dump_target([Block.from_dict(b) for b in item["blocks"]], fence=fence)
    elif item.get("markdown"):
        text = item["markdown"]
    else:
        return None
    return len(tokenizer(text, add_special_tokens=False)["input_ids"])


# ------------------------------------------------------------------ trainer
class GRPOTrainer:
    def __init__(self, model, processor, sources: List[Tuple[List[dict], float]], cfg: GRPOConfig, out: str,
                 dev: Optional[List[dict]] = None, reward_fn: Optional[Callable] = None,
                 reward_cfg: Optional[RewardConfig] = None, ref_adapter: Optional[str] = None, fence: bool = True,
                 amp_dtype=None):
        import torch
        from ..model.generate import PageParser
        self.torch = torch
        self.model, self.processor, self.cfg = model, processor, cfg
        self.tok = processor.tokenizer
        self.sources = [s for s in sources if s[0]]
        tot = sum(w for _, w in self.sources)
        self.weights = [w / tot for _, w in self.sources]
        self.dev = dev or []
        self.out = Path(out)
        self.out.mkdir(parents=True, exist_ok=True)
        self.reward_cfg = reward_cfg or RewardConfig(task=cfg.task)
        self.reward_fn = reward_fn or (lambda item, text, stopped, looped:
                                       reward_score(item, text, self.reward_cfg, stopped=stopped, looped=looped))
        self.fence = fence
        self.prompt = prompts.prompt_for(cfg.task)
        self.parser = PageParser(model, processor, task=cfg.task, max_pixels=cfg.max_pixels,
                                 max_new_tokens=cfg.max_new_tokens, batch_size=1, loop_retry=False)
        self.ids = self.parser.ids
        self.suppress = multimodal_token_ids(self.tok)
        self.device = self.parser._device()
        self.rank = int(os.environ.get("RANK", 0))
        self.world = int(os.environ.get("WORLD_SIZE", 1))
        self.rng = random.Random(cfg.seed * 1009 + self.rank)
        torch.manual_seed(cfg.seed * 1009 + self.rank)
        for m in model.modules():                  # RL wants the policy that samples == the policy trained
            if isinstance(m, torch.nn.Dropout):
                m.p = 0.0
        self.params = [p for p in model.parameters() if p.requires_grad]
        if self.world > 1:                         # every rank starts from rank 0's weights (e.g. a fresh LoRA init)
            import torch.distributed as dist
            for p in self.params:
                dist.broadcast(p.data, 0)
        self.opt = torch.optim.AdamW(self.params, lr=cfg.lr, weight_decay=0.0)
        # mixed precision on GPU (float16 on a T4, bfloat16 where supported); plain float32 on CPU
        self.amp_dtype = amp_dtype if self.device.type == "cuda" else None
        self.scaler = torch.amp.GradScaler("cuda") if self.amp_dtype == torch.float16 else None
        self.ref = None
        if cfg.beta > 0:
            if ref_adapter is None:
                raise ValueError("beta > 0 needs the starting adapter as the KL reference")
            model.load_adapter(ref_adapter, adapter_name="ref", is_trainable=False)
            self.ref = "ref"
        self.step = 0
        self.best = -math.inf
        self.bad_evals = 0
        self._ref_tokens: Dict[str, Optional[int]] = {}

    # ------------------------------------------------------------ utils
    def _autocast(self):
        import contextlib
        if self.amp_dtype is None:
            return contextlib.nullcontext()
        return self.torch.autocast("cuda", dtype=self.amp_dtype)

    def _is_main(self) -> bool:
        return self.rank == 0

    def sample_items(self) -> List[dict]:
        out = []
        for _ in range(self.cfg.prompts_per_step):
            k = self.rng.choices(range(len(self.sources)), weights=self.weights)[0]
            recs = self.sources[k][0]
            out.append(recs[self.rng.randrange(len(recs))])
        return out

    def cap_for(self, item: dict) -> int:
        key = item.get("id", "")
        if key not in self._ref_tokens:
            self._ref_tokens[key] = reference_tokens(item, self.tok, self.fence)
        n = self._ref_tokens[key]
        if n is None:
            return self.cfg.max_new_tokens
        return int(min(self.cfg.max_new_tokens, self.cfg.cap_factor * n + 64))

    def _set_adapter(self, name: str) -> None:
        self.model.set_adapter(name)
        for n_, p in self.model.named_parameters():      # set_adapter flips requires_grad; keep only the policy trainable
            if "lora_" in n_:
                p.requires_grad_(f".{name}." in n_ and name == "default")

    # ---------------------------------------------------------- rollouts
    def rollout(self, item: dict) -> dict:
        from PIL import Image
        from ..loops import is_looping, make_stopping_criteria
        torch = self.torch
        img = Image.open(item["_image"]).convert("RGB")
        enc = self.parser.build_inputs([img], self.prompt)
        dev_enc = {k: (v.to(self.device) if hasattr(v, "to") else v) for k, v in enc.items()}
        n_prompt = dev_enc["input_ids"].shape[1]
        stop = make_stopping_criteria(self.tok)
        # sample from exactly the distribution the loss differentiates: no repetition penalty,
        # no top-k, whatever the checkpoint's generation_config says
        from transformers import GenerationConfig
        gc = GenerationConfig(do_sample=True, temperature=self.cfg.temperature, top_p=self.cfg.top_p, top_k=0,
                              repetition_penalty=1.0, num_return_sequences=self.cfg.group,
                              max_new_tokens=self.cap_for(item), eos_token_id=self.ids["eos"],
                              pad_token_id=self.ids["pad"], use_cache=True, suppress_tokens=self.suppress or None)
        self.model.eval()
        with torch.no_grad(), self._autocast():
            gen = self.model.generate(**dev_enc, generation_config=gc, stopping_criteria=[stop])
        samples = []
        eos = set(self.ids["eos"])
        for g in range(gen.shape[0]):
            new = gen[g, n_prompt:].tolist()
            if g in stop.flagged:                      # stopped for looping: what was written before the padding
                end = next((i for i, t in enumerate(new) if t == self.ids["pad"]), len(new))
                cut, toks, stopped = None, new[:end], False
            else:
                cut = next((i for i, t in enumerate(new) if t in eos), None)
                stopped = cut is not None
                toks = new[:cut + 1] if stopped else [t for t in new if t != self.ids["pad"]]
            text = self.tok.decode(toks[:-1] if stopped else toks, skip_special_tokens=True)
            looped = g in stop.flagged or is_looping(text)
            r = self.reward_fn(item, text, stopped, looped)
            samples.append({"tokens": toks, "text": text, "stopped": stopped, "looped": looped, "reward": r})
        return {"item": item, "enc": enc, "n_prompt": n_prompt, "samples": samples}

    def _seq_inputs(self, ro: dict, toks: List[int]) -> dict:
        torch = self.torch
        enc = ro["enc"]
        ids = torch.cat([enc["input_ids"][0], torch.tensor(toks, dtype=enc["input_ids"].dtype)])[None]
        inp = {"input_ids": ids, "attention_mask": torch.ones_like(ids), "pixel_values": enc["pixel_values"],
               "image_grid_thw": enc["image_grid_thw"]}
        if "mm_token_type_ids" in enc:
            mm = enc["mm_token_type_ids"][0]
            inp["mm_token_type_ids"] = torch.cat([mm, torch.zeros(len(toks), dtype=mm.dtype)])[None]
        return {k: v.to(self.device) for k, v in inp.items()}

    # --------------------------------------------------------------- step
    def train_step(self) -> dict:
        torch = self.torch
        t0 = time.time()
        rollouts = [self.rollout(it) for it in self.sample_items()]
        gen_s = time.time() - t0
        work = []
        skipped = 0
        for ro in rollouts:
            vals = [s["reward"]["reward"] for s in ro["samples"]]
            adv = group_advantages(vals, self.cfg.scale)
            if adv is None:
                skipped += 1
                continue
            for s, a in zip(ro["samples"], adv):
                if a != 0.0 and s["tokens"]:
                    work.append((ro, s, a))
        n_tok = sum(len(s["tokens"]) for _, s, _ in work)
        self.model.train()
        loss_sum, kl_sum = 0.0, 0.0
        if self.ref is not None:                       # KL reference first, without grad
            for ro, s, _ in work:
                self._set_adapter(self.ref)
                with torch.no_grad(), self._autocast():
                    s["ref_lp"] = completion_logprobs(self.model, self._seq_inputs(ro, s["tokens"]), ro["n_prompt"],
                                                      self.cfg.temperature, self.cfg.chunk, self.suppress).float()
                self._set_adapter("default")
        for ro, s, a in work:
            with self._autocast():
                lp = completion_logprobs(self.model, self._seq_inputs(ro, s["tokens"]), ro["n_prompt"],
                                         self.cfg.temperature, self.cfg.chunk, self.suppress).float()
            loss = -(a * lp.sum()) / max(1, n_tok)
            if self.ref is not None:
                d = s["ref_lp"] - lp
                kl = (torch.exp(d) - d - 1).sum() / max(1, n_tok)
                loss = loss + self.cfg.beta * kl
                kl_sum += float(kl.detach())
            (self.scaler.scale(loss) if self.scaler else loss).backward()
            loss_sum += float(loss.detach())
        grad_norm = self._optimizer_step(bool(work))
        self.step += 1
        rewards = [s["reward"] for ro in rollouts for s in ro["samples"]]
        comps: Dict[str, List[float]] = {}
        for r in rewards:
            for k, v in r.get("components", {}).items():
                comps.setdefault(k, []).append(v)
        log = {"step": self.step, "reward": sum(r["reward"] for r in rewards) / len(rewards),
               "components": {k: sum(v) / len(v) for k, v in comps.items()},
               "skipped_groups": skipped, "groups": len(rollouts), "tokens": n_tok,
               "loops": sum(s["looped"] for ro in rollouts for s in ro["samples"]) / len(rewards),
               "no_stop": sum(not s["stopped"] for ro in rollouts for s in ro["samples"]) / len(rewards),
               "loss": loss_sum, "kl": kl_sum, "grad_norm": grad_norm,
               "gen_seconds": round(gen_s, 1), "seconds": round(time.time() - t0, 1)}
        self._last_rollouts = rollouts
        return log

    def _optimizer_step(self, have_grads: bool) -> float:
        torch = self.torch
        if self.world > 1:                             # every rank all-reduces, in the same order, every step
            import torch.distributed as dist
            for p in self.params:
                if p.grad is None:
                    p.grad = torch.zeros_like(p)
                dist.all_reduce(p.grad)
                p.grad /= self.world
            have_grads = True
        if not have_grads:
            self.opt.zero_grad(set_to_none=True)
            return 0.0
        if self.scaler:
            self.scaler.unscale_(self.opt)
        norm = float(torch.nn.utils.clip_grad_norm_(self.params, self.cfg.grad_clip))
        if self.scaler:
            self.scaler.step(self.opt)
            self.scaler.update()
        else:
            self.opt.step()
        self.opt.zero_grad(set_to_none=True)
        return norm

    # --------------------------------------------------------- eval etc.
    def evaluate(self) -> dict:
        from PIL import Image
        from ..textnorm import cer, markdown_to_text
        from .rewards import reference_markdown
        items = self.dev[:self.cfg.eval_items]
        self.model.eval()
        rewards, tests, err, tot = [], [], 0, 0
        for it in items:
            g = self.parser.generate_raw([Image.open(it["_image"]).convert("RGB")], self.prompt,
                                         max_new_tokens=self.cap_for(it))[0]
            r = self.reward_fn(it, g["text"], not g["truncated"] and not g["looped"], g["looped"])
            rewards.append(r["reward"])
            if "tests" in r.get("components", {}):
                tests.append(r["components"]["tests"])
            ref = reference_markdown(it)
            if ref is not None:
                from ..schema import parse_output, to_markdown
                hyp = to_markdown(parse_output(g["text"]).blocks) if self.cfg.task == "doc2json" else g["text"]
                e, n = cer(markdown_to_text(ref), markdown_to_text(hyp))
                err, tot = err + e, tot + n
        m = {"step": self.step, "items": len(items), "reward": sum(rewards) / max(1, len(rewards)),
             "tests": sum(tests) / len(tests) if tests else None, "cer": err / tot if tot else None}
        with open(self.out / "dev_log.jsonl", "a") as f:
            f.write(json.dumps(m) + "\n")
        return m

    def dump_samples(self) -> None:
        d = self.out / "samples"
        d.mkdir(exist_ok=True)
        rows = []
        for ro in getattr(self, "_last_rollouts", []):
            for s in ro["samples"]:
                rows.append({"item": ro["item"].get("id"), "reward": s["reward"].get("reward"),
                             "components": s["reward"].get("components"), "failed_tests": s["reward"].get("failed_tests"),
                             "stopped": s["stopped"], "looped": s["looped"], "text": s["text"][:4000]})
        with open(d / f"step_{self.step:06d}.jsonl", "a", encoding="utf-8") as f:
            for r in rows[:self.cfg.dump_n]:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")

    def save(self, path: Path) -> None:
        torch = self.torch
        path.mkdir(parents=True, exist_ok=True)
        self.model.save_pretrained(str(path), selected_adapters=["default"])
        self.processor.save_pretrained(str(path))
        torch.save({"step": self.step, "opt": self.opt.state_dict(),
                    "scaler": self.scaler.state_dict() if self.scaler else None, "best": self.best,
                    "bad_evals": self.bad_evals, "rng": self.rng.getstate()}, path / "rl_state.pt")

    def resume_from(self, path: Path) -> None:
        torch = self.torch
        from peft import set_peft_model_state_dict
        from safetensors.torch import load_file
        set_peft_model_state_dict(self.model, load_file(str(path / "adapter_model.safetensors")), adapter_name="default")
        st = torch.load(path / "rl_state.pt", map_location="cpu", weights_only=False)
        self.opt.load_state_dict(st["opt"])
        if self.scaler and st.get("scaler"):
            self.scaler.load_state_dict(st["scaler"])
        self.step, self.best, self.bad_evals = st["step"], st["best"], st["bad_evals"]
        self.rng.setstate(st["rng"])

    def latest_checkpoint(self) -> Optional[Path]:
        ck = sorted(self.out.glob("checkpoint-*"), key=lambda p: int(p.name.split("-")[-1]))
        return ck[-1] if ck else None

    def train(self, resume: bool = False) -> dict:
        import shutil
        if resume and (ck := self.latest_checkpoint()) is not None:
            self.resume_from(ck)
            if self._is_main():
                print(f"resumed from {ck} at step {self.step}", flush=True)
        log_f = open(self.out / "train_log.jsonl", "a") if self._is_main() else None
        stop_reason = "steps"
        while self.step < self.cfg.steps:
            log = self.train_step()
            if log_f:
                log_f.write(json.dumps(log) + "\n")
                log_f.flush()
                print(f"[rl {log['step']}] reward {log['reward']:.3f} "
                      + " ".join(f"{k} {v:.2f}" for k, v in log["components"].items())
                      + f" | skipped {log['skipped_groups']}/{log['groups']} | loops {100 * log['loops']:.0f}% "
                      f"| {log['seconds']:.0f}s", flush=True)
            if self._is_main() and self.cfg.dump_every and self.step % self.cfg.dump_every == 0:
                self.dump_samples()
            stop = False
            if self.cfg.eval_every and self.step % self.cfg.eval_every == 0 and self.dev:
                if self._is_main():
                    m = self.evaluate()
                    print(f"[dev {self.step}] reward {m['reward']:.3f} tests {m['tests']} cer {m['cer']}", flush=True)
                    if m["reward"] > self.best + 1e-4:
                        self.best, self.bad_evals = m["reward"], 0
                        self.save(self.out / "best")
                    else:
                        self.bad_evals += 1
                    stop = self.bad_evals >= self.cfg.patience
                stop = self._broadcast_flag(stop)
            if self.cfg.save_every and self.step % self.cfg.save_every == 0 and self._is_main():
                self.save(self.out / f"checkpoint-{self.step}")
                for old in sorted(self.out.glob("checkpoint-*"), key=lambda p: int(p.name.split("-")[-1]))[:-2]:
                    shutil.rmtree(old, ignore_errors=True)
            if stop:
                stop_reason = f"dev reward flat for {self.cfg.patience} evaluations"
                break
        if self._is_main():
            self.save(self.out / "adapter")
            if log_f:
                log_f.close()
        return {"steps": self.step, "best_dev_reward": self.best, "stopped": stop_reason}

    def _broadcast_flag(self, flag: bool) -> bool:
        if self.world <= 1:
            return flag
        import torch.distributed as dist
        t = self.torch.tensor([1.0 if flag else 0.0], device=self.device)
        dist.broadcast(t, 0)
        return bool(t.item())
