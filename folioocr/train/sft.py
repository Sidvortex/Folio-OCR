"""Supervised fine-tuning (Phase 1 resume model, Phase 2 general SFT).

    python scripts/train_sft.py --profile t4 \\
        --data data/resumes/labels.jsonl:3 data/general/labels.jsonl:1 \\
        --dev data/resumes/labels.jsonl --dev-tests data/resumes/tests.jsonl \\
        --out runs/resume-t4 --epochs 1

``--data`` takes label files with optional mixture weights (``path:weight``);
pages whose ``split`` is ``dev`` are held out automatically. Every
``--eval-every`` optimizer steps the model parses ``--eval-pages`` dev pages and
the best adapter (unit-test score, or 1 - CER without tests) is kept in
``<out>/best``. Checkpoints go to ``<out>/checkpoint-*`` and ``--resume``
continues from the newest one (free GPU sessions end after 12 hours).
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
from pathlib import Path


def build_args(argv=None):
    from .profiles import PROFILES
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--profile", default="t4", choices=sorted(PROFILES))
    ap.add_argument("--model", default=None, help="base model id or path (default: Infinity-Parser2-Flash)")
    ap.add_argument("--init-adapter", default=None, help="start from an existing LoRA adapter")
    ap.add_argument("--data", nargs="+", default=None, help="label files, optionally path:weight")
    ap.add_argument("--mixture", default=None, help="mixtures/*.json: label files, shares and task mix")
    ap.add_argument("--allow-noncommercial", action="store_true",
                    help="train on a source whose licence forbids commercial use (e.g. Infinity-Doc2-5M)")
    ap.add_argument("--dev", default=None, help="label file for dev evaluation (split=dev pages are used)")
    ap.add_argument("--dev-tests", default=None, help="tests.jsonl for the dev pages")
    ap.add_argument("--out", required=True)
    ap.add_argument("--task-mix", default=None, help="default doc2json=0.8,doc2md=0.2 (or the mixture's)")
    ap.add_argument("--epochs", type=float, default=1.0)
    ap.add_argument("--max-steps", type=int, default=-1)
    ap.add_argument("--samples-per-epoch", type=int, default=None, help="epoch size for weighted mixtures")
    ap.add_argument("--eval-every", type=int, default=200)
    ap.add_argument("--save-every", type=int, default=200)
    ap.add_argument("--workers", type=int, default=2)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--no-fence", action="store_true", help="train doc2json targets without ```json fences")
    for k in ("load_4bit", "full"):
        ap.add_argument(f"--{k.replace('_', '-')}", dest=k, action="store_true", default=None)
        ap.add_argument(f"--no-{k.replace('_', '-')}", dest=k, action="store_false")
    for k, typ in (("dtype", str), ("lora_r", int), ("lora_alpha", int), ("batch", int), ("grad_accum", int),
                   ("max_pixels", int), ("max_length", int), ("lr", float), ("warmup", float), ("eval_pages", int),
                   ("max_new_tokens", int)):
        ap.add_argument(f"--{k.replace('_', '-')}", dest=k, type=typ, default=None)
    args = ap.parse_args(argv)
    if not args.data and not args.mixture:
        ap.error("--data or --mixture is required")
    if args.mixture:
        from ..data.mixture import load_mixture
        mix = load_mixture(args.mixture)
        args.data = (args.data or []) + mix.specs()
        args.task_mix = args.task_mix or mix.task_mix
    args.task_mix = args.task_mix or "doc2json=0.8,doc2md=0.2"
    from ..data.mixture import licences
    lic = licences(args.data)
    if lic["noncommercial"] and not args.allow_noncommercial:
        ap.error("non-commercial source(s) in the training data: " + ", ".join(lic["noncommercial"])
                 + "; pass --allow-noncommercial to train anyway (the model is then non-commercial too)")
    for u in lic["unknown"]:
        print(f"warning: {u} has no source.json (licence unknown)", file=sys.stderr)
    args.licences = lic
    prof = PROFILES[args.profile]
    for k, v in prof.items():
        if getattr(args, k, None) is None:
            setattr(args, k, v)
    return args


def main(argv=None) -> None:
    args = build_args(argv)
    import torch
    from transformers import Trainer, TrainerCallback, TrainingArguments

    from ..bench.devset import evaluate_records, headline
    from ..bench.unittests import load_tests_jsonl
    from ..data.dataset import Collator, PageDataset, parse_sources, read_labels
    from ..model.generate import PageParser
    from ..model.load import DEFAULT_BASE, ModelConfig, add_lora, freeze_vision, load_model
    from .losses import chunked_causal_lm_loss

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / "train_config.json").write_text(json.dumps(vars(args), indent=2))
    cuda = torch.cuda.is_available()
    if args.load_4bit and not cuda:
        print("no GPU: 4-bit loading needs CUDA, loading in full precision instead", file=sys.stderr)
        args.load_4bit = False

    # ------------------------------------------------------------- model
    local_rank = int(os.environ.get("LOCAL_RANK", -1))
    device_map = ({"": local_rank} if local_rank >= 0 else {"": 0}) if cuda else None
    mcfg = ModelConfig(model=args.model or DEFAULT_BASE, adapter=args.init_adapter, dtype=args.dtype,
                       load_4bit=args.load_4bit, device_map=device_map)
    model, processor = load_model(mcfg, for_training=True)
    model.config.use_cache = False
    if args.full:
        n = freeze_vision(model)
        model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
        model.enable_input_require_grads()
        print(f"full fine-tune; vision tower frozen ({n / 1e6:.0f}M params)")
    elif args.init_adapter is None:
        model = add_lora(model, r=args.lora_r, alpha=args.lora_alpha, quantized=args.load_4bit)
    for p in model.parameters():                 # fp16 AMP needs fp32 master weights for what trains
        if p.requires_grad and p.dtype != torch.float32:
            p.data = p.data.float()
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"trainable parameters: {trainable / 1e6:.1f}M")

    # -------------------------------------------------------------- data
    task_mix = {k: float(v) for k, v in (x.split("=") for x in args.task_mix.split(","))}
    sources = parse_sources(args.data, split="train")
    length = args.samples_per_epoch or sum(len(r) for r, _ in sources)
    train_ds = PageDataset(sources, processor, task_mix, max_pixels=args.max_pixels, max_length=args.max_length,
                           fence=not args.no_fence, length=length, seed=args.seed)
    print(f"train pages: {sum(len(r) for r, _ in sources)} in {len(sources)} source(s); epoch = {length} samples")
    dev = read_labels(args.dev, split="dev") if args.dev else []
    dev_tests = load_tests_jsonl(args.dev_tests) if args.dev_tests else None
    if args.dev and not dev:
        print(f"warning: no split=dev pages in {args.dev}; dev evaluation disabled", file=sys.stderr)

    # ----------------------------------------------------------- trainer
    class FolioTrainer(Trainer):
        def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
            n_items = int(num_items_in_batch) if num_items_in_batch is not None else None
            loss, _ = chunked_causal_lm_loss(model, inputs, num_items=n_items)
            return (loss, None) if return_outputs else loss

    class DevEval(TrainerCallback):
        def __init__(self):
            self.best = -math.inf
            self.log = out / "dev_log.jsonl"
            self.last_step = -1

        def run(self, model, step):
            if not dev or step == self.last_step:
                return
            self.last_step = step
            was = model.training
            model.eval()
            parser = PageParser(model, processor, max_pixels=args.max_pixels, max_new_tokens=args.max_new_tokens,
                                batch_size=1, loop_retry=False)
            t0 = time.time()
            m = evaluate_records(parser, dev, dev_tests, limit=args.eval_pages)
            m.pop("_outputs", None)
            score = m["tests"] if m.get("tests") is not None else 1.0 - m["cer"]
            rec = {"step": step, "score": score, **m, "eval_seconds": round(time.time() - t0, 1)}
            with open(self.log, "a") as f:
                f.write(json.dumps(rec) + "\n")
            print(f"[dev @ step {step}] {headline(m)}", flush=True)
            if score > self.best:
                self.best = score
                model.save_pretrained(out / "best")
                processor.save_pretrained(out / "best")
                print(f"[dev] new best {score:.4f} -> {out / 'best'}", flush=True)
            if was:
                model.train()

        def on_step_end(self, a, state, control, model=None, **kw):
            # with several GPUs the other ranks would wait at the next all-reduce while rank 0
            # generates; evaluate only at the end there
            if (state.is_world_process_zero and a.world_size == 1 and args.eval_every > 0
                    and state.global_step % args.eval_every == 0):
                self.run(model, state.global_step)

        def on_train_end(self, a, state, control, model=None, **kw):
            if state.is_world_process_zero:
                self.run(model, state.global_step)

    targs = TrainingArguments(
        output_dir=str(out), per_device_train_batch_size=args.batch, gradient_accumulation_steps=args.grad_accum,
        learning_rate=args.lr, num_train_epochs=args.epochs, max_steps=args.max_steps, lr_scheduler_type="cosine",
        warmup_steps=args.warmup, weight_decay=0.0, max_grad_norm=1.0, logging_steps=10,
        save_steps=args.save_every, save_total_limit=2, fp16=cuda and args.dtype in ("float16", "fp16"),
        bf16=cuda and args.dtype in ("bfloat16", "bf16"), remove_unused_columns=False,
        dataloader_num_workers=args.workers, report_to="none", seed=args.seed, use_cpu=not cuda,
        optim="paged_adamw_8bit" if args.load_4bit else ("adamw_torch_fused" if cuda else "adamw_torch"),
        ddp_find_unused_parameters=False, dataloader_pin_memory=cuda)
    trainer = FolioTrainer(model=model, args=targs, train_dataset=train_ds,
                           data_collator=Collator(processor.tokenizer.pad_token_id), callbacks=[DevEval()])
    ckpts = sorted(out.glob("checkpoint-*"), key=lambda p: int(p.name.split("-")[-1]))
    trainer.train(resume_from_checkpoint=str(ckpts[-1]) if (args.resume and ckpts) else None)
    final = out / ("model" if args.full else "adapter")
    trainer.save_model(str(final))
    if trainer.is_world_process_zero():
        processor.save_pretrained(final)
        print(f"saved {final}  (dropped trailing blocks to fit max_length: {train_ds.dropped_blocks})")


if __name__ == "__main__":
    main()
