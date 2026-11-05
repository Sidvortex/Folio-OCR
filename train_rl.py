#!/usr/bin/env python3
"""GRPO on top of an SFT adapter (Phase 3), sized for free GPUs by default.

    python train_rl.py --profile t4 --adapter runs/general-t4/best \\
        --data data/rl_regions/labels.jsonl:3 data/rl_synthmix/labels.jsonl:1 --out runs/rl-t4 --steps 200 --resume

    # two T4s on Kaggle: each GPU samples its own items, gradients are averaged
    torchrun --nproc_per_node 2 train_rl.py --profile t4 ...

Watch ``<out>/train_log.jsonl`` (reward and its parts, skipped groups, loops),
``<out>/dev_log.jsonl`` (dev reward, tests, CER every --eval-every steps; the run
stops after --patience flat evaluations) and read ``<out>/samples/step_*.jsonl``
for reward hacking. ``<out>/best`` holds the adapter with the best dev reward.
"""
import argparse
import datetime
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

RL_PROFILES = {
    "t4": dict(load_4bit=True, dtype="float16", group=4, prompts_per_step=2, max_new_tokens=1024,
               max_pixels=1_000_000, lr=2e-5, eval_items=16),
    "a100": dict(load_4bit=False, dtype="bfloat16", group=8, prompts_per_step=4, max_new_tokens=2048,
                 max_pixels=2_000_000, lr=2e-5, eval_items=48),
    "cpu-test": dict(load_4bit=False, dtype="float32", group=4, prompts_per_step=2, max_new_tokens=32,
                     max_pixels=40_000, lr=1e-3, eval_items=2),
}


def build_args(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--profile", default="t4", choices=sorted(RL_PROFILES))
    ap.add_argument("--model", default="infly/Infinity-Parser2-Flash")
    ap.add_argument("--adapter", default=None, help="the SFT adapter to start from (also the KL reference)")
    ap.add_argument("--data", nargs="+", required=True, help="RL item files (labels with tests), path[:weight]")
    ap.add_argument("--dev", default=None, help="dev items (default: split=dev items of --data)")
    ap.add_argument("--out", required=True)
    ap.add_argument("--steps", type=int, default=200)
    ap.add_argument("--task", default="doc2json", choices=["doc2json", "doc2md"])
    ap.add_argument("--beta", type=float, default=0.0)
    ap.add_argument("--scale", default="group", choices=["group", "none"])
    ap.add_argument("--temperature", type=float, default=1.0)
    ap.add_argument("--top-p", type=float, default=1.0)
    ap.add_argument("--eval-every", type=int, default=50)
    ap.add_argument("--dump-every", type=int, default=100)
    ap.add_argument("--dump-n", type=int, default=50)
    ap.add_argument("--save-every", type=int, default=25)
    ap.add_argument("--patience", type=int, default=4)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--no-fence", action="store_true")
    ap.add_argument("--lora-r", type=int, default=32, help="only without --adapter")
    ap.add_argument("--lora-alpha", type=int, default=64)
    ap.add_argument("--load-4bit", dest="load_4bit", action="store_true", default=None)
    ap.add_argument("--no-load-4bit", dest="load_4bit", action="store_false")
    for k, typ in (("dtype", str), ("group", int), ("prompts_per_step", int), ("max_new_tokens", int),
                   ("max_pixels", int), ("lr", float), ("eval_items", int)):
        ap.add_argument(f"--{k.replace('_', '-')}", dest=k, type=typ, default=None)
    args = ap.parse_args(argv)
    for k, v in RL_PROFILES[args.profile].items():
        if getattr(args, k, None) is None:
            setattr(args, k, v)
    return args


def main(argv=None) -> dict:
    args = build_args(argv)
    import torch
    from folioocr.data.dataset import parse_sources, read_labels
    from folioocr.data.mixture import split_spec
    from folioocr.model.load import DEFAULT_BASE, ModelConfig, add_lora, load_model, resolve_dtype
    from folioocr.rl.grpo import GRPOConfig, GRPOTrainer

    world = int(os.environ.get("WORLD_SIZE", 1))
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    cuda = torch.cuda.is_available()
    if world > 1:
        import torch.distributed as dist
        if cuda:
            torch.cuda.set_device(local_rank)
        dist.init_process_group("nccl" if cuda else "gloo", timeout=datetime.timedelta(hours=2))
    if args.load_4bit and not cuda:
        print("no GPU: loading in full precision", file=sys.stderr)
        args.load_4bit = False
    os.makedirs(args.out, exist_ok=True)
    if int(os.environ.get("RANK", 0)) == 0:
        with open(os.path.join(args.out, "rl_config.json"), "w") as f:
            json.dump(vars(args), f, indent=2)
    mcfg = ModelConfig(model=args.model or DEFAULT_BASE, adapter=args.adapter, dtype=args.dtype,
                       load_4bit=args.load_4bit, device_map={"": local_rank} if cuda else None)
    model, processor = load_model(mcfg, for_training=True)
    if args.adapter is None:
        model = add_lora(model, r=args.lora_r, alpha=args.lora_alpha, dropout=0.0, quantized=args.load_4bit)
    for p in model.parameters():                 # fp32 master weights for what trains
        if p.requires_grad and p.dtype != torch.float32:
            p.data = p.data.float()
    sources = parse_sources(args.data, split="train")
    if args.dev:
        dev = read_labels(args.dev, "dev") or read_labels(args.dev)
    else:
        dev = [r for spec in args.data for r in read_labels(split_spec(spec)[0], "dev")]
    cfg = GRPOConfig(group=args.group, prompts_per_step=args.prompts_per_step, steps=args.steps, lr=args.lr,
                     temperature=args.temperature, top_p=args.top_p, beta=args.beta, scale=args.scale,
                     max_new_tokens=args.max_new_tokens, max_pixels=args.max_pixels, eval_every=args.eval_every,
                     eval_items=args.eval_items, dump_every=args.dump_every, dump_n=args.dump_n,
                     save_every=args.save_every, patience=args.patience, seed=args.seed, task=args.task)
    trainer = GRPOTrainer(model, processor, sources, cfg, args.out, dev=dev, ref_adapter=args.adapter,
                          fence=not args.no_fence, amp_dtype=resolve_dtype(args.dtype) if cuda else None)
    print(f"RL items: {sum(len(r) for r, _ in sources)} train, {len(dev)} dev; group {cfg.group} x "
          f"{cfg.prompts_per_step} prompts per step per GPU", flush=True)
    res = trainer.train(resume=args.resume)
    print(json.dumps(res))
    if world > 1:
        import torch.distributed as dist
        dist.destroy_process_group()
    return res


if __name__ == "__main__":
    main()
