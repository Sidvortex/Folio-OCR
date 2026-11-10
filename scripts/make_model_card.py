#!/usr/bin/env python3
"""Write the model card (Phase 4) from the runs and result files, so every number in it is one we measured.

    python scripts/make_model_card.py --name folio-ocr-v1 --model-dir models/folio-v1 \\
        --sft-run runs/general-t4 --rl-run runs/rl-t4 \\
        --olmocr results/olmocr/folio_v1_s300.json --olmocr-baseline results/olmocr/flash_s300.json \\
            results/olmocr/chandra2_s300.json results/olmocr/surya2_s300.json \\
        --resume results/resume/folio_v1.json --resume-baseline results/resume/flash.json results/resume/chandra2.json \\
        --holdout results/olmocr/folio_v1_holdout.json --holdout-baseline results/olmocr/flash_holdout.json \\
        --dev results/dev/folio_v1.json --speed results/speed/folio_v1_t4.json \\
        --contamination results/contamination.json

Writes <model-dir>/README.md (Hugging Face model card format). Every
comparison is a paired bootstrap on the same pages; "beats" is only written
where the 95% interval excludes zero.
"""
import argparse
import datetime
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def _load(p):
    with open(p, encoding="utf-8") as f:
        return json.load(f)


def _name(p, d=None):
    d = d or {}
    return d.get("meta", {}).get("name") or d.get("name") or os.path.splitext(os.path.basename(p))[0]


def _pct(x, digits=1):
    return "–" if x is None else f"{100 * x:.{digits}f}"


def olmocr_section(title, ours, baselines):
    from folioocr.bench.stats import paired_bootstrap
    rows = [(ours, _load(ours))] + [(b, _load(b)) for b in baselines]
    cats = sorted({c for _, d in rows for c in d.get("categories", {})})
    meta = rows[0][1].get("meta", {})
    scope = (f"a stratified sample of {meta.get('pages_scored')} pages (seed {meta.get('seed')})"
             if meta.get("sample") else "all pages")
    head = "| Model | Overall (95% CI) | " + " | ".join(cats) + " |"
    out = [f"### {title}", "",
           f"Our runs, {scope}, {meta.get('render', 'common')} rendering"
           + (f" ({meta.get('longest_side')} px longest side)" if meta.get("render") == "common" else "") + ".", "",
           head, "|" + "---|" * (2 + len(cats))]
    for p, d in rows:
        lo, hi = d.get("ci95", (None, None))
        cells = " | ".join(_pct(d.get("categories", {}).get(c, {}).get("rate")) for c in cats)
        out.append(f"| {_name(p, d)} | {_pct(d.get('overall'))} ({_pct(lo)}–{_pct(hi)}) | {cells} |")
    cmp = []
    for p, d in rows[1:]:
        try:
            r = paired_bootstrap(rows[0][1]["results"], d["results"])
        except (KeyError, ValueError):
            continue
        lo, hi = r["ci95"]
        verdict = ("beats" if r["diff"] > 0 else "is beaten by") if r["significant"] else "is not distinguishable from"
        cmp.append(f"- {_name(ours, rows[0][1])} {verdict} {_name(p, d)}: {100 * r['diff']:+.2f} "
                   f"(95% CI {100 * lo:+.2f} to {100 * hi:+.2f}, paired bootstrap over documents)")
    return out + ([""] + cmp if cmp else []) + [""]


def resume_section(ours, baselines):
    from folioocr.bench.stats import paired_bootstrap_pages
    rows = [(ours, _load(ours))] + [(b, _load(b)) for b in baselines]
    out = ["### Resume-Bench (real resumes; digital, scanned, phone photos)", "",
           "| Model | Page CER | Field accuracy | Extra text | Unit tests |", "|---|---|---|---|---|"]
    for p, d in rows:
        a = d["tiers"].get("all", {})
        out.append(f"| {_name(p, d)} | {_pct(a.get('line_cer'), 2)}% | {_pct(a.get('fields'))}% | "
                   f"{_pct(a.get('extra'))}% | {_pct(d.get('tests'))} |")
    out.append("")
    for p, d in rows[1:]:
        for metric, label in (("line_cer", "page CER"), ("fields", "field accuracy")):
            try:
                r = paired_bootstrap_pages(rows[0][1]["pages"], d["pages"], metric)
            except (KeyError, ValueError):
                continue
            lo, hi = r["ci95"]
            verdict = ("better than" if r["a_better"] else "worse than") if r["significant"] else "not distinguishable from"
            out.append(f"- {label}: {_name(ours, rows[0][1])} is {verdict} {_name(p, d)} "
                       f"({100 * r['diff']:+.2f} points, 95% CI {100 * lo:+.2f} to {100 * hi:+.2f})")
    return out + [""]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--name", required=True)
    ap.add_argument("--model-dir", default=None, help="writes README.md here")
    ap.add_argument("--out", default=None, help="or here")
    ap.add_argument("--base", default="infly/Infinity-Parser2-Flash")
    ap.add_argument("--sft-run", default=None)
    ap.add_argument("--rl-run", default=None)
    ap.add_argument("--olmocr", default=None)
    ap.add_argument("--olmocr-baseline", nargs="*", default=[])
    ap.add_argument("--resume", default=None)
    ap.add_argument("--resume-baseline", nargs="*", default=[])
    ap.add_argument("--holdout", default=None)
    ap.add_argument("--holdout-baseline", nargs="*", default=[])
    ap.add_argument("--dev", nargs="*", default=[])
    ap.add_argument("--speed", nargs="*", default=[])
    ap.add_argument("--contamination", default=None)
    args = ap.parse_args()

    from folioocr.data.mixture import data_card
    sft = _load(os.path.join(args.sft_run, "train_config.json")) if args.sft_run else None
    rl = _load(os.path.join(args.rl_run, "rl_config.json")) if args.rl_run else None
    noncommercial = bool(sft and sft.get("licences", {}).get("noncommercial"))
    lic = "cc-by-nc-sa-4.0" if noncommercial else "apache-2.0"
    today = datetime.date.today().isoformat()
    L = ["---", f"license: {lic}", f"base_model: {args.base}", "pipeline_tag: image-text-to-text",
         "tags: [ocr, document-parsing, layout, resume, folio-ocr]", "---", "", f"# {args.name}", "",
         f"A page parser fine-tuned from [{args.base}](https://huggingface.co/{args.base}) with Folio-OCR: "
         "page image in, layout JSON (boxes, categories, Markdown/HTML/LaTeX text) and Markdown out. "
         f"This card was generated on {today} from the training runs and result files; every score is our own run.",
         ""]
    if noncommercial:
        L += ["**Licence:** the training data includes a non-commercial source (see the data card), so this model "
              "is released for non-commercial use only.", ""]
    L += ["## Use", "", "```bash", f"python scripts/infer.py page.pdf --model {args.model_dir or args.name} --json", "```", "",
          "Prompts are Infinity-Parser2's own (`doc2json`, `doc2md`); generation stops on `<|im_end|>`.", ""]
    L += ["## Training", ""]
    if sft:
        L.append(f"- **SFT** (`{args.sft_run}`): profile {sft.get('profile')}, "
                 + ("full fine-tune" if sft.get("full") else f"LoRA r={sft.get('lora_r')}")
                 + f", lr {sft.get('lr')}, {sft.get('epochs')} epoch(s), max {sft.get('max_pixels')} pixels, "
                 f"task mix {sft.get('task_mix')}"
                 + (f", started from {sft['init_adapter']}" if sft.get("init_adapter") else "") + ".")
    if rl:
        L.append(f"- **RL** (`{args.rl_run}`): GRPO, group {rl.get('group')}, {rl.get('prompts_per_step')} prompts "
                 f"per step per GPU, lr {rl.get('lr')}, beta {rl.get('beta')}, {rl.get('steps')} steps max, rewards: "
                 "unit tests 0.5, text 0.2, TEDS 0.1, fields 0.1, format 0.1 (a loop scores 0).")
    L.append("")
    if sft:
        card = data_card(sft["data"], name="training data", contamination=args.contamination,
                         task_mix=sft.get("task_mix", ""))
        L += ["## Data", ""] + card.split("\n")[2:] + [""]
    L += ["## Evaluation", ""]
    if args.olmocr:
        L += olmocr_section("olmOCR-Bench", args.olmocr, args.olmocr_baseline)
    if args.resume:
        L += resume_section(args.resume, args.resume_baseline)
    if args.holdout:
        L += olmocr_section("Private holdout (fresh pages, hand-checked tests)", args.holdout, args.holdout_baseline)
    if args.dev:
        L += ["### Failure rates (dev pages)", "", "| Run | Pages | Loops | Hit token cap | Parse OK |",
              "|---|---|---|---|---|"]
        for p in args.dev:
            d = _load(p)
            L.append(f"| {_name(p, d)} | {d.get('pages')} | {_pct(d.get('loops'))}% | {_pct(d.get('truncated'))}% | "
                     f"{_pct(d.get('parse_ok'))}% |")
        L.append("")
    if args.speed:
        L += ["### Speed and memory", "", "| Run | GPU | Pages/s | Peak GPU memory | 4-bit |", "|---|---|---|---|---|"]
        for p in args.speed:
            d = _load(p)
            L.append(f"| {_name(p, d)} | {d.get('gpu')} | {d.get('pages_per_second')} | {d.get('peak_gpu_gb')} GB | "
                     f"{d.get('load_4bit')} |")
        L.append("")
    if args.contamination:
        c = _load(args.contamination)
        L += ["### Contamination", "",
              f"{c['total']['dropped']} of {c['total']['records']} training pages matched an evaluation page "
              "(image hash, text MinHash, benchmark test strings, file hash or URL) and were removed before training.",
              ""]
    L += ["## Limitations", "",
          "- Scores are from our harness and can differ from published numbers (rendering size, token caps).",
          "- Handwriting beyond notes on printed forms, charts and scripts other than Latin and the main Indic "
          "ones are out of scope.",
          "- On resumes, emails, phone numbers and dates flagged by `scripts/infer.py --crosscheck` should be checked by a "
          "person: the model can still misread digits.", ""]
    text = "\n".join(L)
    dst = args.out or (os.path.join(args.model_dir, "README.md") if args.model_dir else None)
    if dst:
        os.makedirs(os.path.dirname(os.path.abspath(dst)), exist_ok=True)
        with open(dst, "w", encoding="utf-8") as f:
            f.write(text)
        print(f"-> {dst}")
    else:
        print(text)


if __name__ == "__main__":
    main()
