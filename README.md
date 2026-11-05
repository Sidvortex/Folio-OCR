# Folio-OCR

A page parser fine-tuned from **Infinity-Parser2-Flash** (2B parameters, Apache-2.0): page image in, layout
JSON and Markdown out. It is the code for the plan *"Plan: an OCR that beats Chandra, Surya and Infinity-Parser"*,
built to run on **free GPUs** (Kaggle T4 x2, free Colab T4).

On free GPUs the realistic target is **Tier A**: the best resume reader of every model tested, on real resumes. The general-benchmark tiers (beating Chandra 2 / Infinity-Parser2 on olmOCR-Bench) need roughly 300+ H100-hours. The code for them is here, but on free hardware it runs scaled down.

| Plan phase | What is here | State |
|---|---|---|
| 0. Harness and baselines | `bench_olmocr.py`, `bench_resume.py`, `bench_dev.py`, `bench_omnidoc.py`, `compare.py`, `speed_test.py` | ready |
| 1. Resume model | synthetic page renderer, `train_sft.py` (4-bit LoRA on a T4), `infer.py`, Kaggle/Colab notebook | ready |
| 2. Data engine and SFT | public-dataset adapters, mixtures, dedup | next drop |
| 3. RL with verifiable rewards | rewards, GRPO | next drop |
| 4. Compress, verify, release | merge, quantise, model card | next drop |

## What was checked, and what was not

Checked here, on CPU:

- **Tests:** 28 pass. They include an end-to-end overfit: a tiny random model of the same architecture (Qwen3.5, Gated-DeltaNet layers) trains on one page with this code's dataset, loss and LoRA. It then generates, and the parser recovers exactly the labelled blocks.
- **Labels and tests agree:** on rendered pages, ground-truth Markdown passes 100% of the 3,079 unit tests generated for them. The same pages with 3% character noise score 59–74.
- **Scorer matches olmOCR-Bench:** on a mock benchmark, this repo's scorer reproduces olmOCR's official `benchmark.py` exactly (73.0 vs 73.0, same per-category rates).
- **Prompts are Infinity-Parser2's own:** they are byte-identical to those in the `infinity_parser2` package. Image resizing reproduces qwen-vl-utils' `smart_resize` exactly (0 mismatches in 20,000 random sizes).
- **The memory-lean loss is exact:** the loss used for long pages matches the model's own loss to 1e-6.
- **Rendering:** 1.9 pages/s on 2 CPU cores, about 0.5 MB per page, about 20 blocks, 39 lines and 25 tests per page.

**Not checked, because this workspace had no GPU and could not reach Hugging Face:**

- running the real 2B model
- training speed and memory on a T4
- whether Flash wraps its JSON in a ```` ```json ```` fence (step 3a of the notebook shows you)
- the Chandra and Surya CLIs on real hardware

The notebook's first steps are these checks.

## Setup (EndeavourOS, global Python)

```bash
# rendering, scoring and the tests that don't need torch
pip install --user --break-system-packages -r requirements.txt
pip install --user --break-system-packages --no-deps olmocr==0.4.27
python -m playwright install chromium      # if Chromium won't start: sudo pacman -S chromium (pulls its libraries)
python fetch_assets.py --fonts-from ../ResumeOCR-Lite/fonts   # or just: python fetch_assets.py  (~85 MB)
python -m pytest -q tests                  # model tests skip without torch

# only if you also want to train or run the model locally (CPU works for tests, not for real training)
pip install --user --break-system-packages -r requirements-train.txt
```

Training and real inference run on Kaggle or Colab: open `notebooks/folio_phase0_phase1.ipynb` there.

## Phase 1 in five commands

```bash
# 1. data on your own computer (CPU only), then upload data/ as a Kaggle dataset
python -m folioocr.render.make_dataset --out data/resumes --n 5000 --workers 6 --mix resume=1 --prefix r --preview 20
python -m folioocr.render.make_dataset --out data/general --n 2500 --workers 6 --prefix g --start-seed 1000000 \
    --mix "article=0.35,table=0.3,math=0.15,letter=0.1,indic=0.1"

# 2. (GPU) the base model on held-out dev pages
python bench_dev.py --labels data/resumes/labels.jsonl --tests data/resumes/tests.jsonl --name flash_base --batch-size 4

# 3. (GPU) fine-tune; on Kaggle's two T4s prefix with: accelerate launch --num_processes 2 --mixed_precision fp16
python train_sft.py --profile t4 --data data/resumes/labels.jsonl:3 data/general/labels.jsonl:2 \
    --dev data/resumes/labels.jsonl --dev-tests data/resumes/tests.jsonl --out runs/resume-t4 --epochs 1 --resume

# 4. (GPU) the fine-tune on the same pages, and the paired comparison
python bench_dev.py --labels data/resumes/labels.jsonl --tests data/resumes/tests.jsonl --name resume_v1 \
    --adapter runs/resume-t4/adapter --batch-size 4
python compare.py results/dev/resume_v1.json results/dev/flash_base.json

# 5. use it
python infer.py my_resume.pdf --adapter runs/resume-t4/adapter --json
```

Synthetic dev pages show learning. **Gate G1 is decided on Resume-Bench**: real resumes, built once with ResumeOCR-Lite's ground-truth tools. Every engine keeps page headers and footers there, since a resume's footer can hold its email.

```bash
python bench_resume.py build --gt ../ResumeOCR-Lite/data/realtest/gt.jsonl --out resume_bench
python bench_resume.py run --bench resume_bench --name flash --engine folio
python bench_resume.py run --bench resume_bench --name resume_v1 --engine folio --adapter runs/resume-t4/adapter
python bench_resume.py run --bench resume_bench --name chandra2 --engine chandra
python compare.py results/resume/resume_v1.json results/resume/chandra2.json
```

`docs/PHASES.md` maps every gate of the plan to the commands that decide it.

## The label format

One format for everything, identical to Infinity-Parser2's `doc2json` output, so fine-tuning teaches content, not a
new syntax:

```json
[
{"bbox": [347, 32, 653, 71], "category": "title", "text": "# Sandra Johnson"},
{"bbox": [39, 252, 514, 268], "category": "text", "text": "**Research Intern**, Everspark Solutions Ltd. (Nov 2024 - Apr 2025)"},
{"bbox": [39, 269, 723, 318], "category": "text", "text": "- Prototyped a time-series forecasting service ...\n- Benchmarked ..."}
]
```

- **Boxes** are normalised to 0–1000.
- **Categories** are the 13 Infinity-Parser2 uses.
- **Text** is Markdown, with HTML for tables, `$$...$$` LaTeX for formulas and `""` for figures.
- **Markdown output** is derived from the blocks: headers, footers and page footnotes are dropped, as in Infinity-Parser2's converter. So the JSON and the Markdown can never disagree.

## How the pieces work

- **Renderer** (`folioocr/render`):
  - It writes resumes (Indian, US and UK styles; classic, sidebar and split layouts, with photos, icons, tables, declarations) and general pages (two-column articles with footnotes, tables with merged headers, KaTeX math, letters, Hindi/Tamil/Bengali pages) as HTML.
  - Chromium renders them, and every block and line is measured from the DOM.
  - The visible text is serialised to the label by the same rules everywhere.
  - Pages are then degraded as scans, phone photos (perspective, background, shadows) or old archive scans, and every box moves with the pixels.
  - Unit tests in olmOCR-Bench's format are generated from the same source. Strings that olmOCR's normaliser treats asymmetrically (a lone `_` or `*`) are skipped, so a perfect transcript always passes.
- **Training** (`folioocr/train`):
  - The prompt is built with the model's own chat template, exactly as at inference.
  - Only the target is supervised.
  - Targets too long for `max_length` lose whole blocks from the end, so the JSON stays valid.
  - The loss only computes logits at supervised positions, chunk by chunk, so a 4k-token page target does not need several GB of logits on a 16 GB T4.
  - Profiles: `t4` (4-bit LoRA, fp16), `a100`, `h100-full`.
- **Inference** (`folioocr/model`, `folioocr/infer.py`):
  - Generation stops on `<|im_end|>` or `<|endoftext|>`.
  - A looping generation is cut early and retried once with mild sampling.
  - `--second-look` re-reads tiny-text blocks from the full-resolution page.
  - `--crosscheck` flags emails, phone numbers and dates where ResumeOCR-Lite's CRNN reads something different. It flags only and never "corrects".
- **Benchmarks** (`folioocr/bench`):
  - **olmOCR-Bench:** common 1,288 px or native rendering, sharding and resume for 12-hour sessions, and stratified samples for small budgets.
  - **Resume-Bench:** field accuracy, order-free line CER, extra-text rate and unit tests, per tier.
  - **OmniDocBench:** predictions only; score them with its official code.
  - **Paired bootstrap:** resamples documents within categories.

## Free-GPU expectations (estimates: measure before planning)

| Step | Where | Rough cost |
|---|---|---|
| Render 7,500 pages | your laptop, 6 workers | under an hour; ~3.5 GB |
| Base model on 60 dev pages | 1 T4 | ~15–30 min |
| Phase 1 LoRA, ~6,000 samples | Kaggle 2 x T4 | ~3–6 h (run `--max-steps 20` first and multiply) |
| olmOCR-Bench, 300-page sample | 1 T4 | ~1–2 h per model |

Kaggle gives about 30 GPU-hours a week (users' reports; your account settings show the exact quota) in 12-hour sessions. Everything long resumes where it stopped.

## Third-party pieces

- **Prompts:** copied from `infinity_parser2` (Apache-2.0).
- **Unit-test classes:** used from `olmocr` (Apache-2.0).
- **Resume vocabulary and scan augmentations:** from ResumeOCR-Lite.
- **Downloaded, not shipped:** fonts (OFL, Apache-2.0 and UFL, from google/fonts) and KaTeX (MIT).
- **Base model and data:** Infinity-Parser2-Flash (Apache-2.0). Check each dataset's licence before training on it.
