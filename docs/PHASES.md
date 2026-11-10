# Gates and the commands that decide them

The plan has five go/no-go gates. This page gives the commands that decide each one on free GPUs (Kaggle 2 x T4, free Colab T4).

Some gates need hardware that free tiers don't have:

- Infinity-Parser2-Pro needs an 80 GB GPU.
- The full olmOCR-Bench takes about 4 hours per model on a T4.

Where that happens, the free version of the rule is written next to the plan's rule. The free version is weaker, and any claim must say which one was used.

## Rules for every gate

- **One harness for every model:** `scripts/bench_olmocr.py`, `scripts/bench_resume.py` and `scripts/bench_dev.py`. Each run writes `results/<bench>/<name>.json`, including the settings it used.
- **What "A beats B" means:** `scripts/compare.py A.json B.json` reports *significant*. That is, the 95% paired-bootstrap interval of A − B excludes zero, with documents resampled within categories.
- **Benchmarks only at gates:**
  - Run each public benchmark once per candidate per gate.
  - Choose checkpoints on dev sets (`runs/<run>/dev_log.jsonl`, `scripts/bench_dev.py`), never on a benchmark.
- **One sample for every model:** on free GPUs, olmOCR-Bench runs use `--sample 300 --seed 0` for every model, so every model sees the same pages.
  - On 300 pages, one model's interval is about ±2–3 points; the runner prints it.
  - Paired comparisons on the same pages are much tighter.
  - A larger sample with the same seed contains the smaller one, so going from `--sample 300` to `--sample 600` only runs the new pages.
- **Write the rule down first:** before a phase starts, copy its gate's rule into `results/gates.md`. Write the outcome under it when the gate is decided.

Get the benchmark once:

```bash
hf download allenai/olmOCR-bench --repo-type dataset --local-dir olmOCR-bench
```

## G0: can we trust the ruler? (end of Phase 0)

**Plan:** our runs of Flash, Pro, Chandra 2 and Surya OCR 2 land within 1 point of their published olmOCR-Bench scores, or the gap is explained. Also, 100 Resume-Bench pages are reviewed.

**Free version:**

- Run Flash, Chandra 2 and Surya OCR 2 on the 300-page sample, common setting. Skip Pro.
- **Pass** when each published score (Flash 86.0, Chandra 2 85.8, Surya OCR 2 83.3) lies inside our 95% interval, or the gap has a stated cause. Resolution is a common cause: LlamaIndex's 1,288 px run of Chandra 2 scored 81.2.

```bash
# Flash, the model everything else is measured against (one T4, ~1-2 h)
python scripts/bench_olmocr.py --bench olmOCR-bench/bench_data --name flash_s300 --sample 300 \
    --batch-size 4 --max-new-tokens 6000

# competitors: a separate notebook, because they pin their own library versions
pip install -r requirements-competitors.txt
python scripts/bench_olmocr.py --bench olmOCR-bench/bench_data --name surya2_s300 --sample 300 --engine surya
python scripts/bench_olmocr.py --bench olmOCR-bench/bench_data --name chandra2_s300 --sample 300 --engine chandra
```

**Chandra 2 on a T4:** Chandra 2 loads in bfloat16, which the T4 runs slowly or not at all. If it fails, run it where it works and score its Markdown here with `--engine mddir --md-dir <folder of <key>.md>`. A paid API's output is a different system from the released weights, so label it as such.

**Resolution test.** The plan says the input size is chosen per category in Phase 0. The categories that need pixels:

```bash
for px in 1288 1600 2048; do
  python scripts/bench_olmocr.py --bench olmOCR-bench/bench_data --name flash_px${px} --sample 300 \
      --categories old_scans,multi_column,table_tests,long_tiny_text \
      --longest-side $px --max-pixels $((px * px)) --batch-size 2
done
python scripts/compare.py results/olmocr/flash_px1600.json results/olmocr/flash_px1288.json
```

- **Category names** are the `.jsonl` files in `bench_data`. The runner lists them if a name is wrong.
- **`--max-pixels`:** raise it with the size. At the default 2,000,000, a 2,048 px page would be shrunk back down.

**Speed and memory**, for the efficiency table:

```bash
python scripts/speed_test.py --images bench_cache/long1288 --pages 20 --name flash_t4 --batch-size 4
```

**Resume-Bench v0:**

- Review 100 real pages with ResumeOCR-Lite's tools (`pdf_to_gt.py`, `project_gt.py`, `review_gt.py`; consent and storage rules in its `docs/TEST_SET.md`).
- Then freeze them. The tests never change after this step.

```bash
python scripts/bench_resume.py build --gt ../ResumeOCR-Lite/data/realtest/gt.jsonl --out resume_bench
```

**If G0 fails:** fix the harness. No training starts against a ruler we don't trust.

## G1: the resume win (end of Phase 1)

**Plan:**

- Our model has the best Resume-Bench page CER and field accuracy of every model tested, significantly.
- Its olmOCR-Bench score is at most 0.5 below Flash's.

**Free version:** the same, with olmOCR-Bench on the G0 sample.

- **Pass on the sample:** the point estimate of `ours − Flash` is ≥ −0.5.
- **If the interval reaches below −2:** 300 pages can't rule out a real drop. Grow the sample to 600 and decide there.

Train with the notebook, or with the commands in the README's "Phase 1" section.

On one GPU, `runs/<run>/best` holds the checkpoint with the best dev score. With two GPUs, dev evaluation runs once, at the end, and `runs/<run>/adapter` is the final checkpoint.

```bash
# every model on the same real pages; headers and footers are kept for all of them
python scripts/bench_resume.py run --bench resume_bench --name flash
python scripts/bench_resume.py run --bench resume_bench --name resume_v1 --adapter runs/resume-t4/best
python scripts/bench_resume.py run --bench resume_bench --name surya2 --engine surya
python scripts/bench_resume.py run --bench resume_bench --name chandra2 --engine chandra

# page CER, field accuracy and unit tests, each with its paired interval
python scripts/compare.py results/resume/resume_v1.json results/resume/chandra2.json
python scripts/compare.py results/resume/resume_v1.json results/resume/surya2.json
python scripts/compare.py results/resume/resume_v1.json results/resume/flash.json

# no general regression: the G0 sample
python scripts/bench_olmocr.py --bench olmOCR-bench/bench_data --name resume_v1_s300 --sample 300 \
    --adapter runs/resume-t4/best --batch-size 4 --max-new-tokens 6000
python scripts/compare.py results/olmocr/resume_v1_s300.json results/olmocr/flash_s300.json
```

**Pass:** `resume_v1` beats every other model on both page CER and field accuracy, significantly, and passes the olmOCR rule above.

**If G1 fails:**

- Add more or better resume data, or change the mix.
  - `--data a.jsonl:3 b.jsonl:2` sets the shares of each source.
  - `--samples-per-epoch` sets the number of samples per epoch.
- Phase 2 waits.

**What to read when it fails:**

- **`fields_by_kind`** in the result files (email, phone, URL, date) shows which kind of field is failing.
- **`--crosscheck`** shows whether ResumeOCR-Lite's line reader disagrees on those fields. Example: `scripts/infer.py page.jpg --adapter ... --crosscheck ../ResumeOCR-Lite`.

## G2: general SFT (end of Phase 2)

**Plan:**

- The SFT model matches or beats Flash's reproduced olmOCR-Bench score.
- It beats Flash on the old-scans and multi-column dev sets.

**Free version:**

- olmOCR-Bench on the G0 sample. "Matches" means not significantly worse: the paired interval's upper end is above zero.
- The two dev sets must be beaten significantly.
- Training is a LoRA continuation of the Phase 1 adapter on about 10–20k samples, not the plan's 1–1.5M-page full fine-tune.

**1. Data.** Each `olmOCR-mix` tarball holds a few thousand single-page PDFs (about 1 GB). Kaggle's working disk is 20 GB, so take one or two per subset.

```bash
hf download allenai/olmOCR-mix-1025 --repo-type dataset --local-dir olmocr-mix --include \
    "00_documents_train.parquet" "pdf_tarballs/00_documents_train_00000.tar.gz" \
    "02_loc_transcripts_train.parquet" "pdf_tarballs/02_loc_transcripts_train_00000.tar.gz" \
    "03_national_archives_train.parquet" "pdf_tarballs/03_national_archives_train_00000.tar.gz" \
    "02_loc_transcripts_eval.parquet" "pdf_tarballs/02_loc_transcripts_eval_*"
hf download allenai/olmOCR-synthmix-1025 --repo-type dataset --local-dir synthmix

python scripts/prepare_data.py olmocr-mix --parquet olmocr-mix/00_documents_train.parquet \
    --tarballs olmocr-mix/pdf_tarballs/00_documents_train_00000.tar.gz --out data/olmocr_docs --limit 3000
python scripts/prepare_data.py olmocr-mix --parquet olmocr-mix/02_loc_transcripts_train.parquet \
    --tarballs olmocr-mix/pdf_tarballs/02_loc_transcripts_train_00000.tar.gz --out data/olmocr_hist --limit 1500
python scripts/prepare_data.py olmocr-mix --parquet olmocr-mix/03_national_archives_train.parquet \
    --tarballs olmocr-mix/pdf_tarballs/03_national_archives_train_00000.tar.gz --out data/olmocr_hist --limit 1500
python scripts/prepare_data.py synthmix --root synthmix --out-sft data/synthmix --out-rl data/rl_synthmix
python scripts/prepare_data.py preview --labels data/olmocr_hist/labels.jsonl --n 8      # look before training

# our renders (CPU: run these on your own computer and upload data/)
python -m folioocr.render.make_dataset --out data/render_old --n 4000 --workers 6 --prefix o --start-seed 2000000 \
    --mix "article=0.4,letter=0.25,table=0.2,math=0.15" --conditions "old=0.5,scan=0.4,photo=0.1"
python -m folioocr.render.make_dataset --out data/render_layout --n 4000 --workers 6 --prefix l --start-seed 3000000 \
    --mix "article=0.5,table=0.3,math=0.2" --conditions "clean=0.4,scan=0.4,photo=0.2"
```

**2. Dev sets for the gate.** Every page in them is held out (`--dev-frac 1`).

```bash
python -m folioocr.render.make_dataset --out data/dev_old --n 300 --workers 6 --prefix do --start-seed 9000000 \
    --mix "article=0.5,letter=0.25,table=0.25" --conditions old=1 --dev-frac 1
python -m folioocr.render.make_dataset --out data/dev_multicol --n 300 --workers 6 --prefix dm --start-seed 9500000 \
    --mix article=1 --dev-frac 1
python scripts/prepare_data.py olmocr-mix --parquet olmocr-mix/02_loc_transcripts_eval.parquet \
    --tarballs olmocr-mix/pdf_tarballs/02_loc_transcripts_eval_* --out data/dev_loc --dev-frac 1
```

**3. Contamination, then the data card.**

- Index every evaluation set once.
- Check every training source, and train only on the `labels.clean.jsonl` files.
- The mixture presets in `mixtures/` already point at the clean files.

```bash
python scripts/dedup.py index --out eval_index.npz --olmocr olmOCR-bench/bench_data --resume resume_bench \
    --labels dev_old=data/dev_old/labels.jsonl dev_multicol=data/dev_multicol/labels.jsonl dev_loc=data/dev_loc/labels.jsonl
python scripts/dedup.py check --index eval_index.npz --report results/contamination.json --write-clean --labels \
    data/olmocr_docs/labels.jsonl data/olmocr_hist/labels.jsonl data/synthmix/labels.jsonl \
    data/render_old/labels.jsonl data/render_layout/labels.jsonl
python scripts/make_data_card.py --mixture mixtures/general-free-v1.json --contamination results/contamination.json \
    --out docs/DATA_CARD.md
```

**4. Train.**

- This continues from the Phase 1 adapter, with resumes still 15% of the mix so the resume skill stays.
- `--samples-per-epoch` sets the run's length. Measure with `--max-steps 20` first.
- `scripts/train_sft.py` refuses a mixture with a non-commercial source (Infinity-Doc2-5M is CC BY-NC-SA) unless you pass `--allow-noncommercial`.

```bash
accelerate launch --num_processes 2 --mixed_precision fp16 scripts/train_sft.py --profile t4 \
    --mixture mixtures/general-free-v1.json --init-adapter runs/resume-t4/best \
    --dev data/dev_old/labels.jsonl --dev-tests data/dev_old/tests.jsonl \
    --out runs/general-t4 --samples-per-epoch 12000 --resume
```

**The control (optional on free GPUs).** Train the same recipe from `Qwen/Qwen3.5-2B` (add `--model Qwen/Qwen3.5-2B`, and drop `--init-adapter`). This shows how much of the gain comes from our data rather than Flash's.

**Formulas:** before training on math pages, check one raw Flash output on a page with a display formula (`scripts/infer.py page.png --raw`).

- Our labels write formula blocks as `$$...$$`.
- If Flash writes them bare, decide which convention to keep before training, because olmOCR's math tests read `$$` blocks.

**5. Decide G2.**

```bash
for d in dev_old dev_multicol dev_loc; do
  python scripts/bench_dev.py --labels data/$d/labels.jsonl --tests data/$d/tests.jsonl --name flash_$d --limit 120 --batch-size 4
  python scripts/bench_dev.py --labels data/$d/labels.jsonl --tests data/$d/tests.jsonl --name general_$d --limit 120 \
      --batch-size 4 --adapter runs/general-t4/best
  python scripts/compare.py results/dev/general_$d.json results/dev/flash_$d.json
done
# multi-column only: add --where columns=2 to both bench_dev runs
python scripts/bench_olmocr.py --bench olmOCR-bench/bench_data --name general_s300 --sample 300 \
    --adapter runs/general-t4/best --batch-size 4 --max-new-tokens 6000
python scripts/compare.py results/olmocr/general_s300.json results/olmocr/flash_s300.json
```

- `dev_loc` has no unit tests, so `scripts/compare.py` compares it on page CER only.
- For the rendered dev sets, it compares unit tests and CER.

**If G2 fails:** stop the general climb, and ship Tier A with its efficiency results.

## G3: RL (end of Phase 3)

**Plan:**

- **Tier B:** significantly above Chandra 2 and Surya OCR 2 on olmOCR-Bench.
- **Tier C:** at or above Pro on olmOCR-Bench (87.6) and OmniDocBench v1.6 (93.95).
- OmniDocBench never falls below Flash.

**Free version:**

- **Tier B:** decided on the G0 sample, with paired comparisons against our own Chandra 2 and Surya OCR 2 runs.
- **Tier C:** can't be tested against Pro itself, because it needs an 80 GB GPU. Report our interval next to Pro's published score, and claim nothing unless the interval's lower end clears it.
- **OmniDocBench:** use the same page subset for Flash and for ours.

**1. RL items:** bands of rendered pages, each with its own tests.

- On a T4, keep to bands.
- The full arXiv pages in `data/rl_synthmix` need several thousand output tokens per sample. Use them only with the `a100` profile.

```bash
python scripts/make_rl_set.py --labels data/resumes/labels.jsonl data/render_old/labels.jsonl data/render_layout/labels.jsonl \
    --out data/rl_regions --per-page 2 --max-chars 1200
```

**2. Pilot, read, then run.**

- The pilot is 50 steps. Read `runs/rl-pilot/samples/*.jsonl` before the real run, looking for reward hacking (for example, dropping text to pass absence tests).
- Each GPU samples its own items, and gradients are averaged.
- Check `train_log.jsonl`:
  - **`skipped_groups`:** if most groups are skipped (every sample got the same reward), raise `--group` or `--temperature`.
  - **`loops`:** should stay near zero.

```bash
torchrun --nproc_per_node 2 scripts/train_rl.py --profile t4 --adapter runs/general-t4/best \
    --data data/rl_regions/labels.jsonl --out runs/rl-pilot --steps 50 --eval-every 25 --dump-every 25
torchrun --nproc_per_node 2 scripts/train_rl.py --profile t4 --adapter runs/general-t4/best \
    --data data/rl_regions/labels.jsonl --out runs/rl-t4 --steps 400 --resume
```

- **Saving:** `runs/rl-t4/best` holds the adapter with the best dev reward.
- **Plateau stop:** the run stops by itself after `--patience` flat evaluations.
- **Plan's settings:** 8 samples per page, little or no KL. On the T4 profile that is `--group 8 --beta 0`, at twice the cost of the default `--group 4`.

**3. Decide G3.**

```bash
python scripts/bench_olmocr.py --bench olmOCR-bench/bench_data --name rl_s300 --sample 300 \
    --adapter runs/rl-t4/best --batch-size 4 --max-new-tokens 6000
python scripts/compare.py results/olmocr/rl_s300.json results/olmocr/chandra2_s300.json     # Tier B
python scripts/compare.py results/olmocr/rl_s300.json results/olmocr/surya2_s300.json       # Tier B
python scripts/compare.py results/olmocr/rl_s300.json results/olmocr/flash_s300.json
python scripts/bench_resume.py run --bench resume_bench --name rl_v1 --adapter runs/rl-t4/best   # Tier A must still hold

# OmniDocBench: predictions for the same page subset, scored with its official code
python scripts/bench_omnidoc.py --images OmniDocBench/images --name flash_od --limit 300
python scripts/bench_omnidoc.py --images OmniDocBench/images --name rl_od --limit 300 --adapter runs/rl-t4/best
```

**If G3 fails:** claim only what passed. Run one more RL round only while the dev reward is still rising.

## G4: release (end of Phase 4)

**Plan:**

- The private holdout moves the same way as the public scores.
- The contamination and licence checks are clean.

**1. Merge, optionally average, quantise.**

- **Merging:** happens on the CPU and needs about 5 GB of RAM.
- **Averaging:** keep the average only if it beats the plain merge on the dev sets.
- **Quantising:** needs a CUDA GPU. Re-run every benchmark on the quantised copy, because its numbers are the consumer-GPU numbers.

```bash
python scripts/merge_lora.py --adapter runs/rl-t4/best --out models/folio-v1
python scripts/merge_lora.py --adapter runs/general-t4/best runs/rl-t4/best --weights 0.5 0.5 --out models/folio-avg
python scripts/bench_dev.py --labels data/dev_old/labels.jsonl --tests data/dev_old/tests.jsonl --name v1_dev --model models/folio-v1
python scripts/bench_dev.py --labels data/dev_old/labels.jsonl --tests data/dev_old/tests.jsonl --name avg_dev --model models/folio-avg
python scripts/compare.py results/dev/avg_dev.json results/dev/v1_dev.json

python scripts/merge_lora.py quantize --model models/folio-v1 --out models/folio-v1-nf4 --bits 4
python scripts/bench_olmocr.py --bench olmOCR-bench/bench_data --name v1_nf4_s300 --sample 300 --model models/folio-v1-nf4
python scripts/speed_test.py --images bench_cache/long1288 --pages 20 --name v1_t4 --model models/folio-v1 --batch-size 4
python scripts/speed_test.py --images bench_cache/long1288 --pages 20 --name v1_nf4_t4 --model models/folio-v1-nf4 --batch-size 4
```

**2. The holdout:**

- About 300 fresh pages, collected after the training data was frozen, sorted into `holdout_src/<category>/` by olmOCR-Bench category.
- Draft tests from a strong teacher, then check every test against the page image.
- Every test must be marked `verified` or `rejected`; `--status` says when the set is ready.

```bash
python scripts/draft_holdout_tests.py --src holdout_src --out holdout --engine openai --api-base http://localhost:8000/v1 \
    --api-model infly/Infinity-Parser2-Pro --api-prompt doc2json        # or --engine folio (Flash) on a free GPU
python scripts/draft_holdout_tests.py --out holdout --status
python scripts/bench_olmocr.py --bench holdout/bench_data --name v1_holdout --model models/folio-v1
python scripts/bench_olmocr.py --bench holdout/bench_data --name flash_holdout
python scripts/compare.py results/olmocr/v1_holdout.json results/olmocr/flash_holdout.json
```

**Pass:**

- The holdout difference (ours − Flash) has the same sign as on olmOCR-Bench, and its interval overlaps the public one.
- `scripts/dedup.py check` finds no matches in the clean files you trained on.
- The data card lists no source with an unknown licence.
- Non-commercial sources are named on the model card.

**3. The model card.** Every number on it comes from a result file.

```bash
python scripts/make_model_card.py --name folio-ocr-v1 --model-dir models/folio-v1 --sft-run runs/general-t4 --rl-run runs/rl-t4 \
    --olmocr results/olmocr/rl_s300.json --olmocr-baseline results/olmocr/flash_s300.json \
        results/olmocr/chandra2_s300.json results/olmocr/surya2_s300.json \
    --resume results/resume/rl_v1.json --resume-baseline results/resume/flash.json results/resume/chandra2.json \
    --holdout results/olmocr/v1_holdout.json --holdout-baseline results/olmocr/flash_holdout.json \
    --speed results/speed/v1_t4.json results/speed/v1_nf4_t4.json --contamination results/contamination.json
```

**If G4 fails:** make no public claim until it passes.

## Teacher labels for your own pages (any phase)

Real pages without a source become training data only when two permissively licensed teachers agree:

- Infinity-Parser2-Pro and olmOCR 2 are both Apache-2.0.
- Chandra and Surya are never used as teachers.

On free GPUs, Flash itself can be one of the two teachers (`--engine folio`). The other can be any endpoint whose terms allow training on its output.

```bash
python scripts/label_teachers.py stage --inputs my_scans/ --stage teach/pages
python scripts/label_teachers.py run --stage teach/pages --name flash --licence Apache-2.0 --cache teach/flash --engine folio
python scripts/label_teachers.py run --stage teach/pages --name olmocr2 --licence Apache-2.0 --cache teach/olmocr2 \
    --engine openai --api-base http://localhost:8001/v1 --api-model allenai/olmOCR-2-7B-1025 --api-prompt olmocr
python scripts/label_teachers.py merge --stage teach/pages --a teach/flash --b teach/olmocr2 --out data/consensus
```

- Pages where the teachers disagree go to `data/consensus/review.jsonl`.
- Run `scripts/dedup.py check` on `data/consensus/labels.jsonl` like any other source.
