# Gates and the commands that decide them

The plan has five go/no-go gates. This page gives the commands that decide each one on free GPUs (Kaggle 2 x T4, free Colab T4).

Some gates need hardware that free tiers don't have:

- Infinity-Parser2-Pro needs an 80 GB GPU.
- The full olmOCR-Bench takes about 4 hours per model on a T4.

Where that happens, the free version of the rule is written next to the plan's rule. The free version is weaker, and any claim must say which one was used.

## Rules for every gate

- **One harness for every model:** `bench_olmocr.py`, `bench_resume.py` and `bench_dev.py`. Each run writes `results/<bench>/<name>.json`, including the settings it used.
- **What "A beats B" means:** `compare.py A.json B.json` reports *significant*. That is, the 95% paired-bootstrap interval of A − B excludes zero, with documents resampled within categories.
- **Benchmarks only at gates:**
  - Run each public benchmark once per candidate per gate.
  - Choose checkpoints on dev sets (`runs/<run>/dev_log.jsonl`, `bench_dev.py`), never on a benchmark.
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
python bench_olmocr.py --bench olmOCR-bench/bench_data --name flash_s300 --sample 300 \
    --batch-size 4 --max-new-tokens 6000

# competitors: a separate notebook, because they pin their own library versions
pip install -r requirements-competitors.txt
python bench_olmocr.py --bench olmOCR-bench/bench_data --name surya2_s300 --sample 300 --engine surya
python bench_olmocr.py --bench olmOCR-bench/bench_data --name chandra2_s300 --sample 300 --engine chandra
```

**Chandra 2 on a T4:** Chandra 2 loads in bfloat16, which the T4 runs slowly or not at all. If it fails, run it where it works and score its Markdown here with `--engine mddir --md-dir <folder of <key>.md>`. A paid API's output is a different system from the released weights, so label it as such.

**Resolution test.** The plan says the input size is chosen per category in Phase 0. The categories that need pixels:

```bash
for px in 1288 1600 2048; do
  python bench_olmocr.py --bench olmOCR-bench/bench_data --name flash_px${px} --sample 300 \
      --categories old_scans,multi_column,table_tests,long_tiny_text \
      --longest-side $px --max-pixels $((px * px)) --batch-size 2
done
python compare.py results/olmocr/flash_px1600.json results/olmocr/flash_px1288.json
```

- **Category names** are the `.jsonl` files in `bench_data`. The runner lists them if a name is wrong.
- **`--max-pixels`:** raise it with the size. At the default 2,000,000, a 2,048 px page would be shrunk back down.

**Speed and memory**, for the efficiency table:

```bash
python speed_test.py --images bench_cache/long1288 --pages 20 --name flash_t4 --batch-size 4
```

**Resume-Bench v0:**

- Review 100 real pages with ResumeOCR-Lite's tools (`pdf_to_gt.py`, `project_gt.py`, `review_gt.py`; consent and storage rules in its `docs/TEST_SET.md`).
- Then freeze them. The tests never change after this step.

```bash
python bench_resume.py build --gt ../ResumeOCR-Lite/data/realtest/gt.jsonl --out resume_bench
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
python bench_resume.py run --bench resume_bench --name flash
python bench_resume.py run --bench resume_bench --name resume_v1 --adapter runs/resume-t4/best
python bench_resume.py run --bench resume_bench --name surya2 --engine surya
python bench_resume.py run --bench resume_bench --name chandra2 --engine chandra

# page CER, field accuracy and unit tests, each with its paired interval
python compare.py results/resume/resume_v1.json results/resume/chandra2.json
python compare.py results/resume/resume_v1.json results/resume/surya2.json
python compare.py results/resume/resume_v1.json results/resume/flash.json

# no general regression: the G0 sample
python bench_olmocr.py --bench olmOCR-bench/bench_data --name resume_v1_s300 --sample 300 \
    --adapter runs/resume-t4/best --batch-size 4 --max-new-tokens 6000
python compare.py results/olmocr/resume_v1_s300.json results/olmocr/flash_s300.json
```

**Pass:** `resume_v1` beats every other model on both page CER and field accuracy, significantly, and passes the olmOCR rule above.

**If G1 fails:**

- Add more or better resume data, or change the mix.
  - `--data a.jsonl:3 b.jsonl:2` sets the shares of each source.
  - `--samples-per-epoch` sets the number of samples per epoch.
- Phase 2 waits.

**What to read when it fails:**

- **`fields_by_kind`** in the result files (email, phone, URL, date) shows which kind of field is failing.
- **`--crosscheck`** shows whether ResumeOCR-Lite's line reader disagrees on those fields. Example: `infer.py page.jpg --adapter ... --crosscheck ../ResumeOCR-Lite`.

## G2, G3, G4

These gates come with the Phase 2–4 code (data engine, RL, merge and release), and this page will get their commands then.
