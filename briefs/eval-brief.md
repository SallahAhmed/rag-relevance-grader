You are computing the MISSING HALF of a benchmark table for an existing ML project.
Repo: F:\finetune (git repo, Windows). Python: `python` on PATH (anaconda3).

## Context you need (do not re-derive)
This project fine-tuned Qwen2.5-1.5B-Instruct with QLoRA to classify query-passage
relevance into 3 classes (irrelevant / partial / relevant) on TREC DL 2020.

- Baseline (zero-shot) results ALREADY EXIST: `results/baseline_results.json`
  (accuracy 0.3695, macro_f1 0.2672, n_test_queries 10, n_test_qrels 2122,
   mmlu_baseline 0.4975).
- The fine-tuned adapter is PUBLIC on the Hub:
  `sallahahmed/qwen2.5-1.5b-relevance-grader`
- The test split is already on disk at `data/processed/test.parquet`
  (2122 rows; columns: qid, query, pid, passage, qrel, label). It is verified
  identical to the baseline split (same 10 query ids).
- `peft` 0.21.2 is now installed. `transformers`/`torch` are installed but raise an
  OMP warning — set `KMP_DUPLICATE_LIB_OK=TRUE` in the environment before importing
  torch, or the process may abort.

## THE CRITICAL FACT THAT MAKES THIS CHEAP
Grading is NOT text generation. It is a SINGLE forward pass, argmax over the
three label-token logits. Reference implementation, `notebooks/02_baseline_evaluation.py`
around line 105:

```python
logits = model(**enc).logits[:, -1, :]
sub    = logits[:, torch.tensor(label_ids, device=logits.device)]
probs  = torch.softmax(sub.float(), dim=-1)
```

So NO GPU is required. This runs on CPU. Read notebook 02's `grade_all` and
`resolve_io` carefully and mirror them exactly — including how `label_ids` is
derived from `PROMPT_TEMPLATE` and `LABELS` in `src/dataset_utils.py`.

Use batching (pad to the longest sequence in each batch) to make this fast. The
prompt is `PROMPT_TEMPLATE.format(query=..., passage=...)` then the label token.
Look at `CompletionOnlyCollator` in `notebooks/03_fine_tuning.py` for how the
response position was masked during training — the label logits you need are the
ones at that same position.

## What to build
`scripts/eval_cpu.py`:
1. Load base model `Qwen/Qwen2.5-1.5B-Instruct` in float32 (or bfloat16 if CPU
   supports it well), load the adapter on top with peft. DO NOT use 4-bit/bitsandbytes
   on CPU. Merge the adapter or use PeftModel, whichever notebook 02's shape suggests.
2. Score all 2122 rows of `data/processed/test.parquet`.
3. Write `results/predictions_ft.json` in the SAME shape as
   `results/predictions_baseline.json`. Read that file first to copy the schema
   exactly (keys: qid, pid, true, pred, confidence).
4. Then run the EXISTING `make_comparison.py` to produce
   `results/finetuned_results.json` and `results/before_after_table.md`.
   Do NOT rewrite that script. Read it first to see what inputs it wants.

## Process requirements (these matter more than speed)
- **BENCHMARK FIRST.** Before the full run, score 50 rows, measure seconds/row,
  print the extrapolated full-run ETA, and WRITE IT TO THE LOG. Then proceed.
- Print running progress every ~100 rows with a live ETA so the orchestrator can
  watch progress. Flush stdout.
- Be honest about slowness. If the full 2122 would exceed 90 minutes, STOP and
  report the measured rate plus a recommended subset size. Do not silently run for
  hours.
- Metrics MUST come from `src/evaluation_utils.compute_classification_metrics` and
  `make_comparison.py`. Never hand-compute or hardcode a metric.
- **NEVER invent, guess, extrapolate or copy a metric.** If a step fails, report the
  failure. A missing number is acceptable; a fabricated number is not.

## Hard rules
- DO NOT COMMIT ANYTHING. Leave changes in the working tree.
- DO NOT modify: `notebooks/`, `config/`, `results/baseline_results.json`,
  `results/predictions_baseline.json`, `make_comparison.py`, `make_baseline_results.py`,
  `src/config.py`, `src/dataset_utils.py`, `src/evaluation_utils.py`,
  `src/grader_interface.py`, `src/mmlu_utils.py`, `src/rag_pipeline.py`,
  `src/sample_corpora.py`, `AGENTS.md`, `README.md`, `space/`.
  You MAY create `scripts/eval_cpu.py`.
- Do NOT touch the Colab CLI. Do NOT touch the Hugging Face Space. Do NOT push to
  the Hub. The orchestrator owns those.
- Set `KMP_DUPLICATE_LIB_OK=TRUE` and `TOKENIZERS_PARALLELISM=false` in your own
  process, do not edit global config.

## Report contract — report exactly this
1. Files created or modified.
2. The measured seconds/row and the extrapolated full-run ETA.
3. The FINAL metrics table: accuracy, macro_f1, per-class f1 for all three classes,
   confusion matrix, and n_test_queries / n_test_qrels. Copy the real numbers from
   the JSON that was written — do not retype from memory.
4. The delta vs baseline (macro_f1 after minus before), in percentage points.
5. The exact content of `results/before_after_table.md`.
6. Anything you could not do, and why. Any assumption the orchestrator must check.

Report failures as failures. A truthful "this did not work because X" is far more
useful than an optimistic summary.