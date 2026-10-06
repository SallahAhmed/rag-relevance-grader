---
title: Corrective RAG relevance grader
emoji: 🎛️
colorFrom: indigo
colorTo: blue
sdk: streamlit
sdk_version: streamlit==1.51.0
app_file: app.py
pinned: false
python_version: "3.10"
license: mit
short_description: Toggle a fine-tuned relevance grader between naive and corrective RAG
suggested_hardware: cpu-basic
---

# Corrective RAG with a fine-tuned relevance grader

Retrieval is the weak link in a RAG pipeline: an embedding index happily returns
passages that share vocabulary with the question and answer none of it. A
**relevance grader** placed between retrieval and generation fixes that, and this
Space is the demo of it.

The grader is a QLoRA fine-tune of `Qwen/Qwen2.5-1.5B-Instruct` on TREC DL 2020
graded relevance judgments, bucketed into `irrelevant` / `partial` / `relevant`.
It implements the project's `GraderInterface`, so it drops into any pipeline
that can take a reranker.

## What the toggle does

| Arm | Behaviour |
| --- | --- |
| **Grader OFF** (naive RAG) | Every retrieved passage reaches the generator. No grading call is made at all. |
| **Grader ON** (corrective RAG) | Each passage is graded; `irrelevant` ones are dropped. If nothing survives, the pipeline returns an explicit *no grounded answer* instead of inventing text. |

Both arms always run, side by side, and the "Show retrieval + grading trace"
expander gives you the per-passage retrieval score, label, confidence and
accept/reject decision. `partial` and `relevant` are accepted by default;
`irrelevant` is rejected.

## What you can do

- Run three pre-built sample corpora (a Colab T4 notebook log, an espresso
  troubleshooting log, an Amsterdam travel desk). Each mixes answer-bearing
  passages with decoys that retrieve well and answer nothing, so the difference
  between the two arms is visible on the first click.
- Paste your own documents or upload `.txt` / `.md` files and ask your own
  question.

## Running the numbers

Every metric this app shows is read at runtime from `results/baseline_results.json`
and `results/finetuned_results.json` (written by the evaluation notebooks). No
score is hardcoded here. `N` test queries and `N` test qrels are displayed
alongside every number, because the holdout is query-grouped and small enough
that macro-F1 on it is directional evidence rather than a tight estimate.

## Running it yourself

```bash
pip install -r requirements.txt
streamlit run app.py
```

## Deploying this Space

The Space needs the repository's `src/` folder next to `app.py`, because
`src/rag_pipeline.py` holds the pipeline and `src/requirements.txt` documents
the rest. Two ways to publish it:

1. **Space repo contains `src/`** — create a Space whose repository includes
   both `space/` and `src/`, with this file's front matter at the repository
   root and `app_file: space/app.py`.
2. **Copy `space/` as the Space root** and copy `src/` alongside `app.py`, then
   set `app_file: app.py`.

If `src/rag_pipeline.py` is missing the app shows an explicit error instead of
crashing.

## Notes on the model

- With no GPU the app still runs, on CPU, unquantized: expect minutes rather
  than seconds per answer.
- If the QLoRA adapter (`sallahahmed/qwen2.5-1.5b-relevance-grader`) is not
  available, the app falls back to the base model zero-shot and says so in a
  warning banner. It never silently pretends the fine-tune is running.
- `bitsandbytes` is intentionally not installed: 4-bit NF4 needs CUDA, and
  requiring it would break the CPU-only Space build.
- These pins have **not** been validated by an actual Space build; they are a
  known-good-looking starting set.
