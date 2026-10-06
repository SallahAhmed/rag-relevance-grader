---
title: Fine-tuned RAG relevance grader
emoji: 🎯
colorFrom: blue
colorTo: indigo
sdk: static
license: mit
---

# Fine-tuned RAG relevance grader — static artifact

The always-on, **zero-compute** half of the demo. Static Spaces are free for
everyone and need no GPU, no model download and no quota, so this page keeps
working when the interactive Space is asleep, rate-limited or out of GPU minutes.

## What is on this page

1. **The before/after table** — accuracy, macro-F1 and per-class F1, read from
   `results/baseline_results.json` and `results/finetuned_results.json`, with
   `N` test queries and `N` test qrels next to every number. Nothing is typed by
   hand.
2. **A recorded grader run** — one real ON/OFF comparison from
   `results/demo_trace.json`: the query, both answers, and the per-passage
   retrieval score, label, confidence and accept/reject decision.
3. **The three demo corpora** used by the interactive app.

If a JSON artifact is missing, the page says "not measured yet" instead of
inventing a value.

## Regenerating

The HTML is generated, so re-render it after any evaluation run:

```bash
python space_static/build_static.py     # or: python -m space_static.build_static
```

## Layout

| Path | Role |
|---|---|
| `index.html` | The generated page (this is what the Space serves) |
| `build_static.py` | Generator — reads the JSON artifacts, writes `index.html` |
| `../src/rag_pipeline.py` | Shared pipeline + metrics reader |
| `../results/*.json` | The artifacts every number comes from |
