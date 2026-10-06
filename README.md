# Fine-tuned RAG Relevance Grader (QLoRA) — with a live Corrective RAG demo

> **The table is the project. Everything below exists to make these numbers trustworthy.**

| Metric | Zero-shot | Fine-tuned (QLoRA) | Δ |
|---|---|---|---|
| Accuracy | 0.3695 | **0.6447** | **+27.5 pp** |
| Macro-F1 | 0.2672 | **0.4306** | **+16.3 pp** |
| F1 — irrelevant | 0.4735 | 0.8000 | +32.6 pp |
| F1 — partial | 0.0131 | 0.1590 | +14.6 pp |
| F1 — relevant | 0.3149 | 0.3329 | +1.8 pp |

| Verdict | Bar | Result | |
|---|---|---|---|
| Macro-F1 gain | ≥ +10 pp | **+16.3 pp** | **PASS** |
| Catastrophic forgetting (MMLU retention) | ≥ 0.90 | **0.947** (0.4712 / 0.4975) | **PASS** |

**Eval set: 10 held-out queries / 2122 query-passage qrels** (TREC DL 2020). Split by whole
query, never by row. Macro-F1 over 10 queries is **directional evidence, not a tight estimate** —
quoted to 4 decimals because it is reproducible, not because the 4th decimal is meaningful.

Live demo: **[Corrective RAG Relevance Grader](https://huggingface.co/spaces/SallahAhmed/corrective-rag-relevance-grader)** ·
Weights: **[sallahahmed/qwen2.5-1.5b-relevance-grader](https://huggingface.co/sallahahmed/qwen2.5-1.5b-relevance-grader)**

---

## The one-paragraph version

Most "I fine-tuned an LLM" projects are unverifiable claims with no number attached. This one
attaches the number. I fine-tuned Qwen2.5-1.5B-Instruct with QLoRA (4-bit NF4) to classify
query-passage relevance into three classes, benchmarked it against the *same model, same prompt,
same test queries* with no fine-tuning, and published the before/after table above. Then I put
the fine-tuned model inside a RAG pipeline, where it does something a real system needs: decide
whether a retrieved passage is good enough to answer from.

## Why the baseline was so bad — and why that matters for reading the table

The zero-shot baseline is not merely weak, it is **degenerate**. From its confusion matrix
(rows = truth, columns = prediction) over the same 2122 qrels:

```
                pred: irrelevant  partial  relevant
true irrelevant          483        3        1012
true partial              34        2         264
true relevant              25        0         299
```

Column sums: **542 irrelevant, 5 partial, 1575 relevant.** The un-tuned model answers
"relevant" **74% of the time** while the truth is "irrelevant" **71% of the time**. It is not
predicting the majority class — it is *anti-correlated* with the prior. A 1.5B model that has not
been told how the classes are distributed has not learned the task at all.

So read the +16.3 pp honestly: **a large part of it is learning the label distribution**, which
any competent fine-tune would achieve. The interesting number is not the headline — it is
`F1 — irrelevant: 0.4735 → 0.8000`, which means the model learned *which passages are actually
irrelevant* rather than just which class is common. And it is `F1 — relevant: +1.8 pp`, which is
the honest weak spot: 324 qrels of support is thin, and the model still over-predicts relevant.
Claiming the whole +16.3 pp as "QLoRA magic" would be the wrong read, and it is the first thing a
competent reviewer will check.

## What the fine-tuned model does inside a RAG pipeline

Naive RAG concatenates whatever it retrieved and asks the model to answer. If the retrieval
pulled in three irrelevant passages, the answer inherits their noise. Corrective RAG grades each
retrieved passage first and refuses to answer from the ones that fail.

The Space implements this as a live ON/OFF comparison, same query, same retrieved passages:

| Arm | Behaviour |
|---|---|
| **Grader OFF** — naive RAG | All retrieved passages reach the generator. No grading call is made. |
| **Grader ON** — corrective RAG | Every passage is graded `irrelevant / partial / relevant`. Only `partial` and `relevant` survive. If none survive, the system returns an explicit *no grounded answer* rather than fabricating one. |

On the bundled espresso corpus, the difference is concrete:

- **Grader OFF** keeps all 4 passages and the answer drifts into an irrelevant passage about
  milk-steaming technique.
- **Grader ON** keeps **1 of 4** — dropping 3 passages graded irrelevant at ~0.85 confidence — and
  the answer stays on the question.

### Try it (one click)

Open the Space → leave the **espresso** corpus and the **prefilled query** as they are → press
**Run query**. The trace accordions show per-passage label, confidence and accept/reject decision.

## Architecture: a pluggable contract, not a notebook

`src/grader_interface.py` defines the contract both arms implement:

```python
grade(query: str, passages: list[str], top_k: int = 5) -> list[ScoredPassage]
# ScoredPassage(passage, label, score, confidence)
```

`confidence` is the softmax over the three label-token logits at inference — derived, never
trained. There is deliberately **no `reasoning` field**: TREC qrels contain no reasoning text, so
any explanation the model produced would be an untrained side effect presented as a feature.

Because both the zero-shot grader and the QLoRA adapter satisfy the same interface, either is a
drop-in reranker for any RAG pipeline. That is what makes the ON/OFF comparison above a fair test
rather than two hand-built demos.

## Method, and how to check it

| Choice | Value | Why |
|---|---|---|
| Base model | Qwen2.5-1.5B-Instruct | Fits a free T4; genuinely instruction-tuned |
| Method | QLoRA, 4-bit NF4 + double quant | Adapter-only training in 16 GB VRAM |
| LoRA | r=16, α=32, dropout 0.05 | Agreeing value across 3 independent repos |
| Targets | q,k,v,o,gate,up,down | Same |
| LR / schedule | 2e-4, cosine, 5% warmup | Same |
| Effective batch | 16 (4 × accum 4) | Same |
| Epochs | 1 | Time decision on free T4, not a guideline |
| Loss | completion-only, after `Label:` | Gradients only where the signal is |
| Balancing | seeded replication in the data | Auditable; no custom loss code |
| Seed | 42 | — |

Everything here was validated per-axis against three independent GitHub repositories plus a
smoke test. I make **no claim that any single repo ran this exact batch/sequence pair** — that
pair is my configuration, tested by me, on a T4.

### Reproduce

The prompt template is **byte-identical** between baseline and fine-tuned runs — it is imported
from `src/dataset_utils.py`, never retyped, and its SHA-256 is recorded in the results JSON
(`prompt_sha256`) so a mismatch is detectable.

| File | What |
|---|---|
| `notebooks/01_data_preparation.py` | TREC DL 2020 qrels → (query, passage, label), split by whole query |
| `notebooks/02_baseline_evaluation.py` | Zero-shot baseline on the same 10 queries |
| `notebooks/03_fine_tuning.py` | QLoRA training, smoke test first |
| `notebooks/04_evaluation_comparison.py` | Fine-tuned eval + MMLU forgetting check |
| `notebooks/05_publish_to_hub.py` | Push weights + model card |
| `src/rag_pipeline.py` | Corrective RAG pipeline behind the Space |
| `src/rag_pipeline_smoke_test.py` | 7 checks, no model, no network — `python -m src.rag_pipeline_smoke_test` |
| `make_comparison.py` | Regenerates the table from the JSON; exits non-zero if a bar fails |
| `results/*.json` | Every number above, machine-readable |

Verify the bars yourself:

```bash
python make_comparison.py     # exit 0 == both bars pass
```

## Honest limitations

- **ZeroGPU is unavailable on this account**, so the Space currently serves a recorded run of the
  exact pipeline (`results/demo_trace.json`) rather than executing live. The Space logs the real
  reason: `spaces/zero/wrappers.py → torch.init → RuntimeError: No CUDA GPUs are available`.
  That is account/hardware provisioning, not application code. The replay is labelled as such in
  the UI, and it only serves a recording whose corpus *and* query match what was asked — showing a
  recording for a different question would be a lie.
- **10 test queries is a small evaluation.** TREC DL 2020 only has graded judgments for ~54
  queries total, and 10 were held out. The per-class supports (1498 / 300 / 324) are lopsided by
  the qrels themselves, not by choice. Treat macro-F1 as directional.
- **`relevant` barely improved (+1.8 pp).** The model still over-predicts relevant. This is the
  honest weak spot and the first thing I would attack with more training signal or a rebalanced
  loss.
- **MMLU forgetting was measured on 3 subjects / 607 questions** (world history, mathematics,
  computer science), not the full suite — enough to detect gross collapse, not subtle drift.
- **No hyperparameter search.** One configuration was trained and evaluated. I am reporting the
  number I got, not the best of N tries.

## License

MIT.