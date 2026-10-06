# Internship Final Project — Phases

> **Date:** 30 September 2026
> **Decision:** Ship Project B as prior work + an extension. Never present the existing fine-tune as new.
> **Spine:** Corrective RAG app first (ships fastest), Grounded Answer Checker second (the differentiated headline).
> **Status:** Verified against the repo on 30 Sep 2026. Phase 0 has not started.

---

## 1. Verified state of play

Checked against `results/`, `outputs/`, `notebooks/`, and `git log` on 30 Sep 2026.

| Asset | State |
|---|---|
| `config/sft_config.yaml` | Complete, single source of truth |
| `notebooks/01`–`05` | All five written |
| `results/baseline_results.json` | **Done.** acc 0.3695, macro-F1 0.2672, 2122 qrels / 10 queries, MMLU 0.4975 |
| `results/predictions_baseline.json` | Done (214KB) |
| `outputs/qlora-sft/` | **Empty — no adapter** |
| `results/finetuned_results.json` | **Does not exist** |
| `notebooks/03_fine_tuning.py` | Written, smoke-ready, **never executed** |

**The gap is exactly one thing: the training run never happened.** Everything upstream of it is done and reviewed. `03_fine_tuning.py` already has the smoke-test path, class balancing in the data, the completion-only collator, prompt-template reuse imported from notebook 02 rather than retyped, and graceful W&B degradation when no key is present.

This is good news. There is no research left to do on Phase 0 — only a run to execute.

---

## 2. The headroom argument

Reading `baseline_results.json`'s confusion matrix (rows = true, columns = predicted):

```
                 pred: irrelevant  partial  relevant
true irrelevant          483        3        1012     (support 1498)
true partial              34        2         264     (support  300)
true relevant              25        0         299     (support  324)
```

Column sums: **542 irrelevant, 5 partial, 1575 relevant.**

The zero-shot model answers "relevant" **74% of the time** while the truth is "irrelevant" **71% of the time**. It is not predicting the majority class — it is *anti-correlated* with the prior. That is a classic small-instruct-model instruction-following failure, and it means three useful things:

1. **There is large headroom.** The model has not learned the label distribution at all, so almost any real signal moves the number. The project's +10pp bar (0.2672 → 0.3672) is likely to clear without difficulty.
2. **The baseline is honest, not broken.** For reference, predicting all-irrelevant would score 70.6% accuracy but macro-F1 ≈ 0.112. The baseline's 0.267 already beats a trivial majority predictor, so the before/after comparison is meaningful rather than rigged.
3. **"Partial" is the real target.** F1 0.013 at baseline, 300 qrels of support. It is thin but not empty, and it is where the confusion matrix shows the model failing most.

**What this does not license:** do not claim the improvement is surprising. A model that predicts one class 74% of the time has not been trained. Say that plainly in the write-up — it is more credible than implying 1.5B parameters were rescued by QLoRA.

---

## 3. Phase 0 — Land the fine-tune (blocking)

Nothing ships until an adapter exists. This phase is the whole critical path.

| Step | Action | Done when |
|---|---|---|
| 0.1 | Smoke test on T4: `run(smoke=True)`, 100 steps | No OOM at batch 4 / seq 1024. On OOM, fall back to batch 2 / accum 8 / seq 512 |
| 0.2 | Full run, 1 epoch | Adapter in `outputs/qlora-sft/` |
| 0.3 | Push adapter to HF Hub | `sallahahmed/qwen2.5-1.5b-relevance-grader` exists |
| 0.4 | Evaluate on the same 10 held-out queries | `results/finetuned_results.json` written |
| 0.5 | Catastrophic forgetting check | Fine-tuned MMLU ≥ 90% of 0.4975 (= ≥ 0.4475) |
| 0.6 | Generate before/after table | `results/before_after_table.md` regenerated from both JSON files |

**Carry-over rules from `AGENTS.md` — these are not optional:**
- Identical prompt template in 02 and 03 (it is imported, not retyped — do not change that)
- Split by query, never by row
- Report N test queries alongside every number. **N = 10 queries / 2122 qrels.** A macro-F1 on 10 queries is directional evidence, not a tight estimate. Say so in the README.
- The forgetting bar is *relative* (≥90% of the 0.4975 baseline), not a fixed ≥25% floor. A fixed floor is random chance and proves nothing.

**If the run OOMs repeatedly:** the fallback pair (batch 2 / accum 8 / seq 512) preserves the same effective batch of 16, so the comparison against the baseline stays valid. Do not change the learning rate or LoRA config to escape an OOM — that changes the experiment.

**Artifact discipline:** Colab is free and loses VMs. Your commit log already records a lost VM. Push the adapter to HF Hub immediately after 0.2, before evaluating, so a disconnect does not cost the run.

---

## 4. Phase 1 — Corrective RAG app

Turns the grader into a working system a stranger can operate in ten seconds.

### Reused as-is
`src/grader_interface.py`, `config/sft_config.yaml`, `results/*.json`, notebooks 01–05, the adapter from Phase 0.

### New build

| # | Task | Notes |
|---|---|---|
| 1.1 | `src/rag_pipeline.py` — retriever → grader gate → generate → self-correct | Retriever: FAISS + a **local** embedding model. No API key, no rate limit. Colab is free; a paid embedding API is a dependency you cannot afford |
| 1.2 | HF Space (public) — Streamlit for an app-like demo, Gradio if RAM gets tight | Must load in under 10s or strangers bounce |
| 1.3 | 2–3 pre-built corpora + an upload path | The demo must work before the visitor has files. Ship something genuinely interesting to read |
| 1.4 | **Grader ON/OFF toggle** | The core of the product. See below |

### The ON/OFF toggle is the product

Same query, same retrieval, grader disabled vs enabled. The user sees the answer change.

This is what makes the project shareable instead of merely impressive: it reproduces the before/after table *in the hands of a stranger*, using their own documents. It is the difference between "here is my notebook" and "here is a thing you can try."

### Definition of done
- [ ] Public Space loads in <10s
- [ ] A stranger can run a query without reading docs first
- [ ] ON/OFF delta is visible on screen
- [ ] README leads with the before/after table, not a project description
- [ ] Every number carries N, and the directional-evidence caveat is present
- [ ] Space linked from both the README and the model card

---

## 5. Phase 2 — Grounded Answer Checker

The differentiated claim. This is the one to present as *your* idea.

**Task:** `(question, answer, evidence) → supported | partly_supported | unsupported`, plus which sentence is unsupported.

**Why this beats "3-class relevance" as a headline:** relevance grading is infrastructure. Nobody outside ML knows what a graded-relevance qrel is. "This model told me which sentence in the answer is not backed by the document" needs zero explanation and is a problem every company that has deployed RAG has.

**Why it is feasible so fast:** ~85% pipeline reuse. The machinery — QLoRA config, trainer, collator, split strategy, eval harness, publishing — is already written and already proven in Phase 0. The genuinely new work is dataset selection and label design.

### Dataset candidates — evaluate before committing

| Candidate | Why | Risk |
|---|---|---|
| **RAGTruth** (first choice) | Purpose-built RAG hallucination annotations; the task matches exactly | Check licence and download size before promising it |
| **FEVER** (second) | Claim verification with evidence; large and public | Evidence format differs from RAG context — needs remapping |
| **Synthetic generation** (third) | Full control over label shape, no licensing question | Requires an un-tuned teacher model and an honesty check on the data |

### Definition of done
- [ ] Dataset chosen and justified in writing, with the licence named
- [ ] Label scheme documented, including how "partly supported" is decided
- [ ] Baseline and fine-tuned numbers on the same split
- [ ] Runnable on the grader's evaluation harness, not a parallel one
- [ ] Demoable in the same HF Space as Phase 1, or a second one

---

## 6. Risks

| # | Risk | Severity | Mitigation |
|---|---|---|---|
| R1 | Training OOMs repeatedly | Medium | Fallback pair preserves effective batch 16. Change nothing else |
| R2 | Colab VM lost mid-run | High — has already happened | Push adapter to HF Hub the moment it saves, before evaluating |
| R3 | Improvement lands below the +10pp bar | Low | Headroom analysis (§2) says this is unlikely — the baseline barely learned the prior |
| R4 | Phase 1 demo looks weak if the grader is near-random | Medium | Ship Phase 1 on the ON/OFF framing so the delta is visible regardless of absolute quality |
| R5 | Over-claiming on 10 test queries | High | Rule 13 already covers this. Report N, call it directional, repeat it in the README |
| R6 | Phase 2 dataset turns out unusable | Medium | Decide the fallback before starting, not after |
| R7 | Scope creep — Phase 2 delays Phase 1 shipping | Medium | Phase 1 ships first. Do not start 2 until 1 is live on a public Space |

---

## 7. Open questions — need answers before dates can be set

| # | Question | Why it matters |
|---|---|---|
| Q1 | What is the internship deadline, and is there a demo or presentation day? | Sets the whole timeline |
| Q2 | How many hours per week are genuinely available? | Decides whether Phase 2 happens at all |
| Q3 | What does the company actually grade on — working demo, technical depth, business value, written report? | Changes which phase to prioritise |
| Q4 | Is the internship company in a specific industry? | If logistics, retail, or banking, a use case in *their* world beats a generic one |
| Q5 | Solo or team? Who reviews the work? | Changes review cycles |
| Q6 | Has the company seen Project B, or is it presented cold? | Changes how much context the write-up needs |

---

## 8. Recommended immediate action

**Run the Phase 0 smoke test.** It is the only step that unblocks everything else, it takes about 20 minutes on a free T4, and the code for it is already written and reviewed.

Do not start Phase 1 or 2 design work until 0.1 reports VRAM numbers. Everything above is designed to be resilient to a modest grader, but nothing above is designed to be resilient to no adapter at all.

---

*Companion to `project-b-relevance-grader.md` (the spec) and `AGENTS.md` (the rules). This file covers sequencing and current state; the spec covers design rationale; `AGENTS.md` is binding.*
