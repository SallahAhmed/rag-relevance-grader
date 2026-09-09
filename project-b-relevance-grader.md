# Project B — fine-tuned RAG relevance grader

**Owner**: sallahahmed | **Goal**: Build a portfolio-grade ML project demonstrating fine-tuning with measurable results, integrating real-world architecture patterns.

A reference doc to explore every piece of the stack before writing code. Read top to bottom, follow the links, then come back to scaffold.

## The idea in one paragraph

Most "I fine-tuned an LLM" portfolio projects are unverifiable claims with no number attached. This project fixes that: fine-tune a small open model to classify query-passage relevance into three classes (irrelevant / partial / relevant), benchmark it against the same model used zero-shot with no fine-tuning, and publish the before/after table. The table — not the model — is the actual portfolio artifact. It's also a real, reusable component: this is exactly the kind of grader a Corrective RAG pipeline uses to decide whether a retrieved passage is good enough to answer with.

**Architecture goal**: The grader implements a pluggable `GraderInterface` (modeled after `RerankerInterface` in the ConnexioRag project and `grounded-ai`'s `Evaluator` pattern) so it can be a drop-in replacement for existing rerankers like Jina in any RAG pipeline.

## Why this stack, specifically

- Every decision below was made against one constraint: free-tier Colab/Kaggle T4 GPUs (16GB VRAM, session limits, possible disconnects). Nothing here needs paid compute.
- The goal is a fine-tuning story with a measured result, not the biggest model possible.
- The grader is designed to integrate into a real RAG architecture, not exist as a standalone notebook.
- Hyperparameters, architecture, and workflow validated against 3+ existing GitHub repos (CloudAura, samrat-kar, cw1997).

## Tech stack

| Layer | Tool | Role |
|---|---|---|
| Base model | Qwen2.5-1.5B-Instruct | Small enough for fast iteration on a free T4, still a real instruction-tuned LLM |
| Fine-tuning method | QLoRA (4-bit NF4) | Trains a small adapter instead of the full model — fits in 16GB VRAM |
| Fine-tuning framework | Hugging Face TRL SFTTrainer + PEFT LoRA (Unsloth or standard) | Purpose-built for exactly this setup; validated against multiple repos |
| Quantization backend | bitsandbytes | Handles the 4-bit loading |
| Dataset | TREC DL 2020 graded-relevance judgments (MS MARCO passage corpus) | Larger dataset; solve issues as they arise |
| Experiment tracking | Weights & Biases (free tier) | Logs loss curves, hyperparameters, run comparisons |
| Evaluation | scikit-learn + custom metrics (accuracy, macro-F1, per-class F1, confusion matrix, MMLU forgetting check) | Standard classification metrics — no RAGAS needed here |
| Model hosting | Hugging Face Hub | Where the trained adapter and model card live |
| Compute | Google Colab / Kaggle free T4 | Zero-cost GPU access, the actual constraint on every other choice |
| Architecture pattern | GraderInterface (ConnexioRag RerankerInterface + grounded-ai Evaluator) | Pluggable interface for the grader to slot into any RAG pipeline |
| Task delegation | delegate-skills (OpenCode/Codex delegates) | Parallel execution: one agent plans, another executes, reviewer keeps commit |
| Config format | YAML (`config/sft_config.yaml`) | Single source of truth for hyperparameters, validated against CloudAura |
| Output format | JSON results files (`results/baseline_results.json`, `results/finetuned_results.json`) | Structured artifacts for the before/after table, validated against samrat-kar |

## Component details

### Base model — Qwen2.5-1.5B-Instruct

Small enough to fine-tune multiple times within a single free Colab session without hitting VRAM or time limits, while still being a genuinely capable instruction-following model — the fine-tuning result is a real signal, not a toy. Model card: https://huggingface.co/Qwen/Qwen2.5-1.5B-Instruct

### Fine-tuning method — QLoRA (4-bit NF4)

QLoRA loads the base model in 4-bit precision and trains small low-rank adapter weights on top, instead of updating all 1.5B parameters. Uses `paged_adamw_8bit` optimizer. Standard hyperparameters validated across 3 repos:

```yaml
lora_r: 16
lora_alpha: 32
lora_dropout: 0.05
target_modules: [q_proj, k_proj, v_proj, o_proj, gate_proj, up_proj, down_proj]
learning_rate: 2e-4
lr_scheduler: cosine
warmup_ratio: 0.05
per_device_batch_size: 4
gradient_accumulation_steps: 4  # effective batch = 16
max_seq_length: 1024
optimizer: paged_adamw_8bit
quant_type: nf4
compute_dtype: float16
double_quant: true
```

- Unsloth docs: https://docs.unsloth.ai
- PEFT docs: https://huggingface.co/docs/peft
- Standard PEFT + TRL alternative is equally valid

### Dataset — TREC DL graded relevance on MS MARCO (2020 track)

MS MARCO's passage set on its own only has binary relevance labels. The TREC Deep Learning 2020 track reused the same MS MARCO passage corpus but added NIST human assessors' graded relevance judgments (rated 0–3) for a smaller, high-quality query set. Bucket plan: 0 → irrelevant, 1 → partial, 2–3 → relevant. Using 2020 track (larger dataset); solve issues during execution.
- Track hub (start here): https://microsoft.github.io/msmarco/TREC-Deep-Learning
- 2020 guidelines + file layout: https://microsoft.github.io/msmarco/TREC-Deep-Learning-2020.html
- Dataset size guideline: 500–5000 examples → 3 epochs at lr=2e-4 (from cw1997 fine-tuning-playground)

### Experiment tracking — Weights & Biases

Logs every training run automatically — loss curves, hyperparameters, run comparisons — so you can show *how* you found the best configuration, not just the final number. W&B = Weights & Biases — free tier accounts at https://wandb.ai

### Evaluation — 5-metric approach

Produced twice (once for untouched base model zero-shot, once for fine-tuned), on the same held-out queries:

- **Accuracy** — overall correct-classification rate
- **Macro-F1** — treats all three classes equally, doesn't hide poor performance on the "partial" class
- **Per-class F1** — individual precision/recall/F1 for irrelevant/partial/relevant
- **Confusion matrix** — shows exactly which classes get confused with which
- **MMLU sanity check** (catastrophic forgetting) — verifies fine-tuning didn't break general capabilities

This produces one clean before/after table — the artifact that leads the README. Results saved as structured JSON in `results/`.

### Model hosting

The trained adapter (and optionally the merged model) gets pushed to Hugging Face Hub with a model card documenting the dataset, method, and benchmark numbers — so anyone reviewing your portfolio can inspect it directly.
- Docs: https://huggingface.co/docs/hub/index
- Target model ID: `sallahahmed/qwen2.5-1.5b-relevance-grader`

### Architecture interface — GraderInterface

Modeled after two proven patterns:
1. `RerankerInterface` in the ConnexioRag project — abstract base class: `rerank(query, documents, top_k) → List[RetrievedDocument]`
2. `Evaluator` in grounded-ai — unified interface: `evaluate(response, query, context) → EvaluationOutput(score, label, confidence, reasoning)`

Our GraderInterface contract (v1): `grader(query, passages, top_k) → List[ScoredDocument(score, label, confidence)]` where confidence = softmax probability over the label-token logits at inference. Deliberately NO `reasoning` field in v1 — TREC qrels contain no reasoning text, so any model-generated reasoning would be an untrained side effect and unreliable. (grounded-ai's `reasoning` field comes from frontier-LLM judges, not from graded-relevance training data — not comparable.) The fine-tuned adapter implements this interface, making it a drop-in replacement for existing rerankers like Jina in any RAG pipeline. This is the key architecture decision that turns a notebook into a portfolio-grade engineering artifact.

### Task delegation — delegate-skills

Uses `amElnagdy/delegate-skills` for parallel execution. The workflow:
- **delegate-setup**: Discovers installed CLIs (OpenCode, Codex, Claude Code) and creates a fleet map
- **opencode-delegate**: Delegates notebook writing, data preprocessing, or evaluation code to a separate OpenCode session
- **codex-delegate**: Delegates tasks to Codex CLI for parallel execution
- **claude-delegate**: Delegates to Claude Code for review or planning tasks
- The relay pattern: dispatch → poll → review diff → land commit (the user/orchestrator always keeps the commit)

This allows the project to run multiple workstreams in parallel: one agent handles data prep, another writes training notebooks, another drafts documentation — all while the orchestrator reviews and commits.

**Scope honesty**: the agent-skills/delegate-skills layer (14 skill copies, dispatch→poll→review→land relay) is orchestration practice on top of the portfolio artifact, not a prerequisite for it. For a solo 5-notebook project this specified, it is overhead — kept deliberately for the tooling story, and marked optional wherever it would block the actual training/eval work.

## Pipeline, end to end (5 notebooks as jupytext percent-scripts)

Notebooks are `.py` files with `# %%` cell markers (samrat-kar pattern):
runnable as scripts, openable as notebooks, diffable in git. Convert anytime
with `jupytext --to notebook notebooks/01_data_preparation.py`. Logic lives
in `src/`; notebooks orchestrate and record results.

**Notebook structure** (validated against samrat-kar's 5-notebook workflow):

1. **`01_data_preparation.py`** — Download TREC DL 2020 qrels + MS MARCO passages, join into (query, passage, label) triples, bucket 0–3 ratings into 3 classes. **Split by query, not by row** (whole queries held out — row-wise splitting leaks the same query into train and test). Use a **stratified-by-label** train split and report the class distribution; the "partial" class will be thin, so plan class weights or balanced sampling in notebook 03 rather than handling imbalance only at eval. Save preprocessed data as CSV/Parquet to Hugging Face Hub for Colab continuity.

2. **`02_baseline_evaluation.py`** — **Critical**: Prompt Qwen2.5-1.5B-Instruct zero-shot on the held-out queries with identical template used for fine-tuning. Record accuracy/macro-F1/per-class-F1/confusion matrix **plus N test queries / N qrels and the MMLU baseline score** (needed for the relative forgetting bar in notebook 04). Save `results/baseline_results.json`. Includes pre/post fine-tuning comparison prompts.

3. **`03_fine_tuning.py`** — QLoRA training via TRL SFTTrainer + PEFT LoRA on the train split. Loads `config/sft_config.yaml`. Logs to W&B. Uses `paged_adamw_8bit` optimizer, `gradient_checkpointing: true`, 1 epoch initially (T4-time decision — add 2nd if time permits). **Mandatory smoke test first** (~50–100 steps); on OOM fall back to samrat-kar's exact settings (batch 2 / accum 8 / seq 512). Saves adapter to `outputs/qlora-sft/`.

4. **`04_evaluation_comparison.py`** — Same held-out queries, same metrics, on the fine-tuned model. **Catastrophic forgetting check (relative bar)**: fine-tuned MMLU ≥ 90% of the notebook-02 baseline — a fixed ≥25% floor is random chance and proves nothing. Save `results/finetuned_results.json`. Generate `results/before_after_table.md` from JSON files (report N test queries alongside every number — macro-F1 on ~5–10 queries is noisy, directional evidence only). Side-by-side comparison prompts.

5. **`05_publish_to_hub.py`** — Merge adapter with base model (or push adapter standalone). Push to Hugging Face Hub. Write model card with benchmark numbers. Publish `results/before_after_table.md` and README.

## File structure

```
finetune/
├── project-b-relevance-grader.md    # This spec
├── AGENTS.md                        # Project config + skill mappings
├── config/
│   └── sft_config.yaml             # Single source of truth for hyperparameters
├── notebooks/                       # 5 percent-scripts (.py with # %% cells)
│   ├── 01_data_preparation.py
│   ├── 02_baseline_evaluation.py
│   ├── 03_fine_tuning.py
│   ├── 04_evaluation_comparison.py
│   └── 05_publish_to_hub.py
├── src/
│   ├── config.py                    # All hyperparameters (single source of truth)
│   ├── dataset_utils.py             # Data loading, formatting, filtering
│   ├── evaluation_utils.py          # Metrics (accuracy, macro-F1, per-class F1, confusion matrix, MMLU)
│   ├── grader_interface.py          # GraderInterface implementation
│   └── requirements.txt             # All Python dependencies
├── results/                         # JSON results files
│   ├── baseline_results.json
│   ├── finetuned_results.json
│   ├── before_after_table.md        # Generated README table
│   └── *.png                        # Plots (loss curves, confusion matrix)
├── outputs/                         # Model checkpoints
│   ├── qlora-sft/                   # Training checkpoints
│   └── merged_model/                # Merged model for publishing
├── data/                            # Processed datasets
│   └── processed/                   # CSV/Parquet files
├── .opencode/skills/                # 14 skill copies
├── agent-skills/                    # Upstream repo (do not edit)
├── delegate-skills/                 # Upstream repo (do not edit)
├── references/                      # 7 checklists
└── README.md                        # Portfolio README with before/after table
```

## Timeline (2–3 weeks)

| Week | Focus | Artifacts |
|---|---|---|
| **1** | Data prep + baseline zero-shot evaluation + delegate-skills fleet setup | `01_data_preparation.ipynb`, `02_baseline_evaluation.ipynb`, `results/baseline_results.json`, fleet config |
| **2** | Fine-tuning (1 epoch, then 2nd if time permits), W&B logging, parallel delegate execution | `03_fine_tuning.ipynb`, `outputs/qlora-sft/`, W&B logs |
| **3** | Evaluation + comparison, catastrophic forgetting check, README, HF Hub push | `04_evaluation_comparison.ipynb`, `results/finetuned_results.json`, `results/before_after_table.md`, `05_publish_to_hub.ipynb`, published model |

## What "done" looks like

- A public repo with a README that leads with the before/after benchmark table, not a project description.
- A model card on Hugging Face Hub documenting exactly what was trained on and how.
- A working `GraderInterface` implementation in `src/grader_interface.py` that could plug into any RAG pipeline.
- 5 notebooks following the samrat-kar validated workflow (prep → baseline → train → eval → publish).
- Results saved as structured JSON enabling automatic before/after table generation.
- Evidence of using delegate-skills for parallel execution (fleet config, delegated tasks).
- One sentence you can say in an interview: "I fine-tuned a 1.5B model as a relevance grader and improved macro-F1 from X to Y over the zero-shot baseline." — with real numbers filling in X and Y.
- Minimum target: **+10 percentage points macro-F1** over zero-shot baseline (e.g. 0.45 → 0.55, not 0.45 → 0.495).

## Validated hyperparameters (T4-optimized)

```yaml
# config/sft_config.yaml
model:
  base_model: Qwen/Qwen2.5-1.5B-Instruct
  load_in_4bit: true
  quantization: nf4
  compute_dtype: float16
  double_quant: true

lora:
  r: 16
  alpha: 32
  dropout: 0.05
  target_modules: [q_proj, k_proj, v_proj, o_proj, gate_proj, up_proj, down_proj]

training:
  learning_rate: 2e-4
  lr_scheduler: cosine
  warmup_ratio: 0.05
  epochs: 1  # T4-time decision (NOT from any dataset-size guideline — our ~11k qrels sit above cw1997's 500-5000 band); add 2nd if time permits
  per_device_batch_size: 4
  gradient_accumulation_steps: 4  # effective batch = 16 (matches all 3 repos)
  max_seq_length: 1024
  gradient_checkpointing: true  # mandatory — batch=4 + seq=1024 as a pair is untested by any single source repo; smoke test in notebook 03 first, fallback to batch 2 / accum 8 / seq 512 on OOM
  optimizer: paged_adamw_8bit
  seed: 42

evaluation:
  # SPLIT BY QUERY, not by row: TREC DL 2020 passage judgments cover only ~54 queries.
  # Row-wise splitting leaks the same query into train and test. Hold out whole queries
  # (e.g. ~10 queries) and report N test queries + N qrels alongside every number.
  split_strategy: group_by_query
  test_queries: ~10
  metrics: [accuracy, macro_f1, per_class_f1, confusion_matrix, mmlu_forgetting_check]
  # Single before/after numbers on ~5-10 held-out queries will be noisy — treat them as
  # directional evidence, not tight estimates. Consider repeated splits if eval stays cheap.
  mmlu_forgetting_bar: "fine-tuned MMLU >= 90% of notebook-02 baseline (a fixed >=25% floor is random chance and proves nothing)"

dataset:
  source: TREC_DL_2020
  bucket_plan: "0→irrelevant, 1→partial, 2-3→relevant"
  format: "query, passage, label"

experiment:
  wandb_project: "rag-relevance-grader"
  wandb_entity: "sallahahmed"
  hf_model_id: "sallahahmed/qwen2.5-1.5b-relevance-grader"
```

## Open questions to resolve while executing

- Exact bucket thresholds if the 0–3 scale doesn't split evenly into 3 classes (class imbalance likely — report per-class metrics)
- Whether to fine-tune on full graded set or stratified subset, given Colab session time limits
- Prompt template for zero-shot baseline and fine-tuning data — must be identical between the two
- Training data format: RESOLVED direction — generative SFT (label as text) for v1; confidence derived from label-token softmax/logit probability at inference, NOT from training data. Model-generated "reasoning" is excluded from the v1 contract (untrained side effect, unreliable).
- Colab disconnect strategy: pre-process dataset offline, upload to HF Hub, then in Colab download and train
- W&B account setup required
- Whether to use Unsloth specifically or standard TRL SFTTrainer (both are validated)
- GraderInterface contract (resolved before notebook 03 locks the data format): `grader(query, passages, top_k) → List[ScoredDocument(score, label, confidence)]` where confidence = softmax over label-token logits. No `reasoning` field in v1 — TREC qrels contain no reasoning text, so a reasoning output would be an untrained side effect.
- Whether to include DPO (like CloudAura) in a second iteration or skip it for the initial run
