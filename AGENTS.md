# AGENTS.md — Project B: Fine-tuned RAG Relevance Grader

**Owner**: sallahahmed | **Created**: 2026-09-09 | **Last updated**: 2026-09-09

## Project Overview

Fine-tune Qwen2.5-1.5B-Instruct via QLoRA (4-bit NF4) to classify query-passage relevance into 3 classes (irrelevant / partial / relevant), benchmarked against zero-shot baseline. Target: free Colab/Kaggle T4 (16GB VRAM). Hyperparameters and workflow validated against 3+ existing GitHub repos (CloudAuraOfficial, samrat-kar, cw1997).

## Skill Workflow Mapping

This project follows the agent-skills lifecycle, mapped to ML-specific work, plus delegate-skills for parallel execution:

### agent-skills lifecycle

| Phase | Command | Skill | Project Action |
|---|---|---|---|
| **Define** | — | `spec-driven-development` | Formalize `project-b-relevance-grader.md` into executable spec with acceptance criteria |
| **Define** | — | `constraint-driven-development` | Enforce T4 16GB VRAM, Colab session limits, disconnect resilience |
| **Plan** | — | `planning-and-task-breakdown` | Break 3-week timeline into weekly verifiable tasks with notebook-level artifacts |
| **Build** | — | `incremental-implementation` | 5 notebooks: data prep → baseline → train → eval → publish |
| **Build** | — | `source-driven-development` | Ground every ML decision in official docs + validated GitHub repos |
| **Build** | — | `context-engineering` | Manage experiment context: prompts, hyperparameters, W&B runs, dataset versions |
| **Verify** | — | `debugging-and-error-recovery` | Training crashes, OOM on T4, dataset loading failures, HF upload issues |
| **Ship** | — | `git-workflow-and-versioning` | Track code changes, model adapter versions, experiment commits |
| **Ship** | — | `documentation-and-adrs` | README with benchmark table first, model card on HF Hub, architecture decisions |

### delegate-skills for parallel execution

| Workflow | Skill | Role |
|---|---|---|
| **Fleet setup** | `delegate-setup` | Discover installed CLIs (OpenCode, Codex, Claude Code), create fleet map |
| **Execution** | `opencode-delegate` | Delegate notebook writing, data prep, or eval code to separate OpenCode session |
| **Execution** | `codex-delegate` | Delegate tasks to Codex CLI for parallel execution |
| **Execution** | `claude-delegate` | Delegate to Claude Code for review or planning tasks |

The orchestrator always keeps the commit — delegates produce diffs, the reviewer reviews and lands.

### Execution bridge

| Workflow | Skill | Role |
|---|---|---|
| **Colab GPU** | `colab-operator` | Provision T4 sessions, run scripts, sync artifacts via the `colab` CLI (see skill for binary path + session policy) |

All Colab work goes through `colab-operator`. Always stop sessions when done; `colab run` self-cleans.

### Architecture target

The grader implements a `GraderInterface` (modeled after ConnexioRag's `RerankerInterface` and grounded-ai's `Evaluator`) so it can be a drop-in replacement for existing rerankers in any RAG pipeline.

## Core Rules

1. If a task matches a skill (even 1% chance), invoke it via the `skill` tool before doing anything else
2. Never skip the constraint check — every decision must be validated against the 16GB T4 limit
3. Every experiment iteration must be logged to W&B before touching code
4. The before/after benchmark table is the primary artifact — it leads the README, not the description
5. Prompt template must be identical between zero-shot baseline and fine-tuning for valid comparison
6. Per-class metrics (not just aggregate) must be reported — class imbalance is expected
7. Delegate-skills relay pattern: dispatch → poll → review diff → land commit (user keeps commit)
8. Every task should consider whether it can be delegated to a separate CLI session
9. Follow the 5-notebook workflow: prep → baseline → train → eval → publish
10. Use `config/sft_config.yaml` as single source of truth for hyperparameters
11. Save results as JSON files — the before/after table is generated from these
12. Run a catastrophic forgetting check (fine-tuned MMLU ≥ 90% of notebook-02 baseline — never a fixed ≥25% floor)
13. Split eval data by query, never by row; always report N test queries alongside metrics
14. "Validated against 3+ repos" means per-axis provenance (see ANALYSIS.md table) + smoke test — never claim the batch=4 + seq=1024 pair was tested by any single repo
15. Success bar is +10 percentage points macro-F1 (e.g. 0.45 → 0.55), not +10% relative
16. The skills layer is optional orchestration practice, not a prerequisite — it must never block training/eval work

## Project Files

- `project-b-relevance-grader.md` — Reference spec with validated hyperparameters and architecture
- `ANALYSIS.md` — Deep analysis of GitHub repos and what we adopted from each
- `AGENTS.md` — This file
- `config/sft_config.yaml` — Single source of truth for hyperparameters
- `.opencode/skills/` — 14 skill copies from `agent-skills/skills/` and `delegate-skills/skills/`
- `references/` — Copied checklists from agent-skills
- `agent-skills/` — Upstream agent-skills repo (do not edit)
- `delegate-skills/` — Upstream delegate-skills repo (do not edit)
- `delegate-skills/AGENTS-reference.md` — Copy of delegate-skills docs for reference

## Reference Repos (validated patterns)

| Repo | What We Adopted |
|---|---|
| `CloudAuraOfficial/cloudaura-finetune` | 4-stage pipeline, YAML config, T4 GPU setup, paged_adamw_8bit, LoRA r=16/alpha=32 |
| `samrat-kar/llm-fine-tune` | 5-notebook workflow, baseline evaluation notebook, catastrophic forgetting check, JSON results files |
| `cw1997/fine-tuning-playground` | VRAM presets, dataset size guidelines, troubleshooting, pre/post comparison, AGENTS.md |
| `grounded-ai/grounded_ai` | RAG_RELEVANCE eval pattern, Evaluator interface design |
| `ConnexioRag` | GraderInterface/RerankerInterface pattern |

## Hyperparameters (validated, T4-optimized)

```yaml
lora_r: 16  # ✅ all 3 repos
lora_alpha: 32  # ✅ all 3 repos
lora_dropout: 0.05  # ✅ all 3 repos
target_modules: [q_proj, k_proj, v_proj, o_proj, gate_proj, up_proj, down_proj]  # ✅ all 3 repos
learning_rate: 2e-4  # ✅ all 3 repos
lr_scheduler: cosine  # ✅ all 3 repos
warmup_ratio: 0.05  # ✅ samrat-kar
epochs: 1  # T4-time decision, NOT from any guideline (~11k qrels sit above cw1997's 500-5000 band)
per_device_batch_size: 4  # ⚠️ only CloudAura paired it with seq=1024
gradient_accumulation_steps: 4  # effective batch = 16 ✅ (all 3 converge on 16)
max_seq_length: 1024  # ⚠️ only CloudAura paired it with batch=4
gradient_checkpointing: true  # mandatory — the batch=4 + seq=1024 pair is otherwise unvalidated
optimizer: paged_adamw_8bit  # ✅ CloudAura + cw1997 (samrat-kar used 32bit)
quant_type: nf4  # ✅ all 3
compute_dtype: float16  # ✅ all 3
double_quant: true  # ✅ all 3
seed: 42
# SMOKE TEST notebook 03 (~50-100 steps) before full run; fallback on OOM: batch 2 / accum 8 / seq 512 (samrat-kar exact)
```

## Lifecycle Execution

For every request:
1. Determine which skill applies (check all phases)
2. Invoke the appropriate skill using the `skill` tool
3. Follow the skill workflow strictly
4. For ML-specific steps (data prep, training, evaluation), ensure constraint-driven-development is checked first
5. Only proceed after required steps (spec, plan, constraint check) are complete
6. Always reference ANALYSIS.md for decisions about hyperparameters and architecture

## Anti-Rationalization

- "This is just a notebook cell" → still follows incremental-implementation
- "I'll log W&B later" → constraint-driven-development requires it now
- "The baseline is obvious" → source-driven-development requires documented prompt template
- "One more training run won't hurt" → debugging-and-error-recovery covers runaway iterations
- "The README can wait" → documentation-and-adrs is the deliverable
- "I know the hyperparameters" → ANALYSIS.md validates against 3+ repos, don't guess
- "GraderInterface can wait" → architecture is a portfolio artifact, plan it from the start

## Open Questions (from project-b-relevance-grader.md)

- Exact bucket thresholds if 0–3 scale doesn't split evenly into 3 classes
- Full graded set vs stratified subset given Colab time limits (stratified-by-label split is decided; subset size still open)
- Prompt template for zero-shot baseline and fine-tuning data (must be identical)
- RESOLVED: generative SFT (label as text), confidence from label-token softmax, no `reasoning` in v1 contract
- Colab disconnect strategy: pre-process dataset offline, upload to HF Hub, then in Colab just download and train
- PINNED: +10 percentage points macro-F1; fine-tuned MMLU ≥ 90% of notebook-02 baseline
- W&B account setup required
- GraderInterface v1 contract pinned: `grader(query, passages, top_k) → List[ScoredDocument(score, label, confidence)]`
- Whether to use Unsloth specifically or standard TRL SFTTrainer (both validated)
- Whether to include DPO in a second iteration or skip it for the initial run
