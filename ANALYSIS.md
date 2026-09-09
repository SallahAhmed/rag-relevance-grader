# Deep Analysis — GitHub Repos vs Project B Plan

## Date: 2026-09-09 | Owner: sallahahmed

---

## 1. Repos Analyzed

### 1.1 CloudAuraOfficial/cloudaura-finetune
**Relevance: ⭐⭐⭐⭐⭐ — Exact match for hardware + model**

| Aspect | Their Implementation | What We Should Adopt |
|---|---|---|
| Base model | Qwen/Qwen2.5-1.5B-Instruct | ✅ Same |
| Hardware | NVIDIA T4 GPU (16GB VRAM) via Google Colab | ✅ Same |
| Pipeline | 4 stages: Data Prep → SFT (QLoRA) → DPO → Eval | Use 3 stages (skip DPO for first iteration) |
| Config | YAML files (`pipeline/config/sft_config.yaml`) | Adopt YAML config pattern |
| LoRA r | 16 | Use 16 |
| LoRA alpha | 32 | Use 32 |
| Optimizer | paged_adamw_8bit | Use same |
| SFT LR | 2e-4, cosine, warmup | Use same defaults |
| SFT epochs | 3 | **Start with 1 for T4 time limits, then add 2nd pass** |
| SFT batch size | 4 | Use 4 on T4 |
| Gradient accum | 4 | Use 4 |
| Max seq length | 1024 | Use 1024 on T4 |
| Gradient checkpointing | Not explicitly listed but standard in TRL | **Must add: gradient_checkpointing: true** |
| DPO | Beta=0.1, sigmoid loss, lr=5e-5 | Optional — add after SFT baseline |
| Eval | Base vs SFT vs SFT+DPO comparison | We need Base vs Fine-tuned (our 3-class version) |
| Tracking | W&B (optional) | Required |
| Structure | `pipeline/scripts/`, `notebooks/`, `results/` | Adopt similar structure |
| Colab notebook | `notebooks/finetune_pipeline.ipynb` | Single notebook approach is cleaner for us |
| Docker showcase | nginx:alpine static site | Optional for portfolio |

**Key insight**: Their pipeline is the closest match. However, they do JSON extraction (not relevance grading), and they include DPO (which we don't need for the initial iteration). We can adopt their config structure and training parameters directly.

**Important note on hyperparameters**: CloudAura uses 3 epochs with batch_size=4 and grad_accum=4 (effective batch=16). But on T4 with disconnect limits, we start with 1 epoch. The batch=4 + seq_len=1024 combination was tested by *none* of the three source repos individually — each was conservative on at least one axis (samrat-kar: batch 2 / seq 512; CloudAura: batch 4 / seq 1024 but with 3 epochs and a V100-class GPU). We must add `gradient_checkpointing: true` and run a smoke test before committing to the full run. Fallback: samrat-kar's exact settings (batch 2 / accum 8 / seq 512) if smoke OOMs. See "Hyperparameter Provenance" section below.

---

### Hyperparameter Provenance — What Each Repo Actually Tested

| Hyperparameter | CloudAura | samrat-kar | cw1997 | Our adopted value | Validated by? |
|---|---|---|---|---|---|
| LoRA r | 16 | 16 | 16 | **16** | ✅ All 3 |
| LoRA alpha | 32 | 32 | 32 | **32** | ✅ All 3 |
| LR | 2e-4 | 2e-4 | 2e-4 | **2e-4** | ✅ All 3 |
| Batch size | 4 | 2 | 1-2 | **4** | ⚠️ Only CloudAura (with seq=1024) |
| Grad accum | 4 | 8 | 8 | **4** | ⚠️ Only CloudAura (with batch=4) |
| Effective batch | 16 | 16 | 16 | **16** | ✅ All 3 (same result) |
| Max seq length | 1024 | 512 | 2048/4096 | **1024** | ⚠️ Only CloudAura (with batch=4) |
| Optimizer | paged_adamw_8bit | paged_adamw_32bit | paged_adamw_8bit | **paged_adamw_8bit** | ✅ CloudAura + cw1997 |
| Epochs | 3 | 3 | 3 | **1** (T4-time decision, not from any guideline) | ❌ Unvalidated — see note |
| Gradient checkpointing | Implied | Not mentioned | Default on | **true** | ⚠️ Must verify in smoke test |

**The batch=4 + seq=1024 combination is untested by any single repo as a pair.** CloudAura uses batch=4 with seq=1024 but 3 epochs; samrat-kar uses batch=2 with seq=512; cw1997's VRAM table prescribes batch=1 for larger models at seq=2048. What IS validated across all three is the effective batch size of 16 (4×4, 2×8, 1×16) — the difference is how the batch is distributed across per-device size vs accumulation steps, which changes VRAM pressure on T4. This is why the smoke test in notebook 03 is mandatory before the full run, with samrat-kar's exact settings as fallback.

---

---

### 1.2 samrat-kar/llm-fine-tune
**Relevance: ⭐⭐⭐⭐⭐ — Best workflow pattern to follow**

| Aspect | Their Implementation | What We Should Adopt |
|---|---|---|
| Base model | Qwen/Qwen2.5-1.5B-Instruct | ✅ Same |
| Hardware | T4 GPU (16GB) | ✅ Same |
| Workflow | 5 notebooks: prep → baseline → train → eval → publish | **This is our exact workflow** |
| **Baseline** | **Notebook 2 explicitly evaluates base model before fine-tuning** | **Critical — this is our zero-shot baseline** |
| **Catastrophic forgetting check** | Notebook 4 tests base capabilities retained | Add this as validation |
| Evaluation metrics | ROUGE-L, Exact Match, MMLU | We use accuracy, macro-F1, confusion matrix |
| Framework | HuggingFace PEFT + TRL | Same |
| W&B | Required, link published in README | Same |
| Published model | `samrat-kar/qwen2.5-1.5b-sql-qlora` on HF Hub | Target: `sallahahmed/qwen2.5-1.5b-relevance-grader` |
| Results format | `results/baseline_results.json`, `results/finetuned_results.json` | Adopt this naming |
| Pre/post comparison | Explicit comparison in Notebook 4 | **Our before/after table comes from this** |
| Colab compatible | Upload whole repo to Drive, mount in Colab | ✅ Same approach |
| Config | `src/config.py` (single source of truth) | Adopt single config file |
| Time estimate | ~87 min training on H100, ~2-3 hrs on T4 | Plan accordingly |

**Key insight**: Their 5-notebook structure is **the best template** for our project. Specifically:
- Notebook 2 (baseline evaluation) is exactly what we need before fine-tuning
- The "catastrophic forgetting check" validates our model still works on other tasks
- Results JSON files provide the data for our before/after table
- The published model link is the portfolio artifact

**Their hyperparameters (FALLBACK if smoke test OOMs — NOT the adopted config):**
```python
lora_r = 16
lora_alpha = 32
lora_dropout = 0.05
target_modules = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
learning_rate = 2e-4
lr_scheduler = "cosine"
warmup_ratio = 0.05
epochs = 3  # may need to reduce for T4 time limits
batch_size = 2  # per device
grad_accum = 8  # effective batch = 16
max_seq_length = 512
optimizer = "paged_adamw_32bit"  # note: 32bit, not 8bit
bits = 4
quant_type = "nf4"
compute_dtype = "float16"
double_quant = True
```
**NOTE**: The adopted config (batch=4, accum=4, seq=1024) differs from samrat-kar's (batch=2, accum=8, seq=512). This is a deliberate T4-time tradeoff — larger batch + longer seq per step means fewer steps for the same data, but each step processes more information. See provenance table below for why this specific combination is untested by any single source repo.

---

### 1.3 cw1997/fine-tuning-playground
**Relevance: ⭐⭐⭐⭐ — Best VRAM/preset documentation and troubleshooting**

| Aspect | Their Implementation | What We Should Adopt |
|---|---|---|
| Models | Qwen3/3.5 (newer but same architecture) | Their presets apply to Qwen2.5 too |
| **VRAM presets** | Detailed table for every GPU tier | **Adopt their VRAM analysis for T4** |
| **VRAM/Model table** | 6-8GB: Qwen3.5-4B batch=1, accum=16 | T4 16GB: Qwen2.5-1.5B batch=2, accum=8 |
| Dataset size guidelines | <500 → 5 epochs lr=1e-4; 500-5000 → 3 epochs lr=2e-4 | Our ~11k judged qrels sit ABOVE this band — epochs=1 is a T4-time decision, not justified by this guideline |
| Troubleshooting | OOM fixes, NaN fixes, config fixes | Invaluable for T4 issues |
| Single file train.py | Complete pipeline in one script | We can use notebooks for clarity |
| Pre/post comparison | Built-in `--mode compare` | Our baseline vs fine-tuned comparison |
| Inspect scripts | `inspect_model.py`, `inspect_tokenizer.py` | Useful for debugging |
| Preset system | Named presets (`gpu-12gb-4b`, etc.) | Create our own `t4-16gb` preset |
| Thinking mode | Optional Qwen3 chain-of-thought | Not applicable to our task |
| Formats | ChatML, Alpaca, Text | We'll use custom format |
| AGENTS.md | Included in repo | They practice what they preach |

**Key insight**: Their **VRAM presets table** is the most valuable resource. For our T4 16GB setup:
- Qwen2.5-1.5B is smaller than Qwen3.5-4B, so we have headroom
- Their `gpu-12gb-4b` preset (batch=2, accum=8, eff_batch=16) is directly applicable
- Their dataset size guidelines tell us to use 2-3 epochs for 500-5000 examples
- **Our TREC DL judged set (~11k qrels)** is above that range, so epochs=1 is a T4-time decision, not justified by this guideline
- Their troubleshooting section will save us from OOM issues

---

### 1.4 grounded-ai/grounded_ai
**Relevance: ⭐⭐⭐ — Architecture pattern for RAG Relevance evaluation**

| Aspect | Their Implementation | What We Should Adopt |
|---|---|---|
| Interface | `Evaluator(model, eval_mode="RAG_RELEVANCE")` | Our GraderInterface follows this pattern |
| Output schema | `EvaluationOutput(score, label, confidence, reasoning)` | Our grader follows the shape but drops `reasoning` in v1 (no training signal for it in qrels) |
| Backends | Local SLM, OpenAI, Anthropic, HuggingFace | We use a fine-tuned local model |
| RAG_RELEVANCE mode | `evaluator.evaluate(response, query, context)` | Our grader: `grader(query, passages) → scored_list` |
| Type safety | Pydantic schemas | Consider adding Pydantic schemas |
| Privacy-first | 100% local evaluation | Our fine-tuned model is fully local |
| Phi-4 judge model | `grounded-ai/phi4-mini-judge` for baseline | Could use this as zero-shot baseline reference |

**Key insight**: GroundedAI proves that RAG Relevance evaluation is a recognized category. Their `RAG_RELEVANCE` eval_mode validates our project direction. Our fine-tuned model replaces their cloud API with a local adapter.

---

## 2. Critical Findings & Changes to the Plan

### Finding 1: We need BOTH SFT and baseline notebooks
The samrat-kar repo proves that a **dedicated baseline evaluation notebook** is essential. Our current plan lumps it into "Pipeline step 2" but doesn't give it its own artifact.

**Change to plan**: Add a formal notebook structure:
```
notebooks/
├── 01_data_preparation.ipynb    → Download TREC DL 2020, format data, split
├── 02_baseline_evaluation.ipynb → Zero-shot Qwen2.5-1.5B-Instruct on test split
├── 03_fine_tuning.ipynb         → QLoRA training via Unsloth, W&B logging
├── 04_evaluation_comparison.ipynb → Base vs Fine-tuned, per-class metrics, confusion matrix
└── 05_publish_to_hub.ipynb      → Merge adapter, push to HF Hub, write model card
```

### Finding 2: Training parameters need T4-specific adjustments
The cloudaura-finetune repo uses 3 epochs with batch_size=4 and grad_accum=4 (effective batch=16). But on T4 with disconnect limits, we start with 1 epoch — a T4-time decision, not derived from any repo guideline.

**Recommendation**: Start with 1 epoch, then do a second pass if time allows. Use:
```yaml
sft:
  model: Qwen/Qwen2.5-1.5B-Instruct
  lora_r: 16
  lora_alpha: 32
  lora_dropout: 0.05
  target_modules: [q_proj, k_proj, v_proj, o_proj, gate_proj, up_proj, down_proj]
  learning_rate: 2e-4
  scheduler: cosine
  warmup_ratio: 0.05
  epochs: 1  # T4-time decision; ~11k qrels sit above cw1997's 500-5000 band so that guideline doesn't apply
  batch_size: 4  # per device on T4 — ⚠️ unvalidated pair with seq=1024, smoke test first
  grad_accum: 4  # effective batch = 16 ✅ (all 3 repos)
  max_seq_length: 1024  # ⚠️ unvalidated pair with batch=4, smoke test first
  gradient_checkpointing: true  # mandatory
  optimizer: paged_adamw_8bit
  quant_type: nf4
  compute_dtype: float16
  double_quant: true
```

### Finding 3: The evaluation needs a "catastrophic forgetting check"
The samrat-kar repo includes MMLU accuracy check to ensure fine-tuning didn't break the base model's general capabilities. We should add this.

**Change to plan**: Add a sanity check: after fine-tuning, verify the model still handles general instruction-following tasks (e.g., answer simple questions, follow formatting instructions).

### Finding 4: Use YAML config, not hardcoded parameters
The cloudaura-finetune repo uses YAML config files. This makes hyperparameter sweeps and tracking easier.

**Change to plan**: Create `config/sft_config.yaml` as single source of truth. Reference it from notebooks.

### Finding 5: Results should be JSON files, not just tables
The samrat-kar repo saves `results/baseline_results.json`, `results/finetuned_results.json`. This is the data source for our before/after table.

**Change to plan**: Create `results/` directory with structured JSON outputs. The before/after table in README is generated from these files.

### Finding 6: The grounded-ai RAG_RELEVANCE pattern confirms our approach — with one correction
The `grounded-ai` package proves that RAG relevance evaluation is a real, recognized ML task. Their interface signature:
```python
Evaluator.evaluate(response, query, context) → EvaluationOutput(score, label, confidence, reasoning)
```
Our grader interface follows the shape but deliberately NOT the full contract:
```python
grader(query, passages, top_k) → List[ScoredDocument(score, label, confidence)]
```
**Correction (review 2026-09-09)**: grounded-ai's `reasoning` comes from frontier-LLM judges, not graded-relevance training data — not comparable to our setting. TREC qrels contain no reasoning text and no confidence signal, so v1 trains generative SFT (label as text) and derives confidence from label-token softmax at inference. No `reasoning` field until there is training signal for it.

### Finding 7: Pre/post fine-tuning comparison is critical
The cw1997 repo has `--mode compare` built into inference. We should do the same — explicit side-by-side comparison prompts that demonstrate the model's improvement.

**Change to plan**: Add comparison prompts to the evaluation notebook that show the same query processed by both base and fine-tuned models.

---

## 3. Updated Project Structure

```
finetune/
├── project-b-relevance-grader.md    # Updated spec (this file)
├── AGENTS.md                        # Project config + skill mappings
├── config/
│   └── sft_config.yaml             # Single source of truth for hyperparameters
├── notebooks/                       # 5 Jupyter notebooks
│   ├── 01_data_preparation.ipynb
│   ├── 02_baseline_evaluation.ipynb
│   ├── 03_fine_tuning.ipynb
│   ├── 04_evaluation_comparison.ipynb
│   └── 05_publish_to_hub.ipynb
├── src/
│   ├── config.py                    # All hyperparameters (single source of truth)
│   ├── dataset_utils.py             # Data loading, formatting, filtering
│   ├── evaluation_utils.py          # Metrics (accuracy, macro-F1, confusion matrix)
│   └── grader_interface.py          # GraderInterface implementation
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
├── agent-skills/                    # Upstream repo
├── delegate-skills/                 # Upstream repo
├── references/                      # 7 checklists
├── requirements.txt                 # All dependencies
└── README.md                        # Portfolio README with before/after table
```

---

## 4. Updated Hyperparameters (T4-optimized)

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
  target_modules:
    - q_proj
    - k_proj
    - v_proj
    - o_proj
    - gate_proj
    - up_proj
    - down_proj

training:
  learning_rate: 2e-4
  lr_scheduler: cosine
  warmup_ratio: 0.05
  epochs: 1  # T4-time decision, NOT from any guideline (~11k qrels sit above cw1997's 500-5000 band)
  per_device_batch_size: 4  # ⚠️ only CloudAura paired it with seq=1024
  gradient_accumulation_steps: 4  # effective batch = 16 ✅ (all 3 converge on 16)
  max_seq_length: 1024  # ⚠️ only CloudAura paired it with batch=4
  gradient_checkpointing: true  # mandatory — the batch=4 + seq=1024 pair is otherwise unvalidated
  optimizer: paged_adamw_8bit  # ✅ CloudAura + cw1997 (samrat-kar used 32bit)
  seed: 42
# SMOKE TEST notebook 03 (~50-100 steps) before full run; fallback on OOM: batch 2 / accum 8 / seq 512 (samrat-kar exact)

evaluation:
  split_strategy: group_by_query  # NEVER row-wise: ~54 judged queries; row split leaks queries into both sets
  test_queries: ~10  # report N test queries + N qrels alongside every number; macro-F1 here is noisy, directional only
  metrics:
    - accuracy
    - macro_f1
    - per_class_f1
    - confusion_matrix
  catastrophic_forgetting_check: true  # relative bar: fine-tuned MMLU >= 90% of notebook-02 baseline

dataset:
  source: TREC_DL_2020
  bucket_plan: "0→irrelevant, 1→partial, 2-3→relevant"
  format: "query, passage, label"

experiment:
  wandb_project: "rag-relevance-grader"
  wandb_entity: "sallahahmed"
  hf_model_id: "sallahahmed/qwen2.5-1.5b-relevance-grader"
```

---

## 5. Updated Success Criteria

| Metric | Baseline (zero-shot) | Fine-tuned target | Delta |
|---|---|---|---|
| Accuracy | TBD (notebook 2) | TBD (notebook 4) | +X pp (report N test queries) |
| Macro-F1 | TBD (notebook 2) | TBD (notebook 4) | **+10 percentage points minimum** (e.g. 0.45 → 0.55) |
| Per-class F1 | TBD (per-class report) | TBD (per-class report) | No class below 50% |
| Catastrophic forgetting | MMLU baseline (notebook 2) | Fine-tuned MMLU ≥ 90% of baseline | Relative bar — a fixed ≥25% floor is random chance |

---

## 6. What Changed from Original Plan

| Original | Updated | Reason |
|---|---|---|
| Single notebook approach | 5-notebook workflow | samrat-kar proves this pattern |
| Hardcoded hyperparameters | YAML config file | CloudAura uses YAML configs |
| No baseline notebook | Notebook 2: dedicated baseline evaluation | Essential for valid comparison |
| No catastrophic forgetting check | Added to Notebook 4 | Proves fine-tuning adds value, not just memorization |
| Generic evaluation | JSON results + structured outputs | Reproducible before/after table |
| No pre/post comparison prompts | Built-in comparison in evaluation | Demonstrates real improvement |
| No config management | Single `config/sft_config.yaml` | Hyperparameter sweeps easier |
| Generic timeline | 5-stage timeline with per-stage artifacts | Clear milestones and deliverables |
| "Use Unsloth" only | Unsloth OR TRL SFTTrainer | CloudAura uses TRL, samrat-kar uses TRL |
| No requirements.txt | Added `requirements.txt` | Reproducibility |
| No GraderInterface file yet | `src/grader_interface.py` added | Architecture pattern from ConnexioRag |

---

## 7. Key Lessons from Repos

1. **Always run a baseline before fine-tuning** — samrat-kar does this explicitly; it's the only way to prove your fine-tuning adds value. Also record the MMLU baseline here — the forgetting check needs a pre-finetune number to compare against.
2. **Save results as structured JSON** — enables automatic README table generation and reproducibility
3. **Use YAML configs** — makes hyperparameter sweeps and tracking trivial
4. **Plan for T4 time limits** — start with 1 epoch (a T4-time decision, not a guideline), have a 2nd pass ready
5. **Include a relative catastrophic forgetting check** — fine-tuned MMLU ≥ 90% of baseline; a fixed ≥25% floor is random chance
6. **Pre/post comparison prompts** — the most compelling evidence of improvement
7. **Colab notebook structure** — upload whole repo, mount Drive, run sequentially
8. **cw1997's 500–5000-examples/3-epochs band does NOT cover our ~11k qrels** — don't cite it to justify epochs=1; epochs=1 stands on T4-time grounds alone
9. **Effective batch size 16 is the validated constant** — per_device_batch_size × grad_accum; all 3 repos converge on 16, they differ only in how it's split
10. **Split eval by query, never by row** — TREC DL 2020 covers only ~54 queries; row-wise splits leak queries and ~5 held-out queries of macro-F1 is noisy, directional evidence only
11. **Smoke-test the unvalidated pair** — batch=4 + seq=1024 together was tested by no single repo; gradient_checkpointing: true + short smoke run + samrat-kar fallback is mandatory
10. **LoRA r=16, alpha=32 is the standard starting point** — proven across all repos
