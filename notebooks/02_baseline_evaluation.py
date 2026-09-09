# %% [markdown]
# # 02 — Baseline evaluation (zero-shot, untouched model)
# **Goal:** score Qwen2.5-1.5B-Instruct with NO fine-tuning on the held-out
# queries, using the byte-identical `PROMPT_TEMPLATE` notebook 03 will train on.
# **Acceptance:** `results/baseline_results.json` written with accuracy,
# macro-F1, per-class F1, confusion matrix, N test queries/qrels — PLUS the
# MMLU baseline score (notebook 04's forgetting check needs it).
# Run in Colab T4. This number is the "before" in the before/after table.

# %%
import sys
from pathlib import Path

ROOT = Path.cwd()
sys.path.insert(0, str(ROOT) if (ROOT / "src").is_dir() else str(ROOT.parent))

from src.config import load_config
from src import dataset_utils as du
from src import evaluation_utils as eu

cfg = load_config()
print("template in force:")
print(du.PROMPT_TEMPLATE)
print("test queries:", cfg.evaluation.test_queries)

# %% [markdown]
# ## 1. Load base model (4-bit, eval mode, no adapters)
# RUN: `AutoModelForCausalLM.from_pretrained(cfg.model.base_model,
# load_in_4bit=True, device_map="auto")` + tokenizer with left padding.
# No training, no gradients — `model.eval()` + `torch.no_grad()`.

# %% [markdown]
# ## 2. Grade every held-out pair
# RUN: for each (query, passage) in test: render `du.format_example` prompt,
# greedy-decode ONE label token, take argmax over the three label-token
# logits for the predicted label AND the softmax share as `confidence`.
# Same decoding path the fine-tuned model will use in notebook 04.

# %% [markdown]
# ## 3. Score + save
# RUN: `eu.compute_classification_metrics(y_true, y_pred)` →
# `eu.save_results_json("results/baseline_results.json", {...metrics,
# "n_test_queries": ..., "n_test_qrels": ..., "mmlu_baseline": ...})`.
# ## 4. MMLU baseline (subset, same harness notebook 04 will reuse)
# RUN: 3-subject MMLU slice (e.g. high-school CS / maths / history),
# greedy, record accuracy. This single number is the denominator of the
# ≥90% retention bar — do not skip it.

# %%
# Local sanity check (no GPU needed): metrics plumbing on toy labels.
toy_true = ["relevant", "partial", "irrelevant", "relevant", "partial"]
toy_pred = ["relevant", "irrelevant", "irrelevant", "partial", "partial"]
m = eu.compute_classification_metrics(toy_true, toy_pred)
print({k: (round(v, 3) if isinstance(v, float) else v) for k, v in m.items() if k != "per_class"})
