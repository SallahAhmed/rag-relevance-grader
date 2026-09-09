# %% [markdown]
# # 04 — Evaluation + comparison (the portfolio artifact)
# **Goal:** same held-out queries, same template, same metrics — fine-tuned
# adapter vs notebook-02 baseline. Plus the relative forgetting check.
# **Acceptance:** `results/finetuned_results.json`,
# `results/before_after_table.md` (with N test queries printed beside every
# number), and a pass/fail forgetting verdict. Success bar: **+10 percentage
# points macro-F1** (e.g. 0.45 → 0.55).

# %%
import sys
from pathlib import Path

ROOT = Path.cwd()
sys.path.insert(0, str(ROOT) if (ROOT / "src").is_dir() else str(ROOT.parent))

from src import evaluation_utils as eu

baseline = eu.load_results_json("results/baseline_results.json")
print("baseline keys:", sorted(baseline))
print("baseline N queries:", baseline.get("n_test_queries"))

# %% [markdown]
# ## 1. Grade held-out queries with the adapter
# RUN: load base + `PeftModel.from_pretrained(base, "outputs/qlora-sft")`,
# decode exactly as in notebook 02 (one label token, argmax + softmax
# confidence), then `eu.compute_classification_metrics(...)` →
# `eu.save_results_json("results/finetuned_results.json", {...})`.

# %% [markdown]
# ## 2. Forgetting check (relative bar — fixed 25% is random chance)
# RUN: same MMLU harness/subset as notebook 02 on the adapter, then:

# %%
# Local check of the verdict logic (real scores plugged in at runtime):
verdict = eu.passes_forgetting_check(baseline_mmlu=0.48, finetuned_mmlu=0.45)
print(verdict)  # retention 0.9375 >= 0.90 → passes

# %% [markdown]
# ## 3. Build the before/after table (generated, never hand-typed)
# RUN: read both JSONs, emit `results/before_after_table.md`:
# | Metric | Zero-shot | Fine-tuned | Δ | with `N test queries / N qrels`
# in the caption. Macro-F1 on ~5–10 queries is noisy — the table says so.
# ## 4. Side-by-side prompts
# RUN: 3–5 hand-picked queries showing base vs adapter outputs verbatim.
# This is the most persuasive evidence in the README after the table.
