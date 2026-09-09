# %% [markdown]
# # 05 — Publish to Hugging Face Hub
# **Goal:** adapter (+ optional merge) public with a model card carrying the
# benchmark numbers, so anyone can inspect or run the grader.
# **Acceptance:** public repo `sallahahmed/qwen2.5-1.5b-relevance-grader`
# with model card, plus README table wired to `results/before_after_table.md`.

# %%
import sys
from pathlib import Path

ROOT = Path.cwd()
sys.path.insert(0, str(ROOT) if (ROOT / "src").is_dir() else str(ROOT.parent))

from src.config import load_config

cfg = load_config()
print("publishing:", cfg.experiment.hf_model_id)
# RUN (needs `huggingface-cli login` first):
#   from huggingface_hub import HfApi
#   api = HfApi()
#   api.create_repo(cfg.experiment.hf_model_id, exist_ok=True)
#   api.upload_folder(repo_id=cfg.experiment.hf_model_id, folder_path="outputs/qlora-sft")

# %% [markdown]
# ## 1. Adapter vs merged — which to publish?
# Default: publish the **adapter** (small, honest about what was trained) and
# document the exact base revision in the card. Merge only if a consumer asks
# for a standalone checkpoint — merging is one line either way:
# `PeftModel.from_pretrained(base, adapter).merge_and_unload()`.

# %% [markdown]
# ## 2. Model card (must contain)
# - Base model + revision, QLoRA config (r/alpha/dropout/targets)
# - Dataset: TREC DL 2020, bucket plan, N train queries / N test queries
# - The before/after table (paste from `results/before_after_table.md`)
# - Limits: graded on ~54 judged queries; macro-F1 is directional, not tight;
#   no `reasoning` output in v1; English MS MARCO passages only.

# %% [markdown]
# ## 3. README wiring
# RUN: copy the table into the repo README's leading section and push.
# The README leads with numbers, not description — that ordering IS the project.
