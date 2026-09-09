# %% [markdown]
# # 03 — QLoRA fine-tuning
# **Goal:** train the adapter on the train queries per `config/sft_config.yaml`,
# logging every run to W&B.
# **Acceptance:** adapter saved to `outputs/qlora-sft/` + W&B run linked.
# Run in Colab T4. **Do not skip the smoke test** — batch=4 + seq=1024 as a
# pair is untested by any single source repo (see ANALYSIS.md provenance table).

# %%
import sys
from pathlib import Path

ROOT = Path.cwd()
sys.path.insert(0, str(ROOT) if (ROOT / "src").is_dir() else str(ROOT.parent))

from src.config import load_config

cfg = load_config()
t = cfg.training
print(f"effective batch: {t.per_device_batch_size}x{t.gradient_accumulation_steps}={t.effective_batch_size}")
print("gradient_checkpointing:", t.gradient_checkpointing, "| epochs:", t.epochs)
assert t.gradient_checkpointing, "OOM guard missing — fix config before training."

# %% [markdown]
# ## 0. Install (Colab T4 cell — run first, once per session)
# ```python
# !pip install -q "unsloth[colab-new] @ git+https://github.com/unslothai/unsloth.git"
# !pip install -q -r requirements.txt
# ```
# Standard TRL + PEFT is the fallback if Unsloth fails to install — the
# trainer calls below are written against `trl.SFTTrainer` either way.

# %% [markdown]
# ## 1. SMOKE TEST (mandatory, ~50–100 steps)
# RUN: same trainer, `max_steps=100`, everything else identical.
# Watch VRAM (`!nvidia-smi`). If OOM → apply fallback BEFORE the full run:
# per-device batch 2 / grad accum 8 / seq 512 (samrat-kar exact, same
# effective batch 16), keep `gradient_checkpointing: true`.

# %% [markdown]
# ## 2. Full run
# RUN: `SFTTrainer` with `LoraConfig(r=16, alpha=32, dropout=0.05,
# target_modules=[q,k,v,o,gate,up,down])`, cosine schedule, warmup 0.05,
# `paged_adamw_8bit`, seed 42, W&B project `cfg.experiment.wandb_project`.
# Training rows render prompts via `du.format_example` — the SAME template
# as notebook 02. Consider class weights / balanced sampling for "partial".
# Save adapter every N steps; keep the best by eval loss.

# %% [markdown]
# ## 3. Checkpoint the adapter
# RUN: push `outputs/qlora-sft/` to the Hub (private is fine mid-project)
# so a disconnect never loses more than the current run's tail.
