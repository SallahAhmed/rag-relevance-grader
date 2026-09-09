# %% [markdown]
# # 03 — QLoRA fine-tuning
# **Goal:** train the adapter on the train queries per `config/sft_config.yaml`,
# logging every run to W&B.
# **Acceptance:** adapter saved to `outputs/qlora-sft/` + W&B run linked.
# Run on T4. **Smoke first** (`run(smoke=True)`, 100 steps) — batch=4 +
# seq=1024 as a pair is untested by any single source repo. On OOM, fall back
# to batch 2 / accum 8 / seq 512 (samrat-kar exact, same effective batch 16).
#
# Brilliance levers (why this should beat the 0.267 baseline by >10pp):
# 1. Class balance by seeded replication — relevant is only 15% of pairs;
#    stock cross-entropy on raw data would under-learn it. No custom loss
#    code: balancing lives in the data, where it is auditable.
# 2. Completion-only loss on "Label:" — gradients only where the signal is.
# 3. Prompt template byte-identical to notebook 02 (imported, not retyped).

# %%
import inspect
import json
import os
import sys
from collections import Counter
from pathlib import Path

CANDIDATES = [Path.cwd(), Path.cwd().parent, Path("/content/pb03")]
REPO = next((c for c in CANDIDATES if (c / "src").is_dir()), None)
if REPO is not None and str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

REMOTE_ROOT = Path("/content/pb03")

from src.config import load_config  # noqa: E402
from src.dataset_utils import LABELS, PROMPT_TEMPLATE, split_queries_by_label  # noqa: E402

MODEL_ID = "Qwen/Qwen2.5-1.5B-Instruct"
RESPONSE_TEMPLATE = "Label:"  # completion masking starts after this token span


# %%
def build_balanced_rows(train_df, seed: int) -> list[dict]:
    """Replicate minority classes to 1:1:1 with a seeded shuffle.

    Majority class appears 1x/epoch, minorities more often — equivalent to
    class weights, but visible in row counts instead of hidden in a loss fn.
    """
    import pandas as pd

    counts = Counter(train_df["label"])
    target = max(counts.values())
    print("raw train distribution:", dict(counts), "-> target/class:", target)
    parts = []
    for lab in LABELS:
        block = train_df[train_df["label"] == lab]
        reps = target // len(block)
        rest = target % len(block)
        big = pd.concat([block] * reps + [block.sample(rest, random_state=seed)])
        parts.append(big)
    balanced = (
        pd.concat(parts).sample(frac=1.0, random_state=seed).reset_index(drop=True)
    )
    print("balanced distribution:", dict(Counter(balanced["label"])))
    rows = [
        {"text": PROMPT_TEMPLATE.format(query=q, passage=p) + " " + lab, "label": lab}
        for q, p, lab in zip(balanced["query"], balanced["passage"], balanced["label"])
    ]
    return rows


# %%
def build_training_args(cfg, out_dir: str, use_wandb: bool, smoke: bool):
    """Version-tolerant args: SFTConfig when present, else TrainingArguments.

    Returns (args_object, trainer_kwargs). Trainer-level keys
    (dataset_text_field/packing/max_seq_length) go to SFTTrainer itself —
    whichever of them the installed TRL accepts.
    """
    t = cfg.training
    eff = t.effective_batch_size
    assert eff == 16, f"effective batch drifted: {eff} (all 3 repos converge on 16)"
    want = dict(
        output_dir=out_dir,
        per_device_train_batch_size=t.per_device_batch_size,
        gradient_accumulation_steps=t.gradient_accumulation_steps,
        gradient_checkpointing=t.gradient_checkpointing,
        learning_rate=t.learning_rate,
        lr_scheduler_type=t.lr_scheduler,
        warmup_ratio=t.warmup_ratio,
        num_train_epochs=t.epochs,
        optim=t.optimizer,
        seed=t.seed,
        logging_steps=10,
        save_steps=200,
        save_total_limit=2,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        report_to="wandb" if use_wandb else "none",
        # fp32 adapters (NOT fp16): the fp16 GradScaler path hits an
        # unimplemented bf16 foreach kernel on T4 (sm_75 has no bf16
        # hardware kernels; something in the bnb/autocast path emits bf16
        # grads). samrat-kar likewise trained with a 32bit optimizer.
        # Costs speed, buys correctness. Revisit only with a pinned stack.
        fp16=False,
        bf16=False,
        packing=False,
        dataset_text_field="text",
    )
    if smoke:
        want["max_steps"] = 100
    try:
        from trl import SFTConfig

        params = set(inspect.signature(SFTConfig).parameters)
        arg_cls = SFTConfig
    except ImportError:
        from transformers import TrainingArguments

        params = set(inspect.signature(TrainingArguments).parameters)
        arg_cls = TrainingArguments
    # eval_strategy was renamed evaluation_strategy and back across versions.
    if "eval_strategy" in params:
        want["eval_strategy"] = "steps"
    elif "evaluation_strategy" in params:
        want["evaluation_strategy"] = "steps"
    if "eval_steps" in params:
        want["eval_steps"] = 100
    # warmup_ratio is absent from some TRL builds — fall back to steps (~5% of
    # one balanced epoch ≈ 1126 steps), but never set both (transformers errors).
    if "warmup_ratio" not in params and "warmup_steps" in params:
        want["warmup_steps"] = 50
    kept = {k: v for k, v in want.items() if k in params}
    dropped = sorted(set(want) - set(kept))
    if dropped:
        print("TRL compat: dropped unsupported kwargs:", dropped)
    args = arg_cls(**kept)
    return args, want


# %%
def maybe_wandb_login(project: str) -> bool:
    """Log in from /content/.wbkey if present; else disable W&B (never block)."""
    key_file = REMOTE_ROOT / ".wbkey"
    if not key_file.is_file():
        os.environ["WANDB_DISABLED"] = "true"
        print("wandb: no key file — logging disabled, training proceeds")
        return False
    import wandb

    wandb.login(key=key_file.read_text().strip())
    print(f"wandb: logged in, project={project}")
    return True


# %%
class CompletionOnlyCollator:
    """Masks prompt tokens with -100: loss applies to the label only.

    Hand-rolled because DataCollatorForCompletionOnlyLM was removed from
    both transformers and TRL 1.x. Finds the LAST occurrence of the
    response-template span (rfind — a passage could theoretically contain
    the literal "Label:"; the true template is always the final one since
    the completion is just " <label>").
    """

    def __init__(self, tokenizer, response_template: str = RESPONSE_TEMPLATE,
                 max_length: int = 1024):
        self.tok = tokenizer
        self.max_length = max_length
        tpl = tokenizer(response_template, add_special_tokens=False)["input_ids"]
        assert tpl, "empty response template"
        self.template_ids = tpl

    def _mask_before_template(self, ids: list[int]) -> list[int]:
        t, n, m = ids, len(ids), len(self.template_ids)
        start = -1
        for i in range(n - m, -1, -1):
            if t[i : i + m] == self.template_ids:
                start = i + m
                break
        if start < 0:  # template not found — train on nothing, loudly
            raise ValueError("response template not found in tokenized example")
        return [-100] * start + t[start:]

    def __call__(self, features: list[dict]) -> dict:
        """Masking pass over TRL-1.x-pre-tokenized features.

        Newer SFTTrainer tokenizes/truncates the dataset itself before the
        collator runs, so features arrive as {input_ids, attention_mask, ...}
        with labels already built for FULL-sequence loss. We overwrite labels
        with completion-only masking here. Padding: our own, -100 for labels.
        """
        import torch

        ids: list[list[int]] = []
        for f in features:
            row = f["input_ids"]
            if torch.is_tensor(row):
                row = row.tolist()
            ids.append([int(x) for x in row])
        pad = self.tok.pad_token_id
        unpadded = [[x for x in row if x != pad] for row in ids]
        labels = [self._mask_before_template(row) for row in unpadded]
        maxlen = max(len(r) for r in unpadded)
        input_ids = torch.tensor(
            [r + [pad] * (maxlen - len(r)) for r in unpadded], dtype=torch.long
        )
        attention_mask = torch.tensor(
            [[1] * len(r) + [0] * (maxlen - len(r)) for r in unpadded],
            dtype=torch.long,
        )
        lab = torch.full((len(ids), maxlen), -100, dtype=torch.long)
        for i, l in enumerate(labels):
            lab[i, : len(l)] = torch.tensor(l, dtype=torch.long)
        return {"input_ids": input_ids, "attention_mask": attention_mask,
                "labels": lab}


def run(smoke: bool = False) -> str:
    """Train. smoke=True: 100 steps, no W&B, proves VRAM fit before the full run."""
    import pandas as pd
    import torch
    from datasets import Dataset
    from peft import LoraConfig
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
    from trl import SFTTrainer

    import torch

    cfg = load_config()
    t = cfg.training
    out_dir = str(REMOTE_ROOT / "out" / ("smoke" if smoke else "qlora-sft"))

    train_df = pd.read_parquet(REMOTE_ROOT / "in" / "train.parquet")
    # Val = 2 held-out TRAIN queries (test queries stay untouched for notebook 04).
    per_query: dict = {
        qid: dict(Counter(g["label"])) for qid, g in train_df.groupby("qid")
    }
    train_q, val_q = split_queries_by_label(per_query, 2, seed=t.seed + 1)
    val_df = train_df[train_df["qid"].isin(val_q)].reset_index(drop=True)
    fit_df = train_df[train_df["qid"].isin(train_q)].reset_index(drop=True)
    print(f"fit queries: {len(train_q)} ({len(fit_df)} rows) | val queries: {sorted(val_q)} ({len(val_df)} rows)")

    train_rows = build_balanced_rows(fit_df, t.seed)
    val_rows = [
        {"text": PROMPT_TEMPLATE.format(query=q, passage=p) + " " + lab}
        for q, p, lab in zip(val_df["query"], val_df["passage"], val_df["label"])
    ]
    train_ds = Dataset.from_list(train_rows)
    val_ds = Dataset.from_list(val_rows)

    tok = AutoTokenizer.from_pretrained(MODEL_ID, trust_remote_code=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    tok.padding_side = "right"
    quant = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_use_double_quant=True,
    )
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        quantization_config=quant,
        dtype=torch.float16,
        device_map="auto",
        trust_remote_code=True,
    )
    model.config.use_cache = False
    lora = LoraConfig(
        r=cfg.lora.r,
        lora_alpha=cfg.lora.alpha,
        lora_dropout=cfg.lora.dropout,
        target_modules=list(cfg.lora.target_modules),
        task_type="CAUSAL_LM",
    )
    collator = CompletionOnlyCollator(tok, RESPONSE_TEMPLATE, t.max_seq_length)

    if not smoke:
        use_wandb = maybe_wandb_login(cfg.experiment.wandb_project)
    else:
        os.environ["WANDB_DISABLED"] = "true"
        use_wandb = False

    training_args, want = build_training_args(cfg, out_dir, use_wandb, smoke)
    trainer_extra: dict = {}
    trainer_params = set(inspect.signature(SFTTrainer.__init__).parameters)
    for key in ("dataset_text_field", "packing", "max_seq_length"):
        if key in trainer_params and key in want:
            trainer_extra[key] = want[key]
    if "tokenizer" in trainer_params:
        trainer_extra["tokenizer"] = tok
    if trainer_extra:
        print("TRL trainer kwargs:", sorted(trainer_extra))
    trainer = SFTTrainer(
        model=model,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        peft_config=lora,
        data_collator=collator,
        args=training_args,
        **trainer_extra,
    )
    stats = trainer.train()
    print("train loss:", round(float(stats.training_loss), 4))
    trainer.save_model(out_dir)
    tok.save_pretrained(out_dir)
    (Path(out_dir) / "run_meta.json").write_text(
        json.dumps(
            {
                "smoke": smoke,
                "train_rows": len(train_rows),
                "val_rows": len(val_rows),
                "final_loss": float(stats.training_loss),
                "prompt": "PROMPT_TEMPLATE:v1 (imported from src.dataset_utils)",
            },
            indent=2,
        )
    )
    print("saved:", out_dir)
    return out_dir


# %%
def main() -> None:
    run(smoke=False)


if __name__ == "__main__":
    main()
