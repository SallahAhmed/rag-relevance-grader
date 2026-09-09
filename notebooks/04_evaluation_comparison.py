# %% [markdown]
# # 04 — Evaluation + comparison (the portfolio artifact)
# **Goal:** same held-out queries, same template, same metrics — fine-tuned
# adapter vs notebook-02 baseline. Plus the relative forgetting check.
# **Acceptance:** `results/finetuned_results.json`,
# `results/before_after_table.md` (with N test queries printed beside every
# number), and a pass/fail forgetting verdict. Success bar: **+10 percentage
# points macro-F1** (e.g. 0.45 → 0.55).
#
# Remote part (this file, /content/pb04/): grade test pairs with the adapter
# via the identical constrained path as notebook 02, then MMLU via the shared
# `src/mmlu_utils` protocol. Writes predictions + mmlu JSON.
# Local part (`make_comparison.py`): metrics, verdict, table — with tested code.

# %%
import json
import sys
from pathlib import Path

CANDIDATES = [Path.cwd(), Path.cwd().parent, Path("/content/pb04")]
REPO = next((c for c in CANDIDATES if (c / "src").is_dir()), None)
if REPO is not None and str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

REMOTE_ROOT = Path("/content/pb04")

from src.dataset_utils import LABELS, PROMPT_TEMPLATE  # noqa: E402
from src.mmlu_utils import run_mmlu  # noqa: E402

MODEL_ID = "Qwen/Qwen2.5-1.5B-Instruct"
BATCH = 16
MAX_LEN = 1024


# %%
def load_backbone(adapter_dir: str | None):
    """Base 4-bit model, plus PeftModel when an adapter dir is given."""
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

    tok = AutoTokenizer.from_pretrained(MODEL_ID, trust_remote_code=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    tok.padding_side = "left"
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
    if adapter_dir:
        from peft import PeftModel

        model = PeftModel.from_pretrained(model, adapter_dir)
        print("adapter loaded:", adapter_dir)
    model.eval()
    return model, tok


# %%
def grade_adapter(adapter_dir: str) -> list[dict]:
    """Constrained scoring with the adapter — mirrors notebook 02 exactly."""
    import pandas as pd
    import torch

    model, tok = load_backbone(adapter_dir)
    test = pd.read_parquet(REMOTE_ROOT / "in" / "test.parquet")
    print(f"pairs: {len(test)}, queries: {test['qid'].nunique()}")

    label_ids = []
    for lab in LABELS:
        ids = tok.encode(" " + lab, add_special_tokens=False)
        assert len(ids) == 1, f"label {lab!r} is {len(ids)} tokens — abort"
        label_ids.append(ids[0])

    prompts = [
        PROMPT_TEMPLATE.format(query=q, passage=p)
        for q, p in zip(test["query"], test["passage"])
    ]
    preds, confs = [], []
    with torch.no_grad():
        for i in range(0, len(prompts), BATCH):
            enc = tok(
                prompts[i : i + BATCH], return_tensors="pt", padding=True,
                truncation=True, max_length=MAX_LEN,
            ).to(model.device)
            logits = model(**enc).logits[:, -1, :]
            sub = logits[:, torch.tensor(label_ids, device=logits.device)]
            probs = torch.softmax(sub.float(), dim=-1)
            best = probs.argmax(dim=-1).tolist()
            preds += [LABELS[b] for b in best]
            confs += [round(float(probs[k, b]), 4) for k, b in enumerate(best)]
            if (i // BATCH) % 10 == 0:
                print(f"  scored {min(i + BATCH, len(prompts))}/{len(prompts)}", flush=True)
    out = [
        {"qid": q, "pid": p, "true": t, "pred": pr, "confidence": c}
        for q, p, t, pr, c in zip(test["qid"], test["pid"], test["label"], preds, confs)
    ]
    (REMOTE_ROOT / "out").mkdir(parents=True, exist_ok=True)
    (REMOTE_ROOT / "out" / "predictions_ft.json").write_text(json.dumps(out))
    print("wrote predictions:", len(out))
    return out


# %%
def mmlu_adapter(adapter_dir: str) -> dict:
    """MMLU on the adapter with the shared protocol. Returns results dict."""
    import torch  # noqa: F401 (protocol runs under no_grad inside run_mmlu)

    model, tok = load_backbone(adapter_dir)
    results = run_mmlu(model, tok)
    (REMOTE_ROOT / "out").mkdir(parents=True, exist_ok=True)
    (REMOTE_ROOT / "out" / "mmlu_ft.json").write_text(
        json.dumps(results, indent=2)
    )
    return results


# %%
def main(adapter_dir: str) -> None:
    grade_adapter(adapter_dir)
    mmlu_adapter(adapter_dir)


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "/content/pb04/adapter")
