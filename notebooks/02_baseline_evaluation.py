# %% [markdown]
# # 02 — Baseline evaluation (zero-shot, untouched model)
# **Goal:** score Qwen2.5-1.5B-Instruct with NO fine-tuning on the held-out
# queries, using the byte-identical `PROMPT_TEMPLATE` notebook 03 will train on.
# **Acceptance:** `results/baseline_results.json` written with accuracy,
# macro-F1, per-class F1, confusion matrix, N test queries/qrels — PLUS the
# MMLU baseline score (notebook 04's forgetting check needs it).
# Run in Colab T4. This number is the "before" in the before/after table.
#
# Layout-agnostic: resolves `/content/pb02/{in,out}` on the VM and
# `data/processed/` + `results/` in the repo. Metrics are computed from the
# downloaded predictions with `src/evaluation_utils` (unit-tested locally) —
# the VM only produces raw predictions.

# %%
import json
import sys
from pathlib import Path

CANDIDATES = [Path.cwd(), Path.cwd().parent, Path("/content/pb02"), Path("/content/pb01")]
REPO = next((c for c in CANDIDATES if (c / "src").is_dir()), None)
if REPO is None:
    # Last resort: the uploaded job dir itself carries src/ alongside.
    here = Path("/content/pb02")
    if (here / "src").is_dir():
        REPO = here
if REPO is not None and str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

REMOTE_IN = Path("/content/pb02/in/test.parquet")
REMOTE_OUT = Path("/content/pb02/out/predictions.json")

from src.dataset_utils import LABELS, PROMPT_TEMPLATE  # noqa: E402

MODEL_ID = "Qwen/Qwen2.5-1.5B-Instruct"
BATCH = 16
MAX_LEN = 1024


# %%
def resolve_io():
    """(test.parquet path, predictions path) — VM layout preferred, repo fallback."""
    if REMOTE_IN.is_file():
        return REMOTE_IN, REMOTE_OUT
    if REPO is not None and (REPO / "data" / "processed" / "test.parquet").is_file():
        return REPO / "data" / "processed" / "test.parquet", None
    raise FileNotFoundError("test.parquet found in neither /content/pb02 nor repo")


# %%
def grade_all(test_path: Path, out_path: Path | None) -> list[dict]:
    """Constrained zero-shot scoring: argmax over the 3 label-token logits."""
    import pandas as pd
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

    test = pd.read_parquet(test_path)
    print(f"pairs: {len(test)}, queries: {test['qid'].nunique()}")

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
    model.eval()
    print("model on:", next(model.parameters()).device)

    label_ids = []
    for lab in LABELS:
        ids = tok.encode(" " + lab, add_special_tokens=False)
        print(f"label {lab!r} -> ids {ids}")
        if len(ids) != 1:
            raise SystemExit(
                f"STOP: label {lab!r} is {len(ids)} tokens — "
                "constrained scoring invalid; switch to generation-match path."
            )
        label_ids.append(ids[0])

    prompts = [
        PROMPT_TEMPLATE.format(query=q, passage=p)
        for q, p in zip(test["query"], test["passage"])
    ]
    preds, confs = [], []
    with torch.no_grad():
        for i in range(0, len(prompts), BATCH):
            enc = tok(
                prompts[i : i + BATCH],
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=MAX_LEN,
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
    if out_path is not None:
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(out), encoding="utf-8")
        print("wrote predictions:", len(out))
    return out


# %%
def main() -> None:
    test_path, out_path = resolve_io()
    grade_all(test_path, out_path)


if __name__ == "__main__":
    main()
