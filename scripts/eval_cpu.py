"""CPU evaluation of fine-tuned Qwen2.5-1.5B-Instruct relevance grader.

Loads base model + PEFT adapter on CPU, scores test.parquet via constrained
argmax over 3 label-token logits (identical to notebook 02), writes
predictions_ft.json, runs MMLU, then calls make_comparison.py.
"""
import os
import sys
import json
import time
from pathlib import Path

# Environment MUST be set before importing torch/transformers
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import pandas as pd
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

# Repo root for imports
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.dataset_utils import LABELS, PROMPT_TEMPLATE
from src.evaluation_utils import compute_classification_metrics, save_results_json
from src.mmlu_utils import run_mmlu

MODEL_ID = "Qwen/Qwen2.5-1.5B-Instruct"
ADAPTER_ID = "sallahahmed/qwen2.5-1.5b-relevance-grader"
TEST_PATH = ROOT / "data" / "processed" / "test.parquet"
PRED_OUT = ROOT / "results" / "predictions_ft.json"
MMLU_OUT = ROOT / "results" / "mmlu_ft.json"

BATCH = 16
MAX_LEN = 1024


def load_model_and_tokenizer():
    """Load base model in float32 on CPU, then apply PEFT adapter."""
    print(f"Loading tokenizer: {MODEL_ID}")
    tok = AutoTokenizer.from_pretrained(MODEL_ID, trust_remote_code=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    tok.padding_side = "left"

    print(f"Loading base model: {MODEL_ID} on CPU (float32)")
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        dtype=torch.float32,
        device_map="cpu",
        trust_remote_code=True,
        low_cpu_mem_usage=True,
    )

    print(f"Loading adapter: {ADAPTER_ID}")
    model = PeftModel.from_pretrained(model, ADAPTER_ID)
    model.eval()
    print("Model loaded on:", next(model.parameters()).device)
    return model, tok


def get_label_ids(tok):
    """Derive single-token label IDs exactly like notebook 02."""
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
    return label_ids


def score_batch(model, tok, prompts, label_ids, device):
    """Score a batch of prompts, return (preds, confs)."""
    enc = tok(
        prompts,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=MAX_LEN,
    ).to(device)
    with torch.no_grad():
        logits = model(**enc).logits[:, -1, :]
        sub = logits[:, torch.tensor(label_ids, device=device)]
        probs = torch.softmax(sub.float(), dim=-1)
        best = probs.argmax(dim=-1).tolist()
        preds = [LABELS[b] for b in best]
        confs = [round(float(probs[k, b].item()), 4) for k, b in enumerate(best)]
    return preds, confs


def benchmark_and_score(model, tok, test_df, label_ids, device):
    """Benchmark 50 rows, then score all with progress reporting."""
    prompts = [
        PROMPT_TEMPLATE.format(query=q, passage=p)
        for q, p in zip(test_df["query"], test_df["passage"])
    ]
    n_total = len(prompts)
    print(f"Total pairs to score: {n_total}")

    # Benchmark: first 50 rows (or all if < 50)
    bench_n = min(50, n_total)
    print(f"\n--- Benchmarking on {bench_n} rows ---")
    bench_prompts = prompts[:bench_n]
    start = time.perf_counter()
    _ = score_batch(model, tok, bench_prompts, label_ids, device)
    bench_time = time.perf_counter() - start
    sec_per_row = bench_time / bench_n
    eta_sec = sec_per_row * n_total
    eta_min = eta_sec / 60
    print(f"Benchmark: {bench_time:.2f}s for {bench_n} rows = {sec_per_row:.4f}s/row")
    print(f"Extrapolated full run: {eta_sec:.1f}s ({eta_min:.1f} minutes)")
    sys.stdout.flush()

    if eta_min > 90:
        print(f"\nWARNING: Estimated {eta_min:.1f} minutes exceeds 90-minute limit.")
        print(f"Recommend subset size: {int(90 * 60 / sec_per_row)} rows")
        return None, None, sec_per_row, eta_min

    # Full scoring with progress
    print("\n--- Full scoring ---")
    all_preds = []
    all_confs = []
    total_start = time.perf_counter()

    for i in range(0, n_total, BATCH):
        batch_prompts = prompts[i : i + BATCH]
        preds, confs = score_batch(model, tok, batch_prompts, label_ids, device)
        all_preds.extend(preds)
        all_confs.extend(confs)

        done = min(i + BATCH, n_total)
        if done % 100 == 0 or done == n_total:
            elapsed = time.perf_counter() - total_start
            rate = done / elapsed if elapsed > 0 else 0
            remaining = n_total - done
            eta = remaining / rate if rate > 0 else 0
            print(f"  scored {done}/{n_total} | {rate:.1f} rows/s | ETA: {eta/60:.1f} min", flush=True)

    total_time = time.perf_counter() - total_start
    print(f"\nCompleted {n_total} rows in {total_time:.1f}s ({total_time/60:.1f} min)")
    sys.stdout.flush()
    return all_preds, all_confs, sec_per_row, eta_min


def write_predictions(test_df, preds, confs, out_path):
    """Write predictions in the same schema as predictions_baseline.json."""
    out = [
        {"qid": q, "pid": p, "true": t, "pred": pr, "confidence": c}
        for q, p, t, pr, c in zip(test_df["qid"], test_df["pid"], test_df["label"], preds, confs)
    ]
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(out), encoding="utf-8")
    print(f"Wrote predictions: {len(out)} -> {out_path}")
    return out


def main():
    print("=" * 60)
    print("CPU Evaluation: Fine-tuned Relevance Grader")
    print("=" * 60)

    # Load test data
    print(f"Loading test data: {TEST_PATH}")
    test_df = pd.read_parquet(TEST_PATH)
    print(f"Test pairs: {len(test_df)}, queries: {test_df['qid'].nunique()}")

    # Load model + adapter
    model, tok = load_model_and_tokenizer()
    device = next(model.parameters()).device

    # Get label token IDs
    label_ids = get_label_ids(tok)

    # Benchmark + score
    preds, confs, sec_per_row, eta_min = benchmark_and_score(model, tok, test_df, label_ids, device)

    if preds is None:
        print("\nAborted: estimated runtime exceeds 90 minutes.")
        return 1

    # Write predictions
    write_predictions(test_df, preds, confs, PRED_OUT)

    # Run MMLU
    print("\n--- Running MMLU evaluation ---")
    mmlu_results = run_mmlu(model, tok)
    save_results_json(MMLU_OUT, mmlu_results)
    print(f"MMLU accuracy: {mmlu_results['accuracy']:.4f}")

    # Run make_comparison.py
    print("\n--- Running make_comparison.py ---")
    import subprocess
    result = subprocess.run([sys.executable, str(ROOT / "make_comparison.py")], capture_output=True, text=True)
    print(result.stdout)
    if result.stderr:
        print(result.stderr, file=sys.stderr)
    return result.returncode


if __name__ == "__main__":
    raise SystemExit(main())