"""Hardened Colab orchestrator v2 — after the 2026-09-30 VM loss.

What changed vs v1, and why:
  - NO smoke test. v1 proved batch 4 / seq 1024 fits in 15.6GB T4. That
    8 minutes is not worth re-spending on a deadline run.
  - Push IMMEDIATELY after training, with no 40-minute wait. v1 lost a
    healthy 75-minute run to a wait loop that had nothing to push.
  - Token must ALREADY be at /content/.hfkey. This script refuses to start
    training without it, because an unpushed adapter is a lost adapter.
  - Eval runs AFTER the push, so a failure there can never cost the weights.
"""

import json
import os
import subprocess
import sys
import time
from pathlib import Path

REPO = Path("/content/pb03")
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

for p in (Path("/content/pb01/out"), REPO / "in", REPO / "out"):
    p.mkdir(parents=True, exist_ok=True)

QRELS_DATASET_ID = "msmarco-passage/trec-dl-2020"
HFKEY = Path("/content/.hfkey")


def log(msg: str) -> None:
    print(f"[pb2 {time.strftime('%H:%M:%S')}] {msg}", flush=True)


def step_install() -> None:
    log("install deps")
    subprocess.run([sys.executable, "-m", "pip", "install", "-q",
                    "ir_datasets", "pandas", "pyarrow", "pyyaml", "scikit-learn",
                    "transformers", "peft", "trl", "bitsandbytes", "accelerate",
                    "sentencepiece", "wandb"], check=True)
    import torch, transformers, peft, trl  # noqa: F401
    log(f"deps ok tf={transformers.__version__} trl={trl.__version__}")


def step_data() -> dict:
    import ir_datasets
    import pandas as pd
    from src.config import load_config
    from src.dataset_utils import bucket_qrel, label_distribution, split_queries_by_label

    cfg = load_config()
    ds = ir_datasets.load(QRELS_DATASET_ID)
    qrels = [(q.query_id, q.doc_id, int(q.relevance)) for q in ds.qrels_iter()]
    queries = {q.query_id: q.text for q in ds.queries_iter()}
    log(f"{len(qrels)} qrels / {len({q for q, _, _ in qrels})} judged queries")

    store = ds.docs_store()
    docs = store.get_many([pid for _, pid, _ in qrels])

    rows, missing = [], 0
    for qid, pid, grade in qrels:
        doc = docs.get(pid)
        if doc is None or qid not in queries:
            missing += 1
            continue
        rows.append({"qid": qid, "query": queries[qid], "pid": pid,
                     "passage": doc.text, "qrel": grade, "label": bucket_qrel(grade)})
    log(f"joined {len(rows)} pairs, {missing} missing")

    df = pd.DataFrame(rows)
    per_query = {qid: dict(g["label"].value_counts()) for qid, g in df.groupby("qid")}
    train_q, test_q = split_queries_by_label(
        per_query, cfg.evaluation.test_queries, seed=cfg.training.seed)
    assert not (train_q & test_q), "query leaked across the split!"

    train_df = df[df["qid"].isin(train_q)].reset_index(drop=True)
    test_df = df[df["qid"].isin(test_q)].reset_index(drop=True)
    train_df.to_parquet(REPO / "in" / "train.parquet", index=False)
    test_df.to_parquet(REPO / "in" / "test.parquet", index=False)

    stats = {"n_train_queries": len(train_q), "n_train_pairs": len(train_df),
             "n_test_queries": len(test_q), "n_test_pairs": len(test_df),
             "test_qids": sorted(test_q),
             "train_label_counts": label_distribution(train_df["label"].tolist()),
             "test_label_counts": label_distribution(test_df["label"].tolist())}
    (REPO / "in" / "stats.json").write_text(json.dumps(stats, indent=2), encoding="utf-8")
    log("data ok " + json.dumps({k: stats[k] for k in
        ("n_train_queries", "n_train_pairs", "n_test_queries", "n_test_pairs")}))
    return stats


def load_job03():
    import importlib.util
    spec = importlib.util.spec_from_file_location("pb03", REPO / "job03.py")
    mod = importlib.util.module_from_spec(spec)
    sys.modules["pb03"] = mod
    spec.loader.exec_module(mod)
    return mod


def step_push(adapter_dir: str) -> bool:
    from huggingface_hub import HfApi
    from src.config import load_config
    repo_id = load_config().experiment.hf_model_id
    log(f"push -> {repo_id}")
    api = HfApi(token=HFKEY.read_text().strip())
    api.create_repo(repo_id, exist_ok=True)
    api.upload_folder(folder_path=adapter_dir, repo_id=repo_id, repo_type="model")
    log(f"PUSHED https://huggingface.co/{repo_id}")
    return True


def step_eval() -> None:
    """Post-push. A failure here cannot cost the weights — they are on the Hub."""
    import importlib.util
    log("eval: loading fine-tuned adapter and scoring held-out queries")
    spec = importlib.util.spec_from_file_location("pb04", REPO / "job04.py")
    mod = importlib.util.module_from_spec(spec)
    sys.modules["pb04"] = mod
    spec.loader.exec_module(mod)
    fn = next((getattr(mod, n) for n in ("run", "main", "evaluate")
               if callable(getattr(mod, n, None))), None)
    if fn is None:
        log("eval: no entry point found in job04, skipping")
        return
    fn()
    log("eval done")


def main() -> None:
    import torch
    log(f"gpu={torch.cuda.get_device_name(0)} "
        f"vram={round(torch.cuda.get_device_properties(0).total_memory/1e9,1)}GB")

    if not HFKEY.is_file():
        log("REFUSING TO START: /content/.hfkey missing. An adapter that cannot "
            "be pushed is an adapter that dies with the VM.")
        raise SystemExit(2)

    os.environ.setdefault("WANDB_DISABLED", "true")
    step_install()
    step_data()

    log("train (no smoke — v1 already proved VRAM fit at batch 4 / seq 1024)")
    mod = load_job03()
    t0 = time.time()
    adapter = mod.run(smoke=False)
    log(f"TRAIN DONE in {round(time.time()-t0,1)}s -> {adapter}")

    try:
        step_push(adapter)
    except Exception as exc:
        log(f"PUSH FAILED: {type(exc).__name__}: {exc}")
        log("adapter is still on the VM — download it by hand NOW")
        raise

    try:
        step_eval()
    except Exception as exc:
        log(f"EVAL FAILED (adapter is safe on the Hub): {type(exc).__name__}: {exc}")

    log("PIPELINE COMPLETE")


main()
