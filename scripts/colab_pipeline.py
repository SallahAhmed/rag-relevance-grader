"""End-to-end Colab orchestrator: data prep -> smoke -> full train -> push.

One exec on purpose (2026-09-09 lesson: a 78-min T4 run was reclaimed 40s
after exec ended, before artifacts came down). Nothing here depends on a
second command landing.

Mirrors notebooks/01 build() for the data step, then calls 03's run().
The push step polls for /content/.hfkey so the token can arrive mid-run.
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

PB01 = Path("/content/pb01")
for p in (PB01 / "out", REPO / "in", REPO / "out"):
    p.mkdir(parents=True, exist_ok=True)

QRELS_DATASET_ID = "msmarco-passage/trec-dl-2020"


def log(msg: str) -> None:
    print(f"[pb-final {time.strftime('%H:%M:%S')}] {msg}", flush=True)


def step_install() -> None:
    log("install deps")
    pkgs = [
        "ir_datasets", "pandas", "pyarrow", "pyyaml", "scikit-learn",
        "transformers", "peft", "trl", "bitsandbytes", "accelerate",
        "sentencepiece", "wandb",
    ]
    subprocess.run(
        [sys.executable, "-m", "pip", "install", "-q", *pkgs], check=True
    )
    import torch, transformers, peft, trl  # noqa: F401
    log(f"deps ok torch={torch.__version__} tf={transformers.__version__} "
        f"peft={peft.__version__} trl={trl.__version__}")


def step_data() -> dict:
    log("data prep: loading TREC DL 2020 qrels + queries")
    import ir_datasets
    import pandas as pd
    from src.config import load_config
    from src.dataset_utils import bucket_qrel, label_distribution, split_queries_by_label

    cfg = load_config()
    ds = ir_datasets.load(QRELS_DATASET_ID)
    qrels = [(q.query_id, q.doc_id, int(q.relevance)) for q in ds.qrels_iter()]
    queries = {q.query_id: q.text for q in ds.queries_iter()}
    log(f"{len(qrels)} qrels over {len({q for q, _, _ in qrels})} judged queries")

    log("downloading passage collection (slow, first touch only)")
    store = ds.docs_store()
    pids = [pid for _, pid, _ in qrels]
    docs = store.get_many(pids)

    rows, missing = [], 0
    for qid, pid, grade in qrels:
        doc = docs.get(pid)
        if doc is None or qid not in queries:
            missing += 1
            continue
        rows.append({
            "qid": qid, "query": queries[qid], "pid": pid,
            "passage": doc.text, "qrel": grade, "label": bucket_qrel(grade),
        })
    log(f"joined {len(rows)} pairs, {missing} missing text")

    df = pd.DataFrame(rows)
    per_query = {qid: dict(g["label"].value_counts()) for qid, g in df.groupby("qid")}
    train_q, test_q = split_queries_by_label(
        per_query, cfg.evaluation.test_queries, seed=cfg.training.seed
    )
    assert not (train_q & test_q), "query leaked across the split!"

    train_df = df[df["qid"].isin(train_q)].reset_index(drop=True)
    test_df = df[df["qid"].isin(test_q)].reset_index(drop=True)

    # 03 expects these two paths.
    train_df.to_parquet(REPO / "in" / "train.parquet", index=False)
    test_df.to_parquet(REPO / "in" / "test.parquet", index=False)
    train_df.to_parquet(PB01 / "out" / "train.parquet", index=False)
    test_df.to_parquet(PB01 / "out" / "test.parquet", index=False)

    stats = {
        "n_pairs": len(df), "n_missing_text": missing,
        "label_counts": label_distribution(df["label"].tolist()),
        "n_train_queries": len(train_q), "n_test_queries": len(test_q),
        "n_train_pairs": len(train_df), "n_test_pairs": len(test_df),
        "test_qids": sorted(test_q),
        "train_label_counts": label_distribution(train_df["label"].tolist()),
        "test_label_counts": label_distribution(test_df["label"].tolist()),
    }
    (PB01 / "out" / "stats.json").write_text(json.dumps(stats, indent=2), encoding="utf-8")
    (REPO / "in" / "stats.json").write_text(json.dumps(stats, indent=2), encoding="utf-8")
    log("data ok: " + json.dumps({k: stats[k] for k in
        ("n_train_queries", "n_train_pairs", "n_test_queries", "n_test_pairs")}))
    return stats


def step_train(smoke: bool) -> str:
    log(f"train: run(smoke={smoke})")
    os.environ.setdefault("WANDB_DISABLED", "true")
    import importlib.util
    spec = importlib.util.spec_from_file_location("pb03", REPO / "job03.py")
    mod = importlib.util.module_from_spec(spec)
    sys.modules["pb03"] = mod
    spec.loader.exec_module(mod)
    t0 = time.time()
    out = mod.run(smoke=smoke)
    log(f"train {'smoke ' if smoke else ''}done in {round(time.time() - t0, 1)}s -> {out}")
    return out


def step_push(adapter_dir: str, wait_s: int = 2400) -> bool:
    """Push after training, waiting for the token so it can arrive mid-run."""
    deadline = time.time() + wait_s
    key = Path("/content/.hfkey")
    while not key.is_file() and time.time() < deadline:
        if time.time() > deadline - wait_s + 60:
            log(f"waiting for /content/.hfkey ({(deadline - time.time()) // 60:.0f} min left)")
        time.sleep(20)
    if not key.is_file():
        log("NO TOKEN after wait — adapter stays on VM, download manually")
        return False
    log("token found, pushing adapter to the Hub")
    try:
        from huggingface_hub import HfApi
        from src.config import load_config
        repo_id = load_config().experiment.hf_model_id
        api = HfApi(token=key.read_text().strip())
        api.create_repo(repo_id, exist_ok=True)
        api.upload_folder(folder_path=adapter_dir, repo_id=repo_id, repo_type="model")
        log(f"PUSHED -> https://huggingface.co/{repo_id}")
        return True
    except Exception as exc:
        log(f"push failed: {exc}")
        return False


def main() -> None:
    import torch
    log(f"gpu={torch.cuda.get_device_name(0)} "
        f"vram={round(torch.cuda.get_device_properties(0).total_memory / 1e9, 1)}GB")

    step_install()
    step_data()

    log("smoke test (100 steps) — proves VRAM fit before the real run")
    try:
        step_train(smoke=True)
        log("smoke OK at batch 4 / seq 1024")
    except Exception as exc:
        log(f"SMOKE FAILED: {type(exc).__name__}: {exc}")
        log("see OOM guidance in README; rerun with batch 2 / accum 8 / seq 512")
        raise

    adapter = step_train(smoke=False)
    step_push(adapter)
    log("PIPELINE COMPLETE")


main()
