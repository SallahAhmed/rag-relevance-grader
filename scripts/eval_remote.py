"""Remote eval: verify split -> pull adapter from Hub -> grade -> MMLU -> push results.

Why this exists separately from colab_pipeline_v2.py: the adapter is already
published, so a lost VM no longer costs the weights. Only the predictions are
at risk now, and the last step pushes those to the Hub.

The test split is UPLOADED from local disk (data/processed/test.parquet), not
rebuilt from ir_datasets. Rebuilding re-runs the query-level split, and any drift
there would grade the adapter on a different query set than the zero-shot
baseline -- an invalid comparison that would still produce confident numbers.
verify_test() is the guard against exactly that.
"""

import importlib.util
import json
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path("/content/pb04")
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

HFKEY = Path("/content/.hfkey")
ADAPTER_REPO = "sallahahmed/qwen2.5-1.5b-relevance-grader"
BASE = "Qwen/Qwen2.5-1.5B-Instruct"


def log(m):
    print(f"[eval {time.strftime('%H:%M:%S')}] {m}", flush=True)


def install():
    log("install deps")
    subprocess.run([sys.executable, "-m", "pip", "install", "-q",
                    "pandas", "pyarrow", "pyyaml", "scikit-learn",
                    "transformers", "peft", "bitsandbytes", "accelerate",
                    "sentencepiece"], check=True)
    import transformers, peft  # noqa: F401
    log("deps ok")


def verify_test():
    """Fail loudly unless the uploaded parquet IS the baseline's test split.

    The split is uploaded from local disk rather than rebuilt from ir_datasets:
    re-deriving it re-runs the query-level split, and any drift there would
    silently compare the adapter against a different test set than the baseline.
    The guard below is what makes the upload trustworthy -- it re-checks the two
    properties the comparison actually depends on (2122 qrels over 10 queries,
    same qid set as predictions_baseline.json).
    """
    import pandas as pd

    path = ROOT / "in" / "test.parquet"
    df = pd.read_parquet(path)
    qids = sorted(df["qid"].unique().tolist())

    log(f"test parquet: {len(df)} qrels / {len(qids)} queries")
    log("test_qids=" + json.dumps(qids))
    log("label counts=" + json.dumps(
        {k: int(v) for k, v in df["label"].value_counts().items()}))

    expected_n_qrels = 2122
    expected_n_queries = 10
    if len(df) != expected_n_qrels:
        raise SystemExit(
            f"ABORT: expected {expected_n_qrels} qrels, got {len(df)} -- "
            "this is not the baseline test split; the comparison would be invalid")
    if len(qids) != expected_n_queries:
        raise SystemExit(
            f"ABORT: expected {expected_n_queries} test queries, got {len(qids)}")

    # Cross-check against the baseline predictions actually on the Hub/repo:
    # same qid set means the adapter is graded on exactly the queries the
    # zero-shot baseline was graded on.
    baseline = ROOT / "baseline_predictions.json"
    if baseline.is_file():
        base_qids = sorted({p["qid"] for p in json.loads(baseline.read_text())})
        if base_qids != qids:
            raise SystemExit(
                f"ABORT: qid mismatch vs baseline predictions. "
                f"uploaded={qids} baseline={base_qids}")
        log("qid set matches baseline predictions")
    else:
        log("WARNING: baseline_predictions.json absent -- qid cross-check skipped")


def pull_adapter():
    from huggingface_hub import snapshot_download
    token = HFKEY.read_text().strip()
    p = snapshot_download(repo_id=ADAPTER_REPO, token=token,
                          allow_patterns=["*.json", "*.safetensors", "*.jinja"],
                          local_dir=str(ROOT / "adapter"))
    log(f"adapter downloaded -> {p}")


def push_results():
    from huggingface_hub import HfApi
    token = HFKEY.read_text().strip()
    api = HfApi(token=token)
    out = ROOT / "out"
    files = [f for f in ("predictions_ft.json", "mmlu_ft.json") if (out / f).is_file()]
    for f in files:
        api.upload_file(path_or_fileobj=str(out / f), path_in_repo=f"eval/{f}",
                        repo_id=ADAPTER_REPO, repo_type="model")
        log(f"pushed eval/{f}")
    return files


def run_job04():
    spec = importlib.util.spec_from_file_location("pb04", ROOT / "job04.py")
    mod = importlib.util.module_from_spec(spec)
    sys.modules["pb04"] = mod
    spec.loader.exec_module(mod)
    mod.main(str(ROOT / "adapter"))


def main():
    install()
    verify_test()
    pull_adapter()
    try:
        run_job04()
    except Exception as exc:
        log(f"JOB04 FAILED: {type(exc).__name__}: {exc}")
        try:
            push_results()
        except Exception as e2:
            log(f"push also failed: {e2}")
        raise
    push_results()
    log("EVAL COMPLETE")


main()
