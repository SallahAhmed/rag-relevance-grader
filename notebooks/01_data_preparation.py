# %% [markdown]
# # 01 — Data preparation
# **Goal:** TREC DL 2020 qrels + MS MARCO passages → (query, passage, label) triples,
# bucketed 0→irrelevant / 1→partial / 2–3→relevant, **split by whole query**.
# **Acceptance:** class distribution printed, no query id appears on both sides
# of the split, `train.parquet` + `test.parquet` + `stats.json` saved.
# Runs on CPU (Colab or local). Dataset access via `ir_datasets` — canonical
# URLs and checksums, no hand-rolled download links.
#
# Remote layout on the VM (`/content/pb01/`): this file as `job01.py` plus
# `src/` and `config/` uploaded alongside it. Execute stepwise (kernel persists):
#   step A → `from job01 import inspect_source; inspect_source()`
#   step B → `from job01 import build; build()`
# or full pipeline: `python job01.py` / `colab run job01.py`.

# %%
import json
import sys
from collections import Counter
from pathlib import Path

WORKDIR = Path("/content/pb01")
if (WORKDIR / "src").is_dir() and str(WORKDIR) not in sys.path:
    sys.path.insert(0, str(WORKDIR))

from src.config import load_config  # noqa: E402
from src.dataset_utils import (  # noqa: E402
    bucket_qrel,
    format_example,
    label_distribution,
    split_queries_by_label,
)

OUT_DIR = WORKDIR / "out"
QRELS_DATASET_ID = "msmarco-passage/trec-dl-2020"


# %%
def inspect_source() -> dict:
    """Step A (fast): load qrels + queries, print grade/query counts. No docs download."""
    import ir_datasets

    ds = ir_datasets.load(QRELS_DATASET_ID)
    qrels = [(q.query_id, q.doc_id, int(q.relevance)) for q in ds.qrels_iter()]
    queries = {q.query_id: q.text for q in ds.queries_iter()}
    grades = Counter(g for _, _, g in qrels)
    judged = {qid for qid, _, _ in qrels}
    stats = {
        "n_qrels": len(qrels),
        "grade_counts": dict(sorted(grades.items())),
        "n_queries_released": len(queries),
        "n_queries_judged": len(judged),
    }
    print(json.dumps(stats, indent=2))
    return stats


# %%
def build(cfg_path: str | None = None) -> dict:
    """Step B (slow: downloads the passage collection on first docs access).

    Joins every judged (qid, pid, grade) with query + passage text, buckets
    labels, splits BY QUERY, saves parquet + stats. Returns the stats dict.
    """
    import pandas as pd

    cfg = load_config(cfg_path) if cfg_path else load_config()
    import ir_datasets

    ds = ir_datasets.load(QRELS_DATASET_ID)
    qrels = [(q.query_id, q.doc_id, int(q.relevance)) for q in ds.qrels_iter()]
    queries = {q.query_id: q.text for q in ds.queries_iter()}

    store = ds.docs_store()  # first touch downloads + indexes the collection
    pids = [pid for _, pid, _ in qrels]
    docs = store.get_many(pids)

    rows, missing = [], 0
    for qid, pid, grade in qrels:
        doc = docs.get(pid)
        if doc is None or qid not in queries:
            missing += 1
            continue
        rows.append(
            {
                "qid": qid,
                "query": queries[qid],
                "pid": pid,
                "passage": doc.text,
                "qrel": grade,
                "label": bucket_qrel(grade),
            }
        )
    print(f"joined {len(rows)} pairs, {missing} missing (no text)")

    df = pd.DataFrame(rows)
    print("label distribution:", label_distribution(df["label"].tolist()))

    per_query: dict = {}
    for qid, group in df.groupby("qid"):
        per_query[qid] = dict(Counter(group["label"]))
    train_q, test_q = split_queries_by_label(
        per_query, cfg.evaluation.test_queries, seed=cfg.training.seed
    )
    assert not (train_q & test_q), "query leaked across the split!"

    train_df = df[df["qid"].isin(train_q)].reset_index(drop=True)
    test_df = df[df["qid"].isin(test_q)].reset_index(drop=True)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    train_df.to_parquet(OUT_DIR / "train.parquet", index=False)
    test_df.to_parquet(OUT_DIR / "test.parquet", index=False)
    stats = {
        "n_pairs": len(df),
        "n_missing_text": missing,
        "label_counts": label_distribution(df["label"].tolist()),
        "n_train_queries": len(train_q),
        "n_test_queries": len(test_q),
        "n_train_pairs": len(train_df),
        "n_test_pairs": len(test_df),
        "test_qids": sorted(test_q),
        "train_label_counts": label_distribution(train_df["label"].tolist()),
        "test_label_counts": label_distribution(test_df["label"].tolist()),
    }
    (OUT_DIR / "stats.json").write_text(json.dumps(stats, indent=2), encoding="utf-8")
    print(json.dumps(stats, indent=2))
    return stats


# %%
def main() -> None:
    inspect_source()
    build()


if __name__ == "__main__":
    main()
