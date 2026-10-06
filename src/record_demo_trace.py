"""Record a real ON/OFF comparison for the static page and the quota fallback.

Why this exists: the ZeroGPU Space gives each visitor ~5 GPU-minutes a day, and
the static Space cannot run a model at all. Both need a *recorded* run to show
when no GPU is available. This script produces that recording from a real model
run — the numbers in `results/demo_trace.json` are therefore measured, not
written by hand.

Run it where a model can load (Colab T4 during notebook 03/04, or any machine
with a working env):

    python -m src.record_demo_trace --corpus colab-t4
    python -m src.record_demo_trace --corpus espresso --out results/demo_trace.json

With no adapter available it still records a run, but tags
`"using_adapter": false` so the static page can label it as the zero-shot base
model rather than passing it off as the fine-tune.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

if __package__ in (None, ""):  # direct `python src/record_demo_trace.py`
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.rag_pipeline import (
    Chunk,
    HFRelevanceGrader,
    QwenGenerator,
    answer_with_gate,
    build_index,
    load_metrics,
    retrieve,
)
from src.sample_corpora import CORPORA, get_corpus

DEFAULT_OUT = Path("results/demo_trace.json")
DEFAULT_CORPUS = CORPORA[0].key


def _chunks_for(corpus_key: str) -> list[Chunk]:
    """Sample corpora are authored as discrete passages: one chunk each."""
    corpus = get_corpus(corpus_key)
    return [
        Chunk(chunk_id=i, text=text, source=f"{corpus.key}/p{i + 1}", ordinal=i)
        for i, text in enumerate(corpus.passages)
    ]


def record(
    corpus_key: str = DEFAULT_CORPUS,
    out_path: str | Path = DEFAULT_OUT,
    top_k: int = 4,
    max_new_tokens: int = 180,
    base_model: str | None = None,
    adapter: str | None = None,
) -> dict[str, Any]:
    """Run both arms on one corpus and write the trace as JSON.

    Retrieval runs on CPU (MiniLM is milliseconds per passage); the model is only
    needed for grading and generation. Raises if the model cannot load — a
    missing recording is better than a fabricated one.
    """
    corpus = get_corpus(corpus_key)
    chunks = _chunks_for(corpus_key)
    index = build_index(chunks)
    hits = retrieve(index, corpus.query, top_k=top_k)
    if not hits:
        raise RuntimeError(f"no passages retrieved for corpus {corpus_key!r}")

    grader = HFRelevanceGrader(adapter=adapter, base_model=base_model)
    generator = QwenGenerator(bundle=grader.bundle, max_new_tokens=max_new_tokens)

    off_result = answer_with_gate(corpus.query, hits, False, generator=generator)
    on_result = answer_with_gate(
        corpus.query, hits, True, generator=generator, grader=grader
    )

    payload: dict[str, Any] = {
        "recorded_from": "src/record_demo_trace.py",
        "corpus": {
            "key": corpus.key,
            "title": corpus.title,
            "description": corpus.description,
            "expected_gate": corpus.expected_gate,
            "query": corpus.query,
            "n_passages": len(corpus.passages),
        },
        "model": {
            "base_model": grader.bundle.base_model_id,
            "adapter_ref": grader.bundle.adapter_ref,
            "using_adapter": grader.bundle.using_adapter,
            "quantized": grader.bundle.quantized,
            "device": grader.bundle.device,
            "load_error": grader.bundle.load_error,
            "embedding_model": index.embedding_model,
            "top_k": top_k,
            "max_new_tokens": max_new_tokens,
        },
        "arms": {"off": off_result.to_dict(), "on": on_result.to_dict()},
        "metrics": {
            "baseline": load_metrics().get("baseline"),
            "finetuned": load_metrics().get("finetuned"),
        },
    }
    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return payload


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--corpus", default=DEFAULT_CORPUS, choices=[c.key for c in CORPORA])
    parser.add_argument("--out", default=str(DEFAULT_OUT))
    parser.add_argument("--top-k", type=int, default=4)
    parser.add_argument("--max-new-tokens", type=int, default=180)
    parser.add_argument("--base-model", default=None)
    parser.add_argument("--adapter", default=None)
    args = parser.parse_args(argv)

    payload = record(
        corpus_key=args.corpus,
        out_path=args.out,
        top_k=args.top_k,
        max_new_tokens=args.max_new_tokens,
        base_model=args.base_model,
        adapter=args.adapter,
    )
    model = payload["model"]
    on_arm = payload["arms"]["on"]
    kept = len(on_arm["context_ids"])
    print(f"wrote {args.out}")
    print(f"  corpus: {payload['corpus']['key']} | adapter: {model['adapter_ref'] or 'none (zero-shot)'}")
    print(f"  gate kept {kept} of {len(on_arm['retrieved'])} passage(s); grounded={on_arm['grounded']}")
    if model["load_error"]:
        print(f"  note: {model['load_error']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
