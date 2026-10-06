"""Smoke test for src/rag_pipeline.py — no network, no model, no GPU.

Run either way (both must work):
    python -m src.rag_pipeline_smoke_test
    python src/rag_pipeline_smoke_test.py

The grader and the generator are fakes, so the gate logic is exercised without
loading 1.5B parameters. If any code path tried to reach the model here the run
would hang or raise instead of passing quietly; the final check additionally
asserts no heavy ML module was imported at all.
"""
from __future__ import annotations

import json
import sys
from collections.abc import Sequence
from pathlib import Path
from typing import Any

try:
    from .grader_interface import GraderInterface, ScoredPassage
    from .rag_pipeline import (
        NO_GROUNDED_ANSWER,
        Chunk,
        answer_with_gate,
        before_after_rows,
        chunk_documents,
        grade_passages,
        load_metrics,
    )
except ImportError:  # direct `python src/rag_pipeline_smoke_test.py`
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from src.grader_interface import GraderInterface, ScoredPassage
    from src.rag_pipeline import (
        NO_GROUNDED_ANSWER,
        Chunk,
        answer_with_gate,
        before_after_rows,
        chunk_documents,
        grade_passages,
        load_metrics,
    )

# Captured before any test runs so a pre-imported torch (e.g. inside a notebook
# kernel) is not mistaken for one this test caused.
HEAVY_MODULES = ("torch", "transformers", "peft", "sentence_transformers", "faiss")
_IMPORTS_AT_START = {name: name in sys.modules for name in HEAVY_MODULES}

QUERY = "what VRAM does a free Colab T4 have"
PASSAGES = (
    "A Colab T4 exposes 16 GB of VRAM, the budget every batch size must fit inside.",
    "The A100 on paid Colab offers 80 GB of VRAM, so 1.5B trains unquantized there.",
    "QLoRA loads the base model in 4-bit NF4, so a 1.5B model needs far less than 16 GB.",
)


class CountingGrader(GraderInterface):
    """Fake grader that records its calls and returns fixed labels.

    Confidence decreases with input position so the sort order is predictable.
    """

    name = "counting-grader"

    def __init__(self, labels: Sequence[str] | None = None) -> None:
        self.calls: list[dict[str, Any]] = []
        self.labels = list(labels) if labels is not None else None

    def grade(
        self, query: str, passages: list[str], top_k: int = 5
    ) -> list[ScoredPassage]:
        self.calls.append({"query": query, "passages": list(passages), "top_k": top_k})
        labels = self.labels or ["irrelevant"] * len(passages)
        if len(labels) != len(passages):
            raise AssertionError("CountingGrader: one label per passage required")
        scored = [
            ScoredPassage(
                passage=passage,
                label=labels[i],
                score=round(0.9 - 0.1 * i, 2),
                confidence=round(0.9 - 0.1 * i, 2),
            )
            for i, passage in enumerate(passages)
        ]
        scored.sort(key=lambda s: s.score, reverse=True)
        return scored[:top_k]


class FakeGenerator:
    """Echoes which passages it was handed so the context can be asserted."""

    name = "fake-generator"

    def __init__(self) -> None:
        self.calls: list[dict[str, Any]] = []

    def generate(self, query: str, passages: Sequence[Chunk | str]) -> str:
        texts = [p.text if isinstance(p, Chunk) else str(p) for p in passages]
        self.calls.append({"query": query, "passages": texts})
        return f"FAKE ANSWER from {len(texts)} passage(s): {' | '.join(texts)}"


def _retrieved() -> list[tuple[Chunk, float]]:
    return [
        (Chunk(chunk_id=i, text=text, source="test-corpus", ordinal=i), round(0.9 - 0.1 * i, 3))
        for i, text in enumerate(PASSAGES)
    ]


def check_chunking() -> None:
    """Chunk windows honour chunk_size and long input yields several chunks."""
    short = "one two three"
    long_doc = " ".join(f"w{i}" for i in range(600))
    chunks = chunk_documents([short, long_doc], chunk_size=100, overlap=20)

    assert len(chunks) > 1, f"600 words at 100/chunk must split, got {len(chunks)}"
    assert all(len(c.text.split()) <= 100 for c in chunks), "a chunk exceeded chunk_size"
    assert [c.chunk_id for c in chunks] == list(range(len(chunks))), "chunk ids must be contiguous"
    assert chunks[0].source == "doc-0", "first document must keep its own source label"
    assert chunks[-1].source == "doc-1", "sources must track the document, not the chunk"

    # Overlap is the point of the parameter: consecutive chunks of one document
    # must share boundary words, otherwise a fact split across the cut is lost.
    doc_one = [c for c in chunks if c.source == "doc-1"]
    shared = set(doc_one[0].text.split()) & set(doc_one[1].text.split())
    assert shared, "expected overlapping words between consecutive chunks"

    named = chunk_documents([long_doc], chunk_size=100, sources=["uploaded.md"])
    assert named[0].source == "uploaded.md", "explicit sources must be honoured"

    assert chunk_documents([]) == [], "no documents means no chunks"
    assert chunk_documents(["   "]) == [], "blank documents produce no chunks"

    for bad in (0, -5):
        try:
            chunk_documents([long_doc], chunk_size=bad)
        except ValueError:
            continue
        raise AssertionError(f"chunk_size={bad} should raise ValueError")
    try:
        chunk_documents([long_doc], chunk_size=100, overlap=100)
    except ValueError:
        pass
    else:
        raise AssertionError("overlap == chunk_size must raise ValueError")


def check_off_arm_skips_grader() -> None:
    """use_grader=False answers from everything retrieved and never grades."""
    grader = CountingGrader()  # would reject everything if it were consulted
    generator = FakeGenerator()
    result = answer_with_gate(
        QUERY, _retrieved(), use_grader=False, generator=generator, grader=grader
    )

    assert grader.calls == [], f"OFF arm invoked the grader: {grader.calls}"
    assert result.used_grader is False, "OFF arm must report used_grader=False"
    assert result.graded == [], "OFF arm has no labels, so it must not report any"
    assert len(generator.calls) == 1, f"expected one generation, got {len(generator.calls)}"
    assert len(generator.calls[0]["passages"]) == len(PASSAGES), (
        "OFF arm must answer from ALL retrieved passages"
    )
    assert result.context_ids == [0, 1, 2], f"unexpected context: {result.context_ids}"
    assert result.grounded is True, "OFF arm with a non-empty context is grounded"
    assert NO_GROUNDED_ANSWER not in result.answer, "OFF arm must actually answer"


def check_on_arm_drops_irrelevant() -> None:
    """use_grader=True keeps accepted passages and cuts the rest from context."""
    grader = CountingGrader(["relevant", "irrelevant", "relevant"])
    generator = FakeGenerator()
    result = answer_with_gate(
        QUERY, _retrieved(), use_grader=True, generator=generator, grader=grader
    )

    assert len(grader.calls) == 1, f"expected exactly one grading call, got {len(grader.calls)}"
    assert grader.calls[0]["top_k"] == len(PASSAGES), (
        "every passage must be graded — a truncated trace reads as a bug"
    )
    assert len(result.graded) == len(PASSAGES), "the trace must cover every retrieved passage"
    assert [g.label for g in result.graded] == ["relevant", "irrelevant", "relevant"], (
        f"unexpected graded labels: {[g.label for g in result.graded]}"
    )
    assert [g.accepted for g in result.graded] == [True, False, True], (
        "irrelevant passage must be rejected, the others accepted"
    )
    assert [g.rank for g in result.graded] == [1, 2, 3], "graded trace must be ranked"

    kept_texts = generator.calls[0]["passages"]
    assert len(kept_texts) == 2, f"generator should see 2 passages, saw {len(kept_texts)}"
    assert PASSAGES[1] not in kept_texts, "the irrelevant passage reached the generator"
    assert result.context_ids == [0, 2], f"unexpected context ids: {result.context_ids}"
    assert result.grounded is True, "an accepted context is grounded"

    # The retrieval score must survive grading — the GUI shows both signals.
    assert result.graded[0].retrieval_score == 0.9, "retrieval score was lost during grading"

    # partial is accepted by default policy (it still grounds an answer).
    partial = grade_passages(
        QUERY, [PASSAGES[0]], grader=CountingGrader(["partial"]), retrieval_scores=[0.5]
    )
    assert partial[0].accepted is True, "partial passages must be accepted by default policy"


def check_on_arm_refuses_when_all_rejected() -> None:
    """Reject-everything returns the explicit no-answer result, generating nothing."""
    grader = CountingGrader(["irrelevant", "irrelevant", "irrelevant"])
    generator = FakeGenerator()
    result = answer_with_gate(
        QUERY, _retrieved(), use_grader=True, generator=generator, grader=grader
    )

    assert result.grounded is False, "result must be flagged ungrounded"
    assert result.answer == NO_GROUNDED_ANSWER, (
        f"expected the explicit no-grounded-answer text, got {result.answer!r}"
    )
    assert generator.calls == [], "the generator must not be called when the gate rejects all"
    assert result.context_ids == [], "no passage may be cited when all are rejected"
    for text in PASSAGES:
        assert text not in result.answer, "refusal text must not contain passage content"
    assert len(result.graded) == len(PASSAGES), (
        "the rejection trace must still reach the GUI"
    )
    assert "not called" in result.note, f"note should explain the refusal: {result.note!r}"

    empty = answer_with_gate(QUERY, [], use_grader=True, generator=FakeGenerator())
    assert empty.grounded is False and empty.answer == NO_GROUNDED_ANSWER, (
        "no retrieval must also refuse rather than answer"
    )


def _assert_json_native(value: Any, path: str = "root") -> None:
    allowed = (str, int, float, bool, list, dict, type(None))
    if not isinstance(value, allowed):
        raise AssertionError(f"{path} is {type(value).__name__}, not JSON-native")
    if isinstance(value, dict):
        for key, item in value.items():
            _assert_json_native(item, f"{path}.{key}")
    elif isinstance(value, list):
        for i, item in enumerate(value):
            _assert_json_native(item, f"{path}[{i}]")


def check_serialisation() -> None:
    """AnswerResult.to_dict() is plain JSON and survives a round trip."""
    grader = CountingGrader(["relevant", "partial", "irrelevant"])
    result = answer_with_gate(
        QUERY, _retrieved(), use_grader=True, generator=FakeGenerator(), grader=grader
    )
    payload = result.to_dict()
    _assert_json_native(payload)
    encoded = json.dumps(payload)
    decoded = json.loads(encoded)

    assert decoded["answer"] == result.answer, "answer changed through serialisation"
    assert decoded["grounded"] is True and decoded["used_grader"] is True, (
        "booleans must stay booleans"
    )
    assert decoded["n_retrieved"] == len(PASSAGES), "n_retrieved must be the retriever's count"
    assert decoded["n_kept"] == 2, f"expected 2 kept passages, got {decoded['n_kept']}"
    assert decoded["context_ids"] == [0, 1], f"bad context ids: {decoded['context_ids']}"
    assert len(decoded["retrieved"]) == len(PASSAGES), "retrieval trace must serialise"
    assert len(decoded["graded"]) == len(PASSAGES), "grading trace must serialise"
    first = decoded["graded"][0]
    for key in ("label", "confidence", "accepted", "retrieval_score", "rank", "text"):
        assert key in first, f"graded entry is missing {key!r}"

    off_payload = answer_with_gate(
        QUERY, _retrieved(), use_grader=False, generator=FakeGenerator()
    ).to_dict()
    json.dumps(off_payload)
    assert off_payload["graded"] == [], "the OFF arm must serialise an empty grading trace"
    assert off_payload["n_kept"] == len(PASSAGES), "the OFF arm keeps every retrieved passage"


def check_metrics_come_from_json() -> None:
    """Metrics are read from results/*.json at runtime, never hardcoded."""
    missing_dir = load_metrics(results_dir=Path(__file__).resolve().parent / "__no_such_dir__")
    assert missing_dir["baseline"] is None and missing_dir["finetuned"] is None, (
        "a missing results dir must yield None, never a default number"
    )
    assert missing_dir["missing"], "a missing results dir must be reported"
    _assert_json_native(missing_dir)

    synthetic = {
        "baseline": {"accuracy": 0.37, "macro_f1": 0.267, "per_class": {"partial": {"f1": 0.013}}},
        "finetuned": {"accuracy": 0.45, "macro_f1": 0.412, "per_class": {"partial": {"f1": 0.20}}},
    }
    rows = before_after_rows(synthetic)
    macros = next(r for r in rows if r["metric"] == "macro-F1")
    assert macros["delta"] == round(0.412 - 0.267, 4), f"bad delta: {macros['delta']}"
    assert before_after_rows({"baseline": None, "finetuned": None}) == [], (
        "no measured numbers means no rows"
    )


def check_no_heavy_imports() -> None:
    """Nothing in this test may pull in torch, transformers, faiss or ST."""
    loaded = [
        name
        for name in HEAVY_MODULES
        if name in sys.modules and not _IMPORTS_AT_START[name]
    ]
    assert not loaded, f"gate logic must stay testable without ML imports, but loaded {loaded}"


def main() -> int:
    checks = (
        ("chunking respects chunk_size and splits long input", check_chunking),
        ("OFF arm answers without invoking the grader", check_off_arm_skips_grader),
        ("ON arm drops irrelevant passages", check_on_arm_drops_irrelevant),
        ("ON arm refuses instead of hallucinating", check_on_arm_refuses_when_all_rejected),
        ("AnswerResult.to_dict() is JSON-serializable", check_serialisation),
        ("metrics are read from results/*.json", check_metrics_come_from_json),
        ("no model/ML import happened", check_no_heavy_imports),
    )
    failures = 0
    for name, check in checks:
        try:
            check()
        except AssertionError as exc:
            failures += 1
            print(f"FAIL  {name}\n        {exc}")
        except Exception as exc:  # noqa: BLE001 — a crash is a failure, not a stack trace dump
            failures += 1
            print(f"ERROR {name}\n        {type(exc).__name__}: {exc}")
        else:
            print(f"PASS  {name}")

    total = len(checks)
    print(f"\n{total - failures}/{total} checks passed")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
