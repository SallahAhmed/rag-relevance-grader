"""Regression lock: _run_arms_on_gpu must run the pipeline exactly once.

The refactor shipped a duplicated block that re-built grader/generator/pairs and
re-ran both arms inside one GPU window — double quota burn, double latency, and
a reservation estimated for a single pass. This test fails if the duplication
ever returns, and pins the done-log's correlation id.
"""
from __future__ import annotations


class _FakeResult:
    def __init__(self, used_grader: bool) -> None:
        self.grounded = True
        self.context_ids = [0, 1]
        self.retrieved = [{"chunk_id": 0}, {"chunk_id": 1}]
        self.used_grader = used_grader
        self.answer = "answer"
        self.graded = []
        self.note = None

    def to_dict(self) -> dict:
        return {"grounded": self.grounded, "answer": self.answer}


class _FakeBundle:
    using_adapter = True
    load_error = None

    def describe(self) -> str:
        return "fake bundle"


def test_pipeline_runs_once_per_gpu_call(app, trace, monkeypatch, grader_records):
    grade_calls = []
    monkeypatch.setattr(app, "load_local_model", lambda: _FakeBundle())
    monkeypatch.setattr(app, "HFRelevanceGrader", lambda bundle=None, **k: object())
    monkeypatch.setattr(app, "QwenGenerator", lambda bundle=None, max_new_tokens=None, **k: object())

    def fake_answer(query, pairs, use_grader, generator=None, grader=None):
        grade_calls.append(use_grader)
        return _FakeResult(use_grader)

    monkeypatch.setattr(app, "answer_with_gate", fake_answer)
    hits = trace["arms"]["on"]["retrieved"]

    result = app._run_arms_on_gpu("espresso", "what is pressure?", hits, 50, rid="rid42")

    assert len(grade_calls) == 2, (
        f"both arms must run exactly once per GPU call, got {len(grade_calls)} "
        "answer_with_gate calls (a duplicated block means 4)"
    )
    assert grade_calls == [False, True]
    assert result["mode"] == "live"

    done = [r for r in grader_records if getattr(r, "event", "") == "gpu_worker_done"]
    assert len(done) == 1, f"expected exactly one gpu_worker_done log, got {len(done)}"
    assert getattr(done[0], "rid", None) == "rid42"


def test_estimate_duration_tolerates_gradio_progress_arg(app):
    seconds = app._estimate_duration("espresso", "q", [1, 2], 100, "rid")
    assert isinstance(seconds, int)
    assert 8 <= seconds <= 120
    with_progress = app._estimate_duration(
        "espresso", "q", [1, 2], 100, "rid", progress=0.5
    )
    assert with_progress == seconds
