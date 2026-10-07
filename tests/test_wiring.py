"""Every run_query/_render return path must yield exactly 6 outputs.

The UI rewrite changed the output arity from 5 to 6. A path returning the old
count surfaces to a visitor as an opaque Gradio error, so every branch is
asserted here rather than discovered on the Space. Locks the repair of
scripts/verify_ui_wiring.py (which referenced the removed
`app.NO_GROUNDED_ANSWER` re-export).
"""
from __future__ import annotations

from src.rag_pipeline import NO_GROUNDED_ANSWER

EXPECTED = 6


def _base_result(trace: dict) -> dict:
    return {
        "mode": "replay",
        "model": {"description": "m", "using_adapter": True, "load_error": None},
        "off": trace["arms"]["off"],
        "on": trace["arms"]["on"],
    }


def _refused_result(trace: dict) -> dict:
    result = _base_result(trace)
    result["on"] = {
        "answer": NO_GROUNDED_ANSWER,
        "grounded": False,
        "used_grader": True,
        "context_ids": [],
        "retrieved": trace["arms"]["on"]["retrieved"],
        "graded": [dict(g, accepted=False) for g in trace["arms"]["on"]["graded"]],
        "note": "gate rejected all",
    }
    return result


def _assert_arity(name: str, value) -> None:
    assert isinstance(value, tuple), f"{name}: expected tuple, got {type(value).__name__}"
    assert len(value) == EXPECTED, f"{name}: expected {EXPECTED} outputs, got {len(value)}"


def test_render_live(app, trace):
    _assert_arity("render live", app._render(_base_result(trace), "q", True, "live"))


def test_render_refused(app, trace):
    _assert_arity("render refused", app._render(_refused_result(trace), "q", True, "live"))


def test_fail_helper(app):
    _assert_arity("fail", app._fail("boom", "status"))


def test_empty_query(app):
    _assert_arity("empty query", app.run_query("", "   ", 4, 150, True))


def test_no_hits(app, monkeypatch):
    monkeypatch.setattr(app, "retrieve_for", lambda *a, **k: [])
    _assert_arity("no hits", app.run_query("espresso", "q", 4, 150, True))


def test_retrieval_failure_without_replay(app, monkeypatch):
    def boom(*a, **k):
        raise RuntimeError("no embeddings")

    monkeypatch.setattr(app, "retrieve_for", boom)
    monkeypatch.setattr(app, "_replay_result", lambda *a, **k: None)
    _assert_arity("retrieval failed", app.run_query("espresso", "q", 4, 150, True))


def test_retrieval_failure_with_replay(app, trace, monkeypatch):
    def boom(*a, **k):
        raise RuntimeError("no embeddings")

    monkeypatch.setattr(app, "retrieve_for", boom)
    monkeypatch.setattr(app, "_replay_result", lambda *a, **k: _base_result(trace))
    _assert_arity("retrieval failed + replay", app.run_query("espresso", "q", 4, 150, True))


def test_gpu_failure_without_replay(app, trace, monkeypatch):
    monkeypatch.setattr(
        app, "retrieve_for", lambda *a, **k: trace["arms"]["on"]["retrieved"]
    )
    monkeypatch.setattr(app, "_replay_result", lambda *a, **k: None)
    monkeypatch.setattr(app, "prefetch_weights", lambda: "weights ready (5 files)")

    def gpu_boom(*a, **k):
        raise RuntimeError("No GPU was available in the queue")

    monkeypatch.setattr(app, "run_arms", gpu_boom)
    _assert_arity("gpu failed", app.run_query("espresso", "q", 4, 150, True))


def test_gpu_failure_with_replay(app, trace, monkeypatch):
    monkeypatch.setattr(
        app, "retrieve_for", lambda *a, **k: trace["arms"]["on"]["retrieved"]
    )
    monkeypatch.setattr(app, "prefetch_weights", lambda: "weights ready (5 files)")
    monkeypatch.setattr(app, "_replay_result", lambda *a, **k: _base_result(trace))

    def gpu_boom(*a, **k):
        raise RuntimeError("No GPU was available in the queue")

    monkeypatch.setattr(app, "run_arms", gpu_boom)
    _assert_arity("gpu failed + replay", app.run_query("espresso", "q", 4, 150, True))


def test_happy_path(app, trace, monkeypatch):
    monkeypatch.setattr(
        app, "retrieve_for", lambda *a, **k: trace["arms"]["on"]["retrieved"]
    )
    monkeypatch.setattr(app, "prefetch_weights", lambda: "weights ready (5 files)")
    monkeypatch.setattr(app, "run_arms", lambda *a, **k: _base_result(trace))
    _assert_arity("happy path", app.run_query("espresso", "q", 4, 150, True))


def test_blocks_receives_css(app, ui):
    assert "css" in app.demo.kwargs and app.demo.kwargs["css"], "Blocks must receive css"
    assert app.demo.kwargs["css"] == ui.CSS


def test_import_reexports_resolved(ui):
    assert ui.NO_GROUNDED_ANSWER
