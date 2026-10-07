"""Replay path: a recorded trace must reproduce the canned verdict end-to-end.

Locks _replay_result/_recorded_trace_path behaviour with the real tracked
results/demo_trace.json (and the per-corpus files), so a corrupt or dropped
recording fails CI instead of the demo.
"""
from __future__ import annotations

import json


def test_legacy_trace_replays(app, trace):
    result = app._replay_result("espresso", trace["corpus"]["query"])
    assert result is not None, "espresso has a recorded trace but replay returned None"
    assert result["mode"] == "replay"
    on = result["on"]
    assert on["context_ids"] == trace["arms"]["on"]["context_ids"]
    assert len(on["graded"]) == len(trace["arms"]["on"]["graded"])


def test_replay_matches_file_contents(app, results_dir):
    recorded = json.loads((results_dir / "demo_trace.json").read_text(encoding="utf-8"))
    result = app._replay_result("espresso", recorded["corpus"]["query"])
    assert result is not None
    assert result["on"]["answer"] == recorded["arms"]["on"]["answer"]
    assert result["off"]["answer"] == recorded["arms"]["off"]["answer"]


def test_per_corpus_traces_replay(app, results_dir):
    for corpus in ("amsterdam", "colab-t4"):
        path = results_dir / f"demo_trace_{corpus}.json"
        assert path.exists(), f"{path.name} must be tracked in git"
        recorded = json.loads(path.read_text(encoding="utf-8"))
        result = app._replay_result(corpus, recorded["corpus"]["query"])
        assert result is not None, f"{corpus} recorded trace failed to replay"
        assert result["mode"] == "replay"
        assert result["on"]["answer"] == recorded["arms"]["on"]["answer"]


def test_unknown_corpus_or_query_returns_none(app):
    assert app._replay_result("espresso", "no such query exists anywhere") is None
    assert app._replay_result("nonexistent-corpus", "anything") is None


def test_happy_path_ignores_replay(app, trace, monkeypatch):
    monkeypatch.setattr(
        app, "retrieve_for", lambda *a, **k: trace["arms"]["on"]["retrieved"]
    )
    monkeypatch.setattr(app, "prefetch_weights", lambda: "weights ready")

    def live_run(*a, **k):
        return {
            "mode": "live",
            "model": {"description": "d", "using_adapter": True, "load_error": None},
            "off": trace["arms"]["off"],
            "on": trace["arms"]["on"],
        }

    monkeypatch.setattr(app, "run_arms", live_run)
    out = app.run_query("espresso", "q", 4, 150, True)
    assert "live" in out[5], "happy path must report live mode"
    assert "replay" not in out[5]
