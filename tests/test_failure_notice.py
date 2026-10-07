"""The GPU-failure notice must reach gr.HTML as escaped, non-markdown HTML.

_gpu_failure_notice builds markdown containing the raw exception text. The
refactor moved the failure path onto gr.HTML, which ships strings verbatim:
unescaped, that is a trust-boundary bug and literal `- raw error:` bullets on
screen. Locks _notice_html and the return path that uses it.
"""
from __future__ import annotations

EVIL = "queue <script>alert(1)</script> full"


def test_notice_html_escapes_and_converts(app):
    notice = app._gpu_failure_notice(RuntimeError(EVIL), "espresso")
    rendered = app._notice_html(notice)

    assert "<script>" not in rendered
    assert "&lt;script&gt;" in rendered
    assert "- raw error:" not in rendered, "markdown bullet must be converted, not shipped"
    assert "\n" not in rendered, "gr.HTML gets no line breaks to render — emit tags"
    assert "<li>" in rendered
    assert "<code>" in rendered


def test_notice_html_is_idempotent_on_clean_text(app):
    rendered = app._notice_html("plain failure text")
    assert rendered == "<div>plain failure text</div>"


def test_gpu_failure_return_path_is_html(app, trace, monkeypatch):
    monkeypatch.setattr(
        app, "retrieve_for", lambda *a, **k: trace["arms"]["on"]["retrieved"]
    )
    monkeypatch.setattr(app, "_replay_result", lambda *a, **k: None)
    monkeypatch.setattr(app, "prefetch_weights", lambda: "weights ready")

    def gpu_boom(*a, **k):
        raise RuntimeError(EVIL)

    monkeypatch.setattr(app, "run_arms", gpu_boom)
    out = app.run_query("espresso", "q", 4, 150, True)

    answers, model_line, verdicts, on_table, off_table, status = out
    for slot in (answers, model_line):
        assert "<script>" not in slot
        assert "&lt;script&gt;" in slot
        assert "- raw error:" not in slot
    assert answers.startswith('<div class="warn-box">')
    assert status.startswith('<div class="status-line">')
    assert verdicts == "" and on_table == [] and off_table == []


def test_gpu_failure_status_escapes_prefetch(app, trace, monkeypatch):
    monkeypatch.setattr(
        app, "retrieve_for", lambda *a, **k: trace["arms"]["on"]["retrieved"]
    )
    monkeypatch.setattr(app, "_replay_result", lambda *a, **k: None)
    monkeypatch.setattr(app, "prefetch_weights", lambda: "failed <oops>")

    def gpu_boom(*a, **k):
        raise RuntimeError("no gpu")

    monkeypatch.setattr(app, "run_arms", gpu_boom)
    out = app.run_query("espresso", "q", 4, 150, True)
    assert "<oops>" not in out[5]
    assert "&lt;oops&gt;" in out[5]


def test_fail_helper_escapes(app):
    body, model_line, verdicts, on_table, off_table, status = app._fail(
        "bad <b>query</b>", "no query"
    )
    assert "<b>query</b>" not in body
    assert "&lt;b&gt;" in body
    assert "<b>query</b>" not in status


def test_render_status_escapes_replay_note(app, trace):
    replay = {
        "mode": "replay",
        "provenance": {"recorded_from": "cpu <path>", "device": "cpu", "quantized": False},
        "model": {"description": "m", "using_adapter": True, "load_error": None},
        "off": trace["arms"]["off"],
        "on": trace["arms"]["on"],
    }
    out = app._render(replay, "q", True, "live")
    status = out[5]
    assert "<path>" not in status
    assert "&lt;path&gt;" in status
    assert "&amp;lt;" not in status, "note must be escaped exactly once"
