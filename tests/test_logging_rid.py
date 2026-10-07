"""Correlation-id contract: every log line carries a click's rid, and the
ContextVar never leaks past its click.

A leaked rid stamps an unrelated visitor's later lines with a stale id — the
bug the run_query wrapper (try/finally reset) exists to prevent. The log format
interpolates %(rid)s/%(event)s, so records without them would crash formatting;
the filter's defaults are part of the contract too.
"""
from __future__ import annotations

import logging


def _bare_record(message: str = "hello") -> logging.LogRecord:
    return logging.LogRecord("grader", logging.INFO, __file__, 1, message, None, None)


def test_event_filter_defaults_missing_fields(app):
    record = _bare_record()
    assert not hasattr(record, "event")
    assert app._EventFilter().filter(record) is True
    assert record.event == "-"
    assert record.rid == "-", "unset ContextVar must default to '-' in this thread"


def test_event_filter_keeps_explicit_values(app):
    record = _bare_record()
    record.rid = "abc12345"
    record.event = "explicit"
    app._EventFilter().filter(record)
    assert record.rid == "abc12345"
    assert record.event == "explicit"


def test_log_format_renders(app):
    record = _bare_record("the message")
    app._EventFilter().filter(record)
    out = logging.Formatter(app.LOG_FORMAT, datefmt=app.LOG_DATEFMT).format(record)
    assert "rid=" in out
    assert "event=" in out
    assert "the message" in out


def test_rid_set_and_reset_on_early_return(app):
    token = app._RID.set("outer-click")
    try:
        app.run_query("espresso", "   ", 4, 150, True)
        assert app._RID.get() == "outer-click", "run_query leaked its rid"
    finally:
        app._RID.reset(token)


def test_rid_set_and_reset_on_failure_path(app, monkeypatch):
    def boom(*a, **k):
        raise RuntimeError("no embeddings")

    token = app._RID.set("outer-click")
    try:
        monkeypatch.setattr(app, "retrieve_for", boom)
        monkeypatch.setattr(app, "_replay_result", lambda *a, **k: None)
        out = app.run_query("espresso", "q", 4, 150, True)
        assert "retrieval failed" in out[5]
        assert app._RID.get() == "outer-click", "failure path leaked its rid"
    finally:
        app._RID.reset(token)


def test_rejected_request_log_carries_click_rid(app, grader_records):
    app.run_query("espresso", "   ", 4, 150, True)
    rejected = [r for r in grader_records if getattr(r, "event", "") == "request_rejected"]
    assert rejected, "empty-query rejection must be logged"
    rid = getattr(rejected[0], "rid", None)
    assert rid and rid != "-" and len(rid) == 8 and all(
        c in "0123456789abcdef" for c in rid
    ), f"rejection must carry the click's 8-hex rid, got {rid!r}"


def test_request_log_carries_click_rid(app, grader_records):
    app.run_query("espresso", "q", 4, 150, True)
    requests = [r for r in grader_records if getattr(r, "event", "") == "request"]
    assert requests
    rid = getattr(requests[0], "rid", None)
    assert rid and rid != "-" and len(rid) == 8


def test_two_clicks_get_different_rids(app, grader_records):
    app.run_query("espresso", "   ", 4, 150, True)
    app.run_query("espresso", "   ", 4, 150, True)
    rids = {
        getattr(r, "rid", None)
        for r in grader_records
        if getattr(r, "event", "") == "request_rejected"
    }
    assert len(rids) == 2, f"each click needs its own rid, got {rids}"
