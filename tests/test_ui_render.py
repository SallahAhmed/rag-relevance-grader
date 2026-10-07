"""HTML content checks for the ui.py renderers: real HTML, escaping, benchmarks.

Ported from scripts/verify_ui_render.py into pytest. Gradio is never involved —
ui.py is pure presentation over pipeline dicts.
"""
from __future__ import annotations

import re


def test_arm_card_is_real_html(ui, trace):
    arm = trace["arms"]["on"]
    card = ui._arm_card(arm, "Grader ON", True, naive=False)
    assert "###" not in card
    assert "**" not in card
    assert "<div" in card
    assert "is-primary" in card
    assert "tag-naive" not in card

    naive = ui._arm_card(trace["arms"]["off"], "Grader OFF", False, naive=True)
    assert "tag-naive" in naive
    assert "is-primary" not in naive


def test_refusal_card_styled(ui):
    refused = {
        "answer": ui.NO_GROUNDED_ANSWER,
        "grounded": False,
        "used_grader": True,
        "context_ids": [],
        "retrieved": [{"chunk_id": 0, "source": "s/1", "score": 0.4, "text": "x"}],
        "graded": [
            {
                "chunk_id": 0,
                "source": "s/1",
                "text": "x",
                "retrieval_score": 0.4,
                "label": "irrelevant",
                "score": 0.9,
                "confidence": 0.9,
                "accepted": False,
                "rank": 1,
            }
        ],
        "note": "gate rejected all",
    }
    card = ui._arm_card(refused, "Grader ON", True, naive=False)
    assert "is-refusal" in card
    assert "gate rejected all" in card


def test_gate_summary_counts(ui, trace):
    on = trace["arms"]["on"]
    kept = len(on.get("context_ids") or [])
    total = len(on.get("retrieved") or [])
    summary = ui._gate_summary({"on": on})
    assert f"kept {kept} of {total}" in summary
    assert "**" not in summary


def test_gate_summary_refusal_and_no_grade(ui, trace):
    refused_on = {
        "answer": ui.NO_GROUNDED_ANSWER,
        "grounded": False,
        "used_grader": True,
        "context_ids": [],
        "retrieved": trace["arms"]["on"]["retrieved"],
        "graded": [dict(g, accepted=False) for g in trace["arms"]["on"]["graded"]],
    }
    refused = ui._gate_summary({"on": refused_on})
    assert "rejected every passage" in refused
    assert "is-refused" in refused

    nograde = ui._gate_summary({"on": {**refused_on, "graded": []}})
    assert "did not run" in nograde


def test_verdict_passages(ui, trace):
    on = trace["arms"]["on"]
    rendered = ui._verdict_passages(on)
    assert rendered.count('class="passage') == len(on["retrieved"])
    assert "conf-fill" in rendered
    graded = on.get("graded") or []
    if any(g.get("accepted") for g in graded):
        assert "is-kept" in rendered
    if any(not g.get("accepted") for g in graded):
        assert "is-dropped" in rendered

    off = trace["arms"]["off"]
    rendered_off = ui._verdict_passages(off)
    if not off.get("graded"):
        assert "is-nogate" in rendered_off


def test_verdict_passages_escapes_untrusted_text(ui):
    xss = {
        "chunk_id": 0,
        "source": "<script>alert(1)</script>",
        "text": "<img src=x onerror=alert(2)>",
        "score": 0.5,
    }
    rendered = ui._verdict_passages({"retrieved": [xss], "graded": []})
    assert "<script>" not in rendered
    assert "<img" not in rendered
    assert "&lt;script&gt;" in rendered


def test_model_banner_escapes_load_error(ui):
    banner = ui._model_banner(
        {"model": {"description": "d", "using_adapter": True, "load_error": "<b>x</b>"}}
    )
    assert "<b>x</b>" not in banner
    assert "&lt;b&gt;" in banner


def test_benchmark_panel_from_real_results(ui, results_dir):
    panel = ui.benchmark_html(str(results_dir))
    assert "bench-wrap" in panel
    assert "is-headline" in panel
    assert "0.2672" in panel, "zero-shot macro-F1 missing from panel"
    assert "0.4306" in panel, "fine-tuned macro-F1 missing from panel"
    assert "+0.1635" in panel, "macro-F1 delta missing from panel"
    assert "pred" in panel and "true" in panel
    assert "MMLU retention" in panel
    assert "**" not in panel
    match = re.search(r"macro-F1 Δ ([+-][\d.]+)", panel)
    assert match is not None, "macro-F1 delta not rendered"


def test_benchmark_panel_missing_results_invents_nothing(ui, monkeypatch, tmp_path):
    monkeypatch.setattr(
        ui,
        "load_metrics",
        lambda *a, **k: {"baseline": None, "finetuned": None, "missing": ["x.json"]},
    )
    panel = ui.benchmark_html(str(tmp_path))
    assert "No benchmark JSON" in panel
    assert "0.4306" not in panel and "0.2672" not in panel


def test_benchmark_panel_reads_dir_argument(ui, monkeypatch, tmp_path):
    seen = {}

    def spy(results_dir=None):
        seen["dir"] = results_dir
        return {"baseline": None, "finetuned": None, "missing": ["x.json"]}

    monkeypatch.setattr(ui, "load_metrics", spy)
    ui.benchmark_html(str(tmp_path))
    assert seen["dir"] == str(tmp_path)


def test_hero_and_css(ui):
    assert "<h1>" in ui._hero_html()
    assert "arm-card" in ui.CSS
    assert "conf-fill" in ui.CSS
    assert "gate-summary" in ui.CSS
