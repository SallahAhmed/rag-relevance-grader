"""Benchmark contract: the before/after table reads real numbers from
results/*.json, with no hardcoded score in the UI and both pinned bars honest.

Covers src.rag_pipeline.load_metrics/before_after_rows plus the app's explicit
`benchmark_html(str(RESULTS_DIR))` wiring (a missing str() would pass a Path
into load_metrics, whose probe then needs config/ next to src/ — true in the
repo, not guaranteed on the Space).
"""
from __future__ import annotations

from pathlib import Path


def test_load_metrics_reads_both_jsons(results_dir):
    import src.rag_pipeline as rp

    metrics = rp.load_metrics(str(results_dir))
    assert metrics["missing"] == [], f"tracked benchmark files missing: {metrics['missing']}"
    assert metrics["baseline"] is not None and metrics["finetuned"] is not None


def test_pinned_values(results_dir):
    import src.rag_pipeline as rp

    metrics = rp.load_metrics(str(results_dir))
    base, ft = metrics["baseline"], metrics["finetuned"]
    assert round(base["macro_f1"], 4) == 0.2672, "zero-shot macro-F1 drifted from baseline"
    assert round(ft["macro_f1"], 4) == 0.4306, "fine-tuned macro-F1 drifted from result"
    assert ft["macro_f1"] - base["macro_f1"] >= 0.10, "the +10pp macro-F1 bar is not met"


def test_forgetting_bar_passes_and_is_reported(results_dir):
    import src.rag_pipeline as rp

    metrics = rp.load_metrics(str(results_dir))
    forget = metrics["finetuned"]["forgetting"]
    assert forget["retention"] >= 0.90, f"MMLU retention below the 90% bar: {forget}"
    assert forget["passes"] is True


def test_same_test_queries_on_both_sides(results_dir):
    import src.rag_pipeline as rp

    metrics = rp.load_metrics(str(results_dir))
    base, ft = metrics["baseline"], metrics["finetuned"]
    assert base["n_test_queries"] == ft["n_test_queries"], "eval split drifted between runs"
    assert base["n_test_queries"] > 0


def test_before_after_rows_from_tracked_jsons(results_dir):
    import src.rag_pipeline as rp

    metrics = rp.load_metrics(str(results_dir))
    rows = rp.before_after_rows(metrics)
    names = [r["metric"] for r in rows]
    assert names.count("accuracy") == 1 and names.count("macro-F1") == 1
    assert sum(1 for n in names if n.startswith("F1 — ")) == 3, "per-class rows required"
    macro = next(r for r in rows if r["metric"] == "macro-F1")
    assert macro["baseline"] == 0.2672 and macro["finetuned"] == 0.4306
    expected = round(
        metrics["finetuned"]["macro_f1"] - metrics["baseline"]["macro_f1"], 4
    )
    assert macro["delta"] == expected


def test_missing_results_invent_nothing(tmp_path):
    import src.rag_pipeline as rp

    metrics = rp.load_metrics(str(tmp_path))
    assert metrics["baseline"] is None and metrics["finetuned"] is None
    assert metrics["missing"], "a missing dir must be reported"
    assert rp.before_after_rows(metrics) == [], "missing sides must yield no rows"


def test_app_benchmark_panel_wired_with_explicit_results_dir(root):
    source = (root / "space_gradio" / "app.py").read_text(encoding="utf-8")
    assert "benchmark_html(str(RESULTS_DIR))" in source, (
        "app must pass str(RESULTS_DIR) explicitly; a bare Path or a defaulted "
        "call breaks on the deployed Space"
    )


def test_results_dir_fixture_is_repo_results(results_dir):
    assert results_dir.is_dir()
    for name in ("baseline_results.json", "finetuned_results.json"):
        assert (Path(results_dir) / name).is_file()
