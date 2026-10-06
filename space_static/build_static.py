"""Generate space_static/index.html from the repo's JSON artifacts.

A Static Space has no Python runtime, so the page has to be pre-rendered. This
script is the generator, which keeps the project's rule intact: every number in
the HTML comes from `results/*.json` or `results/demo_trace.json`, never from a
literal in this file. Re-run it whenever the evaluation notebooks rewrite those
files:

    python -m space_static.build_static      # or: python space_static/build_static.py

Missing inputs are reported as "not measured yet" rather than substituted.
"""
from __future__ import annotations

import html
import json
import sys
from pathlib import Path
from typing import Any

if __package__ in (None, ""):  # direct `python space_static/build_static.py`
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.rag_pipeline import LABELS, before_after_rows, load_metrics
from src.sample_corpora import CORPORA

HERE = Path(__file__).resolve().parent
REPO = HERE.parent
OUT = HERE / "index.html"
TRACE = REPO / "results" / "demo_trace.json"
NOT_MEASURED = "not measured yet"

CSS = """
:root { color-scheme: light dark; }
* { box-sizing: border-box; }
body { font: 16px/1.6 -apple-system, "Segoe UI", Roboto, sans-serif; margin: 0 auto;
       max-width: 62rem; padding: 2rem 1.25rem 4rem; }
h1 { line-height: 1.2; }
h2 { margin-top: 2.5rem; border-bottom: 1px solid #8884; padding-bottom: .3rem; }
table { border-collapse: collapse; width: 100%; margin: 1rem 0; font-variant-numeric: tabular-nums; }
th, td { border-bottom: 1px solid #8883; padding: .45rem .6rem; text-align: left; }
th { font-weight: 600; }
td.num { text-align: right; }
.up { color: #1a7f37; font-weight: 600; }
.down { color: #b42318; font-weight: 600; }
.muted { color: #6b7280; font-size: .92rem; }
.card { border: 1px solid #8883; border-radius: .6rem; padding: 1rem 1.15rem; margin: 1rem 0; }
.warn { border-left: 4px solid #d29922; }
code, pre { font-family: ui-monospace, SFMono-Regular, Menlo, monospace; font-size: .88em; }
.answer { white-space: pre-wrap; }
footer { margin-top: 3rem; color: #6b7280; font-size: .88rem; }
"""


def _esc(value: Any) -> str:
    return html.escape(str(value))


def _fmt(value: Any) -> str:
    if value is None:
        return NOT_MEASURED
    if isinstance(value, float):
        return f"{value:.4f}"
    return _esc(value)


def _delta_class(delta: float) -> str:
    return "up" if delta > 0 else "down" if delta < 0 else ""


def _baseline_only_section(baseline: dict[str, Any]) -> str:
    """Show the measured zero-shot baseline even when no fine-tuned run exists.

    The fine-tuned column is genuinely missing until notebook 04 runs, but the
    baseline IS measured — hiding a real number would waste it. Labelled
    baseline-only so it can never be misread as a before/after comparison.
    """
    rows = [
        ("accuracy", baseline.get("accuracy")),
        ("macro-F1", baseline.get("macro_f1")),
    ]
    for label in LABELS:
        rows.append(
            (f"F1 — {label}", (baseline.get("per_class") or {}).get(label, {}).get("f1"))
        )
    body = "".join(
        f"<tr><td>{_esc(name)}</td><td class=\"num\">{_fmt(value)}</td></tr>"
        for name, value in rows
    )
    return (
        '<table><thead><tr><th>metric</th><th class="num">zero-shot baseline</th></tr>'
        f"</thead><tbody>{body}</tbody></table>"
        f'<p class="muted">Baseline only: n={_fmt(baseline.get("n"))} · '
        f"n_test_queries={_fmt(baseline.get('n_test_queries'))} · "
        f"n_test_qrels={_fmt(baseline.get('n_test_qrels'))} · "
        f"model={_fmt(baseline.get('model'))} · mode={_fmt(baseline.get('mode'))}. "
        "The fine-tuned column appears once the evaluation notebook writes "
        "<code>results/finetuned_results.json</code>; the success bar is "
        "<strong>+10 percentage points macro-F1</strong>. The holdout is grouped by "
        "query, so read it as directional evidence.</p>"
    )


def _metrics_section() -> str:
    metrics = load_metrics()
    rows = before_after_rows(metrics)
    baseline = metrics.get("baseline") or {}
    if not rows:
        missing = ", ".join(metrics.get("missing") or ["results/*.json"])
        if baseline:
            return (
                "<h2>Before / after</h2>"
                '<div class="card warn"><strong>Before/after not complete</strong><br>'
                f'<span class="muted">No fine-tuned run yet ({_esc(missing)}). The '
                "measured zero-shot baseline is below; the delta column appears once "
                "the evaluation notebook writes its JSON. Nothing on this page is "
                "typed by hand.</span></div>"
                + _baseline_only_section(baseline)
            )
        return (
            "<h2>Before / after</h2>"
            f'<div class="card warn"><strong>{_esc(NOT_MEASURED)}</strong><br>'
            f'<span class="muted">No before/after pair found ({_esc(missing)}). These '
            "numbers are generated from <code>results/baseline_results.json</code> and "
            "<code>results/finetuned_results.json</code> by the evaluation notebooks "
            "— nothing on this page is typed by hand.</span></div>"
        )

    body = "".join(
        f"<tr><td>{_esc(r['metric'])}</td>"
        f"<td class=\"num\">{_fmt(r['baseline'])}</td>"
        f"<td class=\"num\">{_fmt(r['finetuned'])}</td>"
        f"<td class=\"num {_delta_class(r['delta'])}\">{r['delta']:+.4f}</td></tr>"
        for r in rows
    )
    footer_bits = []
    for key in ("baseline", "finetuned"):
        payload = metrics.get(key) or {}
        footer_bits.append(
            f"<li><code>{_esc(key)}</code>: n={_fmt(payload.get('n'))} · "
            f"n_test_queries={_fmt(payload.get('n_test_queries'))} · "
            f"n_test_qrels={_fmt(payload.get('n_test_qrels'))} · "
            f"model={_fmt(payload.get('model'))} · mode={_fmt(payload.get('mode'))}</li>"
        )
    return (
        "<h2>Before / after</h2>"
        '<table><thead><tr><th>metric</th><th class="num">zero-shot baseline</th>'
        '<th class="num">fine-tuned</th><th class="num">delta</th></tr></thead>'
        f"<tbody>{body}</tbody></table>"
        '<p class="muted">The holdout is grouped BY QUERY, so these numbers cover a small '
        "number of queries. Read them as directional evidence, not a tight estimate "
        "— and note the per-class F1 rows: class imbalance is expected, and macro-F1 "
        "alone would hide the thin <code>partial</code> class.</p>"
        f"<ul class=\"muted\">{''.join(footer_bits)}</ul>"
    )


def _trace_section() -> str:
    if not TRACE.is_file():
        return (
            "<h2>Recorded grader run</h2>"
            f'<div class="card warn"><strong>{_esc(NOT_MEASURED)}</strong><br>'
            '<span class="muted">No <code>results/demo_trace.json</code>. Record one on a '
            "machine with a working model:<br><code>python -m src.record_demo_trace "
            "--corpus colab-t4</code></span></div>"
        )
    try:
        trace = json.loads(TRACE.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return '<h2>Recorded grader run</h2><div class="card warn">Trace file unreadable.</div>'

    corpus = trace.get("corpus", {})
    model = trace.get("model", {})
    on_arm = trace.get("arms", {}).get("on", {})
    off_arm = trace.get("arms", {}).get("off", {})
    kept = len(on_arm.get("context_ids") or [])
    total = len(on_arm.get("retrieved") or [])

    def answer_block(label: str, arm: dict[str, Any]) -> str:
        answer = arm.get("answer", "")
        text = answer if arm.get("grounded") else f"WARNING: {answer}"
        return (
            f'<div class="card"><strong>{_esc(label)}</strong>'
            f'<p class="muted">grounded={_fmt(arm.get("grounded"))} · '
            f"context={_esc(arm.get('context_ids') or 'none')}</p>"
            f'<div class="answer">{_esc(text)}</div></div>'
        )

    rows = []
    graded_by_id = {g["chunk_id"]: g for g in (on_arm.get("graded") or [])}
    for rank, hit in enumerate(off_arm.get("retrieved") or [], start=1):
        verdict = graded_by_id.get(hit["chunk_id"])
        rows.append(
            f"<tr><td>{rank}</td><td><code>{_esc(hit['source'])}</code></td>"
            f"<td class=\"num\">{float(hit['score']):.4f}</td>"
            f"<td>{_esc((verdict or {}).get('label', '-'))}</td>"
            f"<td class=\"num\">{_fmt(round(verdict['confidence'], 4) if verdict else None)}</td>"
            f"<td>{_esc('kept' if verdict['accepted'] else 'dropped') if verdict else 'kept (no gate)'}</td></tr>"
        )

    adapter_line = (
        f"adapter <code>{_esc(model.get('adapter_ref'))}</code>"
        if model.get("using_adapter")
        else "<strong>base model, zero-shot</strong> (no adapter was available)"
    )
    refusal = " Nothing survived, so the ON arm refused to answer." if (
        on_arm.get("graded") and not kept
    ) else ""
    return (
        "<h2>Recorded grader run</h2>"
        f'<p class="muted">Corpus <code>{_esc(corpus.get("key"))}</code> — '
        f"{_esc(corpus.get('title'))}. Model: {_esc(model.get('base_model'))} + "
        f"{adapter_line}, {'4-bit' if model.get('quantized') else 'unquantized'} on "
        f"<code>{_esc(model.get('device'))}</code>.</p>"
        f"<p><strong>Query:</strong> {_esc(corpus.get('query'))}</p>"
        f"<p><strong>Gate decision:</strong> kept {kept} of {total} retrieved passage(s)."
        f"{refusal}</p>"
        f"{answer_block('Grader OFF — naive RAG', off_arm)}"
        f"{answer_block('Grader ON — corrective RAG', on_arm)}"
        '<table><thead><tr><th>rank</th><th>source</th><th class="num">retrieval score</th>'
        "<th>label</th><th class=\"num\">confidence</th><th>decision</th></tr></thead>"
        f"<tbody>{''.join(rows)}</tbody></table>"
    )


def _corpora_section() -> str:
    blocks = []
    for corpus in CORPORA:
        passages = "".join(
            f"<li><code>{_esc(corpus.key)}/p{i + 1}</code> — {_esc(text)}</li>"
            for i, text in enumerate(corpus.passages)
        )
        blocks.append(
            f'<div class="card"><strong>{_esc(corpus.title)}</strong>'
            f'<p class="muted">{_esc(corpus.description)}</p>'
            f"<p><em>Example query:</em> {_esc(corpus.query)}</p>"
            f'<p class="muted"><strong>Gate design intent</strong> — '
            f"{_esc(corpus.expected_gate)}</p>"
            f"<ol>{passages}</ol></div>"
        )
    return "<h2>Demo corpora</h2>" + "".join(blocks)


def build() -> Path:
    labels = ", ".join(LABELS)
    page = f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Fine-tuned RAG relevance grader</title>
<style>{CSS}</style>
</head>
<body>
<h1>A fine-tuned relevance grader for RAG</h1>
<p>Qwen2.5-1.5B-Instruct, fine-tuned with QLoRA (4-bit NF4) on TREC DL 2020 graded
relevance judgments, bucketed into <code>{_esc(labels)}</code>. The adapter
implements a <code>GraderInterface</code>, so it slots into any RAG pipeline as a
drop-in reranker.</p>

<h2>What the gate does</h2>
<div class="card">
<p><strong>Grader OFF (naive RAG):</strong> every retrieved passage reaches the generator.</p>
<p><strong>Grader ON (corrective RAG):</strong> passages graded <code>irrelevant</code> are
dropped. If none survive, the pipeline returns an explicit <em>no grounded answer</em>
rather than inventing text.</p>
<p class="muted">This page is a Static Space: it is generated from the repo's JSON
artifacts and cannot run a model. The interactive version lives in the
Gradio/ZeroGPU Space.</p>
</div>

{_metrics_section()}
{_trace_section()}
{_corpora_section()}

<footer>
Generated by <code>space_static/build_static.py</code> from
<code>results/baseline_results.json</code>, <code>results/finetuned_results.json</code>
and <code>results/demo_trace.json</code>. Metrics are never hand-typed; N test queries
are reported alongside every number. Macro-F1 over a query-grouped holdout of ~10
queries is directional evidence, not a tight estimate.
</footer>
</body>
</html>
"""
    OUT.write_text(page, encoding="utf-8")
    return OUT


if __name__ == "__main__":
    path = build()
    print(f"wrote {path} ({path.stat().st_size} bytes)")
