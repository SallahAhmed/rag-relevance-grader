"""Presentation layer for the Space UI: CSS, HTML renderers, benchmark panel.

Split out of `app.py` so the execution half (CPU retrieval, the ZeroGPU worker,
logging, failure classification) stays readable on its own. Nothing in here runs
a model or touches the network — it is pure presentation over the dicts the
pipeline already returns.

Two rules this module enforces:

1. **No hardcoded score.** Every benchmark number is read at runtime from
   `results/*.json` via `src.rag_pipeline.load_metrics`, the same helper the
   Streamlit app uses. A missing file is reported as missing, never substituted
   with a number written here.
2. **`gr.HTML` is not a markdown renderer.** It ships the string to the browser
   verbatim, so `###` and `**` reach the visitor as literal characters. Every
   piece of markdown in this app goes through `gr.Markdown` instead, and this
   module emits real HTML tags — never markdown syntax — for `gr.HTML`.
"""
from __future__ import annotations

import html
from typing import Any

from src.rag_pipeline import NO_GROUNDED_ANSWER, before_after_rows, load_metrics

# Label presentation. Colour is carried by a CSS class, not inline, so the
# palette lives in one place and stays consistent with the benchmark table.
LABEL_CLASS = {
    "relevant": "verdict-keep",
    "partial": "verdict-partial",
    "irrelevant": "verdict-drop",
}
LABEL_ICON = {"relevant": "✅", "partial": "🟡", "irrelevant": "⛔"}

# Delta thresholds for the benchmark chips. The +10pp macro-F1 bar is the pinned
# project success bar (core rule 15); it is a threshold, not a measurement.
F1_BAR_PP = 0.10
RETENTION_BAR = 0.90


CSS = """
<style>
/* ---- shell ---------------------------------------------------------------- */
.gradio-container {
    max-width: 1400px !important;
    font-family: "Inter", -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
}
.app-hero {
    padding: 1.1rem 1.3rem;
    border-radius: 12px;
    background: linear-gradient(120deg, #1e1b4b 0%, #312e81 55%, #4c1d95 100%);
    color: #f5f3ff;
    margin-bottom: 0.9rem;
}
.app-hero h1 { color: #ffffff; margin: 0 0 0.35rem 0; font-size: 1.65rem; }
.app-hero p { color: #ddd6fe; margin: 0; font-size: 0.98rem; line-height: 1.55; }
.app-hero .hero-metric {
    display: inline-block;
    margin: 0.7rem 0.55rem 0 0;
    padding: 0.4rem 0.75rem;
    border-radius: 8px;
    background: rgba(255,255,255,0.12);
    border: 1px solid rgba(255,255,255,0.22);
    font-size: 0.86rem;
}
.app-hero .hero-metric b { font-size: 1.02rem; }

/* ---- gate summary (the hero result) --------------------------------------- */
.gate-summary {
    border-radius: 12px;
    border: 1px solid #e5e7eb;
    border-left: 5px solid #4c1d95;
    background: #faf5ff;
    padding: 0.85rem 1.1rem;
    margin: 0.75rem 0 0.4rem 0;
}
.gate-summary.is-refused { border-left-color: #b91c1c; background: #fef2f2; }
.gate-summary .gate-headline { font-size: 1.12rem; font-weight: 650; margin-bottom: 0.3rem; }
.gate-summary .gate-sub { color: #4b5563; font-size: 0.9rem; }

/* ---- arm cards ------------------------------------------------------------ */
.arm-card {
    height: 100%;
    border-radius: 12px;
    border: 1px solid #e5e7eb;
    background: #ffffff;
    padding: 0.95rem 1.1rem;
    margin-bottom: 0.7rem;
}
.arm-card.is-primary { border-color: #7c3aed; box-shadow: 0 0 0 3px rgba(124,58,237,0.11); }
.arm-card.is-refused { border-color: #ef4444; background: #fffbfb; }
.arm-card .arm-title {
    display: flex; align-items: center; gap: 0.45rem;
    font-size: 1.02rem; font-weight: 680; margin-bottom: 0.55rem;
    color: #111827;
}
.arm-card .arm-tag {
    font-size: 0.66rem; font-weight: 700; letter-spacing: 0.05em;
    text-transform: uppercase; padding: 0.14rem 0.44rem; border-radius: 5px;
    background: #ede9fe; color: #5b21b6;
}
.arm-card .arm-tag.tag-naive { background: #f3f4f6; color: #374151; }
.arm-card .arm-body { font-size: 0.94rem; line-height: 1.62; color: #1f2937; }
.arm-card .arm-body.is-refusal { color: #991b1b; font-weight: 600; }
.arm-card .arm-meta {
    margin-top: 0.7rem; padding-top: 0.55rem; border-top: 1px dashed #e5e7eb;
    font-size: 0.76rem; color: #6b7280; font-family: ui-monospace, monospace;
}

/* ---- passage verdicts ----------------------------------------------------- */
.passage {
    border-radius: 9px;
    border: 1px solid #e5e7eb;
    background: #ffffff;
    padding: 0.6rem 0.8rem;
    margin-bottom: 0.5rem;
}
.passage.is-kept { border-left: 4px solid #16a34a; background: #f7fdf9; }
.passage.is-dropped { border-left: 4px solid #dc2626; background: #fffafa; }
.passage.is-nogate { border-left: 4px solid #9ca3af; background: #fafafa; }
.passage .p-head {
    display: flex; justify-content: space-between; align-items: center;
    gap: 0.5rem; margin-bottom: 0.32rem;
}
.passage .p-label { font-size: 0.8rem; font-weight: 680; }
.passage .p-conf { font-size: 0.76rem; color: #6b7280; font-family: ui-monospace, monospace; }
.passage .p-text { font-size: 0.85rem; line-height: 1.55; color: #374151; }
.passage .p-src {
    font-size: 0.72rem; color: #6b7280; font-family: ui-monospace, monospace;
    margin-bottom: 0.3rem;
}
.conf-track {
    height: 5px; border-radius: 3px; background: #e5e7eb;
    overflow: hidden; margin: 0.3rem 0 0.4rem 0;
}
.conf-fill { height: 100%; border-radius: 3px; }
.conf-fill.c-keep { background: #16a34a; }
.conf-fill.c-partial { background: #eab308; }
.conf-fill.c-drop { background: #dc2626; }

/* ---- benchmark table ------------------------------------------------------ */
.bench-wrap {
    border-radius: 12px; border: 1px solid #e5e7eb; background: #ffffff;
    padding: 0.9rem 1.05rem; margin-bottom: 0.6rem;
}
.bench-wrap h4 { margin: 0 0 0.6rem 0; color: #111827; font-size: 1rem; }
.bench-table { width: 100%; border-collapse: collapse; font-size: 0.87rem; }
.bench-table th, .bench-table td {
    padding: 0.42rem 0.6rem; text-align: right; border-bottom: 1px solid #f3f4f6;
}
.bench-table th:first-child, .bench-table td:first-child { text-align: left; }
.bench-table thead th {
    color: #6b7280; font-weight: 650; font-size: 0.78rem;
    text-transform: uppercase; letter-spacing: 0.03em;
    border-bottom: 2px solid #e5e7eb;
}
.bench-table td { font-family: ui-monospace, monospace; color: #1f2937; }
.bench-table tr.is-headline { background: #faf5ff; font-weight: 680; }
.bench-table .delta-up { color: #15803d; font-weight: 650; }
.bench-table .delta-down { color: #b91c1c; font-weight: 650; }
.bench-note { font-size: 0.79rem; color: #6b7280; margin-top: 0.65rem; line-height: 1.5; }
.bench-pill {
    display: inline-block; padding: 0.2rem 0.55rem; border-radius: 999px;
    font-size: 0.73rem; font-weight: 680; margin-right: 0.35rem;
}
.bench-pill.pass { background: #dcfce7; color: #166534; }
.bench-pill.fail { background: #fee2e2; color: #991b1b; }
.bench-pill.neutral { background: #f3f4f6; color: #374151; }

.status-line {
    font-size: 0.82rem; color: #6b7280; font-family: ui-monospace, monospace;
    padding: 0.35rem 0;
}
.warn-box {
    border-radius: 9px; border: 1px solid #fcd34d; background: #fffbeb;
    color: #92400e; padding: 0.6rem 0.85rem; font-size: 0.87rem;
    margin-bottom: 0.5rem;
}
</style>
"""


def _esc(text: Any) -> str:
    """Escape model output before it goes anywhere near an HTML tag.

    Passage text and answers come from a corpus the visitor may have uploaded,
    so they are untrusted. Every interpolation below goes through this.
    """
    return html.escape(str(text if text is not None else ""))


def _conf_bar(confidence: float | None, label: str) -> str:
    """Confidence meter. `confidence` is the softmax over the 3 label tokens."""
    if confidence is None:
        return ""
    pct = max(0.0, min(1.0, float(confidence))) * 100.0
    cls = LABEL_CLASS.get(label, "verdict-drop").replace("verdict-", "c-")
    return (
        f'<div class="conf-track"><div class="conf-fill {cls}" '
        f'style="width:{pct:.1f}%"></div></div>'
    )


def _verdict_passages(arm: dict[str, Any]) -> str:
    """Per-passage cards with label, confidence meter and kept/dropped state."""
    graded = {g["chunk_id"]: g for g in (arm.get("graded") or [])}
    blocks: list[str] = []
    for rank, hit in enumerate(arm.get("retrieved") or [], start=1):
        verdict = graded.get(hit["chunk_id"])
        source = _esc(hit.get("source", ""))
        text = _esc(hit.get("text", ""))
        if verdict is None:
            state, css = "kept (no gate)", "is-nogate"
            label_html = (
                '<span class="p-label" style="color:#6b7280">'
                f'retrieval {_esc(round(float(hit.get("score", 0.0)), 4))}</span>'
            )
            bar = ""
        else:
            label = verdict.get("label", "")
            state = "KEPT" if verdict.get("accepted") else "DROPPED"
            css = "is-kept" if verdict.get("accepted") else "is-dropped"
            colour = {"relevant": "#166534", "partial": "#a16207"}.get(label, "#991b1b")
            icon = LABEL_ICON.get(label, "")
            label_html = (
                f'<span class="p-label" style="color:{colour}">{icon} '
                f'{_esc(label)}</span>'
            )
            bar = _conf_bar(verdict.get("confidence"), label)
        blocks.append(
            f'<div class="passage {css}">'
            f'<div class="p-head">{label_html}'
            f'<span class="p-conf">{_esc(state)}</span></div>'
            f'<div class="p-src">#{rank} · {source} · '
            f'retrieval {_esc(round(float(hit.get("score", 0.0)), 4))}</div>'
            f"{bar}"
            f'<div class="p-text">{text}</div>'
            f"</div>"
        )
    if not blocks:
        return '<div class="warn-box">No passages were retrieved for this query.</div>'
    return "".join(blocks)


def _arm_card(arm: dict[str, Any], title: str, primary: bool, naive: bool) -> str:
    """One arm's answer as real HTML.

    `gr.HTML` does not parse markdown, so this emits `<h4>`/`<strong>` tags
    directly. The previous version returned `###` and `**`, which reached the
    browser as literal characters.
    """
    grounded = bool(arm.get("grounded"))
    if grounded:
        body = f'<div class="arm-body">{_esc(arm.get("answer", ""))}</div>'
    else:
        answer = arm.get("answer", NO_GROUNDED_ANSWER)
        body = (
            '<div class="arm-body is-refusal">⚠️ '
            f"{_esc(answer)}</div>"
        )
    tag_cls = "tag-naive" if naive else ""
    tag_text = "naive RAG" if naive else "corrective RAG"
    kept = len(arm.get("context_ids") or [])
    total = len(arm.get("retrieved") or [])
    meta = (
        f"used_grader={_esc(arm.get('used_grader'))} · grounded={_esc(grounded)} · "
        f"context={_esc(arm.get('context_ids') or 'none')} · {kept}/{total} passages"
    )
    note = arm.get("note")
    note_html = f'<div class="arm-meta">note: {_esc(note)}</div>' if note else ""
    return (
        f'<div class="arm-card{" is-primary" if primary else ""}'
        f'{" is-refused" if not grounded else ""}">'
        f'<div class="arm-title">{_esc(title)}'
        f'<span class="arm-tag {tag_cls}">{tag_text}</span></div>'
        f"{body}{note_html}"
        f'<div class="arm-meta">{meta}</div>'
        f"</div>"
    )


def _gate_summary(result: dict[str, Any]) -> str:
    """The single most important line in the app: what the gate decided."""
    on_arm = result.get("on") or {}
    graded = on_arm.get("graded") or []
    kept = len(on_arm.get("context_ids") or [])
    total = len(on_arm.get("retrieved") or [])
    refused = bool(graded) and kept == 0

    if not graded:
        inner = (
            "<b>Grading did not run.</b> The OFF arm trace below shows what the "
            "retriever returned, ungraded."
        )
        return f'<div class="gate-summary">{inner}</div>'

    labels: dict[str, int] = {}
    for g in graded:
        labels[g.get("label", "?")] = labels.get(g.get("label", "?"), 0) + 1
    breakdown = " · ".join(
        f"{LABEL_ICON.get(lab, '')} {lab} ×{count}" for lab, count in sorted(labels.items())
    )

    if refused:
        headline = "🚫 Gate rejected every passage — the ON arm refused to answer."
        sub = (
            "The retriever returned text that matched the query, but none of it "
            "answered it. A naive pipeline would have generated an answer from "
            "that text anyway; this one declined."
        )
    else:
        dropped = total - kept
        headline = f"🚦 Gate kept {kept} of {total} retrieved passage(s), dropped {dropped}."
        sub = (
            f"{breakdown}. Only the kept passages reached the generator — compare "
            "that answer with the naive arm above."
        )
    css = "gate-summary is-refused" if refused else "gate-summary"
    return (
        f'<div class="{css}">'
        f'<div class="gate-headline">{headline}</div>'
        f'<div class="gate-sub">{_esc(sub)}</div>'
        f"</div>"
    )


def _model_banner(result: dict[str, Any]) -> str:
    """Provenance line: which weights actually produced this run."""
    model = result.get("model") or {}
    desc = _esc(model.get("description", "unknown model"))
    if not model.get("using_adapter"):
        desc += ' — <span style="color:#92400e">base model zero-shot, no adapter</span>'
    error = model.get("load_error")
    err_html = (
        f'<div class="warn-box">⚠️ {_esc(error)}</div>' if error else ""
    )
    return f"{err_html}<div>{desc}</div>"


def _confusion_table(ft: dict[str, Any]) -> str:
    """Confusion matrix, oriented so rows are true and columns predicted."""
    block = ft.get("confusion_matrix") or {}
    labels = block.get("labels") or []
    matrix = block.get("matrix") or []
    if not labels or not matrix:
        return ""
    head = "".join(f"<th>pred {LABEL_ICON.get(l, '')} {html.escape(l)}</th>" for l in labels)
    body = []
    for i, lab in enumerate(labels):
        cells = "".join(f"<td>{int(v)}</td>" for v in matrix[i])
        body.append(f'<tr><td>true {html.escape(lab)}</td>{cells}</tr>')
    return (
        '<table class="bench-table"><thead><tr><th></th>'
        f"{head}</tr></thead><tbody>{''.join(body)}</tbody></table>"
    )


def benchmark_html(results_dir: str | None = None) -> str:
    """Headline benchmark panel, read from results/*.json at runtime.

    Core rules 4 and 11: the before/after table is the primary artifact, and no
    metric is ever hardcoded in the UI. When the JSONs are absent (the deployed
    Space ships no `results/`) this says so and shows nothing numeric.

    `results_dir` is passed by the caller rather than left to `load_metrics()`'s
    own probe, which requires BOTH `src/` and `config/` next to each other. That
    is true in this repo and not guaranteed in a Space, where a missing `config/`
    would silently blank the panel rather than raise. The app resolves the folder
    from `src/rag_pipeline.py`, which is the file it already requires.
    """
    metrics = load_metrics(results_dir)
    rows = before_after_rows(metrics)
    if not rows:
        missing = ", ".join(metrics.get("missing") or []) or "unknown"
        return (
            '<div class="bench-wrap"><h4>Measured before / after</h4>'
            '<div class="warn-box">No benchmark JSON found next to this app '
            f"(missing: {_esc(missing)}). These files are produced by notebooks "
            "02 and 04 — no metric is hardcoded in this UI, so there is nothing "
            "to show until they are deployed alongside the app.</div></div>"
        )

    base = metrics["baseline"]
    ft = metrics["finetuned"]

    body: list[str] = []
    for row in rows:
        delta = row["delta"]
        cls = "delta-up" if delta > 0 else ("delta-down" if delta < 0 else "")
        sign = f"{delta:+.4f}"
        hl = ' class="is-headline"' if row["metric"] == "macro-F1" else ""
        body.append(
            f"<tr{hl}><td>{_esc(row['metric'])}</td>"
            f"<td>{row['baseline']:.4f}</td>"
            f"<td>{row['finetuned']:.4f}</td>"
            f'<td class="{cls}">{sign}</td></tr>'
        )

    # Verdicts from the stored numbers, never re-derived from prose.
    pills: list[str] = []
    d_f1 = (ft.get("macro_f1") or 0.0) - (base.get("macro_f1") or 0.0)
    ok = d_f1 >= F1_BAR_PP
    pills.append(
        f'<span class="bench-pill {"pass" if ok else "fail"}">'
        f'macro-F1 Δ {d_f1:+.3f} vs bar {F1_BAR_PP:+.2f} — '
        f'{"PASS" if ok else "FAIL"}</span>'
    )
    forget = ft.get("forgetting") or {}
    if forget.get("retention") is not None:
        keep = bool(forget.get("passes"))
        pills.append(
            f'<span class="bench-pill {"pass" if keep else "fail"}">'
            f'MMLU retention {forget["retention"]:.3f} vs bar '
            f'{RETENTION_BAR:.2f} — {"PASS" if keep else "FAIL"}</span>'
        )

    confusion = _confusion_table(ft)
    confusion_html = (
        f'<div style="margin-top:0.9rem"><b>Fine-tuned confusion matrix</b>'
        f"{confusion}</div>"
        if confusion
        else ""
    )
    n_q = _esc(base.get("n_test_queries", "?"))
    n_r = _esc(ft.get("n_test_qrels", base.get("n_test_qrels", "?")))
    note = (
        f"Held out by whole query: {n_q} test queries / {n_r} graded pairs. "
        "Macro-F1 over that many queries is directional evidence, not a tight "
        "estimate. Read straight from results/baseline_results.json and "
        "results/finetuned_results.json."
    )
    return (
        '<div class="bench-wrap"><h4>Measured before / after '
        "(QLoRA fine-tune vs zero-shot base)</h4>"
        f'<div>{"".join(pills)}</div>'
        '<table class="bench-table" style="margin-top:0.75rem">'
        "<thead><tr><th>Metric</th><th>Zero-shot</th><th>Fine-tuned</th>"
        f"<th>Δ</th></tr></thead><tbody>{''.join(body)}</tbody></table>"
        f"{confusion_html}"
        f'<div class="bench-note">{note}</div>'
        f"</div>"
    )


def _hero_html() -> str:
    """Top banner. Static framing only — every number lives in the panel below."""
    return (
        '<div class="app-hero">'
        "<h1>Corrective RAG with a fine-tuned relevance grader</h1>"
        "<p>Retrieval returns passages that <b>look</b> relevant and answer "
        "nothing. A relevance grader &mdash; a QLoRA fine-tune of "
        "Qwen2.5-1.5B-Instruct &mdash; decides which retrieved passages may reach "
        "the generator. Both arms always run, so the difference is visible rather "
        "than asserted.</p>"
        '<div class="hero-metric">Model <b>Qwen2.5-1.5B</b> + QLoRA adapter</div>'
        '<div class="hero-metric">Gating <b>argmax over 3 label tokens</b></div>'
        '<div class="hero-metric">Retrieval <b>FAISS + MiniLM, on CPU</b></div>'
        "</div>"
    )