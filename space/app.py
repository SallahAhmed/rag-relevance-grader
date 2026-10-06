"""Corrective RAG with the fine-tuned relevance grader in the loop.

The demo is a toggle: with the grader OFF the model answers from every passage
the retriever returned (naive RAG); with it ON the grader decides which passages
are allowed to reach the generator, and if none survive the pipeline says so
instead of inventing an answer. Both arms are always executed so the difference
is visible side by side â€” the comparison is the artifact, not the answer.

Runs entirely on local open models: sentence-transformers for retrieval, and the
local Qwen model plus its QLoRA adapter (when that adapter exists) for grading
and answering. No hosted API, no key.
"""
from __future__ import annotations

import sys
from pathlib import Path

import streamlit as st

HERE = Path(__file__).resolve().parent
for _candidate in (HERE, *HERE.parents):
    if (_candidate / "src" / "rag_pipeline.py").is_file():
        if str(_candidate) not in sys.path:
            sys.path.insert(0, str(_candidate))
        break
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

IMPORT_ERROR: str | None = None
try:
    from src.sample_corpora import CORPORA, corpus_title, get_corpus

    from src.rag_pipeline import (
        DEFAULT_ACCEPT_LABELS,
        Chunk,
        HFRelevanceGrader,
        QwenGenerator,
        RagIndex,
        answer_with_gate,
        before_after_rows,
        build_index,
        chunk_documents,
        load_local_model,
        load_metrics,
        retrieve,
    )
except Exception as exc:  # noqa: BLE001 â€” a broken import must not kill the page
    IMPORT_ERROR = f"{type(exc).__name__}: {exc}"

LABEL_ICON = {"relevant": "âœ…", "partial": "ðŸŸ¡", "irrelevant": "â›”"}


@st.cache_resource(show_spinner=False)
def load_bundle():
    """One model load per session â€” grader and generator share it."""
    return load_local_model()


@st.cache_resource(show_spinner=False)
def get_index(chunks: tuple[Chunk, ...]) -> RagIndex:
    """Cached per corpus; embedding is the slow part of any rerun.

    Takes already-chunked input: the sample corpora are authored as discrete
    passages and re-splitting them would blur the retrieval trace.
    """
    return build_index(list(chunks))


def corpus_chunks(corpus) -> list[Chunk]:
    """Sample corpora arrive pre-chunked: each passage is one retrievable unit."""
    return [
        Chunk(chunk_id=i, text=text, source=f"{corpus.key}/p{i + 1}", ordinal=i)
        for i, text in enumerate(corpus.passages)
    ]


def render_trace(passages, scores, graded=None) -> None:
    """Per-passage table: retrieval score, label, confidence, accept/reject."""
    graded_by_id = {g.chunk.chunk_id: g for g in (graded or [])}
    rows = []
    for rank, (chunk, score) in enumerate(zip(passages, scores), start=1):
        verdict = graded_by_id.get(chunk.chunk_id)
        label = verdict.label if verdict else "-"
        rows.append(
            {
                "rank": rank,
                "source": chunk.source,
                "retrieval score": round(float(score), 4),
                "label": f"{LABEL_ICON.get(label, '')} {label}".strip(),
                "confidence": round(verdict.confidence, 4) if verdict else None,
                "decision": ("kept" if verdict.accepted else "dropped")
                if verdict
                else "kept (no gate)",
            }
        )
        st.dataframe(rows, hide_index=True)


def render_passage_texts(graded) -> None:
    for item in graded:
        icon = LABEL_ICON.get(item.label, "")
        st.markdown(
            f"**{item.rank}. {icon} {item.label} Â· confidence {item.confidence:.3f} Â· "
            f"{'KEPT' if item.accepted else 'DROPPED'}** â€” `{item.chunk.source}`"
        )
        st.text(item.chunk.text)


def render_answer(result, title, primary, passages, scores) -> None:
    st.subheader(f"{'â–¶ ' if primary else ''}{title}")
    if result.grounded:
        st.write(result.answer)
    else:
        st.warning(result.answer)
    if result.note:
        st.caption(result.note)
    st.caption(
        f"used_grader={result.used_grader} Â· grounded={result.grounded} Â· "
        f"context passages: {result.context_ids or 'none'}"
    )
    with st.expander("Retrieval trace for this arm"):
        render_trace(passages, scores, result.graded)
        if result.graded:
            render_passage_texts(result.graded)


def render_metrics_panel() -> None:
    """Every number shown here is read from results/*.json at runtime."""
    metrics = load_metrics()
    rows = before_after_rows(metrics)
    with st.expander("Measured before/after (read from results/*.json at runtime)"):
        if not rows:
            st.info(
                "No before/after JSON found next to this app (missing: "
                f"{', '.join(metrics['missing']) or 'nothing'}). Notebooks 02 and 04 "
                "write those files â€” no metric is hardcoded in this app."
            )
            return
        st.dataframe(rows, hide_index=True)
        for key in ("baseline", "finetuned"):
            payload = metrics.get(key) or {}
            st.caption(
                f"{key}: n={payload.get('n')} Â· n_test_queries="
                f"{payload.get('n_test_queries')} Â· n_test_qrels="
                f"{payload.get('n_test_qrels')} Â· model={payload.get('model')} Â· "
                f"mode={payload.get('mode')}"
            )
        st.caption(
            "Query-grouped holdout, so macro-F1 over that many queries is "
            "directional evidence rather than a tight estimate."
        )


def device_label() -> str:
    try:
        import torch

        if torch.cuda.is_available():
            return f"CUDA: {torch.cuda.get_device_name(0)}"
    except Exception:  # noqa: BLE001 â€” torch may not be importable yet
        pass
    return "CPU only â€” expect minutes per answer"


def main() -> None:
    st.set_page_config(page_title="Corrective RAG Â· relevance grader", layout="wide")
    st.title("Corrective RAG with a fine-tuned relevance grader")
    st.markdown(
        "Retrieval alone returns passages that look similar but answer nothing. "
        "The gate below is a relevance grader â€” a QLoRA fine-tune of "
        "Qwen2.5-1.5B-Instruct â€” deciding which retrieved passages may reach the "
        "generator. Everything runs on local open models."
    )

    if IMPORT_ERROR:
        st.error(f"src/rag_pipeline.py could not be imported â€” {IMPORT_ERROR}")
        st.info(
            "This app needs the repository's `src/` folder next to `space/`. Create "
            "the Space from a repo containing both, or copy `src/` beside `app.py`."
        )
        return

    with st.sidebar:
        st.header("Runtime")
        st.caption(device_label())
        st.caption(f"accepted labels: {', '.join(DEFAULT_ACCEPT_LABELS)}")
        top_k = st.slider("passages to retrieve", 2, 8, 4)
        max_new_tokens = st.slider("max answer tokens", 60, 400, 180, step=20)

    source = st.radio("Corpus", ["Sample corpus", "My documents"], horizontal=True)

    chunks: list[Chunk] = []
    query = ""
    if source == "Sample corpus":
        keys = [c.key for c in CORPORA]
        selected = st.selectbox(
            "Pre-built corpus", keys, format_func=corpus_title
        )
        corpus = get_corpus(selected)
        st.caption(corpus.description)
        st.caption(f"Gate design intent â€” {corpus.expected_gate}")
        query = st.text_input("Query", value=corpus.query)
        chunks = corpus_chunks(corpus)
    else:
        pasted = st.text_area("Paste documents", height=180)
        uploads = st.file_uploader(
            "or upload .txt / .md files", type=["txt", "md"], accept_multiple_files=True
        )
        uploaded = [
            (f.name, f.getvalue().decode("utf-8", errors="replace"))
            for f in uploads or []
        ]
        documents = ([("pasted", pasted)] if pasted.strip() else []) + uploaded
        query = st.text_input("Query", value="")
        if not documents:
            st.info("Paste text or upload a file, then run a query.")
            render_metrics_panel()
            return
        chunk_size = st.slider("chunk size (words)", 60, 400, 220, step=20)
        overlap = st.slider("chunk overlap (words)", 0, 100, 40, step=10)
        chunks = chunk_documents(
            [text for _, text in documents],
            chunk_size=chunk_size,
            overlap=overlap,
            sources=[name for name, _ in documents],
        )

    if not chunks:
        st.warning("Nothing to index: the documents produced no chunks.")
        render_metrics_panel()
        return

    use_grader = st.toggle(
        "Relevance grader ON (corrective gate)",
        value=True,
        help="ON: passages the grader calls irrelevant never reach the generator. "
        "OFF: every retrieved passage is used. Both arms always run so the "
        "comparison is visible; the toggle selects which one the app reports.",
    )

    if not query.strip():
        st.info("Enter a query to run the comparison.")
        render_metrics_panel()
        return

    if st.button("Run query", type="primary"):
        try:
            with st.spinner("Retrieving..."):
                index = get_index(tuple(chunks))
                hits = retrieve(index, query, top_k=top_k)
        except Exception as exc:  # noqa: BLE001 â€” a missing embedding model must not kill the page
            st.error(f"Retrieval failed â€” {type(exc).__name__}: {exc}")
            st.info(
                "The embedding model is downloaded on first use. If this Space is "
                "offline or out of memory, grading and answering are unavailable too."
            )
            render_metrics_panel()
            return
        if not hits:
            st.warning("Nothing retrieved for that query.")
            render_metrics_panel()
            return
        passages = [chunk for chunk, _ in hits]
        scores = [score for _, score in hits]
        st.caption(
            f"indexed {len(chunks)} chunk(s) from {len({c.source for c in chunks})} "
            f"source(s) with {index.embedding_model}"
        )

        try:
            with st.spinner("Loading the local model (first run only)..."):
                bundle = load_bundle()
        except Exception as exc:  # noqa: BLE001 â€” show the error, keep the page alive
            st.error(f"Model failed to load â€” {type(exc).__name__}: {exc}")
            st.info("Retrieval still works; grading and answering are unavailable.")
            render_trace(passages, scores)
            render_metrics_panel()
            return

        if getattr(bundle, "using_adapter", False):
            st.success(f"Grader weights â€” {bundle.describe()}")
        else:
            st.warning(f"Grader weights â€” {bundle.describe()}")
            st.caption(
                "Running the base model zero-shot. Publish the QLoRA adapter to the "
                "Hub (or build it into outputs/qlora-sft) and reload to compare."
            )
        if getattr(bundle, "load_error", None):
            st.warning(str(bundle.load_error))

        grader = HFRelevanceGrader(bundle=bundle)
        generator = QwenGenerator(bundle=bundle, max_new_tokens=max_new_tokens)
        with st.spinner("Running both arms (OFF then ON)..."):
            off_result = answer_with_gate(query, hits, False, generator=generator)
            on_result = answer_with_gate(
                query, hits, True, generator=generator, grader=grader
            )

        left, right = st.columns(2)
        with left:
            render_answer(off_result, "Grader OFF â€” naive RAG", not use_grader, passages, scores)
        with right:
            render_answer(on_result, "Grader ON â€” corrective RAG", use_grader, passages, scores)

        kept = [g for g in on_result.graded if g.accepted]
        dropped = [g for g in on_result.graded if not g.accepted]
        summary = (
            f"**Gate decision:** kept {len(kept)} of {len(on_result.graded)} retrieved "
            f"passage(s), dropped {len(dropped)}."
        )
        if on_result.graded and not kept:
            summary += " Nothing survived, so the ON arm refused to answer."
        st.markdown(summary)

        with st.expander("Show retrieval + grading trace"):
            st.markdown("**ON arm â€” per-passage verdict**")
            if on_result.graded:
                render_trace(passages, scores, on_result.graded)
                render_passage_texts(on_result.graded)
            else:
                st.info("No grading trace: the grader did not run.")
            st.markdown("**OFF arm â€” every retrieved passage reached the model**")
            render_trace(passages, scores)

    render_metrics_panel()


if __name__ == "__main__":
    main()

