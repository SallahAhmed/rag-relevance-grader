"""ZeroGPU Gradio Space: Corrective RAG with the relevance grader in the loop.

Free-GPU hosting design (see ../space_gradio/README.md for the constraints):

  CPU process (always warm)          GPU worker (forked per click)
  -------------------------          --------------------------------
  chunk -> embed -> FAISS retrieve   load Qwen (+ adapter, 4-bit)
                                     grade every retrieved passage
                                     answer OFF arm, answer ON arm

Retrieval deliberately stays on the CPU. MiniLM is milliseconds per passage, and
keeping it out of the `@spaces.GPU` window is what lets us declare a small,
honest `duration` — a 60-second default reservation would burn a visitor's whole
5-minute daily quota on one click.

When the GPU is unavailable (quota exhausted, queue timeout, no adapter) the app
falls back to `results/demo_trace.json`, a recording made by
`src/record_demo_trace.py` from a real model run. It never invents verdicts: with
no recording and no GPU, the app says exactly that.

The container log is the only diagnostic surface a Space has, so it is part of the
design, not an afterthought. Three lines carry the load: `startup: build=...`
identifies the exact deployed file by content hash (a Space serves whatever was
last uploaded, which is often not what the repo says), `gpu worker: device=...`
records what the worker was actually given — unanswerable from the main process,
where `torch.cuda` is hijacked to lie — and `gpu step failed: kind=...` names
the bucket so the log and the visitor notice cannot disagree about whose fault it
is. Everything the platform logs around us is triaged in the README.
"""
from __future__ import annotations

import os

# ZeroGPU rule #1: `spaces` monkey-patches torch.cuda at import time, so it has to
# be imported before anything that pulls in torch. The import is deliberately
# unguarded — HF's image installs `spaces` on every hardware tier, and
# spaces.GPU returns the function unchanged when the platform did not set
# SPACES_ZERO_GPU (spaces/zero/decorator.py). The previous `try: import spaces
# except ImportError` gate treated "the import worked" as "this is a ZeroGPU
# host", which is wrong and wired ZeroGPU scheduling onto Colab and local runs.
import spaces  # isort: skip

# The container log is the only diagnostic surface a Space has. Per-file tqdm bars
# from a 3.1 GB download buried every app line in the previous run, so they are
# off and progress is recorded as start/finish log lines instead. Must be set
# before huggingface_hub is imported; the repo imports it lazily, so here is safe.
os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")

import json
import logging
import sys
import time
from functools import lru_cache
from pathlib import Path
from typing import Any

import gradio as gr

# --------------------------------------------------------------------------- #
# Logging
#
# The container log used to carry nothing but Gradio's startup banner: every
# diagnostic in this app is turned into UI markdown, so a crashed click left no
# trace server-side and the Space was unmonitorable. Everything below writes to
# stdout in a fixed, greppable shape.
#
# UTC timestamps: a cron monitor on another machine reads these, and the Space's
# local time is not the reader's.
# --------------------------------------------------------------------------- #
LOG_FORMAT = "%(asctime)s %(levelname)-7s %(name)-18s %(message)s"
LOG_DATEFMT = "%Y-%m-%dT%H:%M:%SZ"


def _setup_logging() -> logging.Logger:
    """Attach one stdout handler to this app's logger and the pipeline's.

    `propagate = False` on both keeps the Space's uvicorn/gradio logging intact —
    reconfiguring the root logger would silence the platform's own records. The
    "src" logger is named here because `src.rag_pipeline` logs under that parent,
    so the GPU worker's records land in the same stream as the app's.
    """
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(logging.Formatter(LOG_FORMAT, datefmt=LOG_DATEFMT))
    handler.formatter.converter = time.gmtime
    handler.setLevel(logging.INFO)

    for name in ("grader", "src"):
        target = logging.getLogger(name)
        target.setLevel(logging.INFO)
        if not any(getattr(h, "_grader_app", False) for h in target.handlers):
            handler._grader_app = True  # type: ignore[attr-defined]
            target.addHandler(handler)
        target.propagate = False
    return logging.getLogger("grader")


LOG = _setup_logging()

for _candidate in (Path(__file__).resolve().parent, *Path(__file__).resolve().parents):
    if (_candidate / "src" / "rag_pipeline.py").is_file():
        if str(_candidate) not in sys.path:
            sys.path.insert(0, str(_candidate))
        break
from src.rag_pipeline import (  # noqa: E402
    NO_GROUNDED_ANSWER,
    Chunk,
    HFRelevanceGrader,
    QwenGenerator,
    SentenceTransformerEmbedder,
    answer_with_gate,
    build_index,
    load_local_model,
    retrieve,
    zerogpu_active,
)
from src.sample_corpora import CORPORA, corpus_title, get_corpus  # noqa: E402

LABEL_ICON = {"relevant": "✅", "partial": "🟡", "irrelevant": "⛔"}


def _find_repo_root() -> Path:
    """Locate the folder holding `src/`, in either layout.

    In this repo the app lives at `space_gradio/app.py` (root is one level up);
    once deployed the same file sits at the Space repo root. Walking upwards for
    `src/rag_pipeline.py` covers both without a flag.
    """
    here = Path(__file__).resolve().parent
    for candidate in (here, *here.parents):
        if (candidate / "src" / "rag_pipeline.py").is_file():
            return candidate
    return here


REPO = _find_repo_root()
RECORDED_TRACE = REPO / "results" / "demo_trace.json"

# True only when the platform is actually scheduling ZeroGPU workers for this
# Space. `import spaces` succeeding proves nothing — the package ships on CPU,
# T4 and A10G tiers too, and `spaces.GPU` is a no-op unless this is set — so
# gating on the import was the single bug that gave the off-Platform paths
# (Colab, local) ZeroGPU machinery that cannot work there. `zerogpu_active()`
# lives in src/ because `load_local_model` needs the same answer for device_map.
ZEROGPU = zerogpu_active()


# --------------------------------------------------------------------------- #
# CPU side: retrieval, cached across reruns
# --------------------------------------------------------------------------- #
@lru_cache(maxsize=8)
def _cached_index(corpus_key: str):
    corpus = get_corpus(corpus_key)
    chunks = [
        Chunk(chunk_id=i, text=text, source=f"{corpus.key}/p{i + 1}", ordinal=i)
        for i, text in enumerate(corpus.passages)
    ]
    # device="cpu" is mandatory on ZeroGPU: sentence-transformers auto-detects CUDA
    # and would initialise it in the main process, which the platform rejects with
    # "Low-level CUDA init (torch._C._cuda_init) reached". Only the @spaces.GPU
    # function is allowed to touch CUDA.
    started = time.perf_counter()
    index = build_index(chunks, embedder=SentenceTransformerEmbedder(device="cpu"))
    LOG.info(
        "index ready: corpus=%s chunks=%d in %.2fs",
        corpus_key,
        len(chunks),
        time.perf_counter() - started,
    )
    return index


def retrieve_for(corpus_key: str, query: str, top_k: int) -> list[dict[str, Any]]:
    """Embed + search on CPU, then hand plain dicts to the GPU worker."""
    started = time.perf_counter()
    index = _cached_index(corpus_key)
    hits = retrieve(index, query, top_k=top_k)
    rows = [
        {"chunk_id": c.chunk_id, "source": c.source, "text": c.text, "score": float(s)}
        for c, s in hits
    ]
    LOG.info(
        "retrieval: corpus=%s top_k=%d hits=%d best=%.4f in %.2fs",
        corpus_key,
        top_k,
        len(rows),
        rows[0]["score"] if rows else 0.0,
        time.perf_counter() - started,
    )
    return rows


@lru_cache(maxsize=1)
def prefetch_weights() -> str:
    """Download the model weights in the CPU process, before any GPU window.

    Without this, the 1.5B download happens INSIDE the @spaces.GPU call and eats
    the visitor's whole daily quota on the first click. The GPU worker shares
    this container's HF cache, so pre-warming here makes the GPU window pure
    compute. Returns a short status string; failures are non-fatal.
    """
    started = time.perf_counter()
    repo = _base_model_id()
    LOG.info("prefetch weights: start repo=%s (progress bars suppressed)", repo)
    try:
        from huggingface_hub import snapshot_download

        files = snapshot_download(repo)
        LOG.info(
            "prefetch weights: done %d files from %s in %.1fs",
            len(files),
            repo,
            time.perf_counter() - started,
        )
        return f"weights ready ({len(files)} files)"
    except Exception as exc:  # noqa: BLE001 — a cold start is still better than none
        LOG.warning(
            "prefetch weights: FAILED after %.1fs (%s: %s); the GPU window will download instead",
            time.perf_counter() - started,
            type(exc).__name__,
            exc,
            exc_info=True,
        )
        return f"weights not prefetched ({type(exc).__name__})"


def _base_model_id() -> str:
    try:
        from src.config import load_config

        return load_config().model.base_model
    except Exception:  # noqa: BLE001 — config is optional in the Space
        return "Qwen/Qwen2.5-1.5B-Instruct"


@lru_cache(maxsize=1)
def load_recorded_trace() -> dict[str, Any] | None:
    """The offline recording, or None. Absence is reported, never faked."""
    if not RECORDED_TRACE.is_file():
        return None
    try:
        return json.loads(RECORDED_TRACE.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return None


def _build_id() -> str:
    """Fingerprint of the file that is actually running.

    A Space serves whatever `app.py` was last uploaded, which is routinely not
    what the repo says. A content hash in the startup line settles "is the Space
    running my latest change?" without anyone having to remember to bump a
    version string — the previous log had no way to answer that at all, which is
    why a stale build went unnoticed.
    """
    import hashlib

    try:
        return hashlib.sha256(Path(__file__).resolve().read_bytes()).hexdigest()[:12]
    except OSError as exc:  # pragma: no cover - unreadable file is not a reason to stay quiet
        return f"unreadable({type(exc).__name__})"


def _log_startup() -> None:
    """One record per process start — the only proof the app booted at all.

    `recorded_trace=MISSING` matters: the deployed Space ships no `results/`
    directory, so the replay fallback is dead weight there and a GPU-less
    visitor gets the failure notice instead of a recorded run.

    `zerogpu=` is `SPACES_ZERO_GPU`, not "did `import spaces` work". The old
    version logged the import result, which read `True` on Colab and local too
    and sent every reader looking for a GPU that was never scheduled.
    """
    LOG.info(
        "startup: build=%s zerogpu=%s device_api=%s python=%s",
        _build_id(),
        ZEROGPU,
        bool(os.environ.get("SPACES_ZERO_DEVICE_API_URL")),
        sys.version.split()[0],
    )
    LOG.info(
        "startup: repo_root=%s recorded_trace=%s gpu_path=%s",
        REPO,
        "present" if RECORDED_TRACE.is_file() else "MISSING (no replay available)",
        "spaces.GPU worker" if ZEROGPU else "in-process",
    )
    LOG.info("startup: base_model=%s corpora=%d", _base_model_id(), len(CORPORA))
    try:
        import torch

        LOG.info("startup: torch=%s cuda_devices=%s", torch.__version__, os.environ.get("CUDA_VISIBLE_DEVICES", "unset"))
    except ImportError as exc:  # pragma: no cover - torch is a hard dep of the GPU path
        LOG.warning("startup: torch not importable in the main process (%s)", exc)


_log_startup()


# --------------------------------------------------------------------------- #
# GPU side: everything that needs the model
# --------------------------------------------------------------------------- #
def _estimate_duration(corpus_key: str, query: str, hits: list, max_new_tokens: int) -> int:
    """Reserve only what the work needs — shorter reservations rank higher.

    ZeroGPU calls this with the SAME arguments as the decorated function, so the
    signature must mirror `_run_arms_on_gpu` (getting this wrong fails at call
    time, not import time).

    Rough shape: ~12 s to load a 1.5B model in the worker, ~0.5 s per graded
    passage, ~0.04 s per generated token across both arms.
    """
    passages = len(hits) if isinstance(hits, (list, tuple)) else 1
    tokens = int(max_new_tokens) if max_new_tokens else 150
    # Clamped because ZeroGPU compares the RESERVATION against the visitor's
    # remaining quota, not the work's real runtime: an unauthenticated visitor
    # has a 2-minute daily pool, so an oversized reservation reports "quota
    # exceeded" for work that would have finished in a fraction of it.
    return min(120, int(12 + 0.5 * passages + 0.08 * tokens) + 8)


def _log_worker_device() -> None:
    """Record what the worker actually got. Outside @spaces.GPU this is a lie.

    In the main process `torch.cuda` is hijacked to report a device that is not
    there, so `torch.cuda.is_available()` is meaningless there. Inside the worker
    the patch is removed and these are real values — and "which GPU did the
    visitor actually get" was the single biggest gap in the old logs.
    """
    try:
        import torch

        available = torch.cuda.is_available()
        if not available:
            LOG.warning("gpu worker: torch.cuda.is_available() is False inside the worker")
            return
        LOG.info(
            "gpu worker: device=%s count=%s visible=%r",
            torch.cuda.get_device_name(0),
            torch.cuda.device_count(),
            os.environ.get("CUDA_VISIBLE_DEVICES", "unset"),
        )
    except Exception as exc:  # noqa: BLE001 — a probe failure must not mask the real work
        LOG.warning("gpu worker: device probe failed (%s: %s)", type(exc).__name__, exc)


def _run_arms_on_gpu(
    corpus_key: str, query: str, hits: list[dict[str, Any]], max_new_tokens: int
) -> dict[str, Any]:
    """Load the model inside the worker and run both arms. Returns plain dicts."""
    started = time.perf_counter()
    LOG.info(
        "gpu worker: start corpus=%s passages=%d max_new_tokens=%d zerogpu=%s",
        corpus_key,
        len(hits),
        max_new_tokens,
        ZEROGPU,
    )
    _log_worker_device()
    bundle = load_local_model()
    grader = HFRelevanceGrader(bundle=bundle)
    generator = QwenGenerator(bundle=bundle, max_new_tokens=max_new_tokens)
    pairs = [
        (Chunk(chunk_id=h["chunk_id"], text=h["text"], source=h["source"]), h["score"])
        for h in hits
    ]
    off_result = answer_with_gate(query, pairs, False, generator=generator)
    on_result = answer_with_gate(query, pairs, True, generator=generator, grader=grader)
    LOG.info(
        "gpu worker: done in %.1fs | off grounded=%s | on grounded=%s kept=%d/%d",
        time.perf_counter() - started,
        off_result.grounded,
        on_result.grounded,
        len(on_result.context_ids),
        len(on_result.retrieved),
    )
    return {
        "mode": "live",
        "model": {
            "description": bundle.describe(),
            "using_adapter": bundle.using_adapter,
            "load_error": bundle.load_error,
        },
        "off": off_result.to_dict(),
        "on": on_result.to_dict(),
    }


run_arms = (
    spaces.GPU(duration=_estimate_duration)(_run_arms_on_gpu)
    if ZEROGPU
    else _run_arms_on_gpu
)


# --------------------------------------------------------------------------- #
# Shared presentation helpers (also used by the replay path)
# --------------------------------------------------------------------------- #
def _trace_rows(arm: dict[str, Any]) -> list[list[Any]]:
    graded = arm.get("graded") or []
    graded_by_id = {g["chunk_id"]: g for g in graded}
    rows = []
    for rank, hit in enumerate(arm.get("retrieved") or [], start=1):
        verdict = graded_by_id.get(hit["chunk_id"])
        rows.append(
            [
                rank,
                hit["source"],
                round(float(hit["score"]), 4),
                (verdict or {}).get("label", "-"),
                (round(verdict["confidence"], 4) if verdict else None),
                ("kept" if verdict["accepted"] else "dropped")
                if verdict
                else "kept (no gate)",
            ]
        )
    return rows


TRACE_HEADERS = ["rank", "source", "retrieval score", "label", "confidence", "decision"]


def _answer_html(arm: dict[str, Any], title: str, primary: bool) -> str:
    marker = "▶ " if primary else ""
    header = f"### {marker}{title}"
    if arm.get("grounded"):
        body = arm.get("answer", "")
    else:
        body = f"⚠️ **{arm.get('answer', NO_GROUNDED_ANSWER)}**"
    meta = (
        f"`used_grader={arm.get('used_grader')}` · "
        f"`grounded={arm.get('grounded')}` · "
        f"context: {arm.get('context_ids') or 'none'}"
    )
    return f"{header}\n\n{body}\n\n<small>{meta}</small>"


def _replay_result(corpus_key: str, query: str) -> dict[str, Any] | None:
    """Serve the recorded run when the GPU is not available.

    Only replays a recording whose corpus AND query match what was asked —
    showing a recording for a different question would be a lie.
    """
    recorded = load_recorded_trace()
    if not recorded:
        LOG.info("replay unavailable: no recording at %s", RECORDED_TRACE)
        return None
    if recorded.get("corpus", {}).get("key") != corpus_key:
        LOG.info(
            "replay unavailable: recording is for corpus %r, asked for %r",
            recorded.get("corpus", {}).get("key"),
            corpus_key,
        )
        return None
    if recorded.get("corpus", {}).get("query") != query:
        LOG.info("replay unavailable: query differs from the recording")
        return None
    LOG.info("replay served: recorded run for corpus=%s", corpus_key)
    model = recorded.get("model", {})
    return {
        "mode": "replay",
        "model": {
            "description": (
                f"{model.get('base_model')} + adapter {model.get('adapter_ref')}"
                if model.get("using_adapter")
                else f"{model.get('base_model')} — zero-shot base model, no adapter"
            ),
            "using_adapter": bool(model.get("using_adapter")),
            "load_error": model.get("load_error"),
        },
        "off": recorded["arms"]["off"],
        "on": recorded["arms"]["on"],
    }


def _gpu_failure_kind(exc: Exception) -> str:
    """Classify a ZeroGPU failure by whether the message survives the wrapping.

    Shared by the log line and the visitor notice so the two can never disagree
    about whose fault a failure is.

    `spaces/zero/wrappers.py` raises `error("ZeroGPU worker error", exc_class)`
    for ANY exception raised inside the worker, which collapses our own bugs and
    platform failures into the same bare `Error: 'RuntimeError'`. So a wrapped
    error means "the message was lost", NOT "the platform's fault" — the two are
    genuinely indistinguishable from here. Only an unwrapped message naming a
    quota problem can be attributed. Anything else stays `unknown` rather than
    being confidently mislabelled as ours or theirs.

    `gr.Error` stringifies to its *message*, not its title, so the buckets below
    are matched on the message text in `spaces/zero/client.py` — hence "larger
    than the maximum" rather than "illegal duration". Two consequences worth
    knowing: `illegal-duration` is OUR bug (the reservation is too large for the
    tier) and `no-gpu-in-queue` is honestly ambiguous — the same message is raised
    both when the platform has no capacity and when a signed-out visitor's
    two-minute pool is spent, and nothing in it says which.
    """
    lowered = str(exc).lower()
    if "larger than the maximum" in lowered:
        return "illegal-duration"
    if "quota" in lowered or "credits allocated" in lowered:
        return "quota-exhausted"
    if "no cuda gpus are available" in lowered or "worker_init" in lowered:
        return "gpu-not-attached"
    if "no gpu was available" in lowered:
        return "no-gpu-in-queue"
    # A bare wrapped error: the real cause is unknowable from the client side.
    if "error: '" in lowered:
        return "unknown-wrapped"
    return "unknown"


# One sentence per bucket, shared by the log and the visitor notice so the two
# cannot drift. None of these claim fault we cannot prove from the wrapper.
GPU_FAILURE_REASONS = {
    "illegal-duration": (
        "**The GPU reservation was rejected as too large for your tier.** That one "
        "is ours: `_estimate_duration` asked for more GPU time than your plan "
        "allows. The fix is a smaller estimate, not a retry."
    ),
    "quota-exhausted": (
        "**ZeroGPU daily quota exhausted.** Free accounts get a few GPU-minutes "
        "per day and the window resets 24 h after the first request; signed-out "
        "visitors get about two minutes."
    ),
    "gpu-not-attached": (
        "**ZeroGPU did not attach a GPU to the worker.** A platform-side limit "
        "(no capacity in this region, or a scheduling failure) — not an app bug. "
        "Retrieval and the model weights still work; only the grading and "
        "answering step needs the GPU."
    ),
    "no-gpu-in-queue": (
        "**No GPU became available within the 60 s queue window.** That message is "
        "raised both when the platform has no capacity in this region and when a "
        "signed-out visitor's very small daily pool is spent, and it does not say "
        "which — so treat it as platform-side and retry later."
    ),
    "unknown-wrapped": (
        "**The GPU worker raised an error the ZeroGPU wrapper did not forward.** "
        "Only the exception class survived, so the cause is in the container log "
        "rather than in this message — check the lines just above it."
    ),
    "unknown": (
        "**The GPU step failed without a classifiable error.** The raw message is "
        "below; the container log has the full context."
    ),
}


def _gpu_failure_notice(exc: Exception, corpus_key: str) -> str:
    """Turn a ZeroGPU failure into something a human can act on.

    The log line records `repr(exc)` and its args as well as `str(exc)`: for a
    wrapped failure `str` is the only thing that survives, and there is no second
    chance to look at it.
    """
    text = f"{type(exc).__name__}: {exc}"
    kind = _gpu_failure_kind(exc)

    LOG.error(
        "gpu step failed: kind=%s corpus=%s error=%s repr=%r args=%r",
        kind,
        corpus_key,
        text,
        exc,
        getattr(exc, "args", None),
        exc_info=True,
    )

    return (
        f"{GPU_FAILURE_REASONS[kind]}\n\n"
        f"- raw error: `{text}`\n"
        f"- to see a real run without a GPU, record one on any machine with the model: "
        f"`python -m src.record_demo_trace --corpus {corpus_key}` and commit "
        f"`results/demo_trace.json` — this Space will then replay it."
    )


def run_query(
    corpus_key: str, query: str, top_k: int, max_new_tokens: int, use_grader: bool
):
    """Gradio entry point: retrieve on CPU, then grade + answer on the GPU."""
    started = time.perf_counter()
    corpus_key = corpus_key or CORPORA[0].key
    corpus = get_corpus(corpus_key)
    query = (query or corpus.query).strip()
    LOG.info(
        "request: corpus=%s top_k=%s max_new_tokens=%s use_grader=%s query=%r",
        corpus_key,
        top_k,
        max_new_tokens,
        use_grader,
        query[:80],
    )
    if not query:
        LOG.warning("request rejected: empty query")
        return "Enter a query first.", "Enter a query first.", [], [], "no query"

    try:
        hits = retrieve_for(corpus_key, query, int(top_k))
    except Exception as exc:  # noqa: BLE001 — embeddings missing must not kill the UI
        LOG.error(
            "retrieval failed: corpus=%s %s: %s",
            corpus_key,
            type(exc).__name__,
            exc,
            exc_info=True,
        )
        replay = _replay_result(corpus_key, query)
        if replay:
            return _render(replay, query, use_grader, "embeddings unavailable; replaying")
        return (
            f"Retrieval failed — {type(exc).__name__}: {exc}",
            "",
            [],
            [],
            "retrieval failed",
        )
    if not hits:
        LOG.warning("no hits: corpus=%s query=%r", corpus_key, query[:80])
        return "Nothing retrieved for that query.", "", [], [], "no hits"

    # Warm the weight cache in the CPU process so the GPU window is pure compute.
    prefetch_status = prefetch_weights()

    reserved = _estimate_duration(corpus_key, query, hits, int(max_new_tokens))
    LOG.info(
        "gpu window: requesting %.0fs reservation (passages=%d)", reserved, len(hits)
    )
    gpu_started = time.perf_counter()
    try:
        result = run_arms(corpus_key, query, hits, int(max_new_tokens))
    except Exception as exc:  # noqa: BLE001 — quota/queue/load failures land here
        LOG.warning(
            "gpu window failed after %.1fs: kind=%s %s: %s repr=%r",
            time.perf_counter() - gpu_started,
            _gpu_failure_kind(exc),
            type(exc).__name__,
            exc,
            exc,
        )
        replay = _replay_result(corpus_key, query)
        if replay:
            return _render(
                replay, query, use_grader, f"GPU unavailable ({exc}); replaying"
            )
        notice = _gpu_failure_notice(exc, corpus_key)
        # Shown in BOTH slots: the answers panel is where a visitor looks first,
        # and the status line is the only place they would otherwise look.
        return (notice, notice, [], [], f"GPU unavailable · {prefetch_status}")

    LOG.info(
        "request complete: mode=live total=%.1fs", time.perf_counter() - started
    )
    return _render(result, query, use_grader, f"live · {prefetch_status}")


def _render(result: dict[str, Any], query: str, use_grader: bool, status: str):
    off_html = _answer_html(result["off"], "Grader OFF — naive RAG", not use_grader)
    on_html = _answer_html(result["on"], "Grader ON — corrective RAG", use_grader)

    model = result.get("model", {})
    banner = f"**{model.get('description', 'unknown model')}**"
    if not model.get("using_adapter"):
        banner += " — running the base model zero-shot"
    if model.get("load_error"):
        banner += f" — ⚠️ {model['load_error']}"

    on_arm = result["on"]
    kept = len(on_arm.get("context_ids") or [])
    total = len(on_arm.get("retrieved") or [])
    summary = f"**Gate decision:** kept {kept} of {total} retrieved passage(s)."
    if on_arm.get("graded") and not kept:
        summary += " Nothing survived, so the ON arm refused to answer."

    off_rows = _trace_rows(result["off"])
    on_rows = _trace_rows(result["on"])
    LOG.info(
        "render: mode=%s kept=%d/%d %s",
        result.get("mode", "live"),
        kept,
        total,
        "nothing survived the gate" if on_arm.get("graded") and not kept else "",
    )
    note = f"[{result.get('mode', 'live')}] {status}"
    return (
        f"{off_html}\n\n---\n\n{on_html}",
        f"{banner}\n\n{summary}",
        on_rows,
        off_rows,
        note,
    )


# --------------------------------------------------------------------------- #
# UI
# --------------------------------------------------------------------------- #
def _default_query(corpus_key: str) -> str:
    return get_corpus(corpus_key).query


def _corpus_label(key: str) -> str:
    return corpus_title(key)


with gr.Blocks(title="Corrective RAG · relevance grader") as demo:
    gr.Markdown(
        "# Corrective RAG with a fine-tuned relevance grader\n"
        "Retrieval returns passages that look similar but answer nothing. The gate "
        "decides which of them may reach the generator. Toggle it and compare the "
        "two arms. Everything runs on local open models."
    )
    with gr.Row():
        with gr.Column(scale=1):
            # (label, value) tuples: Gradio renders the title, the pipeline sees the key.
            corpus_radio = gr.Radio(
                choices=[(corpus_title(c.key), c.key) for c in CORPORA],
                value=CORPORA[0].key,
                label="Corpus",
            )
            query_box = gr.Textbox(
                value=_default_query(CORPORA[0].key),
                label="Query",
                lines=2,
            )
            top_k_slider = gr.Slider(2, 6, value=4, step=1, label="passages to retrieve")
            tokens_slider = gr.Slider(
                60, 300, value=150, step=10, label="max answer tokens"
            )
            grader_toggle = gr.Checkbox(
                value=True, label="Relevance grader ON (corrective gate)"
            )
            run_button = gr.Button("Run query", variant="primary")
        with gr.Column(scale=2):
            status_line = gr.Markdown("")
            model_line = gr.Markdown("")
            answers = gr.HTML()
            with gr.Accordion("Show retrieval + grading trace", open=False):
                on_table = gr.Dataframe(
                    headers=TRACE_HEADERS,
                    label="ON arm — per-passage verdict",
                    interactive=False,
                    wrap=True,
                )
                off_table = gr.Dataframe(
                    headers=TRACE_HEADERS,
                    label="OFF arm — everything retrieved",
                    interactive=False,
                    wrap=True,
                )

    run_button.click(
        fn=run_query,
        inputs=[corpus_radio, query_box, top_k_slider, tokens_slider, grader_toggle],
        outputs=[answers, model_line, on_table, off_table, status_line],
    )
    corpus_radio.change(fn=_default_query, inputs=[corpus_radio], outputs=[query_box])
    demo.load(
        fn=lambda: (
            "**GPU budget:** each visitor gets ~5 GPU-minutes/day. "
            "With no GPU left, this Space replays a recorded run — never an invented one."
        ),
        outputs=[status_line],
    )


if __name__ == "__main__":
    demo.queue(max_size=8).launch()
