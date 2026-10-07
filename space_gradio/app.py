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

import contextvars
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
import secrets
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
# Shape: human-readable message, machine-stable prefix. Two fields in front of
# every line — `rid` (correlation id) and `event` (stable name) — so a monitor on
# another machine can parse without a regex over prose, while a human reading the
# raw Space log still sees a sentence. Pure JSON was rejected: HF interleaves
# tqdm bars and third-party tracebacks as unstructured text, so a JSON log is not
# reliably parseable anyway.
#
# UTC timestamps: that monitor's clock is not the Space's.
# --------------------------------------------------------------------------- #
LOG_FORMAT = (
    "%(asctime)s %(levelname)-7s %(name)-14s rid=%(rid)-9s event=%(event)-22s %(message)s"
)
LOG_DATEFMT = "%Y-%m-%dT%H:%M:%SZ"

# The current click's correlation id. A ContextVar rather than a parameter so
# every main-process line picks it up without threading `rid` through retrieve /
# render / notice helpers. The GPU worker is the exception: `spaces` REUSES a
# worker across requests, so a value inherited across the fork can be stale by one
# click. `_run_arms_on_gpu` therefore takes `rid` explicitly and overrides it via
# `extra`, and this var is only a fallback.
_RID: contextvars.ContextVar[str] = contextvars.ContextVar("grader_rid", default="-")


def _new_rid() -> str:
    """8 hex chars: short enough to read in a log, wide enough not to collide."""
    return secrets.token_hex(4)


class _EventFilter(logging.Filter):
    """Guarantee `rid` and `event` exist on every record.

    Both are interpolated with `%(...)s`, which raises `KeyError` at format time if
    absent — and records from the `src` logger and any future handler-less logger
    arrive without `extra`. Defaults, not `logging.Formatter.defaults`, because
    the format string is shared by every handler and the defaults would hide the
    omission instead of making it visible.
    """

    def filter(self, record: logging.LogRecord) -> bool:
        if not hasattr(record, "event"):
            record.event = "-"  # type: ignore[attr-defined]
        if not getattr(record, "rid", None):
            record.rid = _RID.get()  # type: ignore[attr-defined]
        return True


# Third-party loggers that reach INFO/WARNING during a load and would otherwise
# arrive unformatted via logging.lastResort (the Space's root logger has no
# handler), interleaved with our lines. Pinned, not silenced: their warnings are
# real diagnostics.
NOISY_LOGGERS = (
    "transformers",
    "sentence_transformers",
    "sentencepiece",
    "faiss",
    "urllib3",
    "filelock",
)


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
    handler.addFilter(_EventFilter())

    for name in ("grader", "src"):
        target = logging.getLogger(name)
        target.setLevel(logging.INFO)
        if not any(getattr(h, "_grader_app", False) for h in target.handlers):
            handler._grader_app = True  # type: ignore[attr-defined]
            target.addHandler(handler)
        target.propagate = False

    for name in NOISY_LOGGERS:
        logging.getLogger(name).setLevel(logging.WARNING)

    return logging.getLogger("grader")


LOG = _setup_logging()

for _candidate in (Path(__file__).resolve().parent, *Path(__file__).resolve().parents):
    if (_candidate / "src" / "rag_pipeline.py").is_file():
        if str(_candidate) not in sys.path:
            sys.path.insert(0, str(_candidate))
        break
from src.rag_pipeline import (  # noqa: E402
    Chunk,
    HFRelevanceGrader,
    QwenGenerator,
    SentenceTransformerEmbedder,
    answer_with_gate,
    before_after_rows,
    build_index,
    load_local_model,
    load_metrics,
    retrieve,
    zerogpu_active,
)
from src.sample_corpora import CORPORA, corpus_title, get_corpus  # noqa: E402

# Presentation lives in ui.py: CSS, HTML renderers and the runtime-read
# benchmark panel. Split out so the execution half of this file — CPU retrieval,
# the ZeroGPU worker, failure classification — stays readable on its own.
sys.path.insert(0, str(Path(__file__).resolve().parent))
from ui import (  # noqa: E402
    CSS,
    _arm_card,
    _gate_summary,
    _hero_html,
    _model_banner,
    _verdict_passages,
    benchmark_html,
)


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
RESULTS_DIR_TRACES = REPO / "results"


def _recorded_trace_path(corpus_key: str) -> Path:
    """Per-corpus recording first, legacy single-file recording as fallback.

    Recordings are made one-per-corpus-query (`demo_trace_<corpus>.json`),
    so a single hardcoded path could only ever serve one corpus — any other
    corpus fell through to the live GPU path even though its recording sat
    right there. The legacy `demo_trace.json` is still honoured so old
    recordings keep working; its own corpus/query checks gate it.
    """
    per_corpus = RESULTS_DIR_TRACES / f"demo_trace_{corpus_key}.json"
    if per_corpus.is_file():
        return per_corpus
    return RESULTS_DIR_TRACES / "demo_trace.json"

# True only when the platform is actually scheduling ZeroGPU workers for this
# Space. `import spaces` succeeding proves nothing — the package ships on CPU,
# T4 and A10G tiers too, and `spaces.GPU` is a no-op unless this is set — so
# gating on the import was the single bug that gave the off-Platform paths
# (Colab, local) ZeroGPU machinery that cannot work there. `zerogpu_active()`
# lives in src/ because `load_local_model` needs the same answer for device_map.
ZEROGPU = zerogpu_active()

# Where the benchmark JSONs live. Passed explicitly to `ui.benchmark_html()`
# rather than left to `load_metrics()`'s probe, which needs both `src/` and
# `config/` side by side — true here, not guaranteed in a Space, where a missing
# `config/` would silently blank the primary artifact instead of raising.
RESULTS_DIR = REPO / "results"


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
        extra={"event": "index_ready"},
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
        extra={"event": "retrieval"},
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
    LOG.info(
        "prefetch weights: start repo=%s (progress bars suppressed)",
        repo,
        extra={"event": "prefetch_start"},
    )
    try:
        from huggingface_hub import snapshot_download

        files = snapshot_download(repo)
        LOG.info(
            "prefetch weights: done %d files from %s in %.1fs",
            len(files),
            repo,
            time.perf_counter() - started,
            extra={"event": "prefetch_done"},
        )
        return f"weights ready ({len(files)} files)"
    except Exception as exc:  # noqa: BLE001 — a cold start is still better than none
        LOG.warning(
            "prefetch weights: FAILED after %.1fs (%s: %s); the GPU window will download instead",
            time.perf_counter() - started,
            type(exc).__name__,
            exc,
            exc_info=True,
            extra={"event": "prefetch_failed"},
        )
        return f"weights not prefetched ({type(exc).__name__})"


def _base_model_id() -> str:
    try:
        from src.config import load_config

        return load_config().model.base_model
    except Exception:  # noqa: BLE001 — config is optional in the Space
        return "Qwen/Qwen2.5-1.5B-Instruct"


@lru_cache(maxsize=8)
def load_recorded_trace(corpus_key: str = "") -> dict[str, Any] | None:
    """The offline recording for this corpus, or None. Absence is reported, never faked."""
    path = _recorded_trace_path(corpus_key)
    if not path.is_file():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
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
        extra={"event": "app_startup"},
    )
    available = sorted(p.name for p in RESULTS_DIR_TRACES.glob("demo_trace*.json"))
    LOG.info(
        "startup: repo_root=%s recorded_traces=%s gpu_path=%s",
        REPO,
        ",".join(available) if available else "MISSING (no replay available)",
        "spaces.GPU worker" if ZEROGPU else "in-process",
        extra={"event": "app_startup"},
    )
    LOG.info(
        "startup: base_model=%s corpora=%d",
        _base_model_id(),
        len(CORPORA),
        extra={"event": "app_startup"},
    )
    # Loaded only to record provenance. `ui.benchmark_html()` owns rendering and
    # reads them again; keeping the payload here too would be a second source of
    # truth for the same two files. The point of logging the resolved paths is
    # drift: a Space whose copied results/ is older than the repo shows numbers
    # with no indication they are stale.
    metrics = load_metrics(RESULTS_DIR)
    LOG.info(
        "startup: metrics rows=%d missing=%s sources=%s",
        len(before_after_rows(metrics)),
        metrics.get("missing") or "none",
        metrics.get("sources") or "none",
        extra={"event": "app_startup"},
    )
    try:
        import torch

        LOG.info(
            "startup: torch=%s cuda_devices=%s",
            torch.__version__,
            os.environ.get("CUDA_VISIBLE_DEVICES", "unset"),
            extra={"event": "app_startup"},
        )
    except ImportError as exc:  # pragma: no cover - torch is a hard dep of the GPU path
        LOG.warning(
            "startup: torch not importable in the main process (%s)",
            exc,
            extra={"event": "app_startup"},
        )


_log_startup()


# --------------------------------------------------------------------------- #
# GPU side: everything that needs the model
# --------------------------------------------------------------------------- #
def _estimate_duration(
    corpus_key: str,
    query: str,
    hits: list,
    max_new_tokens: int,
    *args: Any,
    **kwargs: Any,
) -> int:
    """Reserve only what the work needs — shorter reservations rank higher.

    ZeroGPU calls this with the SAME arguments as the decorated function, so the
    signature must mirror `_run_arms_on_gpu` (getting this wrong fails at call
    time, not import time). `*args, **kwargs` are required, not decorative:
    Gradio passes `progress=` positionally into event handlers, and a strict
    signature raises "takes 5 positional arguments but 6 were given".

    Rough shape: ~12 s to load a 1.5B model in the worker, ~0.5 s per graded
    passage, ~0.04 s per generated token across both arms. These coefficients are
    a guess until `gpu_window` logs a reserved-vs-actual pair — HF's method is
    ship a placeholder, measure 2-3 real calls, then set `round(max x 1.4)`.
    """
    passages = len(hits) if isinstance(hits, (list, tuple)) else 1
    tokens = int(max_new_tokens) if max_new_tokens else 150
    # Clamped because ZeroGPU compares the RESERVATION against the visitor's
    # remaining quota, not the work's real runtime: an unauthenticated visitor
    # has a 2-minute daily pool, so an oversized reservation reports "quota
    # exceeded" for work that would have finished in a fraction of it.
    return min(120, int(12 + 0.5 * passages + 0.08 * tokens) + 8)


def _log_worker_device(rid: str) -> None:
    """Record what the worker actually got. Outside @spaces.GPU this is a lie.

    In the main process `torch.cuda` is hijacked to report a device that is not
    there, so `torch.cuda.is_available()` is meaningless there. Inside the worker
    the patch is removed and these are real values — and "which GPU did the
    visitor actually get" was the single biggest gap in the old logs.

    `rid` is passed explicitly rather than read from the ContextVar: `spaces`
    reuses one worker across requests, so an inherited value can be one click
    stale.
    """
    try:
        import torch

        available = torch.cuda.is_available()
        if not available:
            LOG.warning(
                "gpu worker: torch.cuda.is_available() is False inside the worker",
                extra={"event": "gpu_device", "rid": rid},
            )
            return
        LOG.info(
            "gpu worker: device=%s count=%s visible=%r",
            torch.cuda.get_device_name(0),
            torch.cuda.device_count(),
            os.environ.get("CUDA_VISIBLE_DEVICES", "unset"),
            extra={"event": "gpu_device", "rid": rid},
        )
    except Exception as exc:  # noqa: BLE001 — a probe failure must not mask the real work
        LOG.warning(
            "gpu worker: device probe failed (%s: %s)",
            type(exc).__name__,
            exc,
            extra={"event": "gpu_device", "rid": rid},
        )


def _run_arms_on_gpu(
    corpus_key: str,
    query: str,
    hits: list[dict[str, Any]],
    max_new_tokens: int,
    rid: str = "-",
) -> dict[str, Any]:
    """Load the model inside the worker and run both arms. Returns plain dicts.

    `rid` crosses the fork as a plain str (picklable, bounded). Its presence in
    the signature is why `_estimate_duration` accepts `*args`.
    """
    started = time.perf_counter()
    LOG.info(
        "gpu worker: start corpus=%s passages=%d max_new_tokens=%d zerogpu=%s",
        corpus_key,
        len(hits),
        max_new_tokens,
        ZEROGPU,
        extra={"event": "gpu_worker_start", "rid": rid},
    )
    _log_worker_device(rid)
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
        extra={"event": "gpu_worker_done", "rid": rid},
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
# Shared presentation helpers now live in ui.py. The trace-table builders below
# stay here because make_comparison-style tabular output is what gr.Dataframe
# consumes, and that stays a plain list-of-lists.
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


def _replay_result(corpus_key: str, query: str) -> dict[str, Any] | None:
    """Serve the recorded run when the GPU is not available.

    Only replays a recording whose corpus AND query match what was asked —
    showing a recording for a different question would be a lie.
    """
    recorded = load_recorded_trace(corpus_key)
    if not recorded:
        LOG.info(
            "replay unavailable: no recording at %s",
            _recorded_trace_path(corpus_key),
            extra={"event": "replay_unavailable"},
        )
        return None
    if recorded.get("corpus", {}).get("key") != corpus_key:
        LOG.info(
            "replay unavailable: recording is for corpus %r, asked for %r",
            recorded.get("corpus", {}).get("key"),
            corpus_key,
            extra={"event": "replay_unavailable"},
        )
        return None
    if recorded.get("corpus", {}).get("query") != query:
        LOG.info(
            "replay unavailable: query differs from the recording",
            extra={"event": "replay_unavailable"},
        )
        return None
    model = recorded.get("model", {})
    LOG.info(
        "replay served: corpus=%s recorded_from=%s device=%s quantized=%s adapter=%s",
        corpus_key,
        recorded.get("recorded_from", "unknown"),
        model.get("device", "unknown"),
        model.get("quantized", "unknown"),
        model.get("adapter_ref") or "none",
        extra={"event": "replay_served"},
    )
    # `provenance` is carried through to the status line so a visitor can see that
    # this run was recorded on CPU rather than executed on the GPU they just
    # waited for. Without it the replay reads as a live run, which is the one
    # thing this app must never do.
    return {
        "mode": "replay",
        "provenance": {
            "recorded_from": recorded.get("recorded_from", "unknown"),
            "device": model.get("device", "unknown"),
            "quantized": model.get("quantized"),
        },
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
        "rather than in this message — check the lines just above it. From the "
        "Space's side the two most common causes are `RuntimeError: No CUDA GPUs "
        "are available` (no GPU capacity right now) and bitsandbytes failing to "
        "attach 4-bit weights in the worker."
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
        extra={"event": "gpu_step_failed"},
    )

    recorded_hint = ""
    try:
        trace = load_recorded_trace(corpus_key)
        if trace and trace.get("corpus", {}).get("query"):
            recorded_hint = (
                f"\n- this corpus has a recorded run that replays without a GPU — "
                f"type exactly: `{trace['corpus']['query']}`"
            )
    except Exception:  # noqa: BLE001 — a hint must never break the notice
        pass

    return (
        f"{GPU_FAILURE_REASONS[kind]}\n\n"
        f"- raw error: `{text}`\n"
        f"- to see a real run without a GPU, record one on any machine with the model: "
        f"`python -m src.record_demo_trace --corpus {corpus_key}` and commit "
        f"`results/demo_trace.json` — this Space will then replay it."
        f"{recorded_hint}"
    )


def _fail(message: str, status: str = ""):
    """Uniform failure payload matching the six UI outputs.

    One helper rather than five inline tuples: when the output arity changed the
    inline versions silently drifted, and a length mismatch surfaces to the
    visitor as an opaque Gradio error instead of a readable notice.
    """
    body = f'<div class="warn-box">{_esc_status(message)}</div>'
    note = f'<div class="status-line">{_esc_status(status or message)}</div>'
    return body, "", "", [], [], note


def run_query(
    corpus_key: str, query: str, top_k: int, max_new_tokens: int, use_grader: bool
):
    """Gradio entry point. Owns the click's correlation id.

    Gradio runs handlers on a threadpool whose threads are reused, so the
    ContextVar must be reset when this click ends — a leaked value would stamp
    an unrelated visitor's later lines with this click's id.
    """
    token = _RID.set(_new_rid())
    try:
        return _run_query_impl(corpus_key, query, top_k, max_new_tokens, use_grader)
    finally:
        _RID.reset(token)


def _run_query_impl(
    corpus_key: str, query: str, top_k: int, max_new_tokens: int, use_grader: bool
):
    """Retrieve on CPU, then grade + answer on the GPU. `rid` is read from the
    `_RID` ContextVar (set by `run_query`) so every line below carries this
    click's id without threading it through the retrieve/render/notice helpers.
    """
    started = time.perf_counter()
    rid = _RID.get()
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
        extra={"event": "request", "rid": rid},
    )
    if not query:
        LOG.warning(
            "request rejected: empty query",
            extra={"event": "request_rejected", "rid": rid},
        )
        return _fail("Enter a query first.", "no query")

    try:
        hits = retrieve_for(corpus_key, query, int(top_k))
    except Exception as exc:  # noqa: BLE001 — embeddings missing must not kill the UI
        LOG.error(
            "retrieval failed: corpus=%s %s: %s",
            corpus_key,
            type(exc).__name__,
            exc,
            exc_info=True,
            extra={"event": "retrieval_failed", "rid": rid},
        )
        replay = _replay_result(corpus_key, query)
        if replay:
            return _render(replay, query, use_grader, "embeddings unavailable; replaying")
        LOG.info(
            "click complete: mode=retrieval-failed total=%.1fs",
            time.perf_counter() - started,
            extra={"event": "click_complete", "rid": rid},
        )
        return _fail(
            f"Retrieval failed — {type(exc).__name__}: {exc}", "retrieval failed"
        )
    if not hits:
        LOG.warning(
            "no hits: corpus=%s query=%r",
            corpus_key,
            query[:80],
            extra={"event": "no_hits", "rid": rid},
        )
        return _fail("Nothing retrieved for that query.", "no hits")

    # Warm the weight cache in the CPU process so the GPU window is pure compute.
    prefetch_status = prefetch_weights()

    reserved = _estimate_duration(corpus_key, query, hits, int(max_new_tokens), rid)
    LOG.info(
        "gpu window: requesting %.0fs reservation (passages=%d)",
        reserved,
        len(hits),
        extra={"event": "gpu_window_request", "rid": rid},
    )
    gpu_started = time.perf_counter()
    try:
        result = run_arms(corpus_key, query, hits, int(max_new_tokens), rid)
    except Exception as exc:  # noqa: BLE001 — quota/queue/load failures land here
        gpu_seconds = time.perf_counter() - gpu_started
        LOG.warning(
            "gpu window failed after %.1fs: reserved=%.0fs kind=%s %s: %s repr=%r",
            gpu_seconds,
            reserved,
            _gpu_failure_kind(exc),
            type(exc).__name__,
            exc,
            exc,
            extra={"event": "gpu_window_failed", "rid": rid},
        )
        replay = _replay_result(corpus_key, query)
        if replay:
            return _render(
                replay, query, use_grader, f"GPU unavailable ({exc}); replaying"
            )
        notice = _gpu_failure_notice(exc, corpus_key)
        # Failures are surfaced in the UI, not by raising: raising `gr.Error`
        # would discard these six outputs and replace the notice with a toast.
        # The cost is that HF's own metrics cannot see the failure, so the log
        # carries it instead — `click_complete` below is the countable line, and
        # it is emitted on every path out of this handler.
        LOG.info(
            "click complete: mode=gpu-failed total=%.1fs gpu=%.1fs reserved=%.0fs kind=%s",
            time.perf_counter() - started,
            gpu_seconds,
            reserved,
            _gpu_failure_kind(exc),
            extra={"event": "click_complete", "rid": rid},
        )
        notice_html = _notice_html(notice)
        return (
            f'<div class="warn-box">{notice_html}</div>',
            notice_html,
            "",
            [],
            [],
            f'<div class="status-line">GPU unavailable · '
            f"{_esc_status(prefetch_status)}</div>",
        )

    gpu_seconds = time.perf_counter() - gpu_started
    # The reserved-vs-actual pair is the only evidence that can retune the
    # duration coefficients. Without it `_estimate_duration` stays a guess, and a
    # guess is how a Space ends up reporting "quota exceeded" for work that took
    # 20 s of a 60 s reservation.
    LOG.info(
        "gpu window: actual=%.1fs reserved=%.0fs ratio=%.2f slack=%+.1fs",
        gpu_seconds,
        reserved,
        gpu_seconds / reserved if reserved else 0.0,
        reserved - gpu_seconds,
        extra={"event": "gpu_window_actual", "rid": rid},
    )
    LOG.info(
        "request complete: mode=live total=%.1fs",
        time.perf_counter() - started,
        extra={"event": "request_complete", "rid": rid},
    )
    return _render(result, query, use_grader, f"live · {prefetch_status}")


def _render(result: dict[str, Any], query: str, use_grader: bool, status: str):
    """Assemble the five UI outputs.

    The two arms become real HTML cards rather than one markdown blob: the
    comparison is the artifact, and burying it in prose is what made the old
    version hard to read.
    """
    off_html = _arm_card(
        result["off"], "Grader OFF — naive RAG", not use_grader, naive=True
    )
    on_html = _arm_card(
        result["on"], "Grader ON — corrective RAG", use_grader, naive=False
    )
    answers = (
        f'<div style="display:grid;grid-template-columns:1fr 1fr;gap:0.85rem">'
        f"<div>{off_html}</div><div>{on_html}</div></div>"
        f"{_gate_summary(result)}"
    )

    banner = _model_banner(result)
    on_rows = _trace_rows(result["on"])
    off_rows = _trace_rows(result["off"])
    kept = len((result["on"] or {}).get("context_ids") or [])
    total = len((result["on"] or {}).get("retrieved") or [])
    LOG.info(
        "render: mode=%s kept=%d/%d %s",
        result.get("mode", "live"),
        kept,
        total,
        "nothing survived the gate"
        if (result["on"] or {}).get("graded") and not kept
        else "",
    )
    note = f"[{result.get('mode', 'live')}] {status}"
    # A replay is a recording, not a run. Naming the device it was recorded on
    # keeps a visitor from reading a CPU recording as a live GPU result — the one
    # reading this app must never allow, since the whole point of the fallback is
    # that it never invents evidence.
    if (provenance := result.get("provenance")) and result.get("mode") == "replay":
        note += (
            f" ┬╖ recorded on {provenance.get('device', 'unknown')}"
            f" via {str(provenance.get('recorded_from', 'unknown'))}"
            f"{'' if provenance.get('quantized') else ' (unquantized)'}"
        )
    verdict_block = f"{_verdict_passages(result['on'])}"

    return (
        answers,
        banner,
        verdict_block,
        on_rows,
        off_rows,
        f'<div class="status-line">{_esc_status(note)}</div>',
    )


def _esc_status(text: str) -> str:
    """Escape the status line; it is the one string built from error text."""
    import html as _html

    return _html.escape(str(text))


def _notice_html(text: str) -> str:
    """Render the GPU-failure notice as safe HTML for `gr.HTML`.

    `_gpu_failure_notice` builds markdown — paragraph breaks, `- ` bullets,
    backtick spans — with the raw exception text inside. `gr.HTML` ships strings
    verbatim, so left alone the visitor would see literal `- raw error:` lines
    and unescaped exception text. Escape first (order matters: escaping after
    tag-conversion would mangle our own tags), then convert the shape.
    """
    import html as _html
    import re

    safe = _html.escape(str(text))
    parts: list[str] = []
    in_list = False
    for line in safe.split("\n"):
        stripped = line.strip()
        if stripped.startswith("- "):
            if not in_list:
                parts.append("<ul>")
                in_list = True
            body = re.sub(r"`([^`]*)`", r"<code>\1</code>", stripped[2:])
            parts.append(f"<li>{body}</li>")
        else:
            if in_list:
                parts.append("</ul>")
                in_list = False
            if stripped:
                parts.append(f"<div>{stripped}</div>")
    if in_list:
        parts.append("</ul>")
    return "".join(parts)


# --------------------------------------------------------------------------- #
# UI
# --------------------------------------------------------------------------- #
def _default_query(corpus_key: str) -> str:
    return get_corpus(corpus_key).query


with gr.Blocks(
    title="Corrective RAG · relevance grader", css=CSS, fill_width=True
) as demo:
    gr.HTML(_hero_html())

    with gr.Row():
        with gr.Column(scale=1, min_width=300):
            # (label, value) tuples: Gradio renders the title, the pipeline sees the key.
            corpus_radio = gr.Radio(
                choices=[(corpus_title(c.key), c.key) for c in CORPORA],
                value=CORPORA[0].key,
                label="Corpus",
                info="Each corpus mixes answer-bearing passages with decoys that match on vocabulary.",
            )
            query_box = gr.Textbox(
                value=_default_query(CORPORA[0].key),
                label="Query",
                lines=3,
            )
            top_k_slider = gr.Slider(
                2, 8, value=4, step=1, label="passages to retrieve"
            )
            tokens_slider = gr.Slider(
                60, 300, value=150, step=10, label="max answer tokens"
            )
            grader_toggle = gr.Checkbox(
                value=True,
                label="Emphasise the grader-ON arm",
                info="Both arms always run — this only changes which card is "
                "highlighted. Nothing is hidden either way.",
            )
            run_button = gr.Button("Run comparison", variant="primary", size="lg")
            status_line = gr.HTML("")

        with gr.Column(scale=2):
            model_line = gr.HTML("")
            answers = gr.HTML()

    with gr.Accordion("Per-passage gate verdicts", open=True):
        gr.Markdown(
            "What the grader decided about each retrieved passage, and whether it "
            "was allowed to reach the generator. Confidence is the softmax over the "
            "three label tokens."
        )
        verdicts = gr.HTML("")

    with gr.Accordion("Raw retrieval trace (tables)", open=False):
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

    with gr.Accordion("Measured benchmark: fine-tune vs zero-shot", open=False):
        gr.Markdown(
            "Read from `results/baseline_results.json` and "
            "`results/finetuned_results.json` at runtime. No score in this UI is "
            "hardcoded — if those files are not deployed, this section says so "
            "rather than showing a remembered number."
        )
        bench = gr.HTML(benchmark_html(str(RESULTS_DIR)))

    run_button.click(
        fn=run_query,
        inputs=[corpus_radio, query_box, top_k_slider, tokens_slider, grader_toggle],
        outputs=[answers, model_line, verdicts, on_table, off_table, status_line],
    )
    corpus_radio.change(fn=_default_query, inputs=[corpus_radio], outputs=[query_box])
    demo.load(
        fn=lambda: (
            '<div class="status-line">GPU budget: each visitor gets ~5 '
            "GPU-minutes/day. With no GPU left this Space replays a recorded run "
            "&mdash; never an invented one.</div>"
        ),
        outputs=[status_line],
    )


if __name__ == "__main__":
    demo.queue(max_size=8).launch()
