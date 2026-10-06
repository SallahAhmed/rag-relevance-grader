---
title: Corrective RAG relevance grader
emoji: 🦀
colorFrom: pink
colorTo: purple
sdk: gradio
sdk_version: 6.29.0
python_version: "3.12"
app_file: app.py
pinned: false
license: mit
short_description: Toggle a QLoRA-fine-tuned relevance grader between naive and corrective RAG
suggested_hardware: zero-a10g
---

# Corrective RAG with a fine-tuned relevance grader (ZeroGPU, free)

> **Front matter is pinned to the combination that actually built and ran.**
> `sdk_version: 6.29.0` and `python_version: "3.12"` are what the live Space uses.
> Gradio 6.29 requires `huggingface-hub>=1.16`, which is why `requirements.txt`
> pins `transformers>=5.13` — see the comment there. Do not relax either without
> re-checking the hub constraint intersection.

The hosted twin of the Streamlit app in `../space/`. Same pipeline
(`src/rag_pipeline.py`), different UI layer, because **ZeroGPU only schedules
Gradio Spaces** — a Streamlit Space cannot borrow a GPU on the free tier.

## What the toggle does

| Arm | Behaviour |
| --- | --- |
| **Grader OFF** (naive RAG) | Every retrieved passage reaches the generator. No grading call is made. |
| **Grader ON** (corrective RAG) | Passages the grader calls `irrelevant` are dropped. If none survive, the pipeline returns an explicit *no grounded answer* instead of inventing text. |

Both arms always run and are shown side by side, with a per-passage trace
(retrieval score, label, confidence, accept/reject).

## Why the app is split across two processes

```
CPU process (always warm)        GPU worker (forked per click)
------------------------        --------------------------------
chunk -> embed -> FAISS retrieve  load Qwen 1.5B + adapter, 4-bit
                                  grade every retrieved passage
                                  answer OFF arm, answer ON arm
```

Three ZeroGPU rules are load-bearing here, each learned from a real failure:

1. **Never touch CUDA outside `@spaces.GPU`.** `sentence-transformers`
   auto-detects the GPU, so the CPU-side embedder is pinned to
   `device="cpu"`. Otherwise: `RuntimeError: Low-level CUDA init
   (torch._C._cuda_init) reached`.
2. **The duration estimator must mirror the decorated function's arguments.**
   `spaces.GPU(duration=fn)` calls `fn` with the decorated function's inputs.
3. **Weights are prefetched on the CPU first.** The 3.1 GB download would
   otherwise happen inside the GPU window and burn the visitor's entire daily
   quota before any compute starts.

Retrieval stays on the CPU on purpose. MiniLM costs milliseconds per passage, and
keeping it out of the GPU window is what allows a small
`@spaces.GPU(duration=...)` reservation. The default is 60 s, which would spend a
visitor's entire daily quota on a single click; this app reserves roughly 25–35 s
via a duration estimator, so a visitor's ~5 GPU-minutes/day buys roughly a dozen
queries.

## When the GPU is gone

Each visitor gets about **5 GPU-minutes/day**. When that runs out, the app does
**not** invent verdicts: it replays `results/demo_trace.json`, a recording made
by `src/record_demo_trace.py` from a real model run — and only when the requested
corpus and query match the recording. With no GPU and no matching recording, the
app says so and prints the command to produce one, distinguishing a platform-side
worker failure from an error in our own code.

The always-on, zero-compute version of the same artifact is the static Space in
`../space_static/`.

## Reading the container log

Most of what a ZeroGPU Space prints is not ours. This is the triage table — check
it before changing any code.

| Log line | Verdict | Action |
| --- | --- | --- |
| `WARNING: Running pip as the 'root' user …` | Noise. The image builds as root. | Ignore. |
| `Exception ignored in: <function BaseEventLoop.__del__ …> ValueError: Invalid file descriptor: -1` | Noise. The worker closes every inherited fd after fork, so the parent's already-closed selector socket complains when its loop is GC'd. | Ignore. |
| `gradio.exceptions.Error: "Value: 'ping' … is not in the list of choices"` + `HTTPException: 404` + `Caught handled exception, but response already started` | One event, printed three times, and not ours. `ping` is not sent by Gradio (grepped 6.29.0) and in `spaces` it appears only as an SSE frame in the ZeroGPU scheduler protocol (`spaces/zero/api.py`), which never reaches a component. An outside liveness probe is POSTing `["ping", …]` at API index 0 — `run_query` — whose first input is the corpus `Radio`. `Radio.preprocess` rejects it, and Gradio's error path then trips over the SSE stream it already started. | Ignore. Do **not** add `ping` as a choice; that would put a junk option in the UI to silence a log line. |
| `unauthenticated requests to the HF Hub` | Noise unless the adapter repo is private. | Add `HF_TOKEN` as a Space secret if it is. |
| `RuntimeError: No CUDA GPUs are available` at `spaces/zero/wrappers.py` → `patching.py:417` | **Platform-side.** `torch.init(nvidia_uuid)` sets `CUDA_VISIBLE_DEVICES` to the UUID the scheduler issued, then the first `.cuda()` finds no device — the worker was forked without the device. Byte-identical to [zero-gpu-explorers #179](https://huggingface.co/spaces/zero-gpu-explorers/README/discussions/179), where every documented check had already been done. | Retry; if it persists, it is an HF incident. The replay fallback covers it meanwhile. |

### The lines that are ours

| Log line | Reads as |
| --- | --- |
| `startup: build=<sha256[:12]> zerogpu=<bool> device_api=<bool> python=<ver>` | Which file is actually running, and whether ZeroGPU is really scheduled. **`zerogpu=False` on a ZeroGPU Space means the env gate is wrong, not that the hardware is missing.** |
| `startup: gpu_path=spaces.GPU worker` / `in-process` | Which of the two execution paths the click will take. |
| `gpu worker: device=<name> count=<n> visible=<uuid>` | What the worker was actually given. Unanswerable from the main process — `torch.cuda` is hijacked there and `is_available()` returns a lie. |
| `gpu step failed: kind=<bucket> …` | One of `illegal-duration` (ours: reservation too large), `quota-exhausted`, `gpu-not-attached`, `no-gpu-in-queue`, `unknown-wrapped`, `unknown`. The same `kind` drives the visitor notice, so the two cannot disagree. |

`kind=unknown-wrapped` is the important one to understand: `spaces` reports **any**
exception raised in the worker as a bare `Error: 'RuntimeError'` and discards the
message (`spaces/zero/wrappers.py`, `error("ZeroGPU worker error", exc_class)`).
So a wrapped error means *the cause was lost*, not *the platform's fault* — the
real traceback is above it in the same log, which is why the app logs `repr(exc)`
and its `args` alongside `str(exc)`.

### Why the model loads inside the GPU window

The ZeroGPU guidance is to load at module scope and `.to("cuda")` so the startup
"pack" step can stage weights to disk and cold starts skip the checkpoint read.
This app loads lazily inside the worker instead, because its whole point is
graceful degradation: the adapter repo may be private or missing, and a
module-scope load failure would take the Space down instead of falling back to
zero-shot with a banner. The cost is a cold worker per click, which is the thing
that fails when the platform mis-schedules. `device_map` is therefore `"cuda"`
(not `"auto"`, which routes through `accelerate.set_module_tensor_to_device` →
`torch._C._cuda_init()`) whenever `SPACES_ZERO_GPU` is set.

1. Create a **Gradio** Space with hardware **ZeroGPU** (`zero-a10g`).
2. Copy this directory's `app.py` and `requirements.txt` to the Space root and
   ship `src/` + `config/` alongside: the app imports `src/rag_pipeline.py`, and
   `config/sft_config.yaml` is how it resolves the published adapter id. A Space
   built from this folder alone will show an import error.
3. `models:` is deliberately **absent** above. Add a `models:` block once the
   adapter repo is public; a declared-but-missing repo fails the build.

## Honest limitations

- `accelerate` is required for the `device_map="cpu"` path; it is pinned.
- The adapter is optional. Without it (or if its repo is private) the app falls
  back to the base model zero-shot and says so in a banner.
- The fine-tuning itself still runs on free Colab/Kaggle T4, as the project spec
  requires. This Space is only the demo surface.
