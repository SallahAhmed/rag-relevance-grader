You are implementing the GUI + retrieval layer for an existing ML project.
Repo: F:\finetune (a git repo, Windows).

## Read first (do not skip)
- F:\finetune\AGENTS.md  — binding rules, especially core rules 5, 6, 13, 15
- F:\finetune\project-b-relevance-grader.md — the spec, especially the
  "Architecture interface — GraderInterface" section
- F:\finetune\src\grader_interface.py — the interface you must consume
- F:\finetune\src\dataset_utils.py — PROMPT_TEMPLATE and LABELS
- F:\finetune\src/config.py and config\sft_config.yaml — config style
- F:\finetune\src\evaluation_utils.py — existing code conventions

## Context
A Qwen2.5-1.5B model has been QLoRA fine-tuned to classify query-passage
relevance into 3 classes: irrelevant / partial / relevant. The adapter will be
published to the Hugging Face Hub as `sallahahmed/qwen2.5-1.5b-relevance-grader`.

Zero-shot baseline macro-F1 is 0.267; the fine-tune is running now. The
before/after comparison is the portfolio artifact.

The fine-tuned adapter may NOT exist yet while you work. Your code must load it
if present and fall back to the base model zero-shot if not, without crashing.

## What to build

### 1. `src/rag_pipeline.py`
A Corrective RAG pipeline, importable and testable WITHOUT any model present
(lazy-load the model, never at import time).

Required surface:
- `chunk_documents(texts, chunk_size, overlap) -> list[Chunk]`
- `build_index(chunks, embedding_model_name) -> Index`  (FAISS)
- `retrieve(index, query, top_k) -> list[(chunk, score)]`
- `grade_passages(query, passages, grader=None) -> list[GradedPassage]`
    - wraps `GraderInterface` from src/grader_interface.py
- `answer_with_gate(query, retrieved, use_grader: bool, generator=None) -> AnswerResult`
    - `use_grader=False` must skip the gate entirely and answer from all
      retrieved passages (this is the OFF arm of the demo toggle)
    - `use_grader=True` grades each passage, drops irrelevant ones, and if
      nothing survives, return an explicit "no grounded answer" result rather
      than hallucinating
- `AnswerResult` and `GradedPassage` should be small dataclasses and should
  serialize to plain dicts so the GUI can render them.

HARD CONSTRAINTS:
- Embeddings MUST be a local sentence-transformers model. NO OpenAI, no
  Anthropic, no paid API, no API keys of any kind. Default to
  `sentence-transformers/all-MiniLM-L6-v2`.
- Generation MUST work with the local Qwen model via transformers. No paid API.
- No new required dependencies beyond: sentence-transformers, faiss-cpu,
  streamlit, torch, transformers, peft, numpy. Add these to
  F:\finetune\src\requirements.txt in the existing style.

### 2. `space/` — Hugging Face Space app
Create `F:\finetune\space\` containing:
- `app.py` — Streamlit app
- `requirements.txt` — pinned enough for a HF Space to build
- `README.md` — with the YAML front matter block a Space needs:
  `sdk: streamlit`, `app_file: app.py`, and appropriate `pinned_sdk`/`python_version`

The app must have:
- A text area for the user to paste documents, AND a file uploader for .txt/.md
- 2-3 small PRE-BUILT sample corpora so the demo works before the user
  uploads anything (this matters: a reviewer opening the link must see
  something interesting immediately). Use a topic with genuinely readable
  short passages. Hardcode them in a `space/sample_corpora.py`.
- A query text input
- **A grader ON/OFF toggle** — this is the centerpiece. Run the query twice
  (or once with both arms) and show side by side: which passages the grader
  accepted/rejected, the label + confidence for each, and both final answers.
- A "Show retrieval + grading trace" expander showing per-passage score,
  label, confidence, and accept/reject decision.
- Graceful degradation: if no GPU, say so and still run (slowly). If the model
  fails to load, show the error and keep the UI alive.
- Cache the model with `@st.cache_resource` so it loads once.

### 3. `src/rag_pipeline_smoke_test.py`
A plain-python, no-network, no-model test that runs in under 30 seconds and
exits non-zero on failure. Assert:
- chunking respects chunk_size and produces more than one chunk for long input
- the OFF arm answers without invoking the grader (pass a spy/counting fake)
- the ON arm drops irrelevant passages (use a fake grader, no real model)
- when the fake grader rejects everything, `answer_with_gate` returns the
  explicit no-grounded-answer result and does NOT fabricate text
- serialization: `AnswerResult.to_dict()` returns JSON-serializable values

Run it with `python -m src.rag_pipeline_smoke_test` or
`python src/rag_pipeline_smoke_test.py` — make BOTH work.

## Hard rules
- DO NOT COMMIT ANYTHING. Leave changes in the working tree. The orchestrator commits.
- DO NOT modify: notebooks/, config/sft_config.yaml, results/, src/config.py,
  src/dataset_utils.py, src/grader_interface.py, src/evaluation_utils.py,
  src/mmlu_utils.py, AGENTS.md, ANALYSIS.md, project-b-relevance-grader.md.
  (You MAY append to src/requirements.txt.)
- Do not touch the Colab CLI, do not run anything on Colab, do not run training.
- Follow the existing code style: type hints, `from __future__ import annotations`
  if the neighbours use it, short docstrings that say WHY not WHAT.
- Python 3.10+ syntax is fine (the repo already uses `str | None`).
- Do not invent model performance numbers. If you show any metric in the GUI,
  it must come from results/*.json at runtime, never hardcoded.

## Report contract
When done, report exactly:
1. Files created/modified, with a one-line purpose each.
2. The exact command to run the smoke test, and its actual output.
3. Anything you could not do, and why.
4. Any assumption you made that the orchestrator should double-check.

Be honest about failures. A smoke test that fails and is reported as failing is
worth far more than one that is reported as passing.
