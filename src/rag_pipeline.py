"""Corrective RAG: chunk -> embed -> FAISS retrieve -> relevance gate -> answer.

Every heavy import (torch, transformers, peft, sentence-transformers, faiss)
happens INSIDE the function that needs it, so `import src.rag_pipeline` costs
milliseconds and needs no GPU, no model files and no network. That is what the
smoke test, the notebooks and the Streamlit app all depend on.

The gate is the point of the project: a relevance grader decides which retrieved
passages may reach the generator.
  OFF -> naive RAG: answer from every retrieved passage.
  ON  -> drop the passages the grader calls irrelevant; if none survive, return
         NO_GROUNDED_ANSWER instead of inventing text.

Hard constraints enforced here (see project-b-relevance-grader.md):
  - Embeddings are a LOCAL sentence-transformers model. `_require_local_model`
    rejects anything that looks like a hosted provider, so no API key can leak
    in via a model name.
  - Generation is the local Qwen model through transformers.
  - Grading uses PROMPT_TEMPLATE imported from dataset_utils — never retyped —
    so the live demo and notebooks 02/03/04 are scored by the same prompt
    (core rule 5).
"""
from __future__ import annotations

import json
import logging
import os
import time
from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from .dataset_utils import LABELS, PARTIAL, PROMPT_TEMPLATE, RELEVANT
from .grader_interface import GraderInterface, ScoredPassage

# Stdlib logging only, and deliberately no handler configuration here: this
# module is imported by the smoke test with no ML stack installed, and by the
# notebooks, which bring their own root handlers. The Space attaches one
# handler to the "src" logger (see space_gradio/app.py); everywhere else these
# records propagate to whatever the host process already configured.
LOG = logging.getLogger(__name__)

DEFAULT_EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
DEFAULT_BASE_MODEL = "Qwen/Qwen2.5-1.5B-Instruct"

# Gate policy: partial passages still ground an answer (they address the query
# incompletely), only `irrelevant` is rejected. A stricter policy would hide
# useful context and is a one-line change via `accept_labels`.
DEFAULT_ACCEPT_LABELS: tuple[str, ...] = (PARTIAL, RELEVANT)

DEFAULT_CHUNK_SIZE = 220  # words, not characters: a chunk that cuts mid-word
DEFAULT_CHUNK_OVERLAP = 40  # reads as a bug in the GUI trace

NO_GROUNDED_ANSWER = (
    "NO GROUNDED ANSWER - no retrieved passage was judged relevant enough to "
    "support an answer, so nothing was generated."
)

# Generation prompt. Deliberately NOT PROMPT_TEMPLATE: core rule 5 governs the
# classification prompt (baseline vs fine-tuned comparison), while answering is
# a different task with its own instructions.
REFUSAL_TEXT = "I don't have enough information in the context to answer that."
ANSWER_TEMPLATE = (
    "You are a careful retrieval assistant. Answer the question using ONLY the "
    "numbered context passages below. Cite the passage numbers you used. If the "
    f"context does not contain the answer, reply exactly: {REFUSAL_TEXT}\n\n"
    "Context:\n{context}\n\n"
    "Question: {query}\nAnswer:"
)

_LOCAL_ONLY_MARKERS = (
    "openai",
    "anthropic",
    "cohere",
    "voyage",
    "azure",
    "gemini",
    "vertex",
    "together",
    "http",
)


def _require_local_model(name: str) -> str:
    """Fail loudly on a hosted embedding provider.

    The project runs free (Colab T4 / HF Space CPU) and must never need an API
    key. Checking the name is the cheapest possible guard and catches the
    realistic mistake of pasting a provider id into `embedding_model_name`.
    """
    lowered = name.lower()
    if "://" in lowered or any(marker in lowered for marker in _LOCAL_ONLY_MARKERS):
        raise ValueError(
            f"{name!r} is not a local sentence-transformers model. This project "
            f"uses no hosted embedding API (default: {DEFAULT_EMBEDDING_MODEL})."
        )
    return name


@dataclass(frozen=True)
class Chunk:
    """One retrievable span of one input document."""

    chunk_id: int
    text: str
    source: str = "doc"
    ordinal: int = 0

    def to_dict(self) -> dict[str, Any]:
        return {
            "chunk_id": self.chunk_id,
            "text": self.text,
            "source": self.source,
            "ordinal": self.ordinal,
        }


@dataclass(frozen=True)
class GradedPassage:
    """A retrieved passage plus the gate's verdict on it.

    `score` and `confidence` mirror ScoredPassage (the fine-tune derives
    confidence from the label-token softmax, so the two coincide by
    construction). `retrieval_score` is kept separate: the embedding similarity
    and the grader's judgement are independent signals, and conflating them
    would hide exactly the failure this demo is about.
    """

    chunk: Chunk
    retrieval_score: float
    label: str
    score: float
    confidence: float
    accepted: bool
    rank: int = 0

    def to_dict(self) -> dict[str, Any]:
        return {
            "chunk_id": self.chunk.chunk_id,
            "source": self.chunk.source,
            "text": self.chunk.text,
            "retrieval_score": float(self.retrieval_score),
            "label": self.label,
            "score": float(self.score),
            "confidence": float(self.confidence),
            "accepted": bool(self.accepted),
            "rank": int(self.rank),
        }


@dataclass(frozen=True)
class AnswerResult:
    """What the pipeline decided, in a form the GUI can render without logic.

    `graded` is empty in the OFF arm: no labels were produced, so inventing them
    would be a lie. `retrieved` always carries the ungated retriever output so
    the OFF arm can still show what it answered from.
    """

    query: str
    answer: str
    grounded: bool
    used_grader: bool
    retrieved: list[tuple[Chunk, float]] = field(default_factory=list)
    graded: list[GradedPassage] = field(default_factory=list)
    context_ids: list[int] = field(default_factory=list)
    generator_id: str = ""
    note: str = ""

    @property
    def kept_passages(self) -> list[GradedPassage]:
        return [g for g in self.graded if g.accepted]

    def to_dict(self) -> dict[str, Any]:
        """Plain JSON-serializable payload (str/int/float/bool/list/dict only)."""
        return {
            "query": self.query,
            "answer": self.answer,
            "grounded": bool(self.grounded),
            "used_grader": bool(self.used_grader),
            "generator_id": self.generator_id,
            "note": self.note,
            "n_retrieved": len(self.retrieved),
            "n_kept": len(self.context_ids),
            "context_ids": [int(i) for i in self.context_ids],
            "retrieved": [
                {
                    "chunk_id": int(chunk.chunk_id),
                    "source": chunk.source,
                    "score": float(score),
                    "text": chunk.text,
                }
                for chunk, score in self.retrieved
            ],
            "graded": [g.to_dict() for g in self.graded],
        }


def chunk_documents(
    texts: Sequence[str],
    chunk_size: int = DEFAULT_CHUNK_SIZE,
    overlap: int = DEFAULT_CHUNK_OVERLAP,
    sources: Sequence[str] | None = None,
) -> list[Chunk]:
    """Split documents into overlapping word-window chunks.

    Word windows because the GUI shows raw chunk text to a human reviewer.
    `overlap < chunk_size` is enforced so the cursor always advances — an
    overlap equal to the window would loop forever.
    """
    if chunk_size <= 0:
        raise ValueError(f"chunk_size must be positive, got {chunk_size}")
    if overlap < 0 or overlap >= chunk_size:
        raise ValueError(
            f"overlap must be in [0, chunk_size) to guarantee progress, got "
            f"overlap={overlap}, chunk_size={chunk_size}"
        )
    if sources is not None and len(sources) != len(texts):
        raise ValueError(
            f"sources must align with texts: {len(sources)} vs {len(texts)}"
        )

    step = chunk_size - overlap
    chunks: list[Chunk] = []
    for doc_pos, raw in enumerate(texts):
        words = raw.split()
        if not words:
            continue
        source = sources[doc_pos] if sources is not None else f"doc-{doc_pos}"
        for ordinal, start in enumerate(range(0, len(words), step)):
            window = words[start : start + chunk_size]
            if not window:
                break
            chunks.append(
                Chunk(
                    chunk_id=len(chunks),
                    text=" ".join(window),
                    source=source,
                    ordinal=ordinal,
                )
            )
            if start + chunk_size >= len(words):
                break
    return chunks


class SentenceTransformerEmbedder:
    """Local sentence-transformers embedder.

    The model is loaded on first `encode()` so constructing the object is free —
    the smoke test and the app both create it before deciding whether a model
    run is happening at all.
    """

    def __init__(
        self,
        model_name: str = DEFAULT_EMBEDDING_MODEL,
        device: str | None = None,
        batch_size: int = 64,
    ) -> None:
        self.model_name = _require_local_model(model_name)
        self.device = device
        self.batch_size = batch_size
        self._model: Any = None

    def _ensure(self) -> Any:
        if self._model is None:
            from sentence_transformers import SentenceTransformer

            self._model = SentenceTransformer(
                self.model_name, device=self.device, trust_remote_code=False
            )
        return self._model

    @property
    def dimension(self) -> int:
        return int(self._ensure().get_sentence_embedding_dimension())

    def encode(self, texts: Sequence[str]) -> Any:
        """L2-normalised float32 vectors — normalisation is what makes the
        FAISS inner product equal to cosine similarity."""
        import numpy as np

        if not texts:
            return np.zeros((0, self.dimension), dtype="float32")
        vectors = self._ensure().encode(
            list(texts),
            batch_size=self.batch_size,
            convert_to_numpy=True,
            normalize_embeddings=True,
            show_progress_bar=False,
        )
        return np.asarray(vectors, dtype="float32")


@dataclass(frozen=True, eq=False)
class RagIndex:
    """A built index plus the embedder needed to encode queries against it."""

    chunks: tuple[Chunk, ...]
    vectors: Any  # np.ndarray (n, dim) float32, L2-normalised
    embedder: SentenceTransformerEmbedder
    faiss_index: Any = None
    backend: str = "faiss.IndexFlatIP"
    embedding_model: str = DEFAULT_EMBEDDING_MODEL

    def __len__(self) -> int:
        return len(self.chunks)


def build_index(
    chunks: Sequence[Chunk | str],
    embedding_model_name: str = DEFAULT_EMBEDDING_MODEL,
    embedder: SentenceTransformerEmbedder | None = None,
) -> RagIndex:
    """Encode chunks into a FAISS flat inner-product index.

    Accepts `Chunk` objects or raw strings (a string becomes one chunk), so a
    caller that already chunked — the sample corpora in the Space — does not
    have to re-split its text. Nothing here re-chunks.

    Flat (exact) search, not IVF: the demo corpora are tens of chunks and an
    exact index keeps the retrieval trace explainable. The backend name is kept
    on the object so the GUI can state what it searched.
    """
    import faiss
    import numpy as np

    chunk_list = [
        item
        if isinstance(item, Chunk)
        else Chunk(chunk_id=i, text=str(item), source=f"doc-{i}", ordinal=i)
        for i, item in enumerate(chunks)
    ]
    embedder = embedder or SentenceTransformerEmbedder(embedding_model_name)
    started = time.perf_counter()
    vectors = embedder.encode([c.text for c in chunk_list])
    faiss_index = None
    if chunk_list:
        faiss_index = faiss.IndexFlatIP(vectors.shape[1])
        faiss_index.add(np.ascontiguousarray(vectors))
    LOG.info(
        "index built: chunks=%d dim=%s model=%s backend=%s in %.2fs",
        len(chunk_list),
        vectors.shape[1] if vectors.ndim == 2 else "?",
        embedder.model_name,
        "faiss.IndexFlatIP" if faiss_index is not None else "none",
        time.perf_counter() - started,
    )
    return RagIndex(
        chunks=tuple(chunk_list),
        vectors=vectors,
        embedder=embedder,
        faiss_index=faiss_index,
        embedding_model=embedder.model_name,
    )


def retrieve(index: RagIndex, query: str, top_k: int = 5) -> list[tuple[Chunk, float]]:
    """Top-k (chunk, score) pairs, best first. Score is cosine similarity."""
    if top_k <= 0:
        raise ValueError(f"top_k must be positive, got {top_k}")
    if not index.chunks or index.faiss_index is None:
        return []
    scores, ids = index.faiss_index.search(index.embedder.encode([query]), top_k)
    hits: list[tuple[Chunk, float]] = []
    for score, chunk_pos in zip(scores[0], ids[0]):
        if int(chunk_pos) < 0:  # FAISS pads with -1 when fewer hits than top_k
            continue
        hits.append((index.chunks[int(chunk_pos)], float(score)))
    LOG.debug(
        "retrieve: top_k=%d hits=%d best=%.4f", top_k, len(hits), hits[0][1] if hits else 0.0
    )
    return hits


@dataclass
class LocalQwenModel:
    """A loaded local Qwen model (+ optional QLoRA adapter).

    One instance backs both the grader and the generator so a Space loads 1.5B
    parameters once. `load_error` records why an adapter was skipped — the GUI
    shows it instead of silently pretending the fine-tune is running.
    """

    model: Any
    tokenizer: Any
    base_model_id: str
    device: str
    quantized: bool
    using_adapter: bool
    adapter_ref: str | None = None
    label_ids: tuple[int, ...] = ()
    load_error: str | None = None

    def describe(self) -> str:
        """One-line provenance for the UI: what is actually loaded right now."""
        precision = "4-bit NF4" if self.quantized else "unquantized"
        if self.using_adapter and self.adapter_ref:
            return f"{self.base_model_id} + adapter {self.adapter_ref} ({precision})"
        return f"{self.base_model_id} — zero-shot base model, no adapter ({precision})"

    def score_labels(self, prompts: Sequence[str], batch_size: int = 8) -> list[tuple[str, float]]:
        """Constrained scoring: softmax over the three label-token logits.

        Same protocol as notebook 02 — argmax over label tokens instead of free
        generation, which is what makes confidence a real probability.
        """
        import torch

        out: list[tuple[str, float]] = []
        with torch.no_grad():
            for start in range(0, len(prompts), batch_size):
                batch = list(prompts[start : start + batch_size])
                enc = self.tokenizer(
                    batch,
                    return_tensors="pt",
                    padding=True,
                    truncation=True,
                    max_length=1024,
                ).to(self.model.device)
                logits = self.model(**enc).logits[:, -1, :]
                label_logits = logits[:, list(self.label_ids)]
                probs = torch.softmax(label_logits.float(), dim=-1)
                best = probs.argmax(dim=-1).tolist()
                out += [
                    (LABELS[pos], float(probs[row, pos]))
                    for row, pos in enumerate(best)
                ]
        return out

    def generate_text(self, prompt: str, max_new_tokens: int = 180) -> str:
        """Greedy continuation. No sampling — the demo must be reproducible."""
        import torch

        # `return_tensors="truncation"` is a typo that transformers coerces into
        # TensorType and rejects; it raised inside the Space's ZeroGPU worker,
        # where `spaces` strips the message and reports a bare RuntimeError.
        # Same call shape as the grading path above.
        enc = self.tokenizer(
            prompt, return_tensors="pt", truncation=True, max_length=3072
        )
        enc = {k: v.to(self.model.device) for k, v in enc.items()}
        with torch.no_grad():
            out = self.model.generate(
                **enc,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                pad_token_id=self.tokenizer.pad_token_id,
            )
        generated = out[0, enc["input_ids"].shape[1] :]
        return self.tokenizer.decode(generated, skip_special_tokens=True).strip()


def _repo_root() -> Path | None:
    here = Path(__file__).resolve()
    for parent in (here.parent, *here.parents):
        if (parent / "src").is_dir() and (parent / "config").is_dir():
            return parent
    return None


def resolve_adapter(base_model: str | None = None) -> tuple[str | None, str]:
    """Find an adapter to load, preferring what is already on disk.

    Order: config/sft_config.yaml's `hf_model_id` when `outputs/qlora-sft`
    exists locally, else the Hub id from the same config. Returns
    `(adapter_ref, note)`; `(None, why)` means run the base model zero-shot.
    Never raises — the pipeline must work before the fine-tune lands.
    """
    cfg_base, cfg_hub_id = base_model, None
    try:
        from .config import load_config

        cfg = load_config()
        cfg_base = cfg_base or cfg.model.base_model
        cfg_hub_id = cfg.experiment.hf_model_id
    except Exception:  # noqa: BLE001 — config is optional for the demo path
        pass

    root = _repo_root()
    local_adapter = root / "outputs" / "qlora-sft" if root else None
    if local_adapter is not None and (local_adapter / "adapter_config.json").is_file():
        LOG.info("adapter resolved: %s (local outputs/qlora-sft)", local_adapter)
        return str(local_adapter), "local outputs/qlora-sft"
    if cfg_hub_id:
        LOG.info("adapter resolved: %s (config/sft_config.yaml hf_model_id)", cfg_hub_id)
        return cfg_hub_id, "config/sft_config.yaml hf_model_id"
    LOG.warning("no adapter found; zero-shot base model (%s)", cfg_base)
    return None, f"no adapter found; using base model zero-shot ({cfg_base})"


def _label_token_ids(tokenizer: Any) -> tuple[int, ...]:
    """Label tokens must be single tokens for constrained scoring to be valid.

    Same guard as notebook 02: if Qwen's tokenizer ever splits a label, fail
    loudly rather than scoring the wrong positions.
    """
    ids: list[int] = []
    for label in LABELS:
        encoded = tokenizer.encode(" " + label, add_special_tokens=False)
        if len(encoded) != 1:
            raise RuntimeError(
                f"Label {label!r} tokenizes to {len(encoded)} tokens; constrained "
                "label-logit scoring is invalid for this tokenizer."
            )
        ids.append(encoded[0])
    return tuple(ids)


def zerogpu_active() -> bool:
    """True when the platform is scheduling ZeroGPU workers for this process.

    Read from `SPACES_ZERO_GPU`, the env var `spaces/config.py` keys
    `Config.zero_gpu` off — deliberately not from `import spaces` succeeding, and
    not from `spaces.Config` either, which is package-internal and not exported.
    The import test is the bug this replaces: the `spaces` package ships on CPU,
    T4 and A10G images too and `spaces.GPU` is a no-op there, so "the import
    worked" silently wired ZeroGPU scheduling onto Colab and local runs.

    The loader below needs the answer because HF patches `torch.cuda` on ZeroGPU,
    which makes `torch.cuda.is_available()` lie outside `@spaces.GPU` and changes
    which `device_map` is correct.
    """
    return os.environ.get("SPACES_ZERO_GPU", "").lower() in ("1", "t", "true")


def load_local_model(
    adapter: str | None = None,
    base_model: str | None = None,
    prefer_4bit: bool = True,
    load_adapter: bool = True,
) -> LocalQwenModel:
    """Load Qwen (zero-shot) or Qwen + QLoRA adapter, falling back cleanly.

    4-bit NF4 needs CUDA + bitsandbytes, so on CPU the model loads unquantized
    — the Space's no-GPU mode is slow but correct. If the adapter is missing or
    fails to apply, the base model is reloaded and the reason is recorded.
    """
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    cfg_base = base_model
    if cfg_base is None:
        try:
            from .config import load_config

            cfg_base = load_config().model.base_model
        except Exception:  # noqa: BLE001 — demo path without config
            cfg_base = DEFAULT_BASE_MODEL
    model_id = cfg_base or DEFAULT_BASE_MODEL

    tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"  # so logits[:, -1] is the last real token

    use_4bit = bool(prefer_4bit and torch.cuda.is_available())
    zerogpu = zerogpu_active()
    kwargs: dict[str, Any] = {"trust_remote_code": True}
    if use_4bit:
        from transformers import BitsAndBytesConfig

        kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_use_double_quant=True,
        )
        kwargs["dtype"] = torch.float16
        # "cuda", not "auto". A bare device string is normalised to {"": cuda} and
        # keeps every layer on the GPU, whereas "auto" silently offloads to CPU on
        # VRAM pressure and, on ZeroGPU, routes through accelerate's
        # set_module_tensor_to_device -> torch._C._cuda_init() at load time.
        # transformers/modeling_utils.py does the normalisation.
        kwargs["device_map"] = "cuda" if zerogpu else "auto"
    else:
        kwargs["dtype"] = torch.float32 if not torch.cuda.is_available() else torch.float16

    LOG.info(
        "loading model: id=%s cuda=%s 4bit=%s zerogpu=%s device_map=%s load_adapter=%s",
        model_id,
        torch.cuda.is_available(),
        use_4bit,
        zerogpu,
        kwargs.get("device_map", "<none>"),
        load_adapter,
    )

    def _load_base() -> Any:
        """Load the backbone, degrading 4-bit -> unquantized if the quantised
        path fails.

        On ZeroGPU `torch.cuda.is_available()` is True, so this always takes the
        bitsandbytes 4-bit branch. That combination is the fragile one (the
        worker can fail to attach the quantised weights), and a bare failure here
        propagated straight out of the `@spaces.GPU` function -- where `spaces`
        discards the message, so the visitor only ever saw `Error: 'RuntimeError'`.
        Retrying unquantized costs RAM, not correctness, so it is always worth one
        attempt before giving up.
        """
        nonlocal kwargs, use_4bit
        try:
            loaded = AutoModelForCausalLM.from_pretrained(model_id, **kwargs)
        except Exception as exc:  # noqa: BLE001 — any quantised-load failure degrades
            if not use_4bit:
                raise
            LOG.warning(
                "4-bit load failed (%s: %s); retrying unquantized",
                type(exc).__name__,
                exc,
                exc_info=True,
            )
            use_4bit = False
            kwargs = {"trust_remote_code": True, "dtype": torch.float16}
            loaded = AutoModelForCausalLM.from_pretrained(model_id, **kwargs)
        if not use_4bit:
            loaded = loaded.to("cuda" if torch.cuda.is_available() else "cpu")
        loaded.eval()
        return loaded

    model = _load_base()
    adapter_ref: str | None = None
    load_error: str | None = None
    if load_adapter:
        candidate = adapter if adapter is not None else resolve_adapter(model_id)[0]
        if candidate:
            try:
                from peft import PeftModel

                LOG.info("applying adapter: %s", candidate)
                model = PeftModel.from_pretrained(model, candidate)
                model.eval()
                adapter_ref = candidate
            except Exception as exc:  # noqa: BLE001 — any adapter failure degrades to zero-shot
                load_error = (
                    f"adapter {candidate!r} unavailable "
                    f"({type(exc).__name__}: {exc}); using zero-shot base model"
                )
                LOG.warning(
                    "adapter failed (%s: %s); reloading base model zero-shot",
                    type(exc).__name__,
                    exc,
                    exc_info=True,
                )
                del model
                model = _load_base()
    else:
        LOG.info("adapter loading disabled by caller; zero-shot base model")

    device = str(getattr(model, "device", "unknown"))
    bundle = LocalQwenModel(
        model=model,
        tokenizer=tokenizer,
        base_model_id=model_id,
        device=device,
        quantized=use_4bit,
        using_adapter=adapter_ref is not None,
        adapter_ref=adapter_ref,
        label_ids=_label_token_ids(tokenizer),
        load_error=load_error,
    )
    LOG.info("model ready: %s", bundle.describe())
    return bundle


class HFRelevanceGrader(GraderInterface):
    """GraderInterface over the local Qwen model.

    Wraps whatever `LocalQwenModel` it is given so the app shares one load with
    the generator, and constructs its own only when none is supplied.
    """

    def __init__(
        self,
        bundle: LocalQwenModel | None = None,
        adapter: str | None = None,
        base_model: str | None = None,
    ) -> None:
        self.bundle = bundle or load_local_model(adapter=adapter, base_model=base_model)

    @property
    def name(self) -> str:
        return self.bundle.describe()

    def grade(
        self, query: str, passages: list[str], top_k: int = 5
    ) -> list[ScoredPassage]:
        if not passages:
            return []
        prompts = [
            PROMPT_TEMPLATE.format(query=query.strip(), passage=p.strip())
            for p in passages
        ]
        labelled = self.bundle.score_labels(prompts)
        scored = [
            ScoredPassage(
                passage=passage,
                label=label,
                score=confidence,
                confidence=confidence,
            )
            for passage, (label, confidence) in zip(passages, labelled)
        ]
        # `top_k` truncation is the interface's contract, not ours to second-guess.
        scored.sort(key=lambda s: s.score, reverse=True)
        return scored[:top_k]


class QwenGenerator:
    """Grounded answer generation from the same local model as the grader."""

    name = "qwen-local"

    def __init__(
        self,
        bundle: LocalQwenModel | None = None,
        max_new_tokens: int = 180,
        adapter: str | None = None,
        base_model: str | None = None,
    ) -> None:
        self.bundle = bundle or load_local_model(adapter=adapter, base_model=base_model)
        self.max_new_tokens = max_new_tokens

    def generate(self, query: str, passages: Sequence[Chunk | str]) -> str:
        """Answer from `passages` only. Never sees the raw user corpus."""
        lines = [
            f"[{i}] {p.text if isinstance(p, Chunk) else str(p)}"
            for i, p in enumerate(passages, start=1)
        ]
        prompt = ANSWER_TEMPLATE.format(
            context="\n\n".join(lines) or "(no context)", query=query.strip()
        )
        return self.bundle.generate_text(prompt, self.max_new_tokens)


def _as_chunk(item: Chunk | str, position: int) -> Chunk:
    return item if isinstance(item, Chunk) else Chunk(chunk_id=position, text=str(item))


def grade_passages(
    query: str,
    passages: Sequence[Chunk | str],
    grader: GraderInterface | None = None,
    accept_labels: Sequence[str] = DEFAULT_ACCEPT_LABELS,
    retrieval_scores: Sequence[float] | None = None,
) -> list[GradedPassage]:
    """Grade retrieved passages and mark accept/reject, best first.

    `grader=None` builds the default local Qwen grader (which loads a model).
    Every passage is submitted with `top_k=len(passages)` — a silent truncation
    would leave chunks in the trace with no verdict, which reads as a bug.

    Results are matched back to chunks by passage text rather than position
    because GraderInterface.grade() is free to sort. Identical passages share a
    chunk id: they are indistinguishable to the grader anyway.
    """
    chunks = [_as_chunk(p, i) for i, p in enumerate(passages)]
    if not chunks:
        return []
    scores = list(retrieval_scores) if retrieval_scores is not None else [0.0] * len(chunks)
    if len(scores) != len(chunks):
        raise ValueError(
            f"retrieval_scores must align with passages: {len(scores)} vs {len(chunks)}"
        )
    resolved = grader if grader is not None else HFRelevanceGrader()
    texts = [c.text for c in chunks]
    scored = resolved.grade(query, texts, top_k=len(texts))

    by_text: dict[str, Chunk] = {}
    for chunk in chunks:
        by_text.setdefault(chunk.text, chunk)

    graded: list[GradedPassage] = []
    for item in scored:
        chunk = by_text.get(item.passage)
        if chunk is None:
            continue
        position = next(
            (i for i, c in enumerate(chunks) if c is chunk),
            0,
        )
        graded.append(
            GradedPassage(
                chunk=chunk,
                retrieval_score=float(scores[position]),
                label=item.label,
                score=float(item.score),
                confidence=float(item.confidence),
                accepted=item.label in tuple(accept_labels),
            )
        )
    graded.sort(key=lambda g: g.score, reverse=True)
    LOG.info(
        "gate: kept %d/%d | %s",
        sum(1 for g in graded if g.accepted),
        len(graded),
        ", ".join(f"{g.label}@{g.confidence:.2f}" for g in graded) or "no verdicts",
    )
    return [
        GradedPassage(
            chunk=g.chunk,
            retrieval_score=g.retrieval_score,
            label=g.label,
            score=g.score,
            confidence=g.confidence,
            accepted=g.accepted,
            rank=i + 1,
        )
        for i, g in enumerate(graded)
    ]


def answer_with_gate(
    query: str,
    retrieved: Sequence[tuple[Chunk, float]],
    use_grader: bool,
    generator: QwenGenerator | None = None,
    *,
    grader: GraderInterface | None = None,
    accept_labels: Sequence[str] = DEFAULT_ACCEPT_LABELS,
) -> AnswerResult:
    """Answer a query with the relevance gate ON or OFF.

    `use_grader=False` skips grading entirely — no grader is constructed, called
    or reported — and answers from every retrieved passage (the naive-RAG arm).
    `use_grader=True` grades, keeps only accepted passages, and returns
    NO_GROUNDED_ANSWER without calling the generator if nothing survives.
    """
    pairs = [(chunk, float(score)) for chunk, score in retrieved]
    generator_id = (
        getattr(generator, "name", type(generator).__name__)
        if generator is not None
        else QwenGenerator.name
    )

    LOG.info("answer: grader=%s retrieved=%d", use_grader, len(pairs))

    if not pairs:
        return AnswerResult(
            query=query,
            answer=NO_GROUNDED_ANSWER,
            grounded=False,
            used_grader=use_grader,
            retrieved=[],
            graded=[],
            context_ids=[],
            generator_id=generator_id,
            note="nothing was retrieved for this query",
        )

    graded: list[GradedPassage] = []
    if not use_grader:
        context = [chunk for chunk, _ in pairs]
    else:
        graded = grade_passages(
            query,
            [chunk for chunk, _ in pairs],
            grader=grader,
            accept_labels=accept_labels,
            retrieval_scores=[score for _, score in pairs],
        )
        kept = [g for g in graded if g.accepted]
        if not kept:
            LOG.warning(
                "gate rejected all %d retrieved passage(s); generator not called",
                len(graded),
            )
            return AnswerResult(
                query=query,
                answer=NO_GROUNDED_ANSWER,
                grounded=False,
                used_grader=True,
                retrieved=pairs,
                graded=graded,
                context_ids=[],
                generator_id=generator_id,
                note=(
                    f"gate rejected all {len(graded)} retrieved passage(s); "
                    "the generator was not called"
                ),
            )
        context = [g.chunk for g in kept]

    resolved_generator = generator if generator is not None else QwenGenerator()
    answer = resolved_generator.generate(query, context).strip()
    grounded = bool(answer) and REFUSAL_TEXT.lower() not in answer.lower()
    note = "" if grounded else "generator declined to answer from this context"
    LOG.info(
        "answer done: grounded=%s context_passages=%d answer_chars=%d %s",
        grounded,
        len(context),
        len(answer),
        note or f"generator={generator_id}",
    )
    return AnswerResult(
        query=query,
        answer=answer or NO_GROUNDED_ANSWER,
        grounded=grounded,
        used_grader=use_grader,
        retrieved=pairs,
        graded=graded,
        context_ids=[c.chunk_id for c in context],
        generator_id=getattr(
            resolved_generator, "name", type(resolved_generator).__name__
        ),
        note=note,
    )


def load_metrics(results_dir: str | Path | None = None) -> dict[str, Any]:
    """Read the benchmark JSONs at runtime.

    The GUI is not allowed to contain a hardcoded score: every number it shows
    comes from `results/*.json`, written by notebooks 02 and 04 (core rules 4
    and 11). A missing file is reported in `missing`, never substituted.
    """
    directory = Path(results_dir) if results_dir else None
    if directory is None:
        root = _repo_root()
        directory = (root / "results") if root else None

    payload: dict[str, Any] = {"baseline": None, "finetuned": None, "missing": [], "sources": {}}
    if directory is None or not directory.is_dir():
        payload["missing"] = ["results directory not found"]
        return payload

    for key, filename in (
        ("baseline", "baseline_results.json"),
        ("finetuned", "finetuned_results.json"),
    ):
        path = directory / filename
        if not path.is_file():
            payload["missing"].append(filename)
            continue
        try:
            payload[key] = json.loads(path.read_text(encoding="utf-8"))
            payload["sources"][key] = str(path)
        except json.JSONDecodeError as exc:
            payload["missing"].append(f"{filename} (unreadable: {exc})")
    LOG.info(
        "metrics: baseline=%s finetuned=%s missing=%s",
        "found" if payload["baseline"] else "missing",
        "found" if payload["finetuned"] else "missing",
        payload["missing"] or "none",
    )
    return payload


def before_after_rows(metrics: dict[str, Any]) -> list[dict[str, Any]]:
    """Build the before/after rows from loaded metrics.

    Only metrics present in BOTH runs are emitted — a row with a missing side
    is noise. Per-class F1 is included because class imbalance is expected and
    macro-F1 alone would hide the thin `partial` class (core rule 6).
    """
    baseline = metrics.get("baseline") or {}
    finetuned = metrics.get("finetuned") or {}
    if not baseline or not finetuned:
        return []

    rows: list[dict[str, Any]] = []

    def add(metric: str, before: float | None, after: float | None) -> None:
        if before is None or after is None:
            return
        rows.append(
            {
                "metric": metric,
                "baseline": round(float(before), 4),
                "finetuned": round(float(after), 4),
                "delta": round(float(after) - float(before), 4),
            }
        )

    add("accuracy", baseline.get("accuracy"), finetuned.get("accuracy"))
    add("macro-F1", baseline.get("macro_f1"), finetuned.get("macro_f1"))
    for label in LABELS:
        add(
            f"F1 — {label}",
            (baseline.get("per_class") or {}).get(label, {}).get("f1"),
            (finetuned.get("per_class") or {}).get(label, {}).get("f1"),
        )
    return rows
