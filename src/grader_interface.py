"""Pluggable grader contract (v1).

Modeled on ConnexioRag's RerankerInterface and grounded-ai's Evaluator shape,
minus what our training data cannot support:

  grade(query, passages, top_k) -> list[ScoredPassage(score, label, confidence)]

Deliberate v1 limits (see project-b-relevance-grader.md):
  - NO `reasoning` field. TREC qrels contain no reasoning text, so any
    model-generated reasoning would be an untrained side effect.
  - `confidence` is the softmax probability over the label-token logits at
    inference time — derived, not trained.

Any class implementing this interface (zero-shot baseline today, the QLoRA
adapter after notebook 03) is a drop-in reranker for any RAG pipeline.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass

from .dataset_utils import LABELS


@dataclass(frozen=True)
class ScoredPassage:
    """One graded passage. Sorted descending by score by the caller."""

    passage: str
    label: str
    score: float
    confidence: float

    def __post_init__(self) -> None:
        if self.label not in LABELS:
            raise ValueError(f"Unknown label {self.label!r}; expected one of {LABELS}")
        if not 0.0 <= self.confidence <= 1.0:
            raise ValueError(f"confidence must be in [0, 1], got {self.confidence}")


class GraderInterface(ABC):
    """Contract every relevance grader in this project implements."""

    @abstractmethod
    def grade(
        self, query: str, passages: list[str], top_k: int = 5
    ) -> list[ScoredPassage]:
        """Grade passages for one query, best first, truncated to top_k."""
        raise NotImplementedError
