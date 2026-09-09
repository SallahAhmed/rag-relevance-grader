"""Evaluation metrics for the relevance grader.

Every number in the README before/after table comes from here:
  accuracy, macro-F1, per-class precision/recall/F1, confusion matrix.
Results are saved as JSON (notebooks 02 and 04) so the table is generated,
never hand-typed.

The catastrophic-forgetting bar is RELATIVE by design:
fine-tuned MMLU >= 90% of the notebook-02 baseline. A fixed >=25% floor is
random chance on 4-choice questions and proves nothing.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_recall_fscore_support,
)

from .dataset_utils import LABELS

FORGETTING_RETENTION_BAR = 0.90  # fine-tuned MMLU must keep 90% of baseline


def compute_classification_metrics(
    y_true: list[str],
    y_pred: list[str],
    labels: tuple[str, ...] = LABELS,
) -> dict[str, Any]:
    """Score one eval run. Raises on empty input or unknown labels."""
    if not y_true or len(y_true) != len(y_pred):
        raise ValueError("y_true and y_pred must be non-empty and equal length.")
    unknown = (set(y_true) | set(y_pred)) - set(labels)
    if unknown:
        raise ValueError(f"Labels outside {labels}: {sorted(unknown)}")

    precision, recall, f1, support = precision_recall_fscore_support(
        y_true, y_pred, labels=list(labels), zero_division=0
    )
    return {
        "n": len(y_true),
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "macro_f1": float(f1_score(y_true, y_pred, average="macro")),
        "per_class": {
            label: {
                "precision": float(precision[i]),
                "recall": float(recall[i]),
                "f1": float(f1[i]),
                "support": int(support[i]),
            }
            for i, label in enumerate(labels)
        },
        "confusion_matrix": {
            "labels": list(labels),
            "matrix": confusion_matrix(y_true, y_pred, labels=list(labels)).tolist(),
        },
    }


def passes_forgetting_check(baseline_mmlu: float, finetuned_mmlu: float) -> dict[str, Any]:
    """Relative catastrophic-forgetting verdict with the numbers attached."""
    if baseline_mmlu <= 0:
        raise ValueError("baseline_mmlu must be positive to form a ratio.")
    retention = finetuned_mmlu / baseline_mmlu
    return {
        "baseline_mmlu": float(baseline_mmlu),
        "finetuned_mmlu": float(finetuned_mmlu),
        "retention": float(retention),
        "bar": FORGETTING_RETENTION_BAR,
        "passes": retention >= FORGETTING_RETENTION_BAR,
    }


def save_results_json(path: str | Path, payload: dict[str, Any]) -> Path:
    """Write a results file. Parent dirs are created; keys sorted for diffs."""
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return out


def load_results_json(path: str | Path) -> dict[str, Any]:
    """Read a results file back (used by notebook 04 to build the table)."""
    return json.loads(Path(path).read_text(encoding="utf-8"))
