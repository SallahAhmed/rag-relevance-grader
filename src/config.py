"""Single source of truth for hyperparameters.

`config/sft_config.yaml` is canonical. This module exposes it as typed,
immutable dataclasses so a typo'd key fails fast instead of silently
training with a wrong value. Notebooks must load config from here —
never hardcode hyperparameters inline.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml


@dataclass(frozen=True)
class ModelConfig:
    base_model: str
    load_in_4bit: bool = True
    quantization: str = "nf4"
    compute_dtype: str = "float16"
    double_quant: bool = True


@dataclass(frozen=True)
class LoRAConfig:
    r: int = 16
    alpha: int = 32
    dropout: float = 0.05
    target_modules: tuple = (
        "q_proj",
        "k_proj",
        "v_proj",
        "o_proj",
        "gate_proj",
        "up_proj",
        "down_proj",
    )


@dataclass(frozen=True)
class TrainingConfig:
    learning_rate: float = 2e-4
    lr_scheduler: str = "cosine"
    warmup_ratio: float = 0.05
    epochs: int = 1
    per_device_batch_size: int = 4
    gradient_accumulation_steps: int = 4
    max_seq_length: int = 1024
    gradient_checkpointing: bool = True
    optimizer: str = "paged_adamw_8bit"
    seed: int = 42

    @property
    def effective_batch_size(self) -> int:
        return self.per_device_batch_size * self.gradient_accumulation_steps


@dataclass(frozen=True)
class EvalConfig:
    split_strategy: str = "group_by_query"
    test_queries: int = 10
    mmlu_forgetting_bar: str = "fine-tuned >= 90% of baseline"
    metrics: tuple = ("accuracy", "macro_f1", "per_class_f1", "confusion_matrix")


@dataclass(frozen=True)
class DatasetConfig:
    source: str = "TREC_DL_2020"
    bucket_plan: str = "0->irrelevant, 1->partial, 2-3->relevant"
    format: str = "query, passage, label"  # noqa: A003 — mirrors the YAML key verbatim


@dataclass(frozen=True)
class ExperimentConfig:
    wandb_project: str = "rag-relevance-grader"
    wandb_entity: str = "sallahahmed"
    hf_model_id: str = "sallahahmed/qwen2.5-1.5b-relevance-grader"


@dataclass(frozen=True)
class SFTConfig:
    """Root config object. Immutable — a run's config never changes mid-run."""

    model: ModelConfig
    lora: LoRAConfig
    training: TrainingConfig
    evaluation: EvalConfig
    dataset: DatasetConfig
    experiment: ExperimentConfig


def default_config_path() -> Path:
    """Resolve config/sft_config.yaml from anywhere inside the repo."""
    here = Path(__file__).resolve()
    for parent in (here.parent, *here.parents):
        candidate = parent / "config" / "sft_config.yaml"
        if candidate.is_file():
            return candidate
    raise FileNotFoundError("config/sft_config.yaml not found above " + str(here))


def _coerce(mapping: dict, key: str, kind, section: str):
    """YAML trap: `2e-4` parses as str. Coerce numerics loudly or fail."""
    if key not in mapping:
        return mapping
    try:
        mapping[key] = kind(mapping[key])
    except (TypeError, ValueError) as exc:
        raise ValueError(f"[{section}] {key}={mapping[key]!r} is not a {kind.__name__}") from exc
    return mapping


def load_config(path: str | Path | None = None) -> SFTConfig:
    """Load and validate the YAML config into typed dataclasses."""
    cfg_path = Path(path) if path else default_config_path()
    raw: dict[str, Any] = yaml.safe_load(cfg_path.read_text(encoding="utf-8"))

    try:
        model = ModelConfig(**raw.get("model", {}))
        lora_raw = dict(raw.get("lora", {}))
        lora_raw["target_modules"] = tuple(
            lora_raw.get("target_modules", LoRAConfig.target_modules)
        )
        for key in ("r", "alpha"):
            _coerce(lora_raw, key, int, "lora")
        _coerce(lora_raw, "dropout", float, "lora")
        lora = LoRAConfig(**lora_raw)
        train_raw = dict(raw.get("training", {}))
        for key in ("epochs", "per_device_batch_size", "gradient_accumulation_steps",
                    "max_seq_length", "seed"):
            _coerce(train_raw, key, int, "training")
        for key in ("learning_rate", "warmup_ratio"):
            _coerce(train_raw, key, float, "training")
        training = TrainingConfig(**train_raw)
        eval_raw = dict(raw.get("evaluation", {}))
        if "metrics" in eval_raw:
            eval_raw["metrics"] = tuple(eval_raw["metrics"])
        evaluation = EvalConfig(**eval_raw)
        dataset = DatasetConfig(**raw.get("dataset", {}))
        experiment = ExperimentConfig(**raw.get("experiment", {}))
    except TypeError as exc:
        raise ValueError(f"Invalid key in {cfg_path}: {exc}") from exc

    if evaluation.split_strategy != "group_by_query":
        raise ValueError(
            "split_strategy must be 'group_by_query' — row-wise splits leak "
            "queries between train and test (see project-b-relevance-grader.md)."
        )
    return SFTConfig(
        model=model,
        lora=lora,
        training=training,
        evaluation=evaluation,
        dataset=dataset,
        experiment=experiment,
    )
