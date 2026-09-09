"""Minimal CLI: validate the environment and echo the resolved config.

Usage:
    python -m src.main --print-config
    python -m src.main --check-env        # torch / cuda / key packages
    python -m src.main --config config/sft_config.yaml --print-config

Heavy work lives in notebooks/; this CLI exists so CI or a fresh machine
can verify the setup in one command.
"""
from __future__ import annotations

import argparse
import importlib.util
import sys


def check_env() -> dict[str, str]:
    """Report torch/CUDA/package availability without importing heavy deps."""
    report: dict[str, str] = {}
    for package in ("torch", "transformers", "peft", "trl", "datasets", "sklearn", "wandb", "yaml"):
        report[package] = "ok" if importlib.util.find_spec(package) else "MISSING"
    try:
        import torch

        report["torch_version"] = torch.__version__
        report["cuda_available"] = str(torch.cuda.is_available())
        if torch.cuda.is_available():
            report["cuda_device"] = torch.cuda.get_device_name(0)
    except ImportError:
        report["torch_version"] = "not installed"
    return report


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Relevance grader utilities")
    parser.add_argument("--config", default=None, help="Path to sft_config.yaml")
    parser.add_argument("--print-config", action="store_true")
    parser.add_argument("--check-env", action="store_true")
    args = parser.parse_args(argv)

    if args.check_env:
        for key, value in check_env().items():
            print(f"{key}: {value}")

    if args.print_config:
        from .config import load_config

        cfg = load_config(args.config)
        print(f"base_model: {cfg.model.base_model}")
        print(f"lora: r={cfg.lora.r} alpha={cfg.lora.alpha} dropout={cfg.lora.dropout}")
        t = cfg.training
        print(
            f"training: lr={t.learning_rate} epochs={t.epochs} "
            f"batch={t.per_device_batch_size}x{t.gradient_accumulation_steps}"
            f"={t.effective_batch_size} seq={t.max_seq_length} "
            f"grad_ckpt={t.gradient_checkpointing}"
        )
        print(f"eval: split={cfg.evaluation.split_strategy} test_queries={cfg.evaluation.test_queries}")
        print(f"hf_model_id: {cfg.experiment.hf_model_id}")

    if not args.check_env and not args.print_config:
        parser.print_help()
    return 0


if __name__ == "__main__":
    sys.exit(main())
