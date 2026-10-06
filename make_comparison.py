"""Build results/finetuned_results.json + results/before_after_table.md.

Run locally after downloading predictions_ft.json + mmlu_ft.json from the VM:
  python make_comparison.py
Inputs: results/predictions_ft.json, results/mmlu_ft.json,
        results/baseline_results.json (committed).
Outputs: results/finetuned_results.json, results/before_after_table.md.
Verdicts: +10pp macro-F1 bar + >=90% MMLU retention bar. Exits non-zero
with the table still written if a bar fails (numbers first, verdicts honest).
"""
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from src.evaluation_utils import (  # noqa: E402
    FORGETTING_RETENTION_BAR,
    compute_classification_metrics,
    passes_forgetting_check,
)

F1_BAR_PP = 0.10  # +10 percentage points macro-F1, pinned


def main() -> int:
    baseline = json.loads((ROOT / "results" / "baseline_results.json").read_text())
    ft_preds = json.loads((ROOT / "results" / "predictions_ft.json").read_text())
    mmlu_ft = json.loads((ROOT / "results" / "mmlu_ft.json").read_text())

    y_true = [p["true"] for p in ft_preds]
    y_pred = [p["pred"] for p in ft_preds]
    metrics = compute_classification_metrics(y_true, y_pred)
    forget = passes_forgetting_check(baseline["mmlu_baseline"], mmlu_ft["accuracy"])

    finetuned = {
        "model": "Qwen/Qwen2.5-1.5B-Instruct + qlora-sft",
        "prompt_sha256": baseline["prompt_sha256"],
        "n_test_queries": baseline["n_test_queries"],
        "n_test_qrels": len(ft_preds),
        "mmlu_finetuned": mmlu_ft["accuracy"],
        "forgetting": forget,
        **metrics,
    }
    (ROOT / "results" / "finetuned_results.json").write_text(
        json.dumps(finetuned, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )

    b, f = baseline["macro_f1"], metrics["macro_f1"]
    delta = f - b
    f1_pass = delta >= F1_BAR_PP
    rows = [
        ("Accuracy", baseline["accuracy"], metrics["accuracy"]),
        ("Macro-F1", b, f),
        *[
            (f"F1 — {lab}", baseline["per_class"][lab]["f1"], metrics["per_class"][lab]["f1"])
            for lab in ("irrelevant", "partial", "relevant")
        ],
    ]
    lines = [
        "| Metric | Zero-shot | Fine-tuned (QLoRA) | Δ |",
        "|---|---|---|---|",
        *[f"| {name} | {x:.4f} | {y:.4f} | {(y - x) * 100:+.1f} pp |" for name, x, y in rows],
        "",
        f"_Eval: {baseline['n_test_queries']} test queries / {len(ft_preds)} qrels "
        f"(held out by whole query). Macro-F1 on ~10 queries is directional, "
        f"not tight. MMLU retention {forget['retention']:.3f} "
        f"(bar ≥ {FORGETTING_RETENTION_BAR})._",
    ]
    (ROOT / "results" / "before_after_table.md").write_text(
        "\n".join(lines) + "\n", encoding="utf-8"
    )

    print(f"macro-F1: {b:.4f} → {f:.4f} (Δ {delta:+.4f}, bar +{F1_BAR_PP}) "
          f"→ {'PASS' if f1_pass else 'FAIL'}")
    print(f"MMLU retention: {forget['retention']:.4f} → "
          f"{'PASS' if forget['passes'] else 'FAIL'}")
    return 0 if (f1_pass and forget["passes"]) else 1


if __name__ == "__main__":
    raise SystemExit(main())
