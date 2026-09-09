"""Build results/baseline_results.json from downloaded T4 predictions.

Run locally (no GPU):  python C:\\path\\to\\make_baseline_results.py
Reads results/predictions_baseline.json + data/processed/stats.json,
scores with src/evaluation_utils, writes results/baseline_results.json.
MMLU baseline is a separate slice and merges in later (key: mmlu_baseline).
"""
import hashlib
import json
import sys
from datetime import date
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from src.dataset_utils import PROMPT_TEMPLATE  # noqa: E402
from src.evaluation_utils import compute_classification_metrics  # noqa: E402

preds = json.loads((ROOT / "results" / "predictions_baseline.json").read_text())
stats = json.loads((ROOT / "data" / "processed" / "stats.json").read_text())

y_true = [p["true"] for p in preds]
y_pred = [p["pred"] for p in preds]
metrics = compute_classification_metrics(y_true, y_pred)

payload = {
    "date": date.today().isoformat(),
    "model": "Qwen/Qwen2.5-1.5B-Instruct",
    "mode": "zero-shot",
    "prompt_sha256": hashlib.sha256(PROMPT_TEMPLATE.encode()).hexdigest()[:16],
    "n_test_queries": stats["n_test_queries"],
    "n_test_qrels": stats["n_test_pairs"],
    "test_qids": stats["test_qids"],
    "mmlu_baseline": None,  # filled by the MMLU slice (notebook 02, part 2)
    **metrics,
}
out = ROOT / "results" / "baseline_results.json"
out.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")

print("accuracy :", round(metrics["accuracy"], 4))
print("macro_f1 :", round(metrics["macro_f1"], 4))
for lab, m in metrics["per_class"].items():
    print(f"  {lab:10s} P={m['precision']:.3f} R={m['recall']:.3f} F1={m['f1']:.3f} n={m['support']}")
print("wrote", out)
