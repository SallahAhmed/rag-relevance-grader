# Qwen2.5-1.5B Relevance Grader (QLoRA, TREC DL 2020)

> The table is the project. Everything below exists to make these numbers trustworthy.

| Metric | Zero-shot baseline | Fine-tuned (QLoRA) | Δ |
|---|---|---|---|
| Accuracy | TBD | TBD | TBD |
| Macro-F1 | TBD | TBD | **target: +10 pp** |
| F1 — irrelevant | TBD | TBD | TBD |
| F1 — partial | TBD | TBD | TBD |
| F1 — relevant | TBD | TBD | TBD |
| MMLU retention | 1.00 (baseline) | TBD | **bar: ≥ 0.90** |

*Eval: N test queries / N qrels (held out by whole query — see `results/before_after_table.md`).
Macro-F1 on a ~54-query judged set is directional evidence, not a tight estimate.*

## What this is

A Qwen2.5-1.5B-Instruct adapter fine-tuned with QLoRA (4-bit NF4) to classify
query–passage relevance into **irrelevant / partial / relevant** — the grader
a Corrective RAG pipeline uses to decide whether retrieved passages are good
enough to answer from. It implements a pluggable `GraderInterface`
(`src/grader_interface.py`), so it drops into any RAG pipeline as a reranker.

## Try it

```python
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

base = "Qwen/Qwen2.5-1.5B-Instruct"
tok = AutoTokenizer.from_pretrained(base)
model = AutoModelForCausalLM.from_pretrained(base, load_in_4bit=True, device_map="auto")
model = PeftModel.from_pretrained(model, "sallahahmed/qwen2.5-1.5b-relevance-grader")

from src.dataset_utils import PROMPT_TEMPLATE
prompt = PROMPT_TEMPLATE.format(query="YOUR QUERY", passage="YOUR PASSAGE")
inputs = tok(prompt, return_tensors="pt").to(model.device)
out = model.generate(**inputs, max_new_tokens=5)
print(tok.decode(out[0], skip_special_tokens=True))
```

## Reproduce it

```bash
pip install -r requirements.txt
python -m src.main --check-env
# Notebooks are jupytext percent-scripts: open in Jupyter/VS Code or run directly.
# 01 (CPU ok) → 02 (T4) → 03 (T4, smoke test first) → 04 (T4) → 05
python notebooks/01_data_preparation.py
```

Hyperparameters: `config/sft_config.yaml` (single source of truth).
Method notes + per-axis provenance: `ANALYSIS.md`.

## Limits (read before citing the numbers)

- Graded on ~54 judged TREC DL 2020 queries — before/after deltas are noisy.
- v1 outputs label + softmax confidence only; no `reasoning` field (no training signal for it).
- English MS MARCO passages only.
