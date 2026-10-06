| Metric | Zero-shot | Fine-tuned (QLoRA) | Δ |
|---|---|---|---|
| Accuracy | 0.3695 | 0.6447 | +27.5 pp |
| Macro-F1 | 0.2672 | 0.4306 | +16.3 pp |
| F1 — irrelevant | 0.4735 | 0.8000 | +32.6 pp |
| F1 — partial | 0.0131 | 0.1590 | +14.6 pp |
| F1 — relevant | 0.3149 | 0.3329 | +1.8 pp |

_Eval: 10 test queries / 2122 qrels (held out by whole query). Macro-F1 on ~10 queries is directional, not tight. MMLU retention 0.947 (bar ≥ 0.9)._
