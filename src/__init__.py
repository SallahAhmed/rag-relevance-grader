"""Project B — fine-tuned RAG relevance grader.

Public surface:
  config            typed hyperparameters loaded from config/sft_config.yaml
  dataset_utils     TREC label bucketing, prompt formatting, query-grouped splits
  evaluation_utils  accuracy / macro-F1 / per-class F1 / confusion matrix
  grader_interface  pluggable GraderInterface (v1 contract)

Pipeline notebooks live in notebooks/ and orchestrate these modules —
no training logic lives inline in a notebook cell.
"""

__version__ = "0.1.0"
