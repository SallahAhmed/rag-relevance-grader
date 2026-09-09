"""Dataset helpers: label bucketing, prompt formatting, query-grouped splits.

Design rules enforced here:
  1. PROMPT_TEMPLATE is defined ONCE and imported by both the baseline
     notebook (02) and the fine-tuning notebook (03). An identical template
     is what makes the before/after comparison valid.
  2. Splits are grouped BY QUERY. TREC DL covers ~54 queries; a row-wise
     split leaks the same query into train and test.
  3. Splits are stratified BY LABEL so the thin "partial" class is
     represented on both sides.
"""
from __future__ import annotations

import random
from collections import Counter
from collections.abc import Hashable, Mapping, Sequence

# The three relevance classes, in label-id order.
IRRELEVANT = "irrelevant"
PARTIAL = "partial"
RELEVANT = "relevant"
LABELS: tuple[str, str, str] = (IRRELEVANT, PARTIAL, RELEVANT)

# Single source of truth for the prompt. Import this — never retype it.
PROMPT_TEMPLATE = (
    "Classify the relevance of the passage to the query as exactly one of: "
    "irrelevant, partial, relevant.\n\n"
    "Query: {query}\n"
    "Passage: {passage}\n\n"
    "Label:"
)


def bucket_qrel(qrel: int) -> str:
    """Map a TREC 0–3 grade onto one of the three classes.

    0 → irrelevant, 1 → partial, 2–3 → relevant.
    Raises ValueError on anything outside 0–3 so bad joins fail loudly.
    """
    if qrel == 0:
        return IRRELEVANT
    if qrel == 1:
        return PARTIAL
    if qrel in (2, 3):
        return RELEVANT
    raise ValueError(f"TREC grade out of range 0–3: {qrel!r}")


def format_example(query: str, passage: str, label: str) -> dict[str, str]:
    """Build one training/eval example from the shared template."""
    if label not in LABELS:
        raise ValueError(f"Unknown label {label!r}; expected one of {LABELS}")
    prompt = PROMPT_TEMPLATE.format(query=query.strip(), passage=passage.strip())
    return {"prompt": prompt, "label": label}


def split_queries_by_label(
    query_to_label_counts: Mapping[Hashable, Mapping[str, int]],
    n_test_queries: int,
    seed: int = 42,
) -> tuple[set, set]:
    """Hold out whole queries for test, keeping label mix representative.

    Args:
        query_to_label_counts: {query_id: {label: count}} over judged pairs.
        n_test_queries: how many whole queries to hold out.
        seed: fixed for reproducibility.

    Returns:
        (train_query_ids, test_query_ids). Test queries are sampled to cover
        all three labels when the pool allows it.
    """
    rng = random.Random(seed)
    query_ids = list(query_to_label_counts)
    if n_test_queries >= len(query_ids):
        raise ValueError(
            f"Need fewer test queries ({n_test_queries}) than total ({len(query_ids)})"
        )
    # Greedy cover: first pick queries that carry the rarest label (partial),
    # then fill the remainder at random. Deterministic given seed.
    label_totals: Counter = Counter()
    for counts in query_to_label_counts.values():
        label_totals.update(counts)
    rarest = min(LABELS, key=lambda lab: label_totals.get(lab, 0))

    carriers = [qid for qid in query_ids if query_to_label_counts[qid].get(rarest, 0)]
    rest = [qid for qid in query_ids if qid not in carriers]
    rng.shuffle(carriers)
    rng.shuffle(rest)
    # At least one carrier of the rarest label lands in test when possible.
    test = list(carriers[:1])
    pool = [qid for qid in (carriers[1:] + rest) if qid not in test]
    test += pool[: max(0, n_test_queries - len(test))]
    test_ids = set(test[:n_test_queries])
    return set(query_ids) - test_ids, test_ids


def label_distribution(labels: Sequence[str]) -> dict[str, int]:
    """Count examples per class — call this in notebook 01 and print it."""
    counts = Counter(labels)
    return {label: counts.get(label, 0) for label in LABELS}
