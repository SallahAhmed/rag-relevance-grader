"""Shared MMLU harness — the SAME protocol scores the base model (notebook 02b)
and the adapter (notebook 04): identical subjects, prompt template
(`mmlu_prompt`), greedy 3-token decoding, first-A-D-letter parse.
A forgetting verdict is only valid if both sides ran this protocol.
(The 02b baseline ran an inline copy of this exact protocol; this module is
its canonical home from notebook 04 onward.)
"""
from __future__ import annotations

CHOICES = ("A", "B", "C", "D")
SUBJECTS = (
    "high_school_world_history",  # factual recall
    "high_school_mathematics",  # reasoning
    "high_school_computer_science",  # procedural
)


def mmlu_prompt(question: str, choices: list[str]) -> str:
    body = "".join(f"{c}. {choices[i]}\n" for i, c in enumerate(CHOICES))
    return f"Question: {question}\n{body}Answer:"


def parse_letter(generated: str) -> str:
    text = generated.strip().upper()
    return next((c for c in CHOICES if c in text[:4]), "?")


def run_mmlu(model, tokenizer, subjects: tuple = SUBJECTS) -> dict:
    """Greedy single-letter scoring. Model already loaded; returns results dict."""
    import torch
    from datasets import load_dataset

    results: dict = {"subjects": {}, "n_total": 0, "n_correct": 0}
    with torch.no_grad():
        for subj in subjects:
            ds = load_dataset("cais/mmlu", subj, split="test")
            correct, total = 0, 0
            for row in ds:
                enc = tokenizer(
                    mmlu_prompt(row["question"], list(row["choices"])),
                    return_tensors="pt",
                    truncation=True,
                    max_length=1024,
                ).to(model.device)
                gen = model.generate(
                    **enc, max_new_tokens=3, do_sample=False,
                    pad_token_id=tokenizer.eos_token_id,
                )
                pred = parse_letter(
                    tokenizer.decode(gen[0, enc["input_ids"].shape[1]:])
                )
                total += 1
                correct += pred == "ABCD"[row["answer"]]
            acc = correct / total
            results["subjects"][subj] = {"acc": round(acc, 4), "n": total}
            results["n_total"] += total
            results["n_correct"] += correct
            print(f"{subj}: {correct}/{total} = {acc:.3f}", flush=True)
    results["accuracy"] = round(results["n_correct"] / results["n_total"], 4)
    return results
