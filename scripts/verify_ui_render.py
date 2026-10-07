"""Verify the ui.py renderers against the recorded trace and real results JSON.

Gradio is not installed locally, so this imports ui.py directly (it only needs
src.rag_pipeline, which imports lazily) and asserts the produced HTML. It is a
content test, not a screenshot test.
"""
import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "space_gradio"))

import ui  # noqa: E402

trace = json.loads((ROOT / "results" / "demo_trace.json").read_text(encoding="utf-8"))
result = {
    "mode": "replay",
    "model": {
        "description": "Qwen/Qwen2.5-1.5B-Instruct + adapter x",
        "using_adapter": True,
        "load_error": None,
    },
    "off": trace["arms"]["off"],
    "on": trace["arms"]["on"],
}

failures = []


def check(name, cond, detail=""):
    if cond:
        print(f"  PASS  {name}")
    else:
        print(f"  FAIL  {name} {detail}")
        failures.append(name)


print("== arm card ==")
card = ui._arm_card(result["on"], "Grader ON", True, naive=False)
check("no literal markdown headers", "###" not in card)
check("no literal bold markers", "**" not in card)
check("uses real <h>/<div> tags", "<div" in card)
check("primary gets highlight", "is-primary" in card)
check("naive tag class absent", "tag-naive" not in card)
naive = ui._arm_card(result["off"], "Grader OFF", False, naive=True)
check("naive tag present", "tag-naive" in naive)
check("naive not primary", "is-primary" not in naive)

print("== refusal card ==")
refused = {
    "answer": ui.NO_GROUNDED_ANSWER,
    "grounded": False,
    "used_grader": True,
    "context_ids": [],
    "retrieved": [{"chunk_id": 0, "source": "s/1", "score": 0.4, "text": "x"}],
    "graded": [
        {
            "chunk_id": 0,
            "source": "s/1",
            "text": "x",
            "retrieval_score": 0.4,
            "label": "irrelevant",
            "score": 0.9,
            "confidence": 0.9,
            "accepted": False,
            "rank": 1,
        }
    ],
    "note": "gate rejected all",
}
rcard = ui._arm_card(refused, "Grader ON", True, naive=False)
check("refusal styled", "is-refusal" in rcard)

print("== gate summary ==")
gs = ui._gate_summary(result)
check("keeps count present", "kept 1 of 4" in gs, gs[:200])
check("no markdown", "**" not in gs)
gs_ref = ui._gate_summary({"on": refused})
check("refusal headline", "rejected every passage" in gs_ref)
gs_ref_class = "is-refused" in gs_ref
check("refusal styling", gs_ref_class)
gs_nograde = ui._gate_summary({"on": {**refused, "graded": []}})
check("no-grade message", "did not run" in gs_nograde)

print("== passage verdicts ==")
pv = ui._verdict_passages(result["on"])
check("renders passage cards", pv.count('class="passage') == 4, str(pv.count('class="passage')))
check("kept styled", "is-kept" in pv)
check("dropped styled", "is-dropped" in pv)
check("confidence bar", "conf-fill" in pv)
pv_off = ui._verdict_passages(result["off"])
check("no-gate styling when ungraded", "is-nogate" in pv_off)

print("== HTML escaping (untrusted corpus text) ==")
xss = {"chunk_id": 0, "source": "<script>alert(1)</script>", "text": "<img src=x onerror=alert(2)>", "score": 0.5}
esc = ui._verdict_passages({"retrieved": [xss], "graded": []})
check("script tag escaped", "<script>" not in esc)
check("img onerror escaped", "<img" not in esc)
check("escaped entities present", "&lt;script&gt;" in esc)

print("== benchmark panel (reads real results/*.json) ==")
bench = ui.benchmark_html()
check("panel rendered", "bench-wrap" in bench)
check("macro-F1 headline row", "is-headline" in bench)
check("contains 0.2672 baseline", "0.2672" in bench, "baseline missing")
check("contains 0.4306 finetuned", "0.4306" in bench, "finetuned missing")
check("delta shown", "+0.1634" in bench or "+0.163" in bench)
check("confusion matrix present", "pred" in bench and "true" in bench)
check("mmlu pill present", "MMLU retention" in bench)
check("no literal markdown", "**" not in bench)
m = re.search(r"macro-F1 Δ ([+-][\d.]+)", bench)
print(f"   macro-F1 delta parsed: {m.group(1) if m else 'NOT FOUND'}")

print("== benchmark with missing results (must not invent numbers) ==")
import src.rag_pipeline as rp
orig = rp.load_metrics
rp.load_metrics = lambda *a, **k: {"baseline": None, "finetuned": None, "missing": ["x.json"]}
ui.load_metrics = rp.load_metrics
empty = ui.benchmark_html()
check("reports missing", "No benchmark JSON" in empty)
check("no fabricated numbers", "0.4306" not in empty and "0.2672" not in empty)
rp.load_metrics = orig
ui.load_metrics = orig

print("== hero + css ==")
check("hero has h1", "<h1>" in ui._hero_html())
check("css defines arm-card", "arm-card" in ui.CSS)
check("css defines conf bar", "conf-fill" in ui.CSS)
check("css defines gate summary", "gate-summary" in ui.CSS)

print()
if failures:
    print(f"FAILED: {len(failures)} -> {failures}")
    raise SystemExit(1)
print("ALL UI CONTENT CHECKS PASSED")