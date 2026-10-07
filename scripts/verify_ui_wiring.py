"""Check every run_query/_render return path yields exactly 6 outputs.

The UI rewrite changed the output arity from 5 to 6. A path that returns the
old count surfaces to a visitor as an opaque Gradio error, so every branch is
asserted here rather than discovered on the Space.

gradio and spaces are stubbed: only the control flow and tuple shapes are under
test, not Gradio's rendering.
"""
import json
import os
import sys
import types
from pathlib import Path

# Local anaconda ships two OpenMP runtimes; faiss/torch abort on the second.
# Same workaround the eval scripts use. Not a code change.
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "space_gradio"))

# ---- stub the two Space-only dependencies ---------------------------------- #
fake_spaces = types.ModuleType("spaces")


def _gpu_decorator(duration=None):
    def wrap(fn):
        return fn

    return wrap


fake_spaces.GPU = _gpu_decorator
sys.modules["spaces"] = fake_spaces

fake_gradio = types.ModuleType("gradio")


class _Component:
    """Gradio component/row stand-in: a context manager that swallows nesting."""

    def __init__(self, *a, **k):
        self.kwargs = k

    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False

    def click(self, *a, **k):
        return None

    def change(self, *a, **k):
        return None

    def load(self, *a, **k):
        return None


class Blocks(_Component):
    pass


for name in (
    "Blocks", "Markdown", "HTML", "Row", "Column", "Radio", "Textbox",
    "Slider", "Checkbox", "Button", "Dataframe", "Accordion",
):
    setattr(fake_gradio, name, _Component)
fake_gradio.Blocks = Blocks
sys.modules["gradio"] = fake_gradio

import app  # noqa: E402
from src.rag_pipeline import NO_GROUNDED_ANSWER  # noqa: E402

EXPECTED = 6
failures = []


def arity(name, value):
    ok = isinstance(value, tuple) and len(value) == EXPECTED
    print(f"  {'PASS' if ok else 'FAIL'}  {name} -> {type(value).__name__} len={len(value) if isinstance(value, tuple) else 'n/a'}")
    if not ok:
        failures.append(name)


trace = json.loads((ROOT / "results" / "demo_trace.json").read_text(encoding="utf-8"))
base_result = {
    "mode": "replay",
    "model": {"description": "m", "using_adapter": True, "load_error": None},
    "off": trace["arms"]["off"],
    "on": trace["arms"]["on"],
}

print("== _render: live result ==")
arity("render live", app._render(base_result, "q", True, "live"))

print("== _render: refused gate ==")
refused = dict(base_result)
refused["on"] = {
    "answer": NO_GROUNDED_ANSWER,
    "grounded": False,
    "used_grader": True,
    "context_ids": [],
    "retrieved": trace["arms"]["on"]["retrieved"],
    "graded": [dict(g, accepted=False) for g in trace["arms"]["on"]["graded"]],
    "note": "gate rejected all",
}
arity("render refused", app._render(refused, "q", True, "live"))

print("== _fail helper ==")
arity("fail", app._fail("boom", "status"))

print("== run_query: empty query ==")
arity("empty query", app.run_query("", "   ", 4, 150, True))

print("== run_query: no hits ==")
app.retrieve_for = lambda *a, **k: []
arity("no hits", app.run_query("espresso", "q", 4, 150, True))

print("== run_query: retrieval failure (no replay) ==")


def boom(*a, **k):
    raise RuntimeError("no embeddings")


app.retrieve_for = boom
app._replay_result = lambda *a, **k: None
arity("retrieval failed", app.run_query("espresso", "q", 4, 150, True))

print("== run_query: retrieval failure WITH replay ==")
app.retrieve_for = boom
app._replay_result = lambda *a, **k: base_result
arity("retrieval failed + replay", app.run_query("espresso", "q", 4, 150, True))

print("== run_query: GPU failure (no replay) ==")
app.retrieve_for = lambda *a, **k: trace["arms"]["on"]["retrieved"]
app._replay_result = lambda *a, **k: None


def gpu_boom(*a, **k):
    raise RuntimeError("No GPU was available in the queue")


app.run_arms = gpu_boom
app.prefetch_weights = lambda: "weights ready (5 files)"
arity("gpu failed", app.run_query("espresso", "q", 4, 150, True))

print("== run_query: GPU failure WITH replay ==")
app._replay_result = lambda *a, **k: base_result
arity("gpu failed + replay", app.run_query("espresso", "q", 4, 150, True))

print("== run_query: happy path ==")
app.run_arms = lambda *a, **k: base_result
arity("happy path", app.run_query("espresso", "q", 4, 150, True))

print("== UI wiring ==")
ds = app.demo.kwargs
check_css = "css" in ds and bool(ds["css"])
print(f"  {'PASS' if check_css else 'FAIL'}  Blocks receives css")
if not check_css:
    failures.append("css wiring")

print()
if failures:
    print(f"FAILED: {len(failures)} -> {failures}")
    raise SystemExit(1)
print("ALL RETURN-PATH ARITY CHECKS PASSED")