"""Shared fixtures for the Space test suite.

`gradio` and `spaces` are stubbed BEFORE anything imports `app`: the Space-only
dependencies are not installed locally or in CI, and only the control flow,
wiring and HTML rendering are under test — not Gradio's renderer or ZeroGPU
scheduling. The stubbing is the same proven shape scripts/verify_ui_wiring.py
uses.
"""
from __future__ import annotations

import json
import logging
import os
import sys
import types
from pathlib import Path

import pytest

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

ROOT = Path(__file__).resolve().parents[1]
SPACE = ROOT / "space_gradio"
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(SPACE))

fake_spaces = types.ModuleType("spaces")


def _gpu_decorator(duration=None):
    def wrap(fn):
        return fn

    return wrap


fake_spaces.GPU = _gpu_decorator
sys.modules["spaces"] = fake_spaces

fake_gradio = types.ModuleType("gradio")


class _Component:
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

    def release(self, *a, **k):
        return None


class Blocks(_Component):
    pass


for _name in (
    "Blocks",
    "Markdown",
    "HTML",
    "Row",
    "Column",
    "Radio",
    "Textbox",
    "Slider",
    "Checkbox",
    "Button",
    "Dataframe",
    "Accordion",
    "Error",
    "Update",
):
    setattr(fake_gradio, _name, _Component)
fake_gradio.Blocks = Blocks
sys.modules["gradio"] = fake_gradio


class RecordingHandler(logging.Handler):
    def __init__(self) -> None:
        super().__init__()
        self.records: list[logging.LogRecord] = []

    def emit(self, record: logging.LogRecord) -> None:
        self.records.append(record)


@pytest.fixture(scope="session")
def root() -> Path:
    return ROOT


@pytest.fixture(scope="session")
def results_dir() -> Path:
    return ROOT / "results"


@pytest.fixture(scope="session")
def trace() -> dict:
    return json.loads((ROOT / "results" / "demo_trace.json").read_text(encoding="utf-8"))


@pytest.fixture(scope="session")
def app():
    import app as app_module

    return app_module


@pytest.fixture(scope="session")
def ui():
    import ui as ui_module

    return ui_module


@pytest.fixture
def grader_records() -> list[logging.LogRecord]:
    logger = logging.getLogger("grader")
    handler = RecordingHandler()
    previous_level = logger.level
    logger.addHandler(handler)
    logger.setLevel(logging.INFO)
    yield handler.records
    logger.removeHandler(handler)
    logger.setLevel(previous_level)


@pytest.fixture(autouse=True)
def clear_trace_cache(app):
    app.load_recorded_trace.cache_clear()
    yield
    app.load_recorded_trace.cache_clear()
