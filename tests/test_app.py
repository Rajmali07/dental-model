"""
tests/test_app.py
Unit tests for the Gradio web application (Phase 4).

Run with: uv run pytest tests/test_app.py -v
"""
from __future__ import annotations

import sys
from pathlib import Path

import gradio as gr

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from app.app import build_app  # noqa: E402


def test_build_app_structure() -> None:
    """build_app returns a valid gr.Blocks instance with UI components."""
    demo = build_app()
    assert isinstance(demo, gr.Blocks)
    assert demo.title == "🦷 Dental Model — Clinical AI Screening Demo"
