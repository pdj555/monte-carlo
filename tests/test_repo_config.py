from __future__ import annotations

from pathlib import Path


def test_flake8_excludes_generated_vercel_runtime() -> None:
    text = Path(".flake8").read_text(encoding="utf-8")
    assert ".vercel" in text
