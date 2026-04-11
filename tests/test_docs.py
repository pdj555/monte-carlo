from __future__ import annotations

from pathlib import Path


def test_repo_guides_point_to_public_cli() -> None:
    for path in ("README.md", "AGENTS.md", "CLAUDE.md", "GEMINI.md"):
        text = Path(path).read_text(encoding="utf-8")
        assert "monte-carlo simulate" in text
        assert "python3 -m pip install -e ." in text
