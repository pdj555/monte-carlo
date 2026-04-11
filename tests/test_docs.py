from __future__ import annotations

from pathlib import Path


def test_repo_guides_point_to_public_cli() -> None:
    for path in (
        "README.md",
        "AGENTS.md",
        "CLAUDE.md",
        "GEMINI.md",
        "docs/improvements.md",
    ):
        text = Path(path).read_text(encoding="utf-8")
        assert "monte-carlo simulate" in text
        assert "python3 -m pip install -e ." in text


def test_contributor_guides_describe_public_cli_split() -> None:
    for path in ("README.md", "AGENTS.md", "CLAUDE.md", "GEMINI.md"):
        text = Path(path).read_text(encoding="utf-8")
        assert "public_cli.py" in text


def test_improvements_doc_avoids_retired_command_examples() -> None:
    text = Path("docs/improvements.md").read_text(encoding="utf-8")
    assert "python MonteCarlo.py --ticker" not in text
    assert "python cli.py --ticker" not in text
    assert "pip install -r requirements.txt" not in text
