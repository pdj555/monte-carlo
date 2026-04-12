from __future__ import annotations

from pathlib import Path

from public_cli import main as public_main

REPO_ROOT = Path(__file__).resolve().parents[1]
README_PATH = REPO_ROOT / "README.md"


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


def test_readme_offline_backtest_example_uses_sample_data_friendly_window() -> None:
    text = README_PATH.read_text(encoding="utf-8")
    assert "--lookback 5" in text
    assert "--hold 3" in text
    assert "--rebalance 3" in text
    assert "--top 1" in text
    assert "--scenarios 10" in text


def test_readme_distinguishes_cli_commands_from_browser_entrypoint() -> None:
    text = README_PATH.read_text(encoding="utf-8")
    assert "The main CLI has two commands:" in text
    assert "Optional browser entrypoint:" in text


def test_readme_offline_examples_run(capsys) -> None:
    sample_data = str(REPO_ROOT / "sample_data")

    simulate_exit = public_main(
        ["simulate", "AAPL", "--source", "offline", "--data-path", sample_data]
    )
    simulate_output = capsys.readouterr().out
    assert simulate_exit == 0
    assert "Stance:" in simulate_output

    backtest_exit = public_main(
        [
            "backtest",
            "AAPL",
            "--source",
            "offline",
            "--data-path",
            sample_data,
            "--lookback",
            "5",
            "--hold",
            "3",
            "--rebalance",
            "3",
            "--top",
            "1",
            "--scenarios",
            "10",
        ]
    )
    backtest_output = capsys.readouterr().out
    assert backtest_exit == 0
    assert "Strategy return:" in backtest_output
