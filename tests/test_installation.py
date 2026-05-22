from __future__ import annotations

import importlib.util
import os
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]


def _installed_entrypoint() -> str:
    scripts_dir = Path(sys.executable).resolve().parent
    candidates = [
        scripts_dir / "monte-carlo",
        scripts_dir / "monte-carlo.exe",
    ]

    for candidate in candidates:
        if candidate.exists():
            return str(candidate)

    resolved = shutil.which("monte-carlo")
    if resolved is None:
        raise AssertionError("Installed `monte-carlo` entrypoint was not found on PATH.")
    return resolved


def _run_entrypoint(entrypoint: str, *args: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [entrypoint, *args],
        capture_output=True,
        check=False,
        cwd=REPO_ROOT,
        env={**os.environ, "MPLBACKEND": "Agg"},
        text=True,
    )


def _installed_ui_entrypoint() -> str:
    scripts_dir = Path(sys.executable).resolve().parent
    candidates = [
        scripts_dir / "monte-carlo-ui",
        scripts_dir / "monte-carlo-ui.exe",
    ]

    for candidate in candidates:
        if candidate.exists():
            return str(candidate)

    resolved = shutil.which("monte-carlo-ui")
    if resolved is None:
        raise AssertionError("Installed `monte-carlo-ui` entrypoint was not found on PATH.")
    return resolved


def test_installed_entrypoint_serves_help_commands() -> None:
    entrypoint = _installed_entrypoint()

    for args in (
        ["--help"],
        ["simulate", "--help"],
        ["backtest", "--help"],
    ):
        completed = _run_entrypoint(entrypoint, *args)
        assert completed.returncode == 0, completed.stderr
        assert "monte-carlo" in completed.stdout


def test_installed_entrypoint_without_subcommand_prints_help_hint() -> None:
    entrypoint = _installed_entrypoint()

    completed = _run_entrypoint(entrypoint)
    assert completed.returncode == 1, completed.stderr
    assert "simulate" in completed.stdout
    assert "backtest" in completed.stdout
    assert "Choose `simulate` for current ideas" in completed.stdout


def test_installed_entrypoint_runs_offline_simulate_and_backtest() -> None:
    entrypoint = _installed_entrypoint()
    sample_data = str(REPO_ROOT / "sample_data")

    simulate = _run_entrypoint(
        entrypoint,
        "simulate",
        "AAPL",
        "MSFT",
        "--source",
        "offline",
        "--data-path",
        sample_data,
        "--days",
        "5",
        "--scenarios",
        "10",
        "--details",
    )
    assert simulate.returncode == 0, simulate.stderr
    assert "Stance:" in simulate.stdout
    assert "Ticker ranking" in simulate.stdout
    assert "AAPL" in simulate.stdout
    assert "MSFT" in simulate.stdout

    backtest = _run_entrypoint(
        entrypoint,
        "backtest",
        "AAPL",
        "MSFT",
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
        "--details",
    )
    assert backtest.returncode == 0, backtest.stderr
    assert "Strategy return:" in backtest.stdout
    assert "Backtest summary" in backtest.stdout


def test_pyproject_console_script_points_to_public_cli() -> None:
    text = (REPO_ROOT / "pyproject.toml").read_text(encoding="utf-8")
    assert 'monte-carlo = "public_cli:main"' in text
    assert 'monte-carlo-ui = "app:main"' in text
    assert 'monte-carlo-mcp = "mcp_server:serve"' in text
    assert 'ui = ["Flask"]' in text
    assert 'agents = ["claude-agent-sdk", "openai-agents"]' in text


def test_installed_ui_entrypoint_reports_missing_ui_extra_when_flask_unavailable() -> None:
    if importlib.util.find_spec("flask") is not None:
        pytest.skip("Flask is installed in this environment.")

    entrypoint = _installed_ui_entrypoint()
    completed = _run_entrypoint(entrypoint)

    assert completed.returncode == 2
    combined = f"{completed.stdout}\n{completed.stderr}"
    assert "Install `python3 -m pip install -e .[ui]`" in combined
