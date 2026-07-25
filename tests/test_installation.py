from __future__ import annotations

import os
import shutil
import subprocess
import sys
import zipfile
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


def test_installed_entrypoint_serves_help_commands() -> None:
    entrypoint = _installed_entrypoint()

    for args in (
        ["--help"],
        ["simulate", "--help"],
        ["backtest", "--help"],
        ["evaluate", "--help"],
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
    assert "evaluate" in completed.stdout
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


def test_built_wheel_keeps_reference_assets_source_checkout_only(tmp_path) -> None:
    wheel_dir = tmp_path / "dist"
    wheel_dir.mkdir()
    built = subprocess.run(
        [
            sys.executable,
            "-m",
            "pip",
            "wheel",
            str(REPO_ROOT),
            "--no-deps",
            "--no-build-isolation",
            "--wheel-dir",
            str(wheel_dir),
        ],
        capture_output=True,
        check=False,
        cwd=tmp_path,
        text=True,
    )
    assert built.returncode == 0, built.stderr
    wheel_path = next(wheel_dir.glob("*.whl"))
    with zipfile.ZipFile(wheel_path) as wheel:
        packaged_paths = set(wheel.namelist())
    assert "evaluation.py" in packaged_paths
    assert not any(path.startswith("evaluation_sets/") for path in packaged_paths)
    assert not any(path.startswith("sample_data/") for path in packaged_paths)

    installed_dir = tmp_path / "site"
    installed = subprocess.run(
        [
            sys.executable,
            "-m",
            "pip",
            "install",
            "--no-deps",
            "--target",
            str(installed_dir),
            str(wheel_path),
        ],
        capture_output=True,
        check=False,
        cwd=tmp_path,
        text=True,
    )
    assert installed.returncode == 0, installed.stderr
    evaluated = subprocess.run(
        [
            sys.executable,
            str(installed_dir / "public_cli.py"),
            "evaluate",
            "evaluation_sets/sample-stability.json",
        ],
        capture_output=True,
        check=False,
        cwd=tmp_path,
        env={**os.environ, "MPLBACKEND": "Agg"},
        text=True,
    )
    assert evaluated.returncode == 2
    assert "cannot read" in f"{evaluated.stdout}\n{evaluated.stderr}"

    readme = " ".join(
        (REPO_ROOT / "README.md").read_text(encoding="utf-8").lower().split()
    )
    assert "source-checkout assets" in readme
    assert "not included in the installed wheel" in readme
    assert "supply their own evaluation-set json and local data paths" in readme


def test_pyproject_console_script_points_to_public_cli() -> None:
    text = (REPO_ROOT / "pyproject.toml").read_text(encoding="utf-8")
    assert 'monte-carlo = "public_cli:main"' in text
    assert 'monte-carlo-ui = "web_entrypoint:main"' in text
    assert '"ui_state"' in text
    assert '"ui_bridge"' in text


def test_installed_ui_entrypoint_reports_missing_node_dependencies() -> None:
    if (REPO_ROOT / "node_modules" / "next").exists():
        pytest.skip("Next.js dependencies are installed in this environment.")

    scripts_dir = Path(sys.executable).resolve().parent
    entrypoint = scripts_dir / "monte-carlo-ui"
    if not entrypoint.exists():
        resolved = shutil.which("monte-carlo-ui")
        if resolved is None:
            raise AssertionError("Installed `monte-carlo-ui` entrypoint was not found on PATH.")
        entrypoint = Path(resolved)
    completed = _run_entrypoint(entrypoint)

    assert completed.returncode == 2
    combined = f"{completed.stdout}\n{completed.stderr}"
    assert "Run `npm install` and `npm run dev`" in combined
