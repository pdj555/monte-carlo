from __future__ import annotations

import shutil
import subprocess
import sys
from pathlib import Path


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


def test_installed_entrypoint_serves_help_commands() -> None:
    entrypoint = _installed_entrypoint()

    for args in (
        ["--help"],
        ["simulate", "--help"],
        ["backtest", "--help"],
    ):
        completed = subprocess.run(
            [entrypoint, *args],
            capture_output=True,
            check=False,
            text=True,
        )
        assert completed.returncode == 0, completed.stderr
        assert "monte-carlo" in completed.stdout
