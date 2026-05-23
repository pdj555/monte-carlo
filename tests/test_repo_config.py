from __future__ import annotations

import json
from pathlib import Path


def test_flake8_excludes_generated_vercel_runtime() -> None:
    text = Path(".flake8").read_text(encoding="utf-8")
    assert ".vercel" in text


def test_vercel_config_targets_flask_app_without_legacy_builds() -> None:
    config = json.loads(Path("vercel.json").read_text(encoding="utf-8"))

    assert config["$schema"] == "https://openapi.vercel.sh/vercel.json"
    assert config["framework"] == "nextjs"
    assert "builds" not in config
    assert config["buildCommand"] == "npm run build"
    assert Path("requirements.txt").exists()
    assert Path("api/engine.py").exists()
    assert config["functions"]["api/engine.py"]["maxDuration"] == 60


def test_next_config_traces_python_engine_for_runtime_routes() -> None:
    tracing = Path("lib/engine-tracing.ts").read_text(encoding="utf-8")
    config = Path("next.config.ts").read_text(encoding="utf-8")

    assert "outputFileTracingIncludes" in config
    assert "engineTraceIncludes" in config
    assert "ui_bridge.py" in tracing
    assert "python_packages" in tracing
    assert '"/api/run"' in tracing or "'/api/run'" in tracing


def test_runtime_files_use_next_for_ui_and_python_for_engine() -> None:
    requirements = Path("requirements.txt").read_text(encoding="utf-8")

    assert "Flask" not in requirements
    assert Path("package.json").exists()
    assert Path(".python-version").read_text(encoding="utf-8").strip() == "3.12"


def test_local_agent_runtime_files_are_ignored() -> None:
    gitignore = Path(".gitignore").read_text(encoding="utf-8")

    for entry in (".agent/", ".claude/", "CLAUDE.md", "GEMINI.md", "WARP.md"):
        assert entry in gitignore
