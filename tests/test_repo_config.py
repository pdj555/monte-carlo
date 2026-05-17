from __future__ import annotations

import json
from pathlib import Path


def test_flake8_excludes_generated_vercel_runtime() -> None:
    text = Path(".flake8").read_text(encoding="utf-8")
    assert ".vercel" in text


def test_vercel_config_targets_flask_app_without_legacy_builds() -> None:
    config = json.loads(Path("vercel.json").read_text(encoding="utf-8"))

    assert config["$schema"] == "https://openapi.vercel.sh/vercel.json"
    assert "builds" not in config
    assert "api/*.py" in config["functions"]
    fn_config = config["functions"]["api/*.py"]
    assert "sample_data/**" in fn_config["includeFiles"]
    assert "app.py" in fn_config["includeFiles"]
    assert "tests/**" in fn_config["excludeFiles"]


def test_vercel_runtime_files_include_flask_and_python_version() -> None:
    requirements = Path("requirements.txt").read_text(encoding="utf-8")

    assert "Flask>=3.0" in requirements
    assert Path(".python-version").read_text(encoding="utf-8").strip() == "3.12"
