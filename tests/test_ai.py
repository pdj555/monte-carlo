from __future__ import annotations

import json

import pandas as pd
import pytest

from ai import OpenAIConfigurationError, generate_ai_summary


def test_generate_ai_summary_requires_api_key(monkeypatch):
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)

    summary = pd.Series({"mean": 100.0, "value_at_risk_95": 5.0})
    with pytest.raises(OpenAIConfigurationError):
        generate_ai_summary(
            ticker="AAPL",
            summary=summary,
            simulation_model="historical",
            days=10,
            scenarios=100,
        )


def test_generate_ai_summary_parses_chat_completions(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")

    captured = {}

    class _FakeResponse:
        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            return False

        def read(self):
            payload = {"choices": [{"message": {"content": "Hello"}}]}
            return json.dumps(payload).encode("utf-8")

    def _fake_urlopen(request, timeout):
        captured["url"] = request.full_url
        captured["auth"] = request.headers.get("Authorization")
        captured["timeout"] = timeout
        return _FakeResponse()

    monkeypatch.setattr("urllib.request.urlopen", _fake_urlopen)

    summary = pd.Series({"mean": 100.0, "value_at_risk_95": 5.0})
    text = generate_ai_summary(
        ticker="AAPL",
        summary=summary,
        simulation_model="historical",
        days=10,
        scenarios=100,
        base_url="https://api.openai.com/v1",
    )

    assert text == "Hello"
    assert captured["url"].endswith("/chat/completions")
    assert captured["auth"] == "Bearer test-key"
