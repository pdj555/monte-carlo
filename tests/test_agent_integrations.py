from __future__ import annotations

import json
from types import SimpleNamespace
from unittest.mock import patch

import pandas as pd
import pytest

import agent_integrations
from agent_integrations import (
    AgentDependencyError,
    build_claude_agent_options,
    build_openai_decision_agent,
    scan_tickers_tool,
)


@pytest.fixture()
def sample_prices():
    dates = pd.bdate_range("2023-01-01", periods=120)
    return pd.Series(
        [100.0 + i * 0.1 + (i % 7 - 3) * 0.5 for i in range(120)],
        index=dates,
        name="Close",
    )


def test_scan_tickers_tool_returns_json(sample_prices):
    with patch("sdk.fetch_prices", return_value=sample_prices):
        payload = scan_tickers_tool("AAPL, MSFT", days=10, scenarios=50, seed=42)

    parsed = json.loads(payload)
    assert parsed["universe_size"] == 2
    assert parsed["analyzed"] > 0


def test_build_openai_agent_requires_optional_dependency(monkeypatch):
    def _missing(_name):
        raise ImportError("missing")

    monkeypatch.setattr(agent_integrations, "_import_module", _missing)

    with pytest.raises(AgentDependencyError, match="OpenAI Agents SDK"):
        build_openai_decision_agent()


def test_build_openai_agent_wires_agent_tools(monkeypatch):
    class FakeAgent:
        def __init__(self, **kwargs):
            self.kwargs = kwargs

    def fake_function_tool(func):
        return {"tool_name": func.__name__}

    def fake_import(name):
        assert name == "agents"
        return SimpleNamespace(
            Agent=FakeAgent,
            Runner=object(),
            function_tool=fake_function_tool,
        )

    monkeypatch.setattr(agent_integrations, "_import_module", fake_import)

    agent = build_openai_decision_agent(model="test-model")

    assert agent.kwargs["model"] == "test-model"
    assert agent.kwargs["name"] == "Monte Carlo Decision Agent"
    assert len(agent.kwargs["tools"]) == 4


def test_build_claude_options_requires_optional_dependency(monkeypatch):
    def _missing(_name):
        raise ImportError("missing")

    monkeypatch.setattr(agent_integrations, "_import_module", _missing)

    with pytest.raises(AgentDependencyError, match="Claude Agent SDK"):
        build_claude_agent_options()


def test_build_claude_options_uses_conservative_tools(monkeypatch):
    class FakeOptions:
        def __init__(self, **kwargs):
            self.kwargs = kwargs

    def fake_import(name):
        assert name == "claude_agent_sdk"
        return SimpleNamespace(query=object(), ClaudeAgentOptions=FakeOptions)

    monkeypatch.setattr(agent_integrations, "_import_module", fake_import)

    options = build_claude_agent_options()

    assert options.kwargs["allowed_tools"] == ["Read", "Grep", "Glob", "Bash"]
