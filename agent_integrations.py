"""Optional OpenAI and Claude agent SDK adapters.

The simulation, ranking, and workflow code remains deterministic. These helpers
only wrap that core with agent runtimes when callers explicitly install the
optional agent dependencies and provide provider credentials.
"""

from __future__ import annotations

import importlib
import json
from typing import Any, Iterable

from agent_workflow import (
    opportunity_scan,
    rebalance_signal,
    risk_check,
    what_if,
)

DEFAULT_OPENAI_AGENT_MODEL = "gpt-5.2"
DEFAULT_CLAUDE_TOOLS = ("Read", "Grep", "Glob", "Bash")

OPENAI_AGENT_INSTRUCTIONS = """
You are an agentic quantitative decision assistant. Use the provided tools for
all calculations. Treat tool output as the source of truth, explain uncertainty
plainly, and do not provide personalized financial advice.
""".strip()

CLAUDE_REPO_AGENT_PROMPT = """
You are reviewing the Monte Carlo decision engine as an engineering agent.
Focus on correctness, tests, packaging, observability, and whether agentic
changes preserve the deterministic simulation core.
""".strip()

_import_module = importlib.import_module


class AgentDependencyError(RuntimeError):
    """Raised when an optional agent SDK dependency is not installed."""


def _parse_tickers(tickers: str | Iterable[str]) -> list[str]:
    if isinstance(tickers, str):
        raw = tickers.replace(",", " ").split()
    else:
        raw = list(tickers)
    parsed = [str(ticker).strip().upper() for ticker in raw if str(ticker).strip()]
    if not parsed:
        raise ValueError("At least one ticker is required.")
    return parsed


def scan_tickers_tool(
    tickers: str | Iterable[str],
    *,
    days: int = 252,
    scenarios: int = 1000,
    model: str = "historical",
    seed: int | None = None,
    offline_only: bool = False,
) -> str:
    """Return a JSON opportunity scan for agent tool use."""

    report = opportunity_scan(
        _parse_tickers(tickers),
        days=days,
        scenarios=scenarios,
        model=model,
        seed=seed,
        offline_only=offline_only,
    )
    return report.to_json(sort_keys=True)


def risk_check_tool(
    ticker: str,
    *,
    days: int = 252,
    scenarios: int = 2000,
    model: str = "historical",
    seed: int | None = None,
    offline_only: bool = False,
) -> str:
    """Return a JSON risk check for one ticker."""

    report = risk_check(
        ticker,
        days=days,
        scenarios=scenarios,
        model=model,
        seed=seed,
        offline_only=offline_only,
    )
    return report.to_json(sort_keys=True)


def what_if_tool(
    ticker: str,
    *,
    days: int = 252,
    scenarios: int = 1000,
    seed: int | None = None,
    offline_only: bool = False,
) -> str:
    """Return JSON model-sensitivity analysis for one ticker."""

    report = what_if(
        ticker,
        days=days,
        scenarios=scenarios,
        seed=seed,
        offline_only=offline_only,
    )
    return report.to_json(sort_keys=True)


def rebalance_signal_tool(
    current_holdings_json: str,
    *,
    days: int = 60,
    scenarios: int = 1000,
    model: str = "historical",
    seed: int | None = None,
    capital: float = 100000.0,
    offline_only: bool = False,
) -> str:
    """Return JSON rebalance guidance from a ticker-to-weight JSON object."""

    raw = json.loads(current_holdings_json)
    if not isinstance(raw, dict):
        raise ValueError("current_holdings_json must decode to an object.")
    holdings = {str(ticker).upper(): float(weight) for ticker, weight in raw.items()}
    report = rebalance_signal(
        holdings,
        days=days,
        scenarios=scenarios,
        model=model,
        seed=seed,
        capital=capital,
        offline_only=offline_only,
    )
    return report.to_json(sort_keys=True)


def _load_openai_agents() -> tuple[Any, Any, Any]:
    try:
        module = _import_module("agents")
    except ImportError as exc:
        raise AgentDependencyError(
            "OpenAI Agents SDK is required. Install `monte-carlo-sim[agents]` "
            "or `pip install openai-agents`."
        ) from exc
    return module.Agent, module.Runner, module.function_tool


def build_openai_decision_agent(
    *,
    model: str = DEFAULT_OPENAI_AGENT_MODEL,
) -> Any:
    """Build an OpenAI Agents SDK agent over the deterministic workflow tools."""

    Agent, _Runner, function_tool = _load_openai_agents()

    @function_tool
    def scan_tickers(tickers: str, days: int = 252, scenarios: int = 1000) -> str:
        """Screen ticker symbols and return ranked Monte Carlo opportunities."""

        return scan_tickers_tool(tickers, days=days, scenarios=scenarios)

    @function_tool
    def check_ticker_risk(ticker: str, scenarios: int = 2000) -> str:
        """Run a deeper Monte Carlo risk check for one ticker."""

        return risk_check_tool(ticker, scenarios=scenarios)

    @function_tool
    def compare_model_assumptions(ticker: str, scenarios: int = 1000) -> str:
        """Compare historical bootstrap and GBM assumptions for one ticker."""

        return what_if_tool(ticker, scenarios=scenarios)

    @function_tool
    def check_rebalance(current_holdings_json: str) -> str:
        """Evaluate whether a ticker-to-weight portfolio should rebalance."""

        return rebalance_signal_tool(current_holdings_json)

    return Agent(
        name="Monte Carlo Decision Agent",
        instructions=OPENAI_AGENT_INSTRUCTIONS,
        model=model,
        tools=[
            scan_tickers,
            check_ticker_risk,
            compare_model_assumptions,
            check_rebalance,
        ],
    )


async def run_openai_decision_agent(
    prompt: str,
    *,
    model: str = DEFAULT_OPENAI_AGENT_MODEL,
) -> str:
    """Run the OpenAI decision agent and return its final output."""

    agent = build_openai_decision_agent(model=model)
    _Agent, Runner, _function_tool = _load_openai_agents()
    result = await Runner.run(agent, prompt)
    output = getattr(result, "final_output", result)
    return str(output)


def _load_claude_agent_sdk() -> tuple[Any, Any]:
    try:
        module = _import_module("claude_agent_sdk")
    except ImportError as exc:
        raise AgentDependencyError(
            "Claude Agent SDK is required. Install `monte-carlo-sim[agents]` "
            "or `pip install claude-agent-sdk`."
        ) from exc
    return module.query, module.ClaudeAgentOptions


def build_claude_agent_options(
    *,
    allowed_tools: Iterable[str] | None = None,
) -> Any:
    """Build Claude Agent SDK options with a conservative tool allowlist."""

    _query, ClaudeAgentOptions = _load_claude_agent_sdk()
    return ClaudeAgentOptions(allowed_tools=list(allowed_tools or DEFAULT_CLAUDE_TOOLS))


async def run_claude_repo_agent(
    prompt: str,
    *,
    allowed_tools: Iterable[str] | None = None,
) -> str:
    """Run Claude Agent SDK for repo-aware analysis or maintenance tasks."""

    query, _ClaudeAgentOptions = _load_claude_agent_sdk()
    options = build_claude_agent_options(allowed_tools=allowed_tools)
    result_text = ""
    async for message in query(
        prompt=f"{CLAUDE_REPO_AGENT_PROMPT}\n\nTask:\n{prompt}",
        options=options,
    ):
        if hasattr(message, "result"):
            result_text = str(message.result)
    return result_text


__all__ = [
    "AgentDependencyError",
    "build_claude_agent_options",
    "build_openai_decision_agent",
    "rebalance_signal_tool",
    "risk_check_tool",
    "run_claude_repo_agent",
    "run_openai_decision_agent",
    "scan_tickers_tool",
    "what_if_tool",
]
