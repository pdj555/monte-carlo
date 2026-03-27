"""MCP server exposing Monte Carlo simulation tools to AI agents.

This module implements a Model Context Protocol (MCP) server that allows
any MCP-compatible AI agent (Claude, etc.) to directly invoke simulation,
analysis, and portfolio construction tools via structured tool calls.

Usage
-----
Start the server::

    python mcp_server.py

Or use with Claude Code's MCP configuration::

    {
        "mcpServers": {
            "monte-carlo": {
                "command": "python",
                "args": ["mcp_server.py"],
                "cwd": "/path/to/monte-carlo"
            }
        }
    }

The server exposes the following tools:

- ``analyze_ticker`` -- Run Monte Carlo simulation for a single ticker
- ``analyze_portfolio`` -- Full portfolio analysis with ranking and allocation
- ``screen_tickers`` -- Screen tickers into BUY/WATCH/AVOID categories
- ``compare_tickers`` -- Head-to-head ticker comparison
- ``simulate_prices`` -- Raw price path simulation (for advanced use)
"""

from __future__ import annotations

import json
import sys
from typing import Any

# MCP protocol messages are JSON-RPC 2.0 over stdin/stdout
# We implement a minimal server without external dependencies.


def _read_message() -> dict[str, Any] | None:
    """Read a JSON-RPC message from stdin."""
    line = sys.stdin.readline()
    if not line:
        return None
    return json.loads(line.strip())


def _write_message(msg: dict[str, Any]) -> None:
    """Write a JSON-RPC message to stdout."""
    sys.stdout.write(json.dumps(msg) + "\n")
    sys.stdout.flush()


def _success(id: Any, result: Any) -> dict[str, Any]:
    return {"jsonrpc": "2.0", "id": id, "result": result}


def _error(id: Any, code: int, message: str) -> dict[str, Any]:
    return {"jsonrpc": "2.0", "id": id, "error": {"code": code, "message": message}}


# -- Tool definitions -------------------------------------------------------

TOOLS = [
    {
        "name": "analyze_ticker",
        "description": (
            "Run a Monte Carlo simulation for a single stock ticker and return "
            "comprehensive statistics including expected return, probability metrics, "
            "Value at Risk, Expected Shortfall, Kelly criterion, and drawdown analysis."
        ),
        "inputSchema": {
            "type": "object",
            "properties": {
                "ticker": {
                    "type": "string",
                    "description": "Stock ticker symbol (e.g. AAPL, MSFT)",
                },
                "days": {
                    "type": "integer",
                    "description": "Number of future trading days to simulate",
                    "default": 252,
                },
                "scenarios": {
                    "type": "integer",
                    "description": "Number of Monte Carlo scenarios",
                    "default": 1000,
                },
                "model": {
                    "type": "string",
                    "enum": ["historical", "gbm"],
                    "description": "Simulation model to use",
                    "default": "historical",
                },
                "seed": {
                    "type": "integer",
                    "description": "Random seed for reproducibility",
                },
            },
            "required": ["ticker"],
        },
    },
    {
        "name": "analyze_portfolio",
        "description": (
            "Analyze a portfolio of multiple tickers: simulate each, rank by "
            "risk-adjusted score, compute optimal allocations with risk guardrails, "
            "and produce an actionable investment plan. Returns structured JSON "
            "with rankings, allocations, and action plan."
        ),
        "inputSchema": {
            "type": "object",
            "properties": {
                "tickers": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "List of ticker symbols to analyze",
                },
                "days": {
                    "type": "integer",
                    "description": "Simulation horizon in trading days",
                    "default": 252,
                },
                "scenarios": {
                    "type": "integer",
                    "description": "Monte Carlo scenarios per ticker",
                    "default": 1000,
                },
                "model": {
                    "type": "string",
                    "enum": ["historical", "gbm"],
                    "default": "historical",
                },
                "capital": {
                    "type": "number",
                    "description": "Portfolio capital for share sizing (optional)",
                },
                "seed": {"type": "integer"},
            },
            "required": ["tickers"],
        },
    },
    {
        "name": "screen_tickers",
        "description": (
            "Screen a list of tickers and categorize each as BUY, WATCH, or AVOID "
            "based on Monte Carlo simulation results and risk guardrails. "
            "Returns the top pick and a concise headline."
        ),
        "inputSchema": {
            "type": "object",
            "properties": {
                "tickers": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "Tickers to screen",
                },
                "days": {"type": "integer", "default": 252},
                "scenarios": {"type": "integer", "default": 1000},
                "model": {
                    "type": "string",
                    "enum": ["historical", "gbm"],
                    "default": "historical",
                },
                "seed": {"type": "integer"},
            },
            "required": ["tickers"],
        },
    },
    {
        "name": "compare_tickers",
        "description": (
            "Compare two or more tickers head-to-head on key metrics: "
            "expected return, upside probability, Value at Risk, drawdown, "
            "and Kelly fraction. Returns a compact comparison table."
        ),
        "inputSchema": {
            "type": "object",
            "properties": {
                "tickers": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "Tickers to compare",
                },
                "days": {"type": "integer", "default": 252},
                "scenarios": {"type": "integer", "default": 1000},
                "model": {
                    "type": "string",
                    "enum": ["historical", "gbm"],
                    "default": "historical",
                },
                "seed": {"type": "integer"},
            },
            "required": ["tickers"],
        },
    },
]


# -- Tool handlers -----------------------------------------------------------


def _handle_analyze_ticker(params: dict[str, Any]) -> dict[str, Any]:
    from sdk import MonteCarloSDK

    sdk = MonteCarloSDK()
    result = sdk.analyze(
        params["ticker"],
        days=params.get("days", 252),
        scenarios=params.get("scenarios", 1000),
        model=params.get("model", "historical"),
        seed=params.get("seed"),
    )
    return result.to_dict()


def _handle_analyze_portfolio(params: dict[str, Any]) -> dict[str, Any]:
    from sdk import MonteCarloSDK

    sdk = MonteCarloSDK()
    result = sdk.portfolio(
        params["tickers"],
        days=params.get("days", 252),
        scenarios=params.get("scenarios", 1000),
        model=params.get("model", "historical"),
        seed=params.get("seed"),
        capital=params.get("capital"),
    )
    return result.to_dict()


def _handle_screen_tickers(params: dict[str, Any]) -> dict[str, Any]:
    from sdk import MonteCarloSDK

    sdk = MonteCarloSDK()
    result = sdk.screen(
        params["tickers"],
        days=params.get("days", 252),
        scenarios=params.get("scenarios", 1000),
        model=params.get("model", "historical"),
        seed=params.get("seed"),
    )
    return result.to_dict()


def _handle_compare_tickers(params: dict[str, Any]) -> dict[str, Any]:
    from sdk import MonteCarloSDK

    sdk = MonteCarloSDK()
    return sdk.compare(
        params["tickers"],
        days=params.get("days", 252),
        scenarios=params.get("scenarios", 1000),
        model=params.get("model", "historical"),
        seed=params.get("seed"),
    )


HANDLERS = {
    "analyze_ticker": _handle_analyze_ticker,
    "analyze_portfolio": _handle_analyze_portfolio,
    "screen_tickers": _handle_screen_tickers,
    "compare_tickers": _handle_compare_tickers,
}


# -- Server loop -------------------------------------------------------------


def serve() -> None:
    """Run the MCP server, reading JSON-RPC from stdin and writing to stdout."""

    while True:
        msg = _read_message()
        if msg is None:
            break

        method = msg.get("method", "")
        msg_id = msg.get("id")
        params = msg.get("params", {})

        if method == "initialize":
            _write_message(_success(msg_id, {
                "protocolVersion": "2024-11-05",
                "capabilities": {"tools": {}},
                "serverInfo": {
                    "name": "monte-carlo-sim",
                    "version": "1.0.0",
                },
            }))

        elif method == "notifications/initialized":
            pass  # no response needed for notifications

        elif method == "tools/list":
            _write_message(_success(msg_id, {"tools": TOOLS}))

        elif method == "tools/call":
            tool_name = params.get("name", "")
            arguments = params.get("arguments", {})
            handler = HANDLERS.get(tool_name)

            if handler is None:
                _write_message(_success(msg_id, {
                    "content": [{"type": "text", "text": f"Unknown tool: {tool_name}"}],
                    "isError": True,
                }))
            else:
                try:
                    result = handler(arguments)
                    _write_message(_success(msg_id, {
                        "content": [{
                            "type": "text",
                            "text": json.dumps(result, indent=2, default=str),
                        }],
                    }))
                except Exception as exc:
                    _write_message(_success(msg_id, {
                        "content": [{"type": "text", "text": f"Error: {exc}"}],
                        "isError": True,
                    }))

        else:
            if msg_id is not None:
                _write_message(_error(msg_id, -32601, f"Method not found: {method}"))


if __name__ == "__main__":
    serve()
