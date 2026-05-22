"""Tests for the MCP server protocol handling."""

from __future__ import annotations

import json
from io import StringIO
from unittest.mock import patch

import mcp_server


class TestProtocol:
    def test_initialize_response(self):
        msg = {"jsonrpc": "2.0", "id": 1, "method": "initialize", "params": {}}
        stdin = StringIO(json.dumps(msg) + "\n")
        stdout = StringIO()

        with patch("sys.stdin", stdin), patch("sys.stdout", stdout):
            mcp_server.serve()

        response = json.loads(stdout.getvalue().strip())
        assert response["id"] == 1
        assert "protocolVersion" in response["result"]
        assert response["result"]["serverInfo"]["name"] == "monte-carlo-sim"

    def test_tools_list(self):
        msgs = [
            {"jsonrpc": "2.0", "id": 1, "method": "initialize", "params": {}},
            {"jsonrpc": "2.0", "id": 2, "method": "tools/list", "params": {}},
        ]
        stdin = StringIO("\n".join(json.dumps(m) for m in msgs) + "\n")
        stdout = StringIO()

        with patch("sys.stdin", stdin), patch("sys.stdout", stdout):
            mcp_server.serve()

        lines = [line for line in stdout.getvalue().strip().split("\n") if line]
        assert len(lines) == 2
        tools_response = json.loads(lines[1])
        tool_names = [t["name"] for t in tools_response["result"]["tools"]]
        assert "analyze_ticker" in tool_names
        assert "analyze_portfolio" in tool_names
        assert "screen_tickers" in tool_names
        assert "compare_tickers" in tool_names

    def test_unknown_method_returns_error(self):
        msg = {"jsonrpc": "2.0", "id": 1, "method": "unknown/method", "params": {}}
        stdin = StringIO(json.dumps(msg) + "\n")
        stdout = StringIO()

        with patch("sys.stdin", stdin), patch("sys.stdout", stdout):
            mcp_server.serve()

        response = json.loads(stdout.getvalue().strip())
        assert "error" in response
        assert response["error"]["code"] == -32601

    def test_unknown_tool_returns_tool_error(self):
        msg = {
            "jsonrpc": "2.0",
            "id": 1,
            "method": "tools/call",
            "params": {"name": "nonexistent_tool", "arguments": {}},
        }
        stdin = StringIO(json.dumps(msg) + "\n")
        stdout = StringIO()

        with patch("sys.stdin", stdin), patch("sys.stdout", stdout):
            mcp_server.serve()

        response = json.loads(stdout.getvalue().strip())
        assert response["result"]["isError"] is True


class TestToolExecution:
    def test_analyze_ticker_tool(self):
        import pandas as pd

        prices = pd.Series(
            [100.0 + i * 0.1 for i in range(60)],
            index=pd.bdate_range("2023-01-01", periods=60),
            name="Close",
        )

        msg = {
            "jsonrpc": "2.0",
            "id": 1,
            "method": "tools/call",
            "params": {
                "name": "analyze_ticker",
                "arguments": {"ticker": "AAPL", "days": 5, "scenarios": 10, "seed": 42},
            },
        }
        stdin = StringIO(json.dumps(msg) + "\n")
        stdout = StringIO()

        with (
            patch("sys.stdin", stdin),
            patch("sys.stdout", stdout),
            patch("sdk.fetch_prices", return_value=prices),
        ):
            mcp_server.serve()

        response = json.loads(stdout.getvalue().strip())
        assert "result" in response
        content = response["result"]["content"][0]["text"]
        parsed = json.loads(content)
        assert parsed["ticker"] == "AAPL"
        assert "summary" in parsed
