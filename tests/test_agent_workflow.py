"""Tests for agentic workflow pipelines."""

from __future__ import annotations

from unittest.mock import patch

import pandas as pd
import pytest

from agent_workflow import (
    ScanReport,
    RiskReport,
    WhatIfResult,
    RebalanceSignal,
    opportunity_scan,
    risk_check,
    what_if,
    rebalance_signal,
)


@pytest.fixture()
def sample_prices():
    dates = pd.bdate_range("2023-01-01", periods=120)
    return pd.Series(
        [100.0 + i * 0.1 + (i % 7 - 3) * 0.5 for i in range(120)],
        index=dates,
        name="Close",
    )


class TestOpportunityScan:
    def test_returns_scan_report(self, sample_prices):
        with patch("sdk.fetch_prices", return_value=sample_prices):
            result = opportunity_scan(
                ["AAPL", "MSFT"], days=10, scenarios=50, seed=42,
            )
        assert isinstance(result, ScanReport)
        assert result.universe_size == 2
        assert result.analyzed > 0
        assert isinstance(result.summary, str)

    def test_serializable(self, sample_prices):
        import json

        with patch("sdk.fetch_prices", return_value=sample_prices):
            result = opportunity_scan(["AAPL"], days=10, scenarios=50, seed=42)
        j = result.to_json()
        parsed = json.loads(j)
        assert "buy_signals" in parsed


class TestRiskCheck:
    def test_returns_risk_report(self, sample_prices):
        with patch("sdk.fetch_prices", return_value=sample_prices):
            result = risk_check("AAPL", days=10, scenarios=100, seed=42)
        assert isinstance(result, RiskReport)
        assert result.ticker == "AAPL"
        assert result.risk_level in ("LOW", "MODERATE", "HIGH", "EXTREME")
        assert "value_at_risk_95_pct" in result.metrics
        assert isinstance(result.summary, str)


class TestWhatIf:
    def test_returns_what_if_result(self, sample_prices):
        with patch("sdk.fetch_prices", return_value=sample_prices):
            result = what_if("AAPL", days=10, scenarios=50, seed=42)
        assert isinstance(result, WhatIfResult)
        assert result.ticker == "AAPL"
        assert "historical_bootstrap" in result.scenarios
        assert "geometric_brownian_motion" in result.scenarios
        assert result.best_case_model in (
            "historical_bootstrap",
            "geometric_brownian_motion",
        )


class TestRebalanceSignal:
    def test_returns_rebalance_signal(self, sample_prices):
        with patch("sdk.fetch_prices", return_value=sample_prices):
            result = rebalance_signal(
                {"AAPL": 0.5, "MSFT": 0.5},
                days=10,
                scenarios=50,
                seed=42,
            )
        assert isinstance(result, RebalanceSignal)
        assert result.urgency in ("NONE", "LOW", "MEDIUM", "HIGH")
        assert isinstance(result.should_rebalance, bool)

    def test_empty_holdings(self):
        result = rebalance_signal({})
        assert result.should_rebalance is False
        assert result.urgency == "NONE"
