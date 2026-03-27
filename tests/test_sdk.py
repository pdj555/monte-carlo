"""Tests for the programmatic SDK."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

import pandas as pd
import pytest

from sdk import MonteCarloSDK, TickerResult, PortfolioResult, ScreenResult


@pytest.fixture()
def sample_prices():
    """Return a simple price series for offline testing."""
    dates = pd.bdate_range("2023-01-01", periods=120)
    prices = pd.Series(
        [100.0 + i * 0.1 + (i % 7 - 3) * 0.5 for i in range(120)],
        index=dates,
        name="Close",
    )
    return prices


@pytest.fixture()
def sdk_with_mock(sample_prices):
    """Return an SDK instance with mocked fetch_prices."""
    sdk = MonteCarloSDK(offline_only=True)
    with patch("sdk.fetch_prices", return_value=sample_prices):
        yield sdk


class TestTickerAnalysis:
    def test_analyze_returns_ticker_result(self, sdk_with_mock, sample_prices):
        with patch("sdk.fetch_prices", return_value=sample_prices):
            result = sdk_with_mock.analyze("AAPL", days=10, scenarios=50, seed=42)
        assert isinstance(result, TickerResult)
        assert result.ticker == "AAPL"
        assert result.days == 10
        assert result.scenarios == 50
        assert result.current_price > 0
        assert "expected_return" in result.summary
        assert "prob_above_current" in result.summary

    def test_analyze_to_dict(self, sdk_with_mock, sample_prices):
        with patch("sdk.fetch_prices", return_value=sample_prices):
            result = sdk_with_mock.analyze("AAPL", days=10, scenarios=50, seed=42)
        d = result.to_dict()
        assert d["ticker"] == "AAPL"
        assert isinstance(d["summary"], dict)

    def test_analyze_to_json(self, sdk_with_mock, sample_prices):
        import json

        with patch("sdk.fetch_prices", return_value=sample_prices):
            result = sdk_with_mock.analyze("AAPL", days=10, scenarios=50, seed=42)
        j = result.to_json()
        parsed = json.loads(j)
        assert parsed["ticker"] == "AAPL"

    def test_analyze_reproducible_with_seed(self, sdk_with_mock, sample_prices):
        with patch("sdk.fetch_prices", return_value=sample_prices):
            r1 = sdk_with_mock.analyze("AAPL", days=10, scenarios=50, seed=42)
            r2 = sdk_with_mock.analyze("AAPL", days=10, scenarios=50, seed=42)
        assert r1.summary["mean"] == r2.summary["mean"]


class TestAnalyzeMany:
    def test_concurrent_analysis(self, sdk_with_mock, sample_prices):
        with patch("sdk.fetch_prices", return_value=sample_prices):
            results = sdk_with_mock.analyze_many(
                ["AAPL", "MSFT"], days=10, scenarios=50, seed=42,
            )
        assert "AAPL" in results
        assert "MSFT" in results
        assert isinstance(results["AAPL"], TickerResult)

    def test_failed_ticker_returns_error(self, sdk_with_mock):
        from data import PriceDataError

        def failing_fetch(ticker, **kwargs):
            raise PriceDataError(f"No data for {ticker}")

        with patch("sdk.fetch_prices", side_effect=failing_fetch):
            results = sdk_with_mock.analyze_many(
                ["FAKE"], days=10, scenarios=50,
            )
        assert isinstance(results["FAKE"], PriceDataError)


class TestPortfolio:
    def test_portfolio_returns_structured_result(self, sdk_with_mock, sample_prices):
        with patch("sdk.fetch_prices", return_value=sample_prices):
            result = sdk_with_mock.portfolio(
                ["AAPL", "MSFT"], days=10, scenarios=50, seed=42,
            )
        assert isinstance(result, PortfolioResult)
        assert "AAPL" in result.ticker_results or "MSFT" in result.ticker_results
        assert isinstance(result.action_plan, dict)
        assert "stance" in result.action_plan

    def test_portfolio_with_capital(self, sdk_with_mock, sample_prices):
        with patch("sdk.fetch_prices", return_value=sample_prices):
            result = sdk_with_mock.portfolio(
                ["AAPL"], days=10, scenarios=50, seed=42, capital=10000.0,
            )
        assert isinstance(result.execution_plan, dict)


class TestScreen:
    def test_screen_categorizes_tickers(self, sdk_with_mock, sample_prices):
        with patch("sdk.fetch_prices", return_value=sample_prices):
            result = sdk_with_mock.screen(
                ["AAPL", "MSFT"], days=10, scenarios=50, seed=42,
            )
        assert isinstance(result, ScreenResult)
        all_tickers = result.buy + result.watch + result.avoid
        assert len(all_tickers) > 0
        assert isinstance(result.headline, str)


class TestCompare:
    def test_compare_returns_structured_dict(self, sdk_with_mock, sample_prices):
        with patch("sdk.fetch_prices", return_value=sample_prices):
            result = sdk_with_mock.compare(
                ["AAPL", "MSFT"], days=10, scenarios=50, seed=42,
            )
        assert "tickers" in result
        assert "days" in result
