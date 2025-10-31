from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from analysis import (
    summarize_final_prices,
    calculate_risk_metrics,
    calculate_sharpe_ratio,
    calculate_sortino_ratio,
    calculate_max_drawdown,
)


def test_summarize_final_prices_reports_key_metrics():
    base = np.linspace(100, 120, 50)
    df = pd.DataFrame({i: base * (1 + 0.01 * i) for i in range(1, 6)})

    summary = summarize_final_prices(df, current_price=100.0, quantiles=(0.1, 0.9))

    assert {"mean", "median", "std", "expected_return", "value_at_risk_95"} <= set(summary.index)
    assert summary["prob_above_current"] > 0.0
    assert summary["q10"] <= summary["q90"]


def test_summarize_final_prices_requires_data():
    with pytest.raises(ValueError):
        summarize_final_prices(pd.DataFrame())


def test_summarize_final_prices_includes_cvar():
    """Test that CVaR is included when current_price is provided."""
    base = np.linspace(100, 120, 50)
    df = pd.DataFrame({i: base * (1 + 0.01 * i) for i in range(1, 6)})

    summary = summarize_final_prices(df, current_price=100.0)

    assert "cvar_95" in summary.index
    assert isinstance(summary["cvar_95"], (int, float))


def test_calculate_sharpe_ratio():
    """Test Sharpe ratio calculation."""
    # Create simple upward trending data
    base = np.linspace(100, 110, 100)
    df = pd.DataFrame({i: base * (1 + 0.001 * i) for i in range(1, 6)})

    sharpe = calculate_sharpe_ratio(df, risk_free_rate=0.02)

    assert isinstance(sharpe, float)
    # Sharpe ratio can be quite large for smooth upward trends
    assert not np.isnan(sharpe)


def test_calculate_sortino_ratio():
    """Test Sortino ratio calculation."""
    base = np.linspace(100, 110, 100)
    df = pd.DataFrame({i: base * (1 + 0.001 * i) for i in range(1, 6)})

    sortino = calculate_sortino_ratio(df, target_return=0.0)

    assert isinstance(sortino, float)
    # Sortino can be very large for smooth upward trends with minimal downside
    assert not np.isnan(sortino)
    assert sortino > 0  # Should be positive for upward trending data


def test_calculate_max_drawdown():
    """Test maximum drawdown calculation."""
    # Create data with a known drawdown
    prices = [100, 110, 105, 95, 100, 110, 115]
    df = pd.DataFrame({"scenario1": prices, "scenario2": [p * 1.1 for p in prices]})

    dd_metrics = calculate_max_drawdown(df)

    assert "max_drawdown" in dd_metrics
    assert "avg_drawdown" in dd_metrics
    assert "median_drawdown" in dd_metrics
    # Drawdowns should be negative or zero
    assert dd_metrics["max_drawdown"] <= 0
    assert dd_metrics["avg_drawdown"] <= 0


def test_calculate_risk_metrics():
    """Test comprehensive risk metrics calculation."""
    base = np.linspace(100, 110, 100)
    df = pd.DataFrame({i: base * (1 + 0.001 * i) for i in range(1, 6)})

    metrics = calculate_risk_metrics(df, current_price=100.0, risk_free_rate=0.02)

    # Check that all expected metrics are present
    expected_metrics = {
        "sharpe_ratio",
        "sortino_ratio",
        "max_drawdown",
        "avg_drawdown",
        "median_drawdown",
        "var_90",
        "var_95",
        "var_99",
        "cvar_90",
        "cvar_95",
        "cvar_99",
    }
    assert expected_metrics <= set(metrics.index)

    # Check that metrics are reasonable
    assert isinstance(metrics["sharpe_ratio"], (int, float))
    assert isinstance(metrics["sortino_ratio"], (int, float))
    # VaR can be negative (profit) if prices go up
    assert isinstance(metrics["var_95"], (int, float))
    assert isinstance(metrics["cvar_95"], (int, float))


def test_risk_metrics_empty_dataframe():
    """Test that empty DataFrame raises ValueError."""
    with pytest.raises(ValueError):
        calculate_risk_metrics(pd.DataFrame())
    with pytest.raises(ValueError):
        calculate_sharpe_ratio(pd.DataFrame())
    with pytest.raises(ValueError):
        calculate_sortino_ratio(pd.DataFrame())
    with pytest.raises(ValueError):
        calculate_max_drawdown(pd.DataFrame())
