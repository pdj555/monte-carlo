from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from analysis import summarize_equal_weight_portfolio, summarize_final_prices


def test_summarize_final_prices_reports_key_metrics():
    base = np.linspace(100, 120, 50)
    df = pd.DataFrame({i: base * (1 + 0.01 * i) for i in range(1, 6)})

    summary = summarize_final_prices(df, current_price=100.0, quantiles=(0.1, 0.9))

    assert {
        "mean",
        "median",
        "std",
        "expected_return",
        "value_at_risk_95",
        "expected_shortfall_95",
        "value_at_risk_99",
        "expected_shortfall_99",
    } <= set(summary.index)
    assert 0.0 <= summary["prob_above_current"] <= 1.0
    assert 0.0 <= summary["prob_below_current"] <= 1.0
    assert summary["q10"] <= summary["q90"]
    assert summary["value_at_risk_99"] >= summary["value_at_risk_95"]
    assert summary["expected_shortfall_95"] >= summary["value_at_risk_95"]


def test_summarize_final_prices_requires_data():
    with pytest.raises(ValueError):
        summarize_final_prices(pd.DataFrame())


def test_summarize_equal_weight_portfolio_combines_tickers():
    sims = pd.DataFrame(
        {
            ("AAPL", 0): [100.0, 110.0],
            ("AAPL", 1): [100.0, 120.0],
            ("MSFT", 0): [50.0, 55.0],
            ("MSFT", 1): [50.0, 50.0],
        }
    )
    sims.columns = pd.MultiIndex.from_tuples(sims.columns, names=["ticker", "scenario"])

    summary = summarize_equal_weight_portfolio(
        sims,
        current_prices={"AAPL": 100.0, "MSFT": 50.0},
    )

    assert summary["component_count"] == 2.0
    assert summary["mean"] == pytest.approx(1.1)
    assert summary["expected_return"] == pytest.approx(0.1)
