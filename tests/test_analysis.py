from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from analysis import (
    rank_tickers,
    recommend_allocations,
    summarize_equal_weight_portfolio,
    summarize_final_prices,
)


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


def test_rank_tickers_orders_by_score():
    summaries = pd.DataFrame(
        {
            "expected_return": {"AAPL": 0.15, "MSFT": 0.08, "TSLA": 0.22},
            "prob_above_current": {"AAPL": 0.62, "MSFT": 0.55, "TSLA": 0.49},
            "value_at_risk_95_pct": {"AAPL": 0.09, "MSFT": 0.04, "TSLA": 0.25},
        }
    )

    ranked = rank_tickers(summaries)

    assert list(ranked.index) == ["AAPL", "MSFT", "TSLA"]
    assert ranked.loc["AAPL", "recommendation"] == "BUY"
    assert ranked.loc["TSLA", "recommendation"] == "AVOID"


def test_recommend_allocations_outputs_normalized_weights():
    rankings = pd.DataFrame(
        {
            "score": {"AAPL": 14.0, "MSFT": 7.0, "TSLA": -2.0},
            "value_at_risk_95_pct": {"AAPL": 0.10, "MSFT": 0.04, "TSLA": 0.25},
            "recommendation": {"AAPL": "BUY", "MSFT": "WATCH", "TSLA": "AVOID"},
        }
    )

    allocation = recommend_allocations(rankings, max_weight=0.7)

    assert list(allocation.index) == ["AAPL", "MSFT"]
    assert allocation["weight"].sum() == pytest.approx(1.0)
    assert allocation["weight"].max() <= 0.7


def test_recommend_allocations_returns_empty_when_only_avoid():
    rankings = pd.DataFrame(
        {
            "score": {"TSLA": -5.0},
            "value_at_risk_95_pct": {"TSLA": 0.30},
            "recommendation": {"TSLA": "AVOID"},
        }
    )

    allocation = recommend_allocations(rankings)

    assert allocation.empty
