from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from analysis import (
    summarize_equal_weight_portfolio,
    summarize_final_prices,
    summarize_weighted_portfolio,
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
        "avg_upside_pct",
        "avg_downside_pct",
        "payoff_ratio",
        "kelly_fraction",
    } <= set(summary.index)
    assert 0.0 <= summary["prob_above_current"] <= 1.0
    assert 0.0 <= summary["prob_below_current"] <= 1.0
    assert summary["q10"] <= summary["q90"]
    assert summary["value_at_risk_99"] >= summary["value_at_risk_95"]
    assert summary["expected_shortfall_95"] >= summary["value_at_risk_95"]
    assert 0.0 <= summary["max_drawdown_q95"] <= 1.0
    assert 0.0 <= summary["prob_drawdown_20_pct"] <= 1.0
    assert 0.0 <= summary["kelly_fraction"] <= 1.0


def test_summarize_final_prices_reports_target_and_loss_probabilities():
    df = pd.DataFrame(
        {
            0: [100.0, 110.0],
            1: [100.0, 120.0],
            2: [100.0, 95.0],
            3: [100.0, 85.0],
        }
    )

    summary = summarize_final_prices(
        df,
        current_price=100.0,
        target_return_pct=0.1,
        max_loss_pct=0.1,
    )

    assert summary["target_return_pct"] == pytest.approx(0.1)
    assert summary["max_loss_pct"] == pytest.approx(0.1)
    assert summary["prob_hit_target"] == pytest.approx(0.5)
    assert summary["prob_breach_max_loss"] == pytest.approx(0.25)


def test_summarize_final_prices_reports_path_touch_metrics_for_targets_and_stops():
    df = pd.DataFrame(
        {
            0: [100.0, 108.0, 111.0],
            1: [100.0, 96.0, 89.0],
            2: [100.0, 101.0, 103.0],
            3: [100.0, 112.0, 109.0],
        },
        index=pd.Index([1, 2, 3], name="day"),
    )

    summary = summarize_final_prices(
        df,
        current_price=100.0,
        target_return_pct=0.1,
        max_loss_pct=0.1,
    )

    assert summary["prob_hit_target"] == pytest.approx(0.25)
    assert summary["prob_touch_target"] == pytest.approx(0.5)
    assert summary["median_days_to_target"] == pytest.approx(2.5)
    assert summary["prob_breach_max_loss"] == pytest.approx(0.25)
    assert summary["prob_touch_max_loss"] == pytest.approx(0.25)
    assert summary["median_days_to_max_loss"] == pytest.approx(3.0)


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


def test_summarize_weighted_portfolio_keeps_uninvested_cash_stable():
    sims = pd.DataFrame(
        {
            ("AAPL", 0): [100.0, 80.0],
            ("AAPL", 1): [100.0, 120.0],
            ("MSFT", 0): [50.0, 50.0],
            ("MSFT", 1): [50.0, 60.0],
        }
    )
    sims.columns = pd.MultiIndex.from_tuples(sims.columns, names=["ticker", "scenario"])

    summary = summarize_weighted_portfolio(
        sims,
        current_prices={"AAPL": 100.0, "MSFT": 50.0},
        weights={"AAPL": 0.25, "MSFT": 0.25},
    )

    assert summary["invested_weight"] == pytest.approx(0.5)
    assert summary["cash_weight"] == pytest.approx(0.5)
    assert summary["component_count"] == pytest.approx(2.0)
    assert summary["mean"] == pytest.approx(1.025)


def test_summarize_weighted_portfolio_rejects_leverage():
    sims = pd.DataFrame({("AAPL", 0): [100.0, 101.0]})
    sims.columns = pd.MultiIndex.from_tuples(sims.columns, names=["ticker", "scenario"])

    with pytest.raises(ValueError, match="cannot sum above 1.0"):
        summarize_weighted_portfolio(
            sims,
            current_prices={"AAPL": 100.0},
            weights={"AAPL": 1.2},
        )


def test_summarize_equal_weight_portfolio_uses_full_paths_for_drawdown_metrics():
    sims = pd.DataFrame(
        {
            ("AAPL", 0): [100.0, 0.0, 100.0],
            ("AAPL", 1): [100.0, 0.0, 100.0],
            ("MSFT", 0): [100.0, 100.0, 100.0],
            ("MSFT", 1): [100.0, 100.0, 100.0],
        }
    )
    sims.columns = pd.MultiIndex.from_tuples(sims.columns, names=["ticker", "scenario"])

    summary = summarize_equal_weight_portfolio(
        sims,
        current_prices={"AAPL": 100.0, "MSFT": 100.0},
    )

    assert summary["max_drawdown_q95"] == pytest.approx(0.5)
    assert summary["prob_drawdown_20_pct"] == pytest.approx(1.0)


def test_summarize_final_prices_reports_benchmark_metrics():
    df = pd.DataFrame(
        {
            0: [100.0, 106.0],
            1: [100.0, 102.0],
            2: [100.0, 98.0],
            3: [100.0, 95.0],
        }
    )

    summary = summarize_final_prices(
        df,
        current_price=100.0,
        benchmark_return_pct=0.02,
    )

    assert summary["benchmark_return_pct"] == pytest.approx(0.02)
    assert summary["expected_excess_return"] == pytest.approx(summary["expected_return"] - 0.02)
    assert summary["prob_beat_benchmark"] == pytest.approx(0.5)
