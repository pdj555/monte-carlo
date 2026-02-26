from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from analysis import (
    apply_risk_guards,
    build_action_plan,
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
    assert 0.0 <= summary["max_drawdown_q95"] <= 1.0
    assert 0.0 <= summary["prob_drawdown_20_pct"] <= 1.0


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


def test_rank_tickers_prefers_expected_shortfall_when_available():
    summaries = pd.DataFrame(
        {
            "expected_return": {"AAPL": 0.12, "MSFT": 0.12},
            "prob_above_current": {"AAPL": 0.6, "MSFT": 0.6},
            "value_at_risk_95_pct": {"AAPL": 0.05, "MSFT": 0.09},
            "expected_shortfall_95_pct": {"AAPL": 0.15, "MSFT": 0.06},
        }
    )

    ranked = rank_tickers(summaries)

    assert list(ranked.index) == ["MSFT", "AAPL"]


def test_recommend_allocations_uses_expected_shortfall_when_available():
    rankings = pd.DataFrame(
        {
            "score": {"AAPL": 10.0, "MSFT": 10.0},
            "value_at_risk_95_pct": {"AAPL": 0.05, "MSFT": 0.2},
            "expected_shortfall_95_pct": {"AAPL": 0.25, "MSFT": 0.05},
            "recommendation": {"AAPL": "BUY", "MSFT": "BUY"},
        }
    )

    allocation = recommend_allocations(rankings)

    assert allocation.loc["MSFT", "weight"] > allocation.loc["AAPL", "weight"]


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


def test_build_action_plan_returns_risk_on_for_high_conviction():
    rankings = pd.DataFrame(
        {
            "score": {"AAPL": 16.0, "MSFT": 6.0, "TSLA": -4.0},
            "expected_return": {"AAPL": 0.20, "MSFT": 0.09, "TSLA": -0.01},
            "prob_above_current": {"AAPL": 0.65, "MSFT": 0.58, "TSLA": 0.45},
            "value_at_risk_95_pct": {"AAPL": 0.08, "MSFT": 0.05, "TSLA": 0.25},
            "recommendation": {"AAPL": "BUY", "MSFT": "WATCH", "TSLA": "AVOID"},
        }
    )
    allocations = pd.DataFrame(
        {
            "score": {"AAPL": 16.0, "MSFT": 6.0},
            "value_at_risk_95_pct": {"AAPL": 0.08, "MSFT": 0.05},
            "weight": {"AAPL": 0.62, "MSFT": 0.38},
        }
    )

    plan = build_action_plan(rankings, allocations)

    assert plan["stance"] == "RISK_ON"
    assert plan["primary_pick"]["ticker"] == "AAPL"
    assert plan["avoid_list"] == ["TSLA"]




def test_recommend_allocations_leaves_cash_when_cap_is_binding():
    rankings = pd.DataFrame(
        {
            "score": {"AAPL": 15.0},
            "value_at_risk_95_pct": {"AAPL": 0.08},
            "recommendation": {"AAPL": "BUY"},
        }
    )

    allocation = recommend_allocations(rankings, max_weight=0.6)

    assert allocation.loc["AAPL", "weight"] == pytest.approx(0.6)


def test_build_action_plan_reports_cash_buffer_when_not_fully_invested():
    rankings = pd.DataFrame(
        {
            "score": {"AAPL": 15.0},
            "expected_return": {"AAPL": 0.2},
            "prob_above_current": {"AAPL": 0.62},
            "value_at_risk_95_pct": {"AAPL": 0.08},
            "recommendation": {"AAPL": "BUY"},
        }
    )
    allocations = pd.DataFrame(
        {
            "score": {"AAPL": 15.0},
            "value_at_risk_95_pct": {"AAPL": 0.08},
            "weight": {"AAPL": 0.6},
        }
    )

    plan = build_action_plan(rankings, allocations)

    assert plan["cash_weight"] == pytest.approx(0.4)
    assert "Keep 40.0% in cash." in plan["headline"]

def test_build_action_plan_returns_no_trade_when_empty():
    plan = build_action_plan(pd.DataFrame(), pd.DataFrame())

    assert plan["stance"] == "NO_TRADE"
    assert plan["primary_pick"] is None


def test_apply_risk_guards_marks_failures_as_avoid():
    rankings = pd.DataFrame(
        {
            "score": {"AAPL": 12.0, "TSLA": 11.0},
            "expected_return": {"AAPL": 0.12, "TSLA": 0.09},
            "prob_above_current": {"AAPL": 0.64, "TSLA": 0.49},
            "value_at_risk_95_pct": {"AAPL": 0.09, "TSLA": 0.31},
            "max_drawdown_q95": {"AAPL": 0.12, "TSLA": 0.44},
            "recommendation": {"AAPL": "BUY", "TSLA": "BUY"},
        }
    )

    guarded = apply_risk_guards(
        rankings,
        min_expected_return=0.1,
        min_prob_above_current=0.55,
        max_value_at_risk_95_pct=0.2,
    )

    assert guarded.loc["AAPL", "recommendation"] == "BUY"
    assert guarded.loc["TSLA", "recommendation"] == "AVOID"
    assert "prob_above_current<55%" in guarded.loc["TSLA", "guardrail_reasons"]


def test_apply_risk_guards_enforces_drawdown_cap_when_configured():
    rankings = pd.DataFrame(
        {
            "score": {"AAPL": 13.0, "TSLA": 14.0},
            "expected_return": {"AAPL": 0.13, "TSLA": 0.2},
            "prob_above_current": {"AAPL": 0.63, "TSLA": 0.61},
            "value_at_risk_95_pct": {"AAPL": 0.08, "TSLA": 0.11},
            "max_drawdown_q95": {"AAPL": 0.25, "TSLA": 0.42},
            "recommendation": {"AAPL": "BUY", "TSLA": "BUY"},
        }
    )

    guarded = apply_risk_guards(rankings, max_drawdown_q95=0.3)

    assert guarded.loc["AAPL", "recommendation"] == "BUY"
    assert guarded.loc["TSLA", "recommendation"] == "AVOID"
    assert "max_drawdown_q95>30.0%" in guarded.loc["TSLA", "guardrail_reasons"]
