from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from analysis import (
    apply_risk_guards,
    build_action_plan,
    build_execution_plan,
    enforce_portfolio_risk_budget,
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
        }
    )

    summary = summarize_final_prices(
        df,
        current_price=100.0,
        target_return_pct=0.1,
        max_loss_pct=0.1,
    )

    assert summary["prob_hit_target"] == pytest.approx(0.25)
    assert summary["prob_touch_target"] == pytest.approx(0.5)
    assert summary["median_days_to_target"] == pytest.approx(1.5)
    assert summary["prob_breach_max_loss"] == pytest.approx(0.25)
    assert summary["prob_touch_max_loss"] == pytest.approx(0.25)
    assert summary["median_days_to_max_loss"] == pytest.approx(2.0)

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


def test_summarize_final_prices_reports_benchmark_metrics():
    df = pd.DataFrame({
        0: [100.0, 106.0],
        1: [100.0, 102.0],
        2: [100.0, 98.0],
        3: [100.0, 95.0],
    })

    summary = summarize_final_prices(
        df,
        current_price=100.0,
        benchmark_return_pct=0.02,
    )

    assert summary["benchmark_return_pct"] == pytest.approx(0.02)
    assert summary["expected_excess_return"] == pytest.approx(summary["expected_return"] - 0.02)
    assert summary["prob_beat_benchmark"] == pytest.approx(0.5)


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


def test_rank_tickers_penalizes_negative_excess_return():
    summaries = pd.DataFrame(
        {
            "expected_return": {"AAPL": 0.08, "MSFT": 0.11},
            "expected_excess_return": {"AAPL": 0.03, "MSFT": -0.01},
            "prob_above_current": {"AAPL": 0.58, "MSFT": 0.62},
            "prob_beat_benchmark": {"AAPL": 0.56, "MSFT": 0.42},
            "value_at_risk_95_pct": {"AAPL": 0.07, "MSFT": 0.06},
        }
    )

    ranked = rank_tickers(summaries)

    assert list(ranked.index) == ["AAPL", "MSFT"]


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


def test_rank_tickers_uses_kelly_fraction_as_conviction_boost():
    summaries = pd.DataFrame(
        {
            "expected_return": {"AAPL": 0.1, "MSFT": 0.1},
            "prob_above_current": {"AAPL": 0.58, "MSFT": 0.58},
            "value_at_risk_95_pct": {"AAPL": 0.08, "MSFT": 0.08},
            "kelly_fraction": {"AAPL": 0.75, "MSFT": 0.05},
        }
    )

    ranked = rank_tickers(summaries)

    assert list(ranked.index) == ["AAPL", "MSFT"]


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


def test_recommend_allocations_respects_kelly_signal_boost():
    rankings = pd.DataFrame(
        {
            "score": {"AAPL": 10.0, "MSFT": 10.0},
            "value_at_risk_95_pct": {"AAPL": 0.08, "MSFT": 0.08},
            "kelly_fraction": {"AAPL": 0.7, "MSFT": 0.1},
            "recommendation": {"AAPL": "BUY", "MSFT": "BUY"},
        }
    )

    allocation = recommend_allocations(rankings)

    assert allocation.loc["AAPL", "weight"] > allocation.loc["MSFT", "weight"]


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


def test_apply_risk_guards_can_enforce_target_and_loss_breach_probabilities():
    rankings = pd.DataFrame(
        {
            "score": {"AAPL": 12.0, "TSLA": 13.0},
            "expected_return": {"AAPL": 0.12, "TSLA": 0.14},
            "prob_above_current": {"AAPL": 0.62, "TSLA": 0.61},
            "value_at_risk_95_pct": {"AAPL": 0.08, "TSLA": 0.09},
            "prob_hit_target": {"AAPL": 0.42, "TSLA": 0.30},
            "prob_breach_max_loss": {"AAPL": 0.16, "TSLA": 0.28},
            "recommendation": {"AAPL": "BUY", "TSLA": "BUY"},
        }
    )

    guarded = apply_risk_guards(
        rankings,
        min_prob_hit_target=0.4,
        max_prob_breach_loss=0.2,
    )

    assert guarded.loc["AAPL", "recommendation"] == "BUY"
    assert guarded.loc["TSLA", "recommendation"] == "AVOID"
    assert "prob_hit_target<40%" in guarded.loc["TSLA", "guardrail_reasons"]
    assert "prob_breach_max_loss>20%" in guarded.loc["TSLA", "guardrail_reasons"]


def test_enforce_portfolio_risk_budget_scales_weights_when_budget_exceeded():
    rankings = pd.DataFrame(
        {
            "value_at_risk_95_pct": {"AAPL": 0.20, "MSFT": 0.10},
        }
    )
    allocations = pd.DataFrame(
        {
            "weight": {"AAPL": 0.6, "MSFT": 0.4},
        }
    )

    constrained = enforce_portfolio_risk_budget(
        allocations,
        rankings,
        max_portfolio_var_95_pct=0.08,
    )

    blended = float((constrained["weight"] * rankings["value_at_risk_95_pct"]).sum())
    assert blended == pytest.approx(0.08)
    assert constrained["weight"].sum() < allocations["weight"].sum()


def test_enforce_portfolio_risk_budget_keeps_weights_when_under_budget():
    rankings = pd.DataFrame(
        {
            "value_at_risk_95_pct": {"AAPL": 0.05, "MSFT": 0.03},
        }
    )
    allocations = pd.DataFrame(
        {
            "weight": {"AAPL": 0.5, "MSFT": 0.5},
        }
    )

    constrained = enforce_portfolio_risk_budget(
        allocations,
        rankings,
        max_portfolio_var_95_pct=0.08,
    )

    pd.testing.assert_frame_equal(constrained, allocations)


def test_build_execution_plan_rounds_to_whole_shares_by_default():
    allocations = pd.DataFrame(
        {
            "weight": {"AAPL": 0.6, "MSFT": 0.4},
            "score": {"AAPL": 12.0, "MSFT": 9.0},
            "value_at_risk_95_pct": {"AAPL": 0.1, "MSFT": 0.06},
        }
    )

    plan = build_execution_plan(
        allocations,
        current_prices={"AAPL": 101.0, "MSFT": 50.0},
        capital=1000.0,
    )

    assert plan.loc["AAPL", "shares"] == pytest.approx(5.0)
    assert plan.loc["MSFT", "shares"] == pytest.approx(8.0)
    assert plan.loc["AAPL", "cash_drift"] == pytest.approx(95.0)


def test_build_execution_plan_supports_fractional_shares():
    allocations = pd.DataFrame({"weight": {"AAPL": 0.5}})

    plan = build_execution_plan(
        allocations,
        current_prices={"AAPL": 120.0},
        capital=1000.0,
        allow_fractional_shares=True,
    )

    assert plan.loc["AAPL", "shares"] == pytest.approx(1000.0 * 0.5 / 120.0)
    assert plan.loc["AAPL", "cash_drift"] == pytest.approx(0.0)
