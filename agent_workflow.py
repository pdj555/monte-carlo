"""Pre-built agentic workflows for Monte Carlo simulation.

These are higher-order operations that compose the SDK primitives into
complete analytical pipelines. Each workflow is designed to be invoked as
a single operation by an AI agent, returning a structured result that the
agent can reason about and present to the user.

Workflows
---------
- ``opportunity_scan`` -- Scan a universe of tickers for the best opportunities
- ``risk_check`` -- Deep risk assessment of a single position or portfolio
- ``what_if`` -- Scenario analysis comparing different market conditions
- ``rebalance_signal`` -- Determine whether a portfolio needs rebalancing
"""

from __future__ import annotations

import json
from dataclasses import dataclass, asdict
from typing import Any, Sequence

from sdk import MonteCarloSDK

__all__ = [
    "opportunity_scan",
    "risk_check",
    "what_if",
    "rebalance_signal",
]


@dataclass(frozen=True)
class ScanReport:
    """Structured output from an opportunity scan."""

    universe_size: int
    analyzed: int
    failed: list[str]
    buy_signals: list[dict[str, Any]]
    watch_list: list[str]
    avoid_list: list[str]
    top_pick: dict[str, Any] | None
    summary: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def to_json(self, **kwargs: Any) -> str:
        return json.dumps(self.to_dict(), **kwargs)


@dataclass(frozen=True)
class RiskReport:
    """Structured output from a risk assessment."""

    ticker: str
    risk_level: str  # LOW, MODERATE, HIGH, EXTREME
    metrics: dict[str, float]
    warnings: list[str]
    summary: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def to_json(self, **kwargs: Any) -> str:
        return json.dumps(self.to_dict(), **kwargs)


@dataclass(frozen=True)
class WhatIfResult:
    """Structured output from a what-if scenario analysis."""

    ticker: str
    scenarios: dict[str, dict[str, Any]]
    best_case_model: str
    worst_case_model: str
    summary: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def to_json(self, **kwargs: Any) -> str:
        return json.dumps(self.to_dict(), **kwargs)


@dataclass(frozen=True)
class RebalanceSignal:
    """Structured rebalance recommendation."""

    should_rebalance: bool
    urgency: str  # NONE, LOW, MEDIUM, HIGH
    current_stance: str
    recommended_changes: list[dict[str, Any]]
    summary: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def to_json(self, **kwargs: Any) -> str:
        return json.dumps(self.to_dict(), **kwargs)


def opportunity_scan(
    tickers: Sequence[str],
    *,
    days: int = 252,
    scenarios: int = 1000,
    model: str = "historical",
    seed: int | None = None,
    min_expected_return: float = 0.0,
    offline_only: bool = False,
) -> ScanReport:
    """Scan a universe of tickers for investment opportunities.

    Runs concurrent simulations across all tickers, applies risk guardrails,
    and returns a structured report with categorized results and a natural
    language summary.
    """

    sdk = MonteCarloSDK(offline_only=offline_only)
    screen = sdk.screen(
        tickers,
        days=days,
        scenarios=scenarios,
        model=model,
        seed=seed,
        min_expected_return=min_expected_return,
    )

    failed = [t for t in tickers if t.upper() not in screen.rankings]
    buy_signals = []
    for t in screen.buy:
        r = screen.rankings.get(t, {})
        buy_signals.append({
            "ticker": t,
            "score": r.get("score"),
            "expected_return": r.get("expected_return"),
            "prob_above_current": r.get("prob_above_current"),
            "value_at_risk_95_pct": r.get("value_at_risk_95_pct"),
        })

    buy_signals.sort(key=lambda x: x.get("score", 0), reverse=True)

    top = None
    if buy_signals:
        top = buy_signals[0]

    n_analyzed = len(screen.rankings)
    summary_parts = [f"Scanned {len(tickers)} tickers, analyzed {n_analyzed}."]
    if buy_signals:
        summary_parts.append(f"{len(buy_signals)} BUY signal(s): {', '.join(screen.buy)}.")
    else:
        summary_parts.append("No BUY signals found.")
    if screen.avoid:
        summary_parts.append(f"{len(screen.avoid)} to AVOID: {', '.join(screen.avoid)}.")
    if failed:
        summary_parts.append(f"{len(failed)} ticker(s) failed to fetch data.")

    return ScanReport(
        universe_size=len(tickers),
        analyzed=n_analyzed,
        failed=failed,
        buy_signals=buy_signals,
        watch_list=screen.watch,
        avoid_list=screen.avoid,
        top_pick=top,
        summary=" ".join(summary_parts),
    )


def risk_check(
    ticker: str,
    *,
    days: int = 252,
    scenarios: int = 2000,
    model: str = "historical",
    seed: int | None = None,
    offline_only: bool = False,
) -> RiskReport:
    """Deep risk assessment of a single ticker.

    Runs a higher-scenario simulation and evaluates multiple risk dimensions:
    VaR, CVaR, drawdown probability, and loss probability.
    """

    sdk = MonteCarloSDK(offline_only=offline_only)
    result = sdk.analyze(
        ticker,
        days=days,
        scenarios=scenarios,
        model=model,
        seed=seed,
    )

    s = result.summary
    var_95 = s.get("value_at_risk_95_pct", 0)
    cvar_95 = s.get("expected_shortfall_95_pct", 0)
    dd_mean = s.get("max_drawdown_mean", 0)
    dd_q95 = s.get("max_drawdown_q95", 0)
    prob_down_20 = s.get("prob_drawdown_20_pct", 0)

    warnings: list[str] = []
    if var_95 > 0.20:
        warnings.append(f"High VaR: {var_95:.1%} potential loss at 95th percentile")
    if cvar_95 > 0.25:
        warnings.append(f"Severe tail risk: CVaR95 is {cvar_95:.1%}")
    if dd_q95 > 0.30:
        warnings.append(f"Extreme drawdown risk: 95th pct max drawdown is {dd_q95:.1%}")
    if prob_down_20 > 0.25:
        warnings.append(f"High crash probability: {prob_down_20:.0%} chance of 20%+ drawdown")

    risk_score = var_95 + cvar_95 * 0.5 + dd_q95 * 0.3
    if risk_score > 0.5:
        risk_level = "EXTREME"
    elif risk_score > 0.3:
        risk_level = "HIGH"
    elif risk_score > 0.15:
        risk_level = "MODERATE"
    else:
        risk_level = "LOW"

    metrics = {
        "value_at_risk_95_pct": var_95,
        "expected_shortfall_95_pct": cvar_95,
        "max_drawdown_mean": dd_mean,
        "max_drawdown_q95": dd_q95,
        "prob_drawdown_20_pct": prob_down_20,
        "prob_above_current": s.get("prob_above_current", 0),
        "expected_return": s.get("expected_return", 0),
    }

    summary = (
        f"{ticker} risk level: {risk_level}. "
        f"VaR95={var_95:.1%}, CVaR95={cvar_95:.1%}, "
        f"max drawdown (95th pct)={dd_q95:.1%}. "
        f"{len(warnings)} warning(s)."
    )

    return RiskReport(
        ticker=ticker.upper(),
        risk_level=risk_level,
        metrics=metrics,
        warnings=warnings,
        summary=summary,
    )


def what_if(
    ticker: str,
    *,
    days: int = 252,
    scenarios: int = 1000,
    seed: int | None = None,
    offline_only: bool = False,
) -> WhatIfResult:
    """Run what-if scenario analysis comparing historical vs GBM models.

    This helps agents understand model sensitivity and communicate
    uncertainty to users.
    """

    sdk = MonteCarloSDK(offline_only=offline_only)

    historical = sdk.analyze(
        ticker, days=days, scenarios=scenarios, model="historical", seed=seed,
    )
    gbm = sdk.analyze(
        ticker, days=days, scenarios=scenarios, model="gbm", seed=seed,
    )

    scenarios_dict = {
        "historical_bootstrap": {
            "expected_return": historical.summary.get("expected_return"),
            "prob_above_current": historical.summary.get("prob_above_current"),
            "value_at_risk_95_pct": historical.summary.get("value_at_risk_95_pct"),
            "max_drawdown_q95": historical.summary.get("max_drawdown_q95"),
        },
        "geometric_brownian_motion": {
            "expected_return": gbm.summary.get("expected_return"),
            "prob_above_current": gbm.summary.get("prob_above_current"),
            "value_at_risk_95_pct": gbm.summary.get("value_at_risk_95_pct"),
            "max_drawdown_q95": gbm.summary.get("max_drawdown_q95"),
        },
    }

    h_ret = historical.summary.get("expected_return", 0)
    g_ret = gbm.summary.get("expected_return", 0)

    if h_ret >= g_ret:
        best, worst = "historical_bootstrap", "geometric_brownian_motion"
    else:
        best, worst = "geometric_brownian_motion", "historical_bootstrap"

    summary = (
        f"{ticker} what-if analysis over {days} days. "
        f"Historical model: {h_ret:.1%} expected return. "
        f"GBM model: {g_ret:.1%} expected return. "
        f"Model spread: {abs(h_ret - g_ret):.1%}."
    )

    return WhatIfResult(
        ticker=ticker.upper(),
        scenarios=scenarios_dict,
        best_case_model=best,
        worst_case_model=worst,
        summary=summary,
    )


def rebalance_signal(
    current_holdings: dict[str, float],
    *,
    days: int = 60,
    scenarios: int = 1000,
    model: str = "historical",
    seed: int | None = None,
    capital: float = 100000.0,
    max_var_95_pct: float = 0.25,
    portfolio_risk_budget_pct: float = 0.02,
    offline_only: bool = False,
) -> RebalanceSignal:
    """Determine whether a portfolio needs rebalancing.

    Compares current holdings against fresh simulation-driven optimal
    allocations and recommends changes.

    Parameters
    ----------
    current_holdings : dict
        Mapping of ticker -> current weight (0-1). Must sum to <= 1.0.
    """

    tickers = list(current_holdings.keys())
    if not tickers:
        return RebalanceSignal(
            should_rebalance=False,
            urgency="NONE",
            current_stance="EMPTY",
            recommended_changes=[],
            summary="No holdings to evaluate.",
        )

    sdk = MonteCarloSDK(offline_only=offline_only)
    result = sdk.portfolio(
        tickers,
        days=days,
        scenarios=scenarios,
        model=model,
        seed=seed,
        capital=capital,
        max_var_95_pct=max_var_95_pct,
        portfolio_risk_budget_pct=portfolio_risk_budget_pct,
    )

    changes: list[dict[str, Any]] = []
    total_drift = 0.0

    for t in tickers:
        current_w = current_holdings.get(t, 0.0)
        alloc = result.allocations.get(t, {})
        optimal_w = alloc.get("weight", 0.0)
        drift = optimal_w - current_w
        total_drift += abs(drift)

        if abs(drift) > 0.03:  # >3% weight change threshold
            action = "INCREASE" if drift > 0 else "DECREASE"
            changes.append({
                "ticker": t,
                "action": action,
                "current_weight": round(current_w, 4),
                "optimal_weight": round(optimal_w, 4),
                "drift": round(drift, 4),
            })

    if total_drift > 0.30:
        urgency = "HIGH"
    elif total_drift > 0.15:
        urgency = "MEDIUM"
    elif total_drift > 0.05:
        urgency = "LOW"
    else:
        urgency = "NONE"

    should_rebalance = urgency in ("MEDIUM", "HIGH")
    stance = result.action_plan.get("stance", "UNKNOWN")

    summary = (
        f"Total drift: {total_drift:.1%}. "
        f"Urgency: {urgency}. "
        f"{len(changes)} position(s) need adjustment. "
        f"Current stance: {stance}."
    )

    return RebalanceSignal(
        should_rebalance=should_rebalance,
        urgency=urgency,
        current_stance=stance,
        recommended_changes=changes,
        summary=summary,
    )
