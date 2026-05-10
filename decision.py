"""Decision rules built on top of simulation summaries."""

from __future__ import annotations

from typing import Mapping

import pandas as pd


def rank_tickers(summaries: pd.DataFrame) -> pd.DataFrame:
    """Rank tickers using a simple upside-vs-downside score."""

    if summaries.empty:
        return pd.DataFrame(
            columns=[
                "score",
                "expected_return",
                "prob_above_current",
                "value_at_risk_95_pct",
                "recommendation",
            ]
        )

    required = {"expected_return", "prob_above_current", "value_at_risk_95_pct"}
    missing = sorted(required - set(summaries.columns))
    if missing:
        joined = ", ".join(missing)
        raise ValueError(f"summaries missing required columns: {joined}")

    ranking = summaries.loc[:, sorted(required)].copy()
    if "kelly_fraction" in summaries.columns:
        ranking["kelly_fraction"] = summaries["kelly_fraction"].clip(lower=0.0, upper=1.0)
    if "max_drawdown_q95" in summaries.columns:
        ranking["max_drawdown_q95"] = summaries["max_drawdown_q95"]
    if "prob_hit_target" in summaries.columns:
        ranking["prob_hit_target"] = summaries["prob_hit_target"]
    if "prob_breach_max_loss" in summaries.columns:
        ranking["prob_breach_max_loss"] = summaries["prob_breach_max_loss"]

    downside_col = "expected_shortfall_95_pct"
    if downside_col in summaries.columns:
        ranking[downside_col] = summaries[downside_col]
        downside_penalty = ranking[downside_col]
    else:
        downside_penalty = ranking["value_at_risk_95_pct"]

    drawdown_penalty = (
        summaries["max_drawdown_q95"]
        if "max_drawdown_q95" in summaries.columns
        else 0.0
    )

    expected_return_signal = (
        summaries["expected_excess_return"]
        if "expected_excess_return" in summaries.columns
        else ranking["expected_return"]
    )
    if "expected_excess_return" in summaries.columns:
        ranking["expected_excess_return"] = summaries["expected_excess_return"]
    if "prob_beat_benchmark" in summaries.columns:
        ranking["prob_beat_benchmark"] = summaries["prob_beat_benchmark"]

    ranking["score"] = (
        expected_return_signal * 100.0
        + (ranking["prob_above_current"] - 0.5) * 40.0
        - downside_penalty * 100.0
        - drawdown_penalty * 35.0
    )
    if "prob_beat_benchmark" in ranking.columns:
        ranking["score"] += (ranking["prob_beat_benchmark"] - 0.5) * 20.0
    if "kelly_fraction" in ranking.columns:
        ranking["score"] += ranking["kelly_fraction"] * 20.0

    ranking["recommendation"] = "WATCH"
    ranking.loc[ranking["score"] >= 10.0, "recommendation"] = "BUY"
    ranking.loc[ranking["score"] <= 0.0, "recommendation"] = "AVOID"
    ranking = ranking.sort_values("score", ascending=False)
    ranking.index.name = "ticker"
    return ranking


def recommend_allocations(
    rankings: pd.DataFrame,
    *,
    max_weight: float = 0.6,
) -> pd.DataFrame:
    """Convert ticker rankings into pragmatic portfolio weights."""

    if rankings.empty:
        return pd.DataFrame(columns=["score", "value_at_risk_95_pct", "weight"])

    required = {"score", "value_at_risk_95_pct", "recommendation"}
    missing = sorted(required - set(rankings.columns))
    if missing:
        joined = ", ".join(missing)
        raise ValueError(f"rankings missing required columns: {joined}")
    if not 0 < max_weight <= 1:
        raise ValueError("max_weight must be between 0 and 1")

    eligible = rankings[rankings["recommendation"] != "AVOID"].copy()
    if eligible.empty:
        return pd.DataFrame(columns=["score", "value_at_risk_95_pct", "weight"])

    downside_col = (
        "expected_shortfall_95_pct"
        if "expected_shortfall_95_pct" in eligible.columns
        else "value_at_risk_95_pct"
    )

    signal = eligible["score"].clip(lower=0.0)
    if "kelly_fraction" in eligible.columns:
        signal = signal * (0.5 + eligible["kelly_fraction"].clip(lower=0.0, upper=1.0))
    risk_scale = 1.0 / (1.0 + eligible[downside_col].clip(lower=0.0))
    raw = signal * risk_scale

    if float(raw.sum()) <= 0:
        raw = pd.Series(1.0, index=eligible.index)

    target_invested = min(1.0, len(eligible) * max_weight)
    priorities = raw / raw.sum()

    low = 0.0
    high = 1.0
    while float(priorities.mul(high).clip(upper=max_weight).sum()) < target_invested:
        high *= 2.0

    for _ in range(60):
        alpha = (low + high) / 2.0
        invested = float(priorities.mul(alpha).clip(upper=max_weight).sum())
        if invested < target_invested:
            low = alpha
        else:
            high = alpha

    weights = priorities.mul(high).clip(upper=max_weight)
    if float(weights.sum()) > 0:
        weights *= target_invested / float(weights.sum())

    allocation = eligible.loc[:, ["score", "value_at_risk_95_pct"]].copy()
    allocation["weight"] = weights
    allocation = allocation.sort_values("weight", ascending=False)
    allocation.index.name = "ticker"
    return allocation


def enforce_portfolio_risk_budget(
    allocations: pd.DataFrame,
    rankings: pd.DataFrame,
    *,
    max_portfolio_var_95_pct: float,
    portfolio_var_95_pct: float | None = None,
) -> pd.DataFrame:
    """Scale allocations so portfolio 95% VaR stays within a hard budget.

    When ``portfolio_var_95_pct`` is supplied, the guard uses path-aware
    simulated portfolio VaR without allowing it to understate the conservative
    standalone ticker VaR blend. Otherwise it falls back to the standalone
    blend. Scaling is linear because the excess exposure is moved to cash.
    """

    if allocations.empty:
        return allocations.copy()
    if max_portfolio_var_95_pct < 0:
        raise ValueError("max_portfolio_var_95_pct must be non-negative")
    if portfolio_var_95_pct is not None and portfolio_var_95_pct < 0:
        raise ValueError("portfolio_var_95_pct must be non-negative when provided")
    if "weight" not in allocations.columns:
        raise ValueError("allocations missing required columns: weight")
    if "value_at_risk_95_pct" not in rankings.columns:
        raise ValueError("rankings missing required columns: value_at_risk_95_pct")

    scoped = allocations.copy()
    scoped_var = rankings.reindex(scoped.index)["value_at_risk_95_pct"].fillna(0.0)
    blended_var = float((scoped["weight"] * scoped_var).sum())
    if portfolio_var_95_pct is None:
        budget_var = blended_var
    else:
        budget_var = max(float(portfolio_var_95_pct), blended_var)

    if budget_var <= max_portfolio_var_95_pct or budget_var <= 0.0:
        return scoped

    scale = max_portfolio_var_95_pct / budget_var
    scoped["weight"] = scoped["weight"] * scale
    return scoped


def apply_risk_guards(
    rankings: pd.DataFrame,
    *,
    min_expected_return: float = 0.0,
    min_prob_above_current: float = 0.5,
    max_value_at_risk_95_pct: float = 0.25,
    max_drawdown_q95: float | None = None,
    min_prob_hit_target: float | None = None,
    max_prob_breach_loss: float | None = None,
) -> pd.DataFrame:
    """Apply hard risk/reward filters to ranking output."""

    if rankings.empty:
        return rankings.copy()

    required = {
        "expected_return",
        "prob_above_current",
        "value_at_risk_95_pct",
        "recommendation",
    }
    missing = sorted(required - set(rankings.columns))
    if missing:
        joined = ", ".join(missing)
        raise ValueError(f"rankings missing required columns: {joined}")

    if not 0 <= min_prob_above_current <= 1:
        raise ValueError("min_prob_above_current must be between 0 and 1")
    if min_prob_hit_target is not None and not 0 <= min_prob_hit_target <= 1:
        raise ValueError("min_prob_hit_target must be between 0 and 1 when provided")
    if max_prob_breach_loss is not None and not 0 <= max_prob_breach_loss <= 1:
        raise ValueError("max_prob_breach_loss must be between 0 and 1 when provided")
    if max_value_at_risk_95_pct < 0:
        raise ValueError("max_value_at_risk_95_pct must be non-negative")
    if max_drawdown_q95 is not None and max_drawdown_q95 < 0:
        raise ValueError("max_drawdown_q95 must be non-negative when provided")

    guarded = rankings.copy()
    guarded["guardrail_reasons"] = ""

    fail_expected = guarded["expected_return"] < min_expected_return
    fail_prob = guarded["prob_above_current"] < min_prob_above_current
    fail_var = guarded["value_at_risk_95_pct"] > max_value_at_risk_95_pct
    fail_drawdown = (
        guarded["max_drawdown_q95"] > max_drawdown_q95
        if max_drawdown_q95 is not None and "max_drawdown_q95" in guarded.columns
        else pd.Series(False, index=guarded.index)
    )
    fail_target = (
        guarded["prob_hit_target"] < min_prob_hit_target
        if min_prob_hit_target is not None and "prob_hit_target" in guarded.columns
        else pd.Series(False, index=guarded.index)
    )
    fail_loss_breach = (
        guarded["prob_breach_max_loss"] > max_prob_breach_loss
        if max_prob_breach_loss is not None and "prob_breach_max_loss" in guarded.columns
        else pd.Series(False, index=guarded.index)
    )

    reasons = []
    for ticker in guarded.index:
        ticker_reasons: list[str] = []
        if bool(fail_expected.loc[ticker]):
            ticker_reasons.append(f"expected_return<{min_expected_return:.1%}")
        if bool(fail_prob.loc[ticker]):
            ticker_reasons.append(f"prob_above_current<{min_prob_above_current:.0%}")
        if bool(fail_var.loc[ticker]):
            ticker_reasons.append(f"value_at_risk_95_pct>{max_value_at_risk_95_pct:.1%}")
        if bool(fail_drawdown.loc[ticker]):
            ticker_reasons.append(f"max_drawdown_q95>{max_drawdown_q95:.1%}")
        if bool(fail_target.loc[ticker]):
            ticker_reasons.append(f"prob_hit_target<{min_prob_hit_target:.0%}")
        if bool(fail_loss_breach.loc[ticker]):
            ticker_reasons.append(f"prob_breach_max_loss>{max_prob_breach_loss:.0%}")
        reasons.append("; ".join(ticker_reasons))

    guarded["guardrail_reasons"] = reasons
    failed_any = (
        fail_expected
        | fail_prob
        | fail_var
        | fail_drawdown
        | fail_target
        | fail_loss_breach
    )
    guarded.loc[failed_any, "recommendation"] = "AVOID"
    return guarded


def build_action_plan(
    rankings: pd.DataFrame,
    allocations: pd.DataFrame,
) -> dict[str, object]:
    """Build a lean action plan from ranking and allocation tables."""

    if rankings.empty:
        return {
            "stance": "NO_TRADE",
            "headline": "No valid opportunities found.",
            "primary_pick": None,
            "focus_list": [],
            "avoid_list": [],
            "cash_weight": 1.0,
        }

    avoid_list = rankings.index[rankings["recommendation"] == "AVOID"].tolist()
    focus = rankings[rankings["recommendation"] != "AVOID"]

    if focus.empty or allocations.empty:
        return {
            "stance": "DEFENSIVE",
            "headline": "All candidates are high-risk or low-conviction. Hold cash.",
            "primary_pick": None,
            "focus_list": [],
            "avoid_list": avoid_list,
            "cash_weight": 1.0,
        }

    top_ticker = allocations.index[0]
    top_row = rankings.loc[top_ticker]
    top_weight = float(allocations.loc[top_ticker, "weight"])
    top_score = float(top_row["score"])

    if top_score >= 10 and top_weight >= 0.5:
        stance = "RISK_ON"
        verb = "Concentrate"
    elif top_score > 0:
        stance = "SELECTIVE"
        verb = "Accumulate"
    else:
        stance = "DEFENSIVE"
        verb = "Stay light"

    focus_list = allocations.index.tolist()
    headline = (
        f"{verb} in {top_ticker} ({top_weight:.1%} weight, score {top_score:.1f}). "
        "Avoid weak names."
    )
    cash_weight = max(0.0, 1.0 - float(allocations["weight"].sum()))
    if cash_weight > 0:
        headline = f"{headline} Keep {cash_weight:.1%} in cash."

    return {
        "stance": stance,
        "headline": headline,
        "primary_pick": {
            "ticker": top_ticker,
            "weight": top_weight,
            "score": top_score,
            "expected_return": float(top_row["expected_return"]),
            "prob_above_current": float(top_row["prob_above_current"]),
            "value_at_risk_95_pct": float(top_row["value_at_risk_95_pct"]),
        },
        "focus_list": focus_list,
        "avoid_list": avoid_list,
        "cash_weight": cash_weight,
    }


def build_execution_plan(
    allocations: pd.DataFrame,
    *,
    current_prices: Mapping[str, float],
    capital: float,
    allow_fractional_shares: bool = False,
) -> pd.DataFrame:
    """Translate allocation weights into executable order sizes."""

    if allocations.empty:
        return pd.DataFrame(
            columns=["weight", "price", "target_dollars", "shares", "est_cost", "cash_drift"]
        )
    if capital <= 0:
        raise ValueError("capital must be positive")
    if "weight" not in allocations.columns:
        raise ValueError("allocations missing required columns: weight")

    plan = allocations.copy()
    plan["price"] = pd.Series(current_prices).reindex(plan.index)
    if plan["price"].isna().any():
        missing = ", ".join(plan.index[plan["price"].isna()].tolist())
        raise ValueError(f"missing current prices for: {missing}")
    if (plan["price"] <= 0).any():
        raise ValueError("current_prices must be positive")

    plan["target_dollars"] = plan["weight"].clip(lower=0.0) * float(capital)
    if allow_fractional_shares:
        plan["shares"] = plan["target_dollars"] / plan["price"]
    else:
        plan["shares"] = (plan["target_dollars"] / plan["price"]).floordiv(1)
    plan["shares"] = plan["shares"].astype(float)
    plan["est_cost"] = plan["shares"] * plan["price"]
    plan["cash_drift"] = plan["target_dollars"] - plan["est_cost"]
    return plan


__all__ = [
    "apply_risk_guards",
    "build_action_plan",
    "build_execution_plan",
    "enforce_portfolio_risk_budget",
    "rank_tickers",
    "recommend_allocations",
]
