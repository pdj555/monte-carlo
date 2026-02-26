"""High-level analytics for simulated price paths."""

from __future__ import annotations

from typing import Mapping, Sequence

import pandas as pd

_DEFAULT_QUANTILES: tuple[float, ...] = (0.05, 0.25, 0.75, 0.95)


def summarize_final_prices(
    df: pd.DataFrame,
    *,
    current_price: float | None = None,
    quantiles: Sequence[float] | None = None,
    target_return_pct: float | None = None,
    max_loss_pct: float | None = None,
) -> pd.Series:
    """Return summary statistics for the final simulated prices.

    Parameters
    ----------
    df : pandas.DataFrame
        Simulation results where the final row represents the final price of
        each scenario.
    current_price : float, optional
        Observed current price. When provided the summary includes expected
        return, win/loss probabilities and downside risk metrics (VaR/CVaR).
    quantiles : sequence of float, optional
        Additional quantiles to report. Values must be between ``0`` and ``1``.
    target_return_pct : float, optional
        Decision target expressed as simple return from ``current_price``. When
        provided, include probability of finishing at or above target.
    max_loss_pct : float, optional
        Maximum acceptable loss expressed as a positive fraction from
        ``current_price``. When provided, include probability of breaching this
        loss threshold.

    Returns
    -------
    pandas.Series
        Series containing descriptive statistics for the simulated prices.
    """

    if df.empty:
        raise ValueError("df must contain simulation results")

    final_prices = df.iloc[-1]
    summary = {
        "mean": float(final_prices.mean()),
        "median": float(final_prices.median()),
        "std": float(final_prices.std(ddof=1)),
        "min": float(final_prices.min()),
        "max": float(final_prices.max()),
    }

    quantiles = tuple(_DEFAULT_QUANTILES if quantiles is None else quantiles)
    for q in quantiles:
        if not 0 <= q <= 1:
            raise ValueError("quantiles must lie between 0 and 1")
        key = f"q{int(q * 100):02d}"
        summary[key] = float(final_prices.quantile(q))

    if current_price is not None:
        if current_price <= 0:
            raise ValueError("current_price must be positive when provided")
        current_price = float(current_price)
        summary["expected_return"] = float(final_prices.mean() / current_price - 1)
        summary["prob_above_current"] = float((final_prices > current_price).mean())
        summary["prob_below_current"] = float((final_prices < current_price).mean())

        simple_returns = final_prices / current_price - 1.0
        wins = simple_returns[simple_returns > 0]
        losses = simple_returns[simple_returns < 0]
        avg_win = float(wins.mean()) if not wins.empty else 0.0
        avg_loss = float((-losses).mean()) if not losses.empty else 0.0
        payoff_ratio = avg_win / avg_loss if avg_loss > 0 else 10.0
        kelly_fraction = 0.0
        if avg_loss > 0 and avg_win > 0:
            p_win = float((simple_returns > 0).mean())
            p_loss = 1.0 - p_win
            kelly_fraction = p_win - (p_loss / payoff_ratio)
            kelly_fraction = float(min(max(kelly_fraction, 0.0), 1.0))

        summary["avg_upside_pct"] = avg_win
        summary["avg_downside_pct"] = avg_loss
        summary["payoff_ratio"] = float(payoff_ratio)
        summary["kelly_fraction"] = float(kelly_fraction)

        q05 = float(final_prices.quantile(0.05))
        q01 = float(final_prices.quantile(0.01))

        var_95 = max(0.0, current_price - q05)
        var_99 = max(0.0, current_price - q01)

        tail_95 = final_prices[final_prices <= q05]
        tail_99 = final_prices[final_prices <= q01]

        es_95 = current_price - float(tail_95.mean()) if not tail_95.empty else 0.0
        es_99 = current_price - float(tail_99.mean()) if not tail_99.empty else 0.0

        summary["value_at_risk_95"] = float(max(0.0, var_95))
        summary["expected_shortfall_95"] = float(max(0.0, es_95))
        summary["value_at_risk_99"] = float(max(0.0, var_99))
        summary["expected_shortfall_99"] = float(max(0.0, es_99))
        summary["value_at_risk_95_pct"] = float(summary["value_at_risk_95"] / current_price)
        summary["expected_shortfall_95_pct"] = float(summary["expected_shortfall_95"] / current_price)
        summary["value_at_risk_99_pct"] = float(summary["value_at_risk_99"] / current_price)
        summary["expected_shortfall_99_pct"] = float(summary["expected_shortfall_99"] / current_price)

        if target_return_pct is not None:
            target_return_pct = float(target_return_pct)
            summary["target_return_pct"] = target_return_pct
            realized_returns = final_prices / current_price - 1.0
            summary["prob_hit_target"] = float((realized_returns >= target_return_pct).mean())

        if max_loss_pct is not None:
            if max_loss_pct < 0:
                raise ValueError("max_loss_pct must be non-negative when provided")
            loss_floor = current_price * (1.0 - float(max_loss_pct))
            summary["max_loss_pct"] = float(max_loss_pct)
            summary["prob_breach_max_loss"] = float((final_prices <= loss_floor).mean())

    if len(df.index) > 1:
        running_peaks = df.cummax()
        drawdown = 1.0 - df.div(running_peaks)
        max_drawdown = drawdown.max(axis=0)

        summary["max_drawdown_mean"] = float(max_drawdown.mean())
        summary["max_drawdown_median"] = float(max_drawdown.median())
        summary["max_drawdown_q95"] = float(max_drawdown.quantile(0.95))
        summary["prob_drawdown_10_pct"] = float((max_drawdown >= 0.10).mean())
        summary["prob_drawdown_20_pct"] = float((max_drawdown >= 0.20).mean())

    return pd.Series(summary)


def summarize_equal_weight_portfolio(
    simulations: pd.DataFrame,
    *,
    current_prices: Mapping[str, float],
    quantiles: Sequence[float] | None = None,
) -> pd.Series:
    """Return summary statistics for an equal-weight portfolio.

    Parameters
    ----------
    simulations : pandas.DataFrame
        Combined simulation frame with ``ticker`` and ``scenario`` column levels.
    current_prices : mapping of str to float
        Mapping from ticker to observed current price.
    quantiles : sequence of float, optional
        Additional quantiles to report. Values must be between ``0`` and ``1``.
    """

    if simulations.empty:
        raise ValueError("simulations must contain scenario paths")
    if not isinstance(simulations.columns, pd.MultiIndex):
        raise ValueError("simulations must use a ticker/scenario MultiIndex")

    tickers = list(simulations.columns.get_level_values("ticker").unique())
    if not tickers:
        raise ValueError("simulations must include at least one ticker")

    missing_prices = [ticker for ticker in tickers if ticker not in current_prices]
    if missing_prices:
        joined = ", ".join(sorted(missing_prices))
        raise ValueError(f"current_prices missing entries for: {joined}")

    for ticker in tickers:
        if current_prices[ticker] <= 0:
            raise ValueError("all current prices must be positive")

    final_row = simulations.iloc[-1]
    final_prices = final_row.unstack("ticker")
    weights = pd.Series(1.0 / len(tickers), index=tickers)
    initial_prices = pd.Series(
        {ticker: float(current_prices[ticker]) for ticker in tickers}, dtype=float
    )
    shares = weights / initial_prices
    portfolio_final = final_prices.mul(shares, axis="columns").sum(axis=1)

    summary = summarize_final_prices(
        pd.DataFrame([portfolio_final.to_numpy()]),
        current_price=1.0,
        quantiles=quantiles,
    )
    summary["component_count"] = float(len(tickers))
    return summary


def rank_tickers(summaries: pd.DataFrame) -> pd.DataFrame:
    """Rank tickers using a simple upside-vs-downside score.

    Parameters
    ----------
    summaries : pandas.DataFrame
        Summary table where rows are tickers and columns contain at least
        ``expected_return``, ``prob_above_current`` and ``value_at_risk_95_pct``.
        When available, ``expected_shortfall_95_pct`` is used as the downside
        penalty because it captures tail severity better than VaR. When
        available, ``kelly_fraction`` boosts ranking for setups with stronger
        asymmetric payoff.

    Returns
    -------
    pandas.DataFrame
        Table ordered by descending ``score`` with a lean recommendation label.
    """

    if summaries.empty:
        return pd.DataFrame(
            columns=["score", "expected_return", "prob_above_current", "value_at_risk_95_pct", "recommendation"]
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

    ranking["score"] = (
        ranking["expected_return"] * 100.0
        + (ranking["prob_above_current"] - 0.5) * 40.0
        - downside_penalty * 100.0
        - drawdown_penalty * 35.0
    )
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
    """Convert ticker rankings into pragmatic portfolio weights.

    The strategy is intentionally simple and conservative:

    * Drop tickers labeled ``AVOID``.
    * Use only positive score signal.
    * Scale signal by downside risk so high-risk names receive smaller weights.
      When available, expected shortfall is preferred over VaR.
    * Normalize to weights that sum to ``1.0`` when feasible and cap concentration.
      If the cap is too strict (e.g. one eligible ticker with ``max_weight=0.6``),
      the remaining capital is intentionally left as cash.

    Parameters
    ----------
    rankings : pandas.DataFrame
        Output of :func:`rank_tickers`.
    max_weight : float, default 0.6
        Maximum weight assigned to a single ticker.
    """

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
    """Apply hard risk/reward filters to ranking output.

    This function turns soft ranking into explicit go/no-go gates. Any ticker
    failing one or more constraints is marked ``AVOID`` regardless of score.
    """

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
            ticker_reasons.append(
                f"expected_return<{min_expected_return:.1%}"
            )
        if bool(fail_prob.loc[ticker]):
            ticker_reasons.append(
                f"prob_above_current<{min_prob_above_current:.0%}"
            )
        if bool(fail_var.loc[ticker]):
            ticker_reasons.append(
                f"value_at_risk_95_pct>{max_value_at_risk_95_pct:.1%}"
            )
        if bool(fail_drawdown.loc[ticker]):
            ticker_reasons.append(
                f"max_drawdown_q95>{max_drawdown_q95:.1%}"
            )
        if bool(fail_target.loc[ticker]):
            ticker_reasons.append(
                f"prob_hit_target<{min_prob_hit_target:.0%}"
            )
        if bool(fail_loss_breach.loc[ticker]):
            ticker_reasons.append(
                f"prob_breach_max_loss>{max_prob_breach_loss:.0%}"
            )
        reasons.append("; ".join(ticker_reasons))

    guarded["guardrail_reasons"] = reasons
    failed_any = fail_expected | fail_prob | fail_var | fail_drawdown | fail_target | fail_loss_breach
    guarded.loc[failed_any, "recommendation"] = "AVOID"
    return guarded


__all__ = [
    "apply_risk_guards",
    "build_action_plan",
    "rank_tickers",
    "recommend_allocations",
    "summarize_final_prices",
    "summarize_equal_weight_portfolio",
]


def build_action_plan(
    rankings: pd.DataFrame,
    allocations: pd.DataFrame,
) -> dict[str, object]:
    """Build a lean action plan from ranking and allocation tables.

    The output is intentionally direct so the CLI can produce decision-grade
    guidance instead of only descriptive statistics.
    """

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
