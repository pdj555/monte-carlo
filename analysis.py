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
    ranking["score"] = (
        ranking["expected_return"] * 100.0
        + (ranking["prob_above_current"] - 0.5) * 40.0
        - ranking["value_at_risk_95_pct"] * 100.0
    )
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
    * Scale signal by downside risk so high-VaR names receive smaller weights.
    * Normalize to weights that sum to ``1.0`` and cap concentration.

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

    signal = eligible["score"].clip(lower=0.0)
    risk_scale = 1.0 / (1.0 + eligible["value_at_risk_95_pct"].clip(lower=0.0))
    raw = signal * risk_scale

    if float(raw.sum()) <= 0:
        raw = pd.Series(1.0, index=eligible.index)

    weights = raw / raw.sum()
    weights = weights.clip(upper=max_weight)
    weights = weights / weights.sum()

    allocation = eligible.loc[:, ["score", "value_at_risk_95_pct"]].copy()
    allocation["weight"] = weights
    allocation = allocation.sort_values("weight", ascending=False)
    allocation.index.name = "ticker"
    return allocation


__all__ = [
    "rank_tickers",
    "recommend_allocations",
    "summarize_final_prices",
    "summarize_equal_weight_portfolio",
]
