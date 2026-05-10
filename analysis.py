"""High-level analytics for simulated price paths."""

from __future__ import annotations

from typing import Mapping, Sequence

import pandas as pd

_DEFAULT_QUANTILES: tuple[float, ...] = (0.05, 0.25, 0.75, 0.95)


def _median_first_hit_day(hit_frame: pd.DataFrame) -> float:
    """Return the median first-hit day using the frame index values."""

    hit_mask = hit_frame.to_numpy(dtype=bool)
    touched = hit_mask.any(axis=0)
    if not bool(touched.any()):
        return float("nan")

    first_positions = hit_mask.argmax(axis=0)[touched]
    first_days = pd.Index(hit_frame.index).take(first_positions)
    return float(pd.Series(first_days, dtype=float).median())


def summarize_final_prices(
    df: pd.DataFrame,
    *,
    current_price: float | None = None,
    quantiles: Sequence[float] | None = None,
    target_return_pct: float | None = None,
    max_loss_pct: float | None = None,
    benchmark_return_pct: float | None = None,
) -> pd.Series:
    """Return summary statistics for the final simulated prices."""

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
        summary[f"q{int(q * 100):02d}"] = float(final_prices.quantile(q))

    if current_price is not None:
        if current_price <= 0:
            raise ValueError("current_price must be positive when provided")

        current_price = float(current_price)
        summary["expected_return"] = float(final_prices.mean() / current_price - 1.0)
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
            kelly_fraction = p_win - ((1.0 - p_win) / payoff_ratio)
            kelly_fraction = float(min(max(kelly_fraction, 0.0), 1.0))

        summary["avg_upside_pct"] = avg_win
        summary["avg_downside_pct"] = avg_loss
        summary["payoff_ratio"] = float(payoff_ratio)
        summary["kelly_fraction"] = float(kelly_fraction)

        q05 = float(final_prices.quantile(0.05))
        q01 = float(final_prices.quantile(0.01))
        tail_95 = final_prices[final_prices <= q05]
        tail_99 = final_prices[final_prices <= q01]

        summary["value_at_risk_95"] = float(max(0.0, current_price - q05))
        summary["expected_shortfall_95"] = float(
            max(0.0, current_price - float(tail_95.mean())) if not tail_95.empty else 0.0
        )
        summary["value_at_risk_99"] = float(max(0.0, current_price - q01))
        summary["expected_shortfall_99"] = float(
            max(0.0, current_price - float(tail_99.mean())) if not tail_99.empty else 0.0
        )
        summary["value_at_risk_95_pct"] = float(summary["value_at_risk_95"] / current_price)
        summary["expected_shortfall_95_pct"] = float(
            summary["expected_shortfall_95"] / current_price
        )
        summary["value_at_risk_99_pct"] = float(summary["value_at_risk_99"] / current_price)
        summary["expected_shortfall_99_pct"] = float(
            summary["expected_shortfall_99"] / current_price
        )

        realized_returns = final_prices / current_price - 1.0
        if benchmark_return_pct is not None:
            benchmark_return_pct = float(benchmark_return_pct)
            summary["benchmark_return_pct"] = benchmark_return_pct
            summary["expected_excess_return"] = float(
                summary["expected_return"] - benchmark_return_pct
            )
            summary["prob_beat_benchmark"] = float(
                (realized_returns >= benchmark_return_pct).mean()
            )

        if target_return_pct is not None:
            target_return_pct = float(target_return_pct)
            summary["target_return_pct"] = target_return_pct
            summary["prob_hit_target"] = float((realized_returns >= target_return_pct).mean())

            target_price = current_price * (1.0 + target_return_pct)
            target_hits = df >= target_price
            summary["prob_touch_target"] = float(target_hits.any(axis=0).mean())
            summary["median_days_to_target"] = _median_first_hit_day(target_hits)

        if max_loss_pct is not None:
            if max_loss_pct < 0:
                raise ValueError("max_loss_pct must be non-negative when provided")

            max_loss_pct = float(max_loss_pct)
            loss_floor = current_price * (1.0 - max_loss_pct)
            loss_hits = df <= loss_floor

            summary["max_loss_pct"] = max_loss_pct
            summary["prob_breach_max_loss"] = float((final_prices <= loss_floor).mean())
            summary["prob_touch_max_loss"] = float(loss_hits.any(axis=0).mean())
            summary["median_days_to_max_loss"] = _median_first_hit_day(loss_hits)

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


def summarize_weighted_portfolio(
    simulations: pd.DataFrame,
    *,
    current_prices: Mapping[str, float],
    weights: Mapping[str, float] | pd.Series,
    quantiles: Sequence[float] | None = None,
    benchmark_return_pct: float | None = None,
) -> pd.Series:
    """Return path-aware summary statistics for a weighted portfolio.

    The portfolio starts at ``1.0``. Supplied weights are invested weights; any
    unused capital stays in cash at ``1.0`` for the simulated horizon. Scenario
    columns are matched by scenario id across tickers, preserving the simulated
    cross-name outcomes instead of adding standalone VaR numbers together.
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

    initial_prices = pd.Series(
        {ticker: float(current_prices[ticker]) for ticker in tickers},
        dtype=float,
    )
    if (initial_prices <= 0).any():
        raise ValueError("all current prices must be positive")

    weight_series = pd.Series(weights, dtype=float).reindex(tickers).fillna(0.0)
    if (weight_series < 0).any():
        raise ValueError("portfolio weights must be non-negative")
    if float(weight_series.sum()) > 1.0 + 1e-9:
        raise ValueError("portfolio weights cannot sum above 1.0")

    cash_weight = max(0.0, 1.0 - float(weight_series.sum()))
    scenarios = simulations.columns.get_level_values("scenario").unique()
    portfolio_columns: dict[object, pd.Series] = {}
    for scenario in scenarios:
        scenario_frame = simulations.xs(scenario, axis=1, level="scenario")
        normalized = scenario_frame.div(initial_prices, axis="columns")
        portfolio_columns[scenario] = (
            normalized.mul(weight_series, axis="columns").sum(axis=1) + cash_weight
        )

    portfolio_paths = pd.DataFrame(portfolio_columns, index=simulations.index)
    portfolio_paths.columns.name = "scenario"
    summary = summarize_final_prices(
        portfolio_paths,
        current_price=1.0,
        quantiles=quantiles,
        benchmark_return_pct=benchmark_return_pct,
    )
    summary["component_count"] = float((weight_series > 0).sum())
    summary["invested_weight"] = float(weight_series.sum())
    summary["cash_weight"] = float(cash_weight)
    return summary


def summarize_equal_weight_portfolio(
    simulations: pd.DataFrame,
    *,
    current_prices: Mapping[str, float],
    quantiles: Sequence[float] | None = None,
    benchmark_return_pct: float | None = None,
) -> pd.Series:
    """Return summary statistics for an equal-weight portfolio."""

    if simulations.empty:
        raise ValueError("simulations must contain scenario paths")
    if not isinstance(simulations.columns, pd.MultiIndex):
        raise ValueError("simulations must use a ticker/scenario MultiIndex")

    tickers = list(simulations.columns.get_level_values("ticker").unique())
    if not tickers:
        raise ValueError("simulations must include at least one ticker")

    weights = pd.Series(1.0 / len(tickers), index=tickers, dtype=float)
    return summarize_weighted_portfolio(
        simulations,
        current_prices=current_prices,
        weights=weights,
        quantiles=quantiles,
        benchmark_return_pct=benchmark_return_pct,
    )


__all__ = [
    "summarize_final_prices",
    "summarize_equal_weight_portfolio",
    "summarize_weighted_portfolio",
]
