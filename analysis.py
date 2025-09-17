"""High-level analytics for simulated price paths."""

from __future__ import annotations

from typing import Sequence

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
        return, probability of finishing above the current price and the 95%
        value at risk.
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
        summary["expected_return"] = float(final_prices.mean() / current_price - 1)
        summary["prob_above_current"] = float((final_prices > current_price).mean())
        summary["value_at_risk_95"] = float(
            current_price - final_prices.quantile(0.05)
        )

    return pd.Series(summary)


__all__ = ["summarize_final_prices"]
