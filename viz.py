"""Plotting utilities for Monte Carlo simulation results.

This module standardizes charts used across notebooks and the CLI. The
functions return :class:`matplotlib.figure.Figure` objects, allowing users to
save or display them as needed.

Example
-------
>>> from simulation import simulate_prices
>>> import pandas as pd
>>> from viz import plot_distribution
>>> returns = pd.Series([0.01, -0.02, 0.03, 0.04])
>>> sims = simulate_prices(returns, days=10, scenarios=100, seed=42)
>>> fig = plot_distribution(sims, ticker="AAPL")
>>> fig.show()
"""

from __future__ import annotations

from typing import Optional

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

__all__ = ["plot_distribution", "plot_paths"]


def _select_ticker(df: pd.DataFrame, ticker: str) -> pd.DataFrame:
    """Return a DataFrame containing only columns for ``ticker``.

    Parameters
    ----------
    df : pandas.DataFrame
        Simulation results with scenarios as columns. Columns may be a
        ``MultiIndex`` with the first level containing tickers.
    ticker : str
        Ticker symbol to select.

    Returns
    -------
    pandas.DataFrame
        DataFrame with columns for ``ticker`` only. If ``df`` contains a
        single ticker already, it is returned unchanged.
    """

    if isinstance(df.columns, pd.MultiIndex):
        if ticker not in df.columns.get_level_values(0):
            raise ValueError(f"Ticker '{ticker}' not found in DataFrame")
        sub = df.xs(ticker, axis=1, level=0)
    else:
        if ticker in df.columns:
            sub = df[ticker]
        else:
            # DataFrames from :func:`simulate_prices` may not label tickers.
            # Treat them as a single-ticker result and return as-is.
            sub = df

    if isinstance(sub, pd.Series):
        sub = sub.to_frame()
    return sub


def plot_distribution(
    df: pd.DataFrame,
    ticker: str,
    *,
    title: Optional[str] = None,
    palette: str = "tab10",
    current_price: float | None = None,
) -> plt.Figure:
    """Plot a histogram of simulated final prices.

    Parameters
    ----------
    df : pandas.DataFrame
        Simulation results where rows represent days and columns represent
        scenarios. ``df`` may contain multiple tickers using a ``MultiIndex``
        on the columns.
    ticker : str
        The ticker symbol to plot.
    title : str, optional
        Custom plot title. Defaults to ``"Distribution of {ticker}"``.
    palette : str, optional
        Name of the seaborn color palette to use.
    current_price : float, optional
        When provided, draw a vertical reference line at the current price.

    Returns
    -------
    matplotlib.figure.Figure
        Figure object containing the histogram.

    Examples
    --------
    >>> from simulation import simulate_prices
    >>> import pandas as pd
    >>> from viz import plot_distribution
    >>> returns = pd.Series([0.01, -0.02, 0.03])
    >>> sims = simulate_prices(returns, days=5, scenarios=500, seed=0)
    >>> fig = plot_distribution(sims, ticker="AAPL")
    >>> fig.show()
    """

    current = None
    if current_price is not None:
        current = float(current_price)
        if current <= 0:
            raise ValueError("current_price must be positive")

    prices = _select_ticker(df, ticker).iloc[-1]
    color = sns.color_palette(palette)[0]

    fig, ax = plt.subplots()
    sns.histplot(prices, bins=30, kde=True, stat="density", ax=ax, color=color)
    if current is not None:
        ax.axvline(
            current,
            color="black",
            linestyle="--",
            linewidth=1.2,
            alpha=0.8,
            label="Current price",
        )
    ax.set_xlabel("Price")
    ax.set_ylabel("Density")
    ax.set_title(title or f"Distribution of {ticker}")
    if current is not None:
        ax.legend()
    return fig


def plot_paths(
    df: pd.DataFrame,
    ticker: str,
    *,
    title: Optional[str] = None,
    palette: str = "tab10",
    max_paths: int | None = 100,
    current_price: float | None = None,
) -> plt.Figure:
    """Plot simulated price paths for ``ticker``.

    Parameters
    ----------
    df : pandas.DataFrame
        Simulation results where rows are days and columns are scenarios.
        Columns may use a ``MultiIndex`` to store multiple tickers.
    ticker : str
        Ticker symbol to plot.
    title : str, optional
        Custom plot title. Defaults to ``"Simulated Paths for {ticker}"``.
    palette : str, optional
        Name of the seaborn color palette to use.
    max_paths : int, optional
        Maximum number of individual paths to draw (use ``None`` to draw all).
        When the simulation includes more scenarios than this cap the function
        selects evenly spaced paths for a representative view.
    current_price : float, optional
        When provided, draw a horizontal reference line at the current price.

    Returns
    -------
    matplotlib.figure.Figure
        Figure object containing the line plot.

    Examples
    --------
    >>> from simulation import simulate_prices
    >>> import pandas as pd
    >>> from viz import plot_paths
    >>> returns = pd.Series([0.01, -0.02, 0.03])
    >>> sims = simulate_prices(returns, days=5, scenarios=3, seed=0)
    >>> fig = plot_paths(sims, ticker="AAPL")
    >>> fig.show()
    """

    prices_all = _select_ticker(df, ticker)
    if max_paths is not None and max_paths <= 0:
        raise ValueError("max_paths must be a positive integer or None")

    current = None
    if current_price is not None:
        current = float(current_price)
        if current <= 0:
            raise ValueError("current_price must be positive")

    prices_to_plot = prices_all
    if max_paths is not None and prices_all.shape[1] > max_paths:
        selection = np.linspace(
            0, prices_all.shape[1] - 1, num=max_paths, dtype=int, endpoint=True
        )
        prices_to_plot = prices_all.iloc[:, selection]

    color = sns.color_palette(palette)[0]

    fig, ax = plt.subplots()
    if prices_all.shape[1] > 1:
        q05 = prices_all.quantile(0.05, axis=1).to_numpy()
        q95 = prices_all.quantile(0.95, axis=1).to_numpy()
        ax.fill_between(
            prices_all.index,
            q05,
            q95,
            color="grey",
            alpha=0.2,
            label="5-95% band",
        )

    prices_to_plot.plot(ax=ax, legend=False, color=color, linewidth=1.0, alpha=0.25)

    if prices_all.shape[1] > 1:
        mean_path = prices_all.mean(axis=1)
        ax.plot(
            mean_path.index,
            mean_path.to_numpy(),
            color="black",
            linewidth=2.0,
            label="Mean",
        )

    if current is not None:
        ax.axhline(
            current,
            color="black",
            linestyle="--",
            linewidth=1.2,
            alpha=0.8,
            label="Current price",
        )
    ax.set_xlabel("Days")
    ax.set_ylabel("Price")
    ax.set_title(title or f"Simulated Paths for {ticker}")
    if prices_all.shape[1] > 1 or current is not None:
        ax.legend()
    return fig
