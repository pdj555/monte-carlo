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

    prices = _select_ticker(df, ticker).iloc[-1]
    color = sns.color_palette(palette)[0]

    fig, ax = plt.subplots()
    sns.histplot(prices, bins=30, kde=True, stat="density", ax=ax, color=color)
    ax.set_xlabel("Price")
    ax.set_ylabel("Density")
    ax.set_title(title or f"Distribution of {ticker}")
    return fig


def plot_paths(
    df: pd.DataFrame,
    ticker: str,
    *,
    title: Optional[str] = None,
    palette: str = "tab10",
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

    prices = _select_ticker(df, ticker)
    colors = sns.color_palette(palette, n_colors=prices.shape[1])

    fig, ax = plt.subplots()
    prices.plot(ax=ax, legend=False, color=colors, linewidth=1)
    ax.set_xlabel("Days")
    ax.set_ylabel("Price")
    ax.set_title(title or f"Simulated Paths for {ticker}")
    return fig
