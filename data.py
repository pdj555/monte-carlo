"""Utilities for downloading price data.

This module wraps ``yfinance`` with a thin convenience layer that retries
requests and raises :class:`PriceDataError` on failure.

Example
-------
>>> from data import fetch_prices
>>> prices = fetch_prices("AAPL")
>>> print(prices.head())
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import Optional

import pandas as pd
import yfinance as yf

# Default directory containing fallback CSV data
_FALLBACK_DIR = Path(__file__).resolve().parent / "sample_data"


class PriceDataError(Exception):
    """Raised when price data cannot be retrieved."""


def fetch_prices(
    ticker: str,
    start: Optional[str] = None,
    end: Optional[str] = None,
    *,
    offline_path: Optional[Path | str] = None,
    prefer_local: bool = False,
) -> pd.Series:
    """Download daily closing prices for ``ticker``.

    Parameters
    ----------
    ticker : str
        Stock ticker symbol to fetch.
    start : str, optional
        Start date in ``YYYY-MM-DD`` format.
    end : str, optional
        End date in ``YYYY-MM-DD`` format.
    offline_path : pathlib.Path or str, optional
        Local CSV file or directory to use when network requests fail. If a
        directory is supplied the file ``{ticker}.csv`` inside that directory
        will be used. When omitted the function falls back to
        ``sample_data/{ticker}.csv`` relative to this module.
    prefer_local : bool, optional
        When ``True`` skip network requests entirely and use local CSV data.

    Returns
    -------
    pandas.Series
        Series of closing prices indexed by date.

    Raises
    ------
    PriceDataError
        If the ticker is invalid or data cannot be retrieved after retries.
    """
    if not isinstance(ticker, str) or not ticker.strip():
        raise PriceDataError("Ticker must be a non-empty string")

    offline_path = Path(offline_path) if offline_path is not None else None

    attempts = 0
    last_error: Optional[Exception] = None
    if not prefer_local:
        while attempts < 3:
            try:
                data = yf.download(ticker, start=start, end=end, progress=False)
                close = data.get("Close")
                if close is None or close.empty:
                    raise PriceDataError(
                        f"No price data returned for ticker '{ticker}'"
                    )
                return close
            except Exception as exc:  # network error or other issues
                last_error = exc
                attempts += 1
                time.sleep(1)

    # If online retrieval fails or local data is preferred, attempt to load CSV
    if offline_path is None:
        fallback = _FALLBACK_DIR / f"{ticker}.csv"
    else:
        fallback = offline_path / f"{ticker}.csv" if offline_path.is_dir() else offline_path

    if fallback.exists():
        series = pd.read_csv(fallback, index_col=0, parse_dates=True)["Close"]
        series = series.sort_index()
        if not series.empty:
            return series

    message = str(last_error) if last_error else "offline data not found"
    raise PriceDataError(f"Failed to fetch price data for '{ticker}': {message}")
