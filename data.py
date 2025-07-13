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
    offline_path: Optional[Path] = None,
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

    Parameters
    ----------
    offline_path : pathlib.Path, optional
        Local CSV file to use when network requests fail. If not provided,
        ``data.py`` will look for ``sample_data/{ticker}.csv`` relative to the
        module location.

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

    attempts = 0
    last_error: Optional[Exception] = None
    while attempts < 3:
        try:
            data = yf.download(ticker, start=start, end=end, progress=False)
            close = data.get("Close")
            if close is None or close.empty:
                raise PriceDataError(f"No price data returned for ticker '{ticker}'")
            return close
        except Exception as exc:  # network error or other issues
            last_error = exc
            attempts += 1
            time.sleep(1)

    # If online retrieval fails, attempt to load from a local CSV
    fallback = offline_path or _FALLBACK_DIR / f"{ticker}.csv"
    if fallback.exists():
        series = pd.read_csv(fallback, index_col=0, parse_dates=True)["Close"]
        if not series.empty:
            return series

    message = str(last_error) if last_error else "Unknown error"
    raise PriceDataError(f"Failed to fetch price data for '{ticker}': {message}")
