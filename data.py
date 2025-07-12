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
from typing import Optional

import pandas as pd
import yfinance as yf


class PriceDataError(Exception):
    """Raised when price data cannot be retrieved."""


def fetch_prices(ticker: str, start: Optional[str] = None, end: Optional[str] = None) -> pd.Series:
    """Download daily closing prices for ``ticker``.

    Parameters
    ----------
    ticker : str
        Stock ticker symbol to fetch.
    start : str, optional
        Start date in ``YYYY-MM-DD`` format.
    end : str, optional
        End date in ``YYYY-MM-DD`` format.

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

    message = str(last_error) if last_error else "Unknown error"
    raise PriceDataError(f"Failed to fetch price data for '{ticker}': {message}")
