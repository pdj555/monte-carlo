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


def _parse_date(value: Optional[str], *, label: str) -> Optional[pd.Timestamp]:
    """Parse a user-supplied date string into a pandas Timestamp."""

    if value is None:
        return None
    try:
        return pd.to_datetime(value)
    except Exception as exc:
        raise PriceDataError(f"Invalid {label} date '{value}': {exc}") from exc


def _slice_prices(
    prices: pd.Series, start: Optional[str], end: Optional[str]
) -> pd.Series:
    start_ts = _parse_date(start, label="start")
    end_ts = _parse_date(end, label="end")
    if start_ts is not None and end_ts is not None and start_ts > end_ts:
        raise PriceDataError("start date must be on or before end date")

    prices = prices.sort_index()
    if start_ts is not None:
        prices = prices.loc[start_ts:]
    if end_ts is not None:
        prices = prices.loc[:end_ts]
    if prices.empty:
        raise PriceDataError(
            f"No price data available for the requested date range (start={start!r}, end={end!r})."
        )
    return prices


def _load_prices_from_csv(path: Path) -> pd.Series:
    """Load a price series from a CSV file with common column conventions."""

    try:
        df = pd.read_csv(path)
    except Exception as exc:
        raise PriceDataError(f"Failed to read CSV at '{path}': {exc}") from exc

    if df.empty:
        raise PriceDataError(f"CSV at '{path}' is empty")

    date_column = "Date" if "Date" in df.columns else df.columns[0]
    df[date_column] = pd.to_datetime(df[date_column], errors="coerce")
    df = df.dropna(subset=[date_column])
    df = df.set_index(date_column).sort_index()

    close_column = None
    for candidate in ("Close", "Adj Close", "close", "adj_close", "adjclose"):
        if candidate in df.columns:
            close_column = candidate
            break

    if close_column is None:
        numeric_candidates = [
            col
            for col in df.columns
            if col != date_column and pd.api.types.is_numeric_dtype(df[col])
        ]
        if len(numeric_candidates) == 1:
            close_column = numeric_candidates[0]

    if close_column is None:
        raise PriceDataError(
            f"CSV at '{path}' must include a 'Close' (or 'Adj Close') column"
        )

    series = pd.to_numeric(df[close_column], errors="coerce").dropna()
    series.name = "Close"
    if series.empty:
        raise PriceDataError(f"CSV at '{path}' does not contain usable close prices")
    return series


def fetch_prices(
    ticker: str,
    start: Optional[str] = None,
    end: Optional[str] = None,
    *,
    offline_path: Optional[Path | str] = None,
    prefer_local: bool = False,
    cache_dir: Optional[Path | str] = None,
    refresh_cache: bool = False,
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
    cache_dir : pathlib.Path or str, optional
        Directory used to cache downloaded CSV data keyed by ticker. When a
        cached file exists it is used before attempting network access. When
        caching is enabled, network downloads ignore ``start``/``end`` so the
        cache remains reusable across date ranges; slicing happens after load.
    refresh_cache : bool, optional
        When ``True`` ignore any cached data and attempt a fresh download.

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
    raw_ticker = ticker.strip()
    ticker = raw_ticker.upper()

    offline_path = Path(offline_path) if offline_path is not None else None
    cache_dir = Path(cache_dir) if cache_dir is not None else None
    cache_file = cache_dir / f"{ticker}.csv" if cache_dir is not None else None

    if cache_file is not None and cache_file.exists() and not refresh_cache:
        return _slice_prices(_load_prices_from_csv(cache_file), start, end)

    attempts = 0
    last_error: Optional[Exception] = None
    if not prefer_local:
        while attempts < 3:
            try:
                download_start = None if cache_file is not None else start
                download_end = None if cache_file is not None else end
                data = yf.download(
                    ticker, start=download_start, end=download_end, progress=False
                )
                close = data.get("Close")
                if close is None or close.empty:
                    raise PriceDataError(
                        f"No price data returned for ticker '{ticker}'"
                    )
                close = close.sort_index()
                close.index.name = "Date"

                if cache_file is not None:
                    try:
                        cache_file.parent.mkdir(parents=True, exist_ok=True)
                        close.to_frame(name="Close").to_csv(cache_file, index_label="Date")
                    except Exception:
                        pass

                return _slice_prices(close, start, end)
            except Exception as exc:  # network error or other issues
                last_error = exc
                attempts += 1
                if attempts < 3:
                    time.sleep(2 ** (attempts - 1))

    # If online retrieval fails or local data is preferred, attempt to load CSV
    local_candidates: list[Path] = []
    if cache_file is not None:
        local_candidates.append(cache_file)

    if offline_path is None:
        local_candidates.append(_FALLBACK_DIR / f"{ticker}.csv")
    else:
        if offline_path.is_dir():
            local_candidates.extend(
                [
                    offline_path / f"{ticker}.csv",
                    offline_path / f"{raw_ticker}.csv",
                    offline_path / f"{raw_ticker.lower()}.csv",
                ]
            )
        else:
            local_candidates.append(offline_path)

    seen: set[Path] = set()
    deduped_candidates: list[Path] = []
    for candidate in local_candidates:
        if candidate in seen:
            continue
        deduped_candidates.append(candidate)
        seen.add(candidate)
    local_candidates = deduped_candidates

    for candidate in local_candidates:
        if candidate.exists():
            return _slice_prices(_load_prices_from_csv(candidate), start, end)

    attempted = ", ".join(str(path) for path in local_candidates)
    if last_error is None:
        raise PriceDataError(
            f"Failed to fetch price data for '{ticker}': offline CSV not found. "
            f"Tried: {attempted}"
        )
    raise PriceDataError(
        f"Failed to fetch price data for '{ticker}'. "
        f"Last network error: {type(last_error).__name__}: {last_error}. "
        f"Tried local CSVs: {attempted}"
    )
