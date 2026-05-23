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
_PRICE_SOURCE_ATTR = "price_source"


class PriceDataError(Exception):
    """Raised when price data cannot be retrieved."""


def _with_price_source(
    prices: pd.Series,
    *,
    kind: str,
    path: Path | None = None,
    used_fallback: bool = False,
    is_sample_data: bool = False,
) -> pd.Series:
    tagged = prices.copy()
    tagged.attrs[_PRICE_SOURCE_ATTR] = {
        "kind": kind,
        "path": str(path) if path is not None else None,
        "used_fallback": used_fallback,
        "is_sample_data": is_sample_data,
    }
    return tagged


def get_price_source(prices: pd.Series) -> dict[str, object] | None:
    """Return normalized provenance metadata attached to a fetched price series."""

    raw = prices.attrs.get(_PRICE_SOURCE_ATTR)
    if not isinstance(raw, dict):
        return None

    return {
        "kind": str(raw.get("kind", "")),
        "path": raw.get("path"),
        "used_fallback": bool(raw.get("used_fallback", False)),
        "is_sample_data": bool(raw.get("is_sample_data", False)),
    }


def _is_bundled_sample_path(path: Path) -> bool:
    try:
        return path.resolve().is_relative_to(_FALLBACK_DIR.resolve())
    except OSError:
        return False


def _parse_date(value: Optional[str], *, label: str) -> Optional[pd.Timestamp]:
    """Parse a user-supplied date string into a pandas Timestamp."""

    if value is None:
        return None
    try:
        return pd.to_datetime(value)
    except Exception as exc:
        raise PriceDataError(
            f"{label.title()} date '{value}' is not valid. "
            "Use YYYY-MM-DD, for example 2024-01-31."
        ) from exc


def _slice_prices(
    prices: pd.Series, start: Optional[str], end: Optional[str]
) -> pd.Series:
    attrs = dict(prices.attrs)
    start_ts = _parse_date(start, label="start")
    end_ts = _parse_date(end, label="end")
    if start_ts is not None and end_ts is not None and start_ts > end_ts:
        raise PriceDataError(
            "Start date must be on or before end date. "
            "Choose an earlier start date or a later end date."
        )

    prices = prices.sort_index()
    if start_ts is not None:
        prices = prices.loc[start_ts:]
    if end_ts is not None:
        prices = prices.loc[:end_ts]
    if prices.empty:
        raise PriceDataError(
            "No price data is available for the requested date range. "
            "Try a wider range or remove the date filter."
        )
    prices.attrs = attrs
    return prices


def _load_prices_from_csv(path: Path) -> pd.Series:
    """Load a price series from a CSV file with common column conventions."""

    try:
        df = pd.read_csv(path)
    except Exception as exc:
        raise PriceDataError(
            f"Couldn't read CSV at '{path}'. "
            "Check that the file exists and is a valid CSV."
        ) from exc

    if df.empty:
        raise PriceDataError(
            f"CSV at '{path}' is empty. Add rows with Date and Close columns."
        )

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
            f"CSV at '{path}' is missing a Close column. "
            "Add a 'Close' or 'Adj Close' column."
        )

    series = pd.to_numeric(df[close_column], errors="coerce").dropna()
    series.name = "Close"
    if series.empty:
        raise PriceDataError(
            f"CSV at '{path}' does not contain usable close prices. "
            "Check that the Close column has numeric values."
        )
    return series


def _normalize_download_close(
    close: pd.Series | pd.DataFrame,
    *,
    ticker: str,
) -> pd.Series:
    """Normalize yfinance close data into a single close-price series."""

    if isinstance(close, pd.DataFrame):
        normalized: pd.Series | pd.DataFrame = close
        if isinstance(normalized.columns, pd.MultiIndex):
            for level in (0, -1):
                labels = normalized.columns.get_level_values(level)
                if ticker in labels:
                    normalized = normalized.xs(ticker, axis=1, level=level, drop_level=True)
                    break
        if isinstance(normalized, pd.DataFrame):
            if ticker in normalized.columns:
                normalized = normalized[ticker]
            elif normalized.shape[1] == 1:
                normalized = normalized.iloc[:, 0]
        if isinstance(normalized, pd.DataFrame):
            raise PriceDataError(
                f"Price data for '{ticker}' came back in an unexpected shape. "
                "Try again later or switch to local CSV data."
            )
        close = normalized

    if not isinstance(close, pd.Series):
        raise PriceDataError(
            f"Price data for '{ticker}' came back in an unexpected format. "
            "Try again later or switch to local CSV data."
        )

    series = pd.to_numeric(close, errors="coerce").dropna().sort_index()
    series.name = "Close"
    if series.empty:
        raise PriceDataError(
            f"No usable close prices were returned for '{ticker}'. "
            "Try again later or switch to local CSV data."
        )
    return series


def fetch_prices(
    ticker: str,
    start: Optional[str] = None,
    end: Optional[str] = None,
    *,
    offline_path: Optional[Path | str] = None,
    prefer_local: bool = False,
    allow_local_fallback: bool = True,
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
    allow_local_fallback : bool, optional
        When ``False`` do not fall back to local CSV files after a failed
        network request.
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
        raise PriceDataError("Ticker must be a non-empty string. Provide at least one symbol.")
    raw_ticker = ticker.strip()
    ticker = raw_ticker.upper()

    offline_path = Path(offline_path) if offline_path is not None else None
    cache_dir = Path(cache_dir) if cache_dir is not None else None
    cache_file = cache_dir / f"{ticker}.csv" if cache_dir is not None else None

    if cache_file is not None and cache_file.exists() and not refresh_cache:
        return _slice_prices(
            _with_price_source(
                _load_prices_from_csv(cache_file),
                kind="cache",
                path=cache_file,
            ),
            start,
            end,
        )

    attempts = 0
    last_error: Optional[Exception] = None
    if not prefer_local:
        while attempts < 3:
            try:
                download_start = None if cache_file is not None else start
                download_end = None if cache_file is not None else end
                if download_start is None and download_end is None:
                    raw = yf.download(
                        ticker,
                        period="max",
                        progress=False,
                        auto_adjust=True,
                    )
                else:
                    raw = yf.download(
                        ticker,
                        start=download_start,
                        end=download_end,
                        progress=False,
                        auto_adjust=True,
                    )
                close = raw.get("Close")
                if close is None or close.empty:
                    raise PriceDataError(
                        f"No price data was returned for '{ticker}'. "
                        "Check the symbol and try again."
                    )
                close = _normalize_download_close(close, ticker=ticker)
                close.index.name = "Date"

                if cache_file is not None:
                    try:
                        cache_file.parent.mkdir(parents=True, exist_ok=True)
                        close.to_frame(name="Close").to_csv(cache_file, index_label="Date")
                    except Exception:
                        pass

                return _slice_prices(
                    _with_price_source(close, kind="live"),
                    start,
                    end,
                )
            except Exception as exc:  # network error or other issues
                last_error = exc
                attempts += 1
                if attempts < 3:
                    time.sleep(2 ** (attempts - 1))

    if not allow_local_fallback and not prefer_local:
        if last_error is None:
            raise PriceDataError(
                f"Couldn't download price data for '{ticker}'. "
                "Try again later or switch to local CSV data."
            )
        raise PriceDataError(
            f"Couldn't download price data for '{ticker}'. "
            f"Last network error: {type(last_error).__name__}: {last_error}. "
            "Try again later or switch to local CSV data."
        )

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
            if cache_file is not None and candidate == cache_file:
                tagged = _with_price_source(
                    _load_prices_from_csv(candidate),
                    kind="cache",
                    path=candidate,
                )
            else:
                tagged = _with_price_source(
                    _load_prices_from_csv(candidate),
                    kind="local",
                    path=candidate,
                    used_fallback=last_error is not None and not prefer_local,
                    is_sample_data=_is_bundled_sample_path(candidate),
                )
            return _slice_prices(tagged, start, end)

    attempted = ", ".join(str(path) for path in local_candidates)
    if last_error is None:
        raise PriceDataError(
            f"Couldn't load price data for '{ticker}' from local CSVs. "
            f"Use --data-path to point at a directory containing '{ticker}.csv', "
            f"or switch --source to auto or online. Tried: {attempted}"
        )
    raise PriceDataError(
        f"Couldn't load price data for '{ticker}'. "
        f"Last network error: {type(last_error).__name__}: {last_error}. "
        f"Use --data-path to point at a directory containing '{ticker}.csv', "
        f"or switch --source to auto or online. Tried: {attempted}"
    )
