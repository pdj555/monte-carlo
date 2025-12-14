from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

import data
from data import PriceDataError, fetch_prices


def _write_sample_csv(path: Path, ticker: str) -> Path:
    dates = pd.date_range("2024-01-01", periods=5, freq="D")
    close = pd.Series([100, 101, 102, 101, 103], index=dates, name="Close")
    df = close.to_frame()
    csv_path = path / f"{ticker}.csv"
    df.to_csv(csv_path, index_label="Date")
    return csv_path


def test_fetch_prices_prefers_cache_before_network(tmp_path, monkeypatch):
    cache_dir = tmp_path / "cache"
    cache_dir.mkdir()
    _write_sample_csv(cache_dir, "AAPL")

    def _unexpected_download(*_args, **_kwargs):
        raise AssertionError("yfinance.download should not be called when cache exists")

    monkeypatch.setattr(data.yf, "download", _unexpected_download)

    prices = fetch_prices("AAPL", cache_dir=cache_dir)
    assert not prices.empty
    assert prices.name == "Close"


def test_fetch_prices_saves_cache_after_network(tmp_path, monkeypatch):
    cache_dir = tmp_path / "cache"

    dates = pd.date_range("2024-01-01", periods=4, freq="D")
    downloaded = pd.DataFrame({"Close": [10.0, 11.0, 12.0, 13.0]}, index=dates)

    def _download(_ticker, start=None, end=None, progress=False):
        assert start is None
        assert end is None
        assert progress is False
        return downloaded

    monkeypatch.setattr(data.yf, "download", _download)

    prices = fetch_prices("MSFT", cache_dir=cache_dir, refresh_cache=True)
    assert not prices.empty
    assert (cache_dir / "MSFT.csv").exists()

    def _unexpected_download(*_args, **_kwargs):
        raise AssertionError("yfinance.download should not be called on cache hits")

    monkeypatch.setattr(data.yf, "download", _unexpected_download)
    cached = fetch_prices("MSFT", cache_dir=cache_dir)
    pd.testing.assert_series_equal(prices, cached, check_freq=False)


def test_fetch_prices_supports_offline_directory(tmp_path):
    offline_dir = tmp_path / "offline"
    offline_dir.mkdir()
    _write_sample_csv(offline_dir, "AAPL")

    prices = fetch_prices("AAPL", offline_path=offline_dir, prefer_local=True)
    assert not prices.empty
    assert prices.index.is_monotonic_increasing


def test_fetch_prices_offline_directory_is_case_insensitive(tmp_path):
    offline_dir = tmp_path / "offline"
    offline_dir.mkdir()
    _write_sample_csv(offline_dir, "AAPL")

    prices = fetch_prices("aapl", offline_path=offline_dir, prefer_local=True)
    assert not prices.empty


def test_fetch_prices_rejects_start_after_end(tmp_path):
    offline_dir = tmp_path / "offline"
    offline_dir.mkdir()
    _write_sample_csv(offline_dir, "AAPL")

    with pytest.raises(PriceDataError):
        fetch_prices(
            "AAPL",
            start="2024-01-04",
            end="2024-01-02",
            offline_path=offline_dir,
            prefer_local=True,
        )


def test_fetch_prices_cache_is_not_date_range_dependent(tmp_path, monkeypatch):
    cache_dir = tmp_path / "cache"

    dates = pd.date_range("2024-01-01", periods=10, freq="D")
    downloaded = pd.DataFrame({"Close": range(10)}, index=dates)

    captured = {}

    def _download(_ticker, start=None, end=None, progress=False):
        captured["start"] = start
        captured["end"] = end
        captured["progress"] = progress
        return downloaded

    monkeypatch.setattr(data.yf, "download", _download)

    sliced = fetch_prices(
        "MSFT",
        start="2024-01-05",
        end="2024-01-07",
        cache_dir=cache_dir,
        refresh_cache=True,
    )
    assert captured["start"] is None
    assert captured["end"] is None
    assert captured["progress"] is False
    assert sliced.index.min() >= pd.Timestamp("2024-01-05")
    assert sliced.index.max() <= pd.Timestamp("2024-01-07")

    def _unexpected_download(*_args, **_kwargs):
        raise AssertionError("yfinance.download should not be called on cache hits")

    monkeypatch.setattr(data.yf, "download", _unexpected_download)

    earlier_slice = fetch_prices(
        "MSFT",
        start="2024-01-02",
        end="2024-01-03",
        cache_dir=cache_dir,
    )
    assert not earlier_slice.empty


def test_fetch_prices_rejects_invalid_dates(tmp_path):
    offline_dir = tmp_path / "offline"
    offline_dir.mkdir()
    _write_sample_csv(offline_dir, "AAPL")

    with pytest.raises(PriceDataError):
        fetch_prices("AAPL", start="not-a-date", offline_path=offline_dir, prefer_local=True)
