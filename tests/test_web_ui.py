from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

pytest.importorskip("flask")

import data  # noqa: E402
import app as web_app  # noqa: E402


def test_build_public_argv_for_demo_simulate_uses_sample_data_and_seed() -> None:
    argv = web_app.build_public_argv(web_app.UIRequest())

    assert argv[:2] == ["simulate", "AAPL"]
    assert "--source" in argv
    assert "offline" in argv
    assert str(web_app.SAMPLE_DATA_DIR) in argv
    assert "--seed" in argv
    assert web_app.DEMO_SEED in argv
    assert "--days" in argv
    assert "20" in argv


def test_build_public_argv_for_demo_backtest_uses_short_window() -> None:
    argv = web_app.build_public_argv(web_app.UIRequest(job="backtest"))

    for expected in ("--lookback", "5", "--hold", "3", "--rebalance", "3", "--top", "1"):
        assert expected in argv


def test_validate_request_requires_existing_local_path(tmp_path: Path) -> None:
    missing = tmp_path / "missing"
    request = web_app.UIRequest(source="local", data_path=str(missing))

    assert web_app.validate_request(request) == (
        "That path was not found. Choose a CSV file or folder that exists."
    )


def test_create_page_state_for_default_demo_returns_chart() -> None:
    state = web_app.build_default_state()

    assert state.error is None
    assert state.request.source == "demo"
    assert state.chart_data_url is not None
    assert state.chart_data_url.startswith("data:image/png;base64,")
    assert state.title


def test_create_page_state_live_first_handles_download_shape(monkeypatch) -> None:
    dates = pd.date_range("2024-01-01", periods=40, freq="D")
    columns = pd.MultiIndex.from_product([["Close"], ["AAPL"]])
    downloaded = pd.DataFrame(
        [[100.0 + idx] for idx in range(len(dates))],
        index=dates,
        columns=columns,
    )

    def _download(_ticker, start=None, end=None, progress=False):
        assert progress is False
        return downloaded

    monkeypatch.setattr(data.yf, "download", _download)

    state = web_app.create_page_state(web_app.UIRequest(source="auto"))

    assert state.error is None
    assert state.chart_data_url is not None
    assert "Live prices weren’t available" not in state.summary


def test_create_page_state_local_sample_data_handles_multiple_tickers() -> None:
    state = web_app.create_page_state(
        web_app.UIRequest(
            source="local",
            tickers="AAPL MSFT",
            data_path=str(web_app.SAMPLE_DATA_DIR),
        )
    )

    assert state.error is None
    assert state.chart_data_url is not None
    assert "Summary for AAPL" in state.details_text
    assert "Summary for MSFT" in state.details_text


def test_flask_app_renders_default_demo() -> None:
    client = web_app.app.test_client()

    response = client.get("/")
    body = response.get_data(as_text=True)

    assert response.status_code == 200
    assert "Simulate current ideas or backtest history." in body
    assert "Live first" in body
    assert "Terminal output" in body
    assert web_app.SOURCE_NOTES["auto"] in body
    assert "data:image/png;base64," in body


def test_flask_app_surfaces_local_path_guidance() -> None:
    client = web_app.app.test_client()

    response = client.post("/", data={"job": "simulate", "source": "local", "tickers": "AAPL"})
    body = response.get_data(as_text=True)

    assert response.status_code == 200
    assert "Choose a CSV file or folder before running Local CSV." in body
    assert "CSV file or folder" in body


def test_healthz_and_css_routes() -> None:
    client = web_app.app.test_client()

    health = client.get("/healthz")
    css = client.get("/app.css")
    favicon = client.get("/favicon.ico")

    assert health.status_code == 200
    assert health.get_data(as_text=True) == "ok"
    assert css.status_code == 200
    assert ".masthead" in css.get_data(as_text=True)
    assert favicon.status_code == 204
