from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

import data
import ui_bridge
import ui_state


def test_build_public_argv_for_demo_simulate_uses_sample_data_and_seed() -> None:
    argv = ui_state.build_public_argv(ui_state.UIRequest())

    assert argv[:2] == ["simulate", "AAPL"]
    assert "--source" in argv
    assert "offline" in argv
    assert str(ui_state.SAMPLE_DATA_DIR) in argv
    assert "--seed" in argv
    assert ui_state.DEMO_SEED in argv
    assert "--days" in argv
    assert "20" in argv


def test_build_public_argv_for_demo_backtest_uses_short_window() -> None:
    argv = ui_state.build_public_argv(ui_state.UIRequest(job="backtest"))

    for expected in ("--lookback", "5", "--hold", "3", "--rebalance", "3", "--top", "1"):
        assert expected in argv


def test_build_public_argv_for_auto_uses_live_source_with_fallback_path() -> None:
    argv = ui_state.build_public_argv(ui_state.UIRequest(source="auto"))

    assert "--source" in argv
    assert "auto" in argv
    assert "--data-path" in argv
    assert str(ui_state.SAMPLE_DATA_DIR) in argv


def test_build_public_argv_for_online_uses_live_only() -> None:
    argv = ui_state.build_public_argv(ui_state.UIRequest(source="online"))

    assert argv.count("--source") == 1
    assert "online" in argv
    assert "--data-path" not in argv


def test_build_public_argv_for_backtest_auto_uses_live_window() -> None:
    argv = ui_state.build_public_argv(ui_state.UIRequest(job="backtest", source="auto"))

    assert "--lookback" in argv
    assert "60" in argv
    assert "--scenarios" in argv
    assert "100" in argv


def test_build_public_argv_for_backtest_local_sample_uses_short_window() -> None:
    argv = ui_state.build_public_argv(
        ui_state.UIRequest(
            job="backtest",
            source="local",
            data_path=str(ui_state.SAMPLE_DATA_DIR),
        )
    )

    assert "--lookback" in argv
    assert "5" in argv


def test_validate_request_requires_existing_local_path(tmp_path: Path) -> None:
    missing = tmp_path / "missing"
    request = ui_state.UIRequest(source="local", data_path=str(missing))

    assert ui_state.validate_request(request) == (
        "That path was not found. Choose a CSV file or folder that exists."
    )


def test_create_page_state_for_default_demo_returns_chart() -> None:
    state = ui_state.build_default_state()

    assert state.error is None
    assert state.request.source == "demo"
    assert state.chart_svg is not None
    assert state.chart_svg.startswith("<svg")
    assert state.title
    assert state.source_note == "Data source: bundled sample data."


def test_create_page_state_live_first_handles_download_shape(monkeypatch) -> None:
    dates = pd.date_range("2024-01-01", periods=40, freq="D")
    columns = pd.MultiIndex.from_product([["Close"], ["AAPL"]])
    downloaded = pd.DataFrame(
        [[100.0 + idx] for idx in range(len(dates))],
        index=dates,
        columns=columns,
    )

    def _download(_ticker, start=None, end=None, progress=False, **kwargs):
        assert progress is False
        assert kwargs.get("period") == "max"
        return downloaded

    monkeypatch.setattr(data.yf, "download", _download)

    state = ui_state.create_page_state(ui_state.UIRequest(source="auto"))

    assert state.error is None
    assert state.chart_svg is not None
    assert "Live prices weren't available" not in state.summary
    assert state.source_note == "Data source: live download."


def test_create_page_state_local_sample_data_handles_multiple_tickers() -> None:
    state = ui_state.create_page_state(
        ui_state.UIRequest(
            source="local",
            tickers="AAPL MSFT",
            data_path=str(ui_state.SAMPLE_DATA_DIR),
        )
    )

    assert state.error is None
    assert state.chart_svg is not None
    assert state.source_note == "Data source: bundled sample data for AAPL, MSFT."
    assert "Summary for AAPL" in state.details_text
    assert "Summary for MSFT" in state.details_text


def test_create_page_state_auto_fallback_reports_bundled_sample(monkeypatch) -> None:
    def _download(*_args, **_kwargs):
        raise RuntimeError("network unavailable")

    monkeypatch.setattr(data.yf, "download", _download)

    state = ui_state.create_page_state(ui_state.UIRequest(source="auto"))

    assert state.error is None
    assert state.source_note == "Data source: bundled sample data (fallback)."


def test_ui_bridge_serializes_default_state() -> None:
    payload = ui_bridge.create_payload({"job": "simulate", "source": "demo"})

    assert payload["request"]["job"] == "simulate"
    assert payload["request"]["source"] == "demo"
    assert payload["chartSvg"].startswith("<svg")
    assert payload["metrics"]
    assert payload["sourceNote"] == "Data source: bundled sample data."


def test_ui_bridge_cli_returns_json(capsys, monkeypatch) -> None:
    monkeypatch.setattr("sys.stdin.read", lambda: json.dumps({"source": "demo"}))

    assert ui_bridge.main() == 0
    payload = json.loads(capsys.readouterr().out)

    assert payload["request"]["source"] == "demo"
    assert payload["chartSvg"].startswith("<svg")


def test_next_ui_files_present_the_public_surface() -> None:
    package_json = json.loads(Path("package.json").read_text(encoding="utf-8"))
    page = Path("app/page.tsx").read_text(encoding="utf-8")
    workbench = Path("components/workbench/workbench.tsx").read_text(encoding="utf-8")
    bridge = Path("lib/python-bridge.ts").read_text(encoding="utf-8")
    route = Path("app/api/run/route.ts").read_text(encoding="utf-8")

    assert package_json["dependencies"]["next"].startswith("16.")
    assert "Workbench" in page
    assert 'source: "auto"' in page
    assert "RunResults" in workbench
    assert 'source: "auto"' in page
    assert 'runtime = "nodejs"' in route
    assert "runViaVercelEngine" in bridge
    assert Path("api/engine.py").exists()
