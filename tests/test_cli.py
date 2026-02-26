from __future__ import annotations

import numpy as np
import pandas as pd
import matplotlib
import pytest

matplotlib.use("Agg")

from cli import main, parse_args, run  # noqa: E402


def _write_sample_csv(directory: str, ticker: str, trend: float) -> None:
    dates = pd.date_range("2024-01-01", periods=40, freq="D")
    close = 100 + trend * np.arange(len(dates))
    df = pd.DataFrame({"Date": dates, "Close": close})
    df.to_csv(f"{directory}/{ticker}.csv", index=False)


def test_cli_runs_multi_ticker(tmp_path):
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    _write_sample_csv(str(data_dir), "AAPL", trend=0.5)
    _write_sample_csv(str(data_dir), "MSFT", trend=0.3)

    output_dir = tmp_path / "out"

    args = parse_args(
        [
            "--tickers",
            "AAPL,MSFT",
            "--days",
            "20",
            "--scenarios",
            "25",
            "--seed",
            "123",
            "--no-show",
            "--output",
            str(output_dir),
            "--offline-path",
            str(data_dir),
            "--offline-only",
        ]
    )

    result = run(args)

    combined = result["simulations"]
    summaries = result["summaries"]

    assert not combined.empty
    assert set(summaries.index) == {"AAPL", "MSFT"}
    assert (output_dir / "AAPL_distribution.png").exists()
    assert (output_dir / "MSFT_paths.png").exists()
    assert set(combined.columns.get_level_values(0)) == {"AAPL", "MSFT"}

    assert result["report"]["portfolio_summary"] is not None
    assert result["report"]["portfolio_summary"]["component_count"] == 2.0
    assert set(result["report"]["rankings"]) == {"AAPL", "MSFT"}
    assert result["report"]["allocations"]
    assert result["report"]["action_plan"]["stance"] in {"RISK_ON", "SELECTIVE", "DEFENSIVE"}
    assert (output_dir / "rankings.csv").exists()
    assert (output_dir / "allocations.csv").exists()
    assert (output_dir / "action_plan.md").exists()


def test_cli_main_respects_strict_mode(tmp_path):
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    _write_sample_csv(str(data_dir), "AAPL", trend=0.5)

    output_dir = tmp_path / "out"

    base_args = [
        "--tickers",
        "AAPL,MSFT",
        "--days",
        "20",
        "--scenarios",
        "25",
        "--seed",
        "123",
        "--no-show",
        "--no-plots",
        "--output",
        str(output_dir),
        "--offline-path",
        str(data_dir),
        "--offline-only",
    ]

    assert main(base_args) == 0
    assert main([*base_args, "--strict"]) == 1


def test_cli_rejects_negative_seed():
    with pytest.raises(SystemExit):
        parse_args(["--seed", "-1"])


def test_cli_no_plots_skips_plot_files(tmp_path):
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    _write_sample_csv(str(data_dir), "AAPL", trend=0.5)

    output_dir = tmp_path / "out"

    args = parse_args(
        [
            "--tickers",
            "AAPL",
            "--days",
            "10",
            "--scenarios",
            "25",
            "--seed",
            "123",
            "--no-show",
            "--no-plots",
            "--output",
            str(output_dir),
            "--offline-path",
            str(data_dir),
            "--offline-only",
        ]
    )

    result = run(args)

    assert not result["summaries"].empty
    assert (output_dir / "AAPL_distribution.png").exists() is False
    assert (output_dir / "AAPL_paths.png").exists() is False


def test_cli_guardrails_can_force_defensive_plan(tmp_path):
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    _write_sample_csv(str(data_dir), "AAPL", trend=0.05)

    args = parse_args(
        [
            "--tickers",
            "AAPL",
            "--days",
            "15",
            "--scenarios",
            "50",
            "--seed",
            "1",
            "--no-show",
            "--no-plots",
            "--offline-path",
            str(data_dir),
            "--offline-only",
            "--min-expected-return",
            "0.5",
        ]
    )

    result = run(args)
    ranking = result["report"]["rankings"]["AAPL"]

    assert ranking["recommendation"] == "AVOID"
    assert "expected_return<50.0%" in ranking["guardrail_reasons"]
    assert result["report"]["action_plan"]["stance"] == "DEFENSIVE"
