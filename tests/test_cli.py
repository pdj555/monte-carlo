from __future__ import annotations

import matplotlib

matplotlib.use("Agg")

import numpy as np
import pandas as pd

from cli import parse_args, run


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
