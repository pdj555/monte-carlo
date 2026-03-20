from __future__ import annotations

import pandas as pd

from backtest import parse_args, run, run_walk_forward_backtest


def _write_price_csv(path, ticker: str, closes: list[float]) -> None:
    dates = pd.date_range("2024-01-01", periods=len(closes), freq="D")
    pd.DataFrame({"Date": dates, "Close": closes}).to_csv(path / f"{ticker}.csv", index=False)


def _build_offline_dataset(path) -> None:
    _write_price_csv(
        path,
        "AAPL",
        [
            100,
            102,
            104,
            106,
            108,
            110,
            112,
            114,
            116,
            118,
            120,
            118,
            116,
            114,
            112,
            110,
            108,
            106,
            104,
            102,
            100,
            98,
            96,
            94,
        ],
    )
    _write_price_csv(
        path,
        "MSFT",
        [
            100,
            98,
            96,
            94,
            92,
            90,
            88,
            86,
            84,
            82,
            80,
            82,
            84,
            86,
            88,
            90,
            92,
            94,
            96,
            98,
            100,
            102,
            104,
            106,
        ],
    )


def test_run_walk_forward_backtest_applies_transaction_cost_drag(tmp_path):
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    _build_offline_dataset(data_dir)

    common_kwargs = {
        "tickers": ["AAPL", "MSFT"],
        "lookback_days": 5,
        "holding_days": 3,
        "rebalance_every": 3,
        "top_k": 1,
        "model": "gbm",
        "scenarios": 200,
        "seed": 7,
        "offline_path": data_dir,
        "offline_only": True,
        "min_expected_return": -1.0,
        "min_prob_up": 0.0,
        "max_var_95_pct": 1.0,
        "portfolio_risk_budget_pct": 1.0,
        "annual_cash_yield": 0.01,
    }

    frictionless = run_walk_forward_backtest(
        **common_kwargs,
        transaction_cost_bps=0.0,
    )
    costly = run_walk_forward_backtest(
        **common_kwargs,
        transaction_cost_bps=200.0,
    )

    frictionless_equity = frictionless["equity_curve"]
    costly_equity = costly["equity_curve"]
    frictionless_log = frictionless["rebalance_log"]
    frictionless_summary = frictionless["summary"]

    assert list(frictionless_equity.columns) == ["strategy", "equal_weight", "cash"]
    assert not frictionless_log.empty
    assert (frictionless_log["turnover"] > 0).any()
    assert costly_equity["strategy"].iloc[-1] < frictionless_equity["strategy"].iloc[-1]
    assert frictionless_summary["excess_return_vs_cash"] != 0.0


def test_backtest_cli_saves_summary_log_curve_and_plot(tmp_path):
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    _build_offline_dataset(data_dir)

    output_dir = tmp_path / "out"
    args = parse_args(
        [
            "--tickers",
            "AAPL,MSFT",
            "--lookback-days",
            "5",
            "--holding-days",
            "3",
            "--rebalance-every",
            "3",
            "--top-k",
            "1",
            "--model",
            "gbm",
            "--scenarios",
            "100",
            "--seed",
            "11",
            "--offline-path",
            str(data_dir),
            "--offline-only",
            "--output",
            str(output_dir),
            "--transaction-cost-bps",
            "0",
            "--annual-cash-yield",
            "0.01",
            "--min-expected-return",
            "-1",
            "--min-prob-up",
            "0",
            "--max-var-95-pct",
            "1",
            "--portfolio-risk-budget-pct",
            "1",
        ]
    )

    result = run(args)
    equity_curve = result["equity_curve"]

    assert not result["summary"].empty
    assert not result["rebalance_log"].empty
    assert equity_curve.index.is_monotonic_increasing
    assert (output_dir / "backtest_summary.csv").exists()
    assert (output_dir / "rebalance_log.csv").exists()
    assert (output_dir / "equity_curve.csv").exists()
    assert (output_dir / "equity_curve.png").exists()
