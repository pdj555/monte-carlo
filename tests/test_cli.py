from __future__ import annotations

import json
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
    assert "max_drawdown_q95" in result["report"]["rankings"]["AAPL"]
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


def test_cli_rejects_negative_portfolio_risk_budget_pct():
    with pytest.raises(SystemExit):
        parse_args(["--portfolio-risk-budget-pct", "-0.01"])


def test_cli_rejects_negative_annual_cash_yield():
    with pytest.raises(SystemExit):
        parse_args(["--annual-cash-yield", "-0.01"])


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


def test_cli_drawdown_guardrail_can_force_avoid(tmp_path):
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    dates = pd.date_range("2024-01-01", periods=60, freq="D")
    close = np.tile([100.0, 130.0, 90.0, 125.0, 88.0], 12)
    pd.DataFrame({"Date": dates, "Close": close}).to_csv(data_dir / "AAPL.csv", index=False)

    args = parse_args(
        [
            "--tickers",
            "AAPL",
            "--days",
            "25",
            "--scenarios",
            "80",
            "--seed",
            "2",
            "--no-show",
            "--no-plots",
            "--offline-path",
            str(data_dir),
            "--offline-only",
            "--max-drawdown-q95-pct",
            "0.2",
        ]
    )

    result = run(args)
    ranking = result["report"]["rankings"]["AAPL"]

    assert ranking["recommendation"] == "AVOID"
    assert "max_drawdown_q95>20.0%" in ranking["guardrail_reasons"]


def test_cli_shock_inputs_validation():
    with pytest.raises(SystemExit):
        parse_args(["--shock-probability", "1.1"])

    with pytest.raises(SystemExit):
        parse_args(["--shock-return", "-1.0"])


def test_cli_target_and_loss_guardrail_argument_validation():
    with pytest.raises(SystemExit):
        parse_args(["--min-prob-hit-target", "0.6"])

    with pytest.raises(SystemExit):
        parse_args(["--max-prob-breach-loss", "0.2"])

    with pytest.raises(SystemExit):
        parse_args(["--max-loss-pct", "-0.1"])


def test_cli_shock_mode_lowers_expected_return(tmp_path):
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    _write_sample_csv(str(data_dir), "AAPL", trend=0.4)

    base = run(
        parse_args(
            [
                "--tickers",
                "AAPL",
                "--days",
                "30",
                "--scenarios",
                "300",
                "--seed",
                "7",
                "--no-show",
                "--no-plots",
                "--offline-path",
                str(data_dir),
                "--offline-only",
            ]
        )
    )
    shocked = run(
        parse_args(
            [
                "--tickers",
                "AAPL",
                "--days",
                "30",
                "--scenarios",
                "300",
                "--seed",
                "7",
                "--no-show",
                "--no-plots",
                "--offline-path",
                str(data_dir),
                "--offline-only",
                "--shock-probability",
                "0.02",
                "--shock-return",
                "-0.05",
            ]
        )
    )

    base_er = base["report"]["results"]["AAPL"]["summary"]["expected_return"]
    shock_er = shocked["report"]["results"]["AAPL"]["summary"]["expected_return"]
    assert shock_er < base_er


def test_cli_reports_target_and_loss_probabilities(tmp_path):
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    _write_sample_csv(str(data_dir), "AAPL", trend=0.3)

    result = run(
        parse_args(
            [
                "--tickers",
                "AAPL",
                "--days",
                "20",
                "--scenarios",
                "200",
                "--seed",
                "123",
                "--no-show",
                "--no-plots",
                "--offline-path",
                str(data_dir),
                "--offline-only",
                "--target-return-pct",
                "0.05",
                "--max-loss-pct",
                "0.08",
                "--min-prob-hit-target",
                "0.1",
                "--max-prob-breach-loss",
                "0.95",
            ]
        )
    )

    summary = result["report"]["results"]["AAPL"]["summary"]
    ranking = result["report"]["rankings"]["AAPL"]

    assert 0.0 <= summary["prob_hit_target"] <= 1.0
    assert 0.0 <= summary["prob_breach_max_loss"] <= 1.0
    assert "prob_hit_target" in ranking
    assert "prob_breach_max_loss" in ranking


def test_cli_applies_portfolio_risk_budget_scaling(tmp_path):
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    _write_sample_csv(str(data_dir), "AAPL", trend=0.4)
    _write_sample_csv(str(data_dir), "MSFT", trend=0.35)

    unconstrained = run(
        parse_args(
            [
                "--tickers",
                "AAPL,MSFT",
                "--days",
                "25",
                "--scenarios",
                "250",
                "--seed",
                "123",
                "--no-show",
                "--no-plots",
                "--offline-path",
                str(data_dir),
                "--offline-only",
                "--portfolio-risk-budget-pct",
                "1.0",
                "--shock-probability",
                "0.02",
                "--shock-return",
                "-0.05",
                "--min-expected-return",
                "-1.0",
                "--min-prob-up",
                "0.0",
                "--max-var-95-pct",
                "1.0",
            ]
        )
    )

    constrained = run(
        parse_args(
            [
                "--tickers",
                "AAPL,MSFT",
                "--days",
                "25",
                "--scenarios",
                "250",
                "--seed",
                "123",
                "--no-show",
                "--no-plots",
                "--offline-path",
                str(data_dir),
                "--offline-only",
                "--portfolio-risk-budget-pct",
                "0.01",
                "--shock-probability",
                "0.02",
                "--shock-return",
                "-0.05",
                "--min-expected-return",
                "-1.0",
                "--min-prob-up",
                "0.0",
                "--max-var-95-pct",
                "1.0",
            ]
        )
    )

    unconstrained_weight = sum(item["weight"] for item in unconstrained["report"]["allocations"].values())
    constrained_weight = sum(item["weight"] for item in constrained["report"]["allocations"].values())

    assert constrained_weight < unconstrained_weight
    assert constrained["report"]["portfolio_risk_budget_pct"] == pytest.approx(0.01)


def test_cli_generates_execution_plan_when_capital_is_provided(tmp_path):
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
            "--capital",
            "10000",
        ]
    )

    result = run(args)

    assert result["report"]["execution_plan"]
    first = next(iter(result["report"]["execution_plan"].values()))
    assert "shares" in first
    assert (output_dir / "execution_plan.csv").exists()


def test_cli_accepts_block_bootstrap_for_historical_model(tmp_path):
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    _write_sample_csv(str(data_dir), "AAPL", trend=0.2)

    result = run(
        parse_args(
            [
                "--tickers",
                "AAPL",
                "--days",
                "20",
                "--scenarios",
                "120",
                "--seed",
                "123",
                "--no-show",
                "--no-plots",
                "--offline-path",
                str(data_dir),
                "--offline-only",
                "--model",
                "historical",
                "--block-size",
                "5",
            ]
        )
    )

    assert result["report"]["results"]["AAPL"]["summary"]["mean"] > 0


def test_cli_report_includes_benchmark_metrics(tmp_path):
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    _write_sample_csv(str(data_dir), "AAPL", trend=0.25)

    result = run(
        parse_args(
            [
                "--tickers",
                "AAPL",
                "--days",
                "20",
                "--scenarios",
                "200",
                "--seed",
                "123",
                "--no-show",
                "--no-plots",
                "--offline-path",
                str(data_dir),
                "--offline-only",
                "--annual-cash-yield",
                "0.10",
            ]
        )
    )

    summary = result["report"]["results"]["AAPL"]["summary"]

    assert "benchmark_return_pct" in summary
    assert "expected_excess_return" in summary
    assert "prob_beat_benchmark" in summary


def test_cli_policy_file_applies_defaults_and_reports_fingerprint(tmp_path):
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    _write_sample_csv(str(data_dir), "AAPL", trend=0.3)

    policy_file = tmp_path / "policy.json"
    policy_file.write_text(
        json.dumps(
            {
                "min_expected_return": 0.5,
                "min_prob_up": 0.99,
                "max_var_95_pct": 0.05,
            }
        ),
        encoding="utf-8",
    )

    result = run(
        parse_args(
            [
                "--tickers",
                "AAPL",
                "--days",
                "20",
                "--scenarios",
                "100",
                "--seed",
                "123",
                "--no-show",
                "--no-plots",
                "--offline-path",
                str(data_dir),
                "--offline-only",
                "--policy-file",
                str(policy_file),
            ]
        )
    )

    ranking = result["report"]["rankings"]["AAPL"]
    assert ranking["recommendation"] == "AVOID"
    assert result["report"]["policy"]["min_expected_return"] == 0.5
    assert result["report"]["policy_crc32"] is not None


def test_cli_explicit_flags_override_policy_file(tmp_path):
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    _write_sample_csv(str(data_dir), "AAPL", trend=0.3)

    policy_file = tmp_path / "policy.json"
    policy_file.write_text(json.dumps({"min_expected_return": 0.9}), encoding="utf-8")

    args = parse_args(
        [
            "--tickers",
            "AAPL",
            "--days",
            "20",
            "--scenarios",
            "100",
            "--seed",
            "123",
            "--no-show",
            "--no-plots",
            "--offline-path",
            str(data_dir),
            "--offline-only",
            "--policy-file",
            str(policy_file),
            "--min-expected-return",
            "-1.0",
        ]
    )

    result = run(args)
    ranking = result["report"]["rankings"]["AAPL"]

    assert ranking["recommendation"] in {"BUY", "WATCH"}
