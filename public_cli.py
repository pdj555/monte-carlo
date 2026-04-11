"""Public CLI surface for the ``monte-carlo`` command."""

from __future__ import annotations

import argparse
import logging
from pathlib import Path
from typing import Any, Iterable, Optional

import pandas as pd

import backtest as backtest_cli
from cli_shared import (
    IntentionalDefaultsHelpFormatter,
    non_negative_int,
    package_version,
    positive_int,
    render_detailed_simulation_tables,
)
from simulate_cli import maybe_show_simulation_plots, run as run_simulation

LOGGER = logging.getLogger(__name__)


def build_public_parser() -> argparse.ArgumentParser:
    """Create the simplified public CLI parser."""

    parser = argparse.ArgumentParser(
        prog="monte-carlo",
        description="Monte Carlo tools for current ideas and historical validation.",
        formatter_class=IntentionalDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--version",
        action="version",
        version=f"%(prog)s {package_version()}",
    )

    subparsers = parser.add_subparsers(dest="command")

    simulate_parser = subparsers.add_parser(
        "simulate",
        help="Rank current opportunities from simulated future paths.",
        formatter_class=IntentionalDefaultsHelpFormatter,
    )
    simulate_parser.add_argument(
        "tickers",
        nargs="*",
        help="Ticker symbols to simulate. Defaults to AAPL when omitted.",
    )
    simulate_parser.add_argument(
        "--days",
        type=positive_int,
        default=252,
        help="Trading days to simulate into the future.",
    )
    simulate_parser.add_argument(
        "--scenarios",
        type=positive_int,
        default=1000,
        help="Number of Monte Carlo paths to run per ticker.",
    )
    simulate_parser.add_argument(
        "--model",
        choices=("historical", "gbm"),
        default="historical",
        help="Simulation model to use.",
    )
    simulate_parser.add_argument(
        "--seed",
        type=non_negative_int,
        default=None,
        help="Random seed for reproducible runs.",
    )
    simulate_parser.add_argument(
        "--source",
        choices=("auto", "offline", "online"),
        default="auto",
        help="Where to load price data from.",
    )
    simulate_parser.add_argument(
        "--data-path",
        type=str,
        default=None,
        help="Directory or CSV file for local price data.",
    )
    simulate_parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Directory where reports and plots are saved.",
    )
    simulate_parser.add_argument(
        "--show",
        action="store_true",
        help="Display plots after the run finishes.",
    )
    simulate_parser.add_argument(
        "--details",
        action="store_true",
        help="Print tables and secondary metrics.",
    )

    backtest_parser = subparsers.add_parser(
        "backtest",
        help="Validate the process with walk-forward backtesting.",
        formatter_class=IntentionalDefaultsHelpFormatter,
    )
    backtest_parser.add_argument(
        "tickers",
        nargs="*",
        help="Ticker symbols to evaluate. Defaults to AAPL when omitted.",
    )
    backtest_parser.add_argument(
        "--lookback",
        type=positive_int,
        default=60,
        help="Trading days of history to use before each rebalance.",
    )
    backtest_parser.add_argument(
        "--hold",
        type=positive_int,
        default=20,
        help="Trading days to hold each position after a rebalance.",
    )
    backtest_parser.add_argument(
        "--rebalance",
        type=positive_int,
        default=20,
        help="Trading days between rebalances.",
    )
    backtest_parser.add_argument(
        "--top",
        type=positive_int,
        default=1,
        help="Number of ranked tickers to hold after each rebalance.",
    )
    backtest_parser.add_argument(
        "--model",
        choices=("historical", "gbm"),
        default="historical",
        help="Simulation model to use at each rebalance.",
    )
    backtest_parser.add_argument(
        "--scenarios",
        type=positive_int,
        default=1000,
        help="Number of Monte Carlo paths to run at each rebalance.",
    )
    backtest_parser.add_argument(
        "--seed",
        type=non_negative_int,
        default=None,
        help="Random seed for reproducible runs.",
    )
    backtest_parser.add_argument(
        "--source",
        choices=("auto", "offline", "online"),
        default="auto",
        help="Where to load price data from.",
    )
    backtest_parser.add_argument(
        "--data-path",
        type=str,
        default=None,
        help="Directory or CSV file for local price data.",
    )
    backtest_parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Directory where reports and plots are saved.",
    )
    backtest_parser.add_argument(
        "--details",
        action="store_true",
        help="Print tables and secondary metrics.",
    )

    return parser


def _parse_public_args_with_parser(
    argv: Optional[Iterable[str]] = None,
) -> tuple[argparse.ArgumentParser, argparse.Namespace]:
    parser = build_public_parser()
    args = parser.parse_args(list(argv) if argv is not None else None)
    if getattr(args, "command", None) and not args.tickers:
        args.tickers = ["AAPL"]
    return parser, args


def parse_public_args(argv: Optional[Iterable[str]] = None) -> argparse.Namespace:
    """Return parsed arguments for the public CLI."""

    _, args = _parse_public_args_with_parser(argv)
    return args


def _source_settings(source: str) -> tuple[bool, bool]:
    """Map a public source mode to fetch settings."""

    if source == "offline":
        return True, True
    if source == "online":
        return False, False
    return False, True


def _public_tickers_to_csv_arg(tickers: list[str]) -> str:
    return ",".join(tickers or ["AAPL"])


def _build_public_simulate_legacy_args(args: argparse.Namespace) -> argparse.Namespace:
    prefer_local, allow_local_fallback = _source_settings(str(args.source))
    should_make_plots = bool(args.show or args.output)
    return argparse.Namespace(
        journal_file=None,
        policy_file=None,
        tickers=_public_tickers_to_csv_arg(list(args.tickers)),
        days=int(args.days),
        scenarios=int(args.scenarios),
        max_paths=100,
        no_plots=not should_make_plots,
        dt=1.0,
        block_size=1,
        shock_probability=0.0,
        shock_return=-0.15,
        seed=args.seed,
        model=str(args.model),
        fundamental_probability=None,
        market_price=None,
        fundamental_certainty=100.0,
        prob_mean_reversion=0.2,
        prob_daily_volatility=0.03,
        start=None,
        end=None,
        output=args.output,
        cache_dir=None,
        refresh_cache=False,
        save_simulations=False,
        offline_path=args.data_path,
        offline_only=prefer_local,
        allow_local_fallback=allow_local_fallback,
        show=bool(args.show),
        ai_summary=False,
        ai_model="gpt-4o-mini",
        annual_cash_yield=0.04,
        min_expected_return=0.0,
        min_prob_up=0.5,
        portfolio_risk_budget_pct=0.02,
        max_var_95_pct=0.25,
        max_drawdown_q95_pct=None,
        target_return_pct=None,
        max_loss_pct=None,
        min_prob_hit_target=None,
        max_prob_breach_loss=None,
        capital=None,
        allow_fractional_shares=False,
        minimal=False,
        strict=False,
        verbose=False,
        details=bool(args.details),
    )


def _build_public_backtest_legacy_args(args: argparse.Namespace) -> argparse.Namespace:
    prefer_local, allow_local_fallback = _source_settings(str(args.source))
    return argparse.Namespace(
        tickers=_public_tickers_to_csv_arg(list(args.tickers)),
        lookback_days=int(args.lookback),
        holding_days=int(args.hold),
        rebalance_every=int(args.rebalance),
        top_k=int(args.top),
        model=str(args.model),
        scenarios=int(args.scenarios),
        seed=args.seed,
        start=None,
        end=None,
        offline_path=args.data_path,
        offline_only=prefer_local,
        allow_local_fallback=allow_local_fallback,
        output=args.output,
        transaction_cost_bps=10.0,
        annual_cash_yield=0.04,
        min_expected_return=0.0,
        min_prob_up=0.5,
        max_var_95_pct=0.25,
        max_drawdown_q95_pct=None,
        portfolio_risk_budget_pct=0.02,
        verbose=False,
        details=bool(args.details),
    )


def _render_public_simulation_output(
    result: dict[str, Any],
    *,
    details: bool,
    output: str | None,
) -> None:
    report = result["report"]
    action_plan = report["action_plan"]

    print(f"Stance: {action_plan['stance']}")
    print(action_plan["headline"])

    if action_plan["primary_pick"] is not None:
        pick = action_plan["primary_pick"]
        print(
            f"Top idea: {pick['ticker']} at {pick['weight']:.1%} weight "
            f"(expected return {pick['expected_return']:.1%})."
        )
    if action_plan["avoid_list"]:
        print(f"Avoid: {', '.join(action_plan['avoid_list'])}")
    if action_plan.get("cash_weight", 0.0) > 0:
        print(f"Cash buffer: {action_plan['cash_weight']:.1%}")

    for item in report["errors"]:
        print(f"Skipped {item['ticker']}: {item['error']}")

    if details:
        summary_df = result["summaries"]
        portfolio_summary = result["portfolio_summary"]
        rankings_payload = report["rankings"]
        allocations_payload = report["allocations"]
        rankings = (
            pd.DataFrame.from_dict(rankings_payload, orient="index")
            if rankings_payload
            else pd.DataFrame()
        )
        allocations = (
            pd.DataFrame.from_dict(allocations_payload, orient="index")
            if allocations_payload
            else pd.DataFrame()
        )
        render_detailed_simulation_tables(
            summary_df,
            portfolio_summary,
            rankings,
            allocations,
        )

    if output:
        print(f"Saved outputs to {Path(output).expanduser()}")


def _render_public_backtest_output(
    result: dict[str, pd.DataFrame | pd.Series],
    *,
    details: bool,
    output: str | None,
) -> None:
    summary = result["summary"]
    if not isinstance(summary, pd.Series):
        raise ValueError("summary output must be a pandas Series")

    print(
        "Strategy return: "
        f"{float(summary['strategy_total_return']):.1%} "
        f"({float(summary['strategy_annualized_return']):.1%} annualized)"
    )
    print(f"Max drawdown: {float(summary['strategy_max_drawdown']):.1%}")
    print(f"vs equal weight: {float(summary['excess_return_vs_equal_weight']):.1%}")
    print(f"vs cash: {float(summary['excess_return_vs_cash']):.1%}")

    if details:
        print("\nBacktest summary")
        print(summary.to_frame(name="value").to_string(float_format=lambda value: f"{value:0.4f}"))

    if output:
        print(f"Saved outputs to {Path(output).expanduser()}")


def run_public_simulate(args: argparse.Namespace) -> dict[str, Any]:
    """Execute the simplified simulate command."""

    legacy_args = _build_public_simulate_legacy_args(args)
    result = run_simulation(legacy_args, render=False, display_plots=False)
    _render_public_simulation_output(
        result,
        details=bool(args.details),
        output=args.output,
    )
    maybe_show_simulation_plots(legacy_args, result)
    return result


def run_public_backtest(args: argparse.Namespace) -> dict[str, pd.DataFrame | pd.Series]:
    """Execute the simplified backtest command."""

    legacy_args = _build_public_backtest_legacy_args(args)
    result = backtest_cli.run(legacy_args, render=False)
    _render_public_backtest_output(
        result,
        details=bool(args.details),
        output=args.output,
    )
    return result


def main(argv: Optional[Iterable[str]] = None) -> int:
    """Entrypoint for the public ``monte-carlo`` command."""

    parser, args = _parse_public_args_with_parser(argv)
    if args.command is None:
        parser.print_help()
        print("\nChoose `simulate` for current ideas or `backtest` for historical validation.")
        return 1
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

    try:
        if args.command == "simulate":
            result = run_public_simulate(args)
            return 0 if not result["summaries"].empty else 1
        if args.command == "backtest":
            run_public_backtest(args)
            return 0
    except Exception as exc:
        LOGGER.error("%s", exc)
        return 2

    return 2
