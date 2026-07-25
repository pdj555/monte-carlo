"""Public CLI surface for the ``monte-carlo`` command."""

from __future__ import annotations

import argparse
import contextlib
import io
import logging
from pathlib import Path
from typing import Any, Iterable, Optional

import pandas as pd

import backtest as backtest_cli
from data import PriceDataError
from evaluation import (
    EvaluationReport,
    EvaluationRun,
    evaluate_scenario_set,
    format_evaluation_scorecard,
    load_evaluation_set,
    save_evaluation_report,
)
from cli_shared import (
    IntentionalDefaultsHelpFormatter,
    non_negative_int,
    package_version,
    positive_int,
    render_detailed_simulation_tables,
)
from simulate_cli import (
    build_simulation_args,
    maybe_show_simulation_plots,
    run as run_simulation,
)

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
        help=(
            "Price source. auto tries live first, then falls back to local CSVs; "
            "offline uses local CSVs only, online uses live data only."
        ),
    )
    simulate_parser.add_argument(
        "--data-path",
        type=str,
        default=None,
        help=(
            "Path to a CSV file, or a directory of <TICKER>.csv files with Date "
            "and Close columns. Used for offline runs and as the local fallback "
            "for auto."
        ),
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
        help=(
            "Price source. auto tries live first, then falls back to local CSVs; "
            "offline uses local CSVs only, online uses live data only."
        ),
    )
    backtest_parser.add_argument(
        "--data-path",
        type=str,
        default=None,
        help=(
            "Path to a CSV file, or a directory of <TICKER>.csv files with Date "
            "and Close columns. Used for offline runs and as the local fallback "
            "for auto."
        ),
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

    evaluate_parser = subparsers.add_parser(
        "evaluate",
        help="Test decision stability across a reproducible scenario set.",
        formatter_class=IntentionalDefaultsHelpFormatter,
    )
    evaluate_parser.add_argument(
        "set_file",
        help="Versioned JSON evaluation-set file.",
    )
    evaluate_parser.add_argument(
        "--output",
        default=None,
        help="Directory for scorecard.md, runs.csv, and report.json.",
    )

    return parser


def _parse_public_args_with_parser(
    argv: Optional[Iterable[str]] = None,
) -> tuple[argparse.ArgumentParser, argparse.Namespace]:
    parser = build_public_parser()
    args = parser.parse_args(list(argv) if argv is not None else None)
    if getattr(args, "command", None) in {"simulate", "backtest"} and not args.tickers:
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
    return build_simulation_args(
        tickers=_public_tickers_to_csv_arg(list(args.tickers)),
        days=int(args.days),
        scenarios=int(args.scenarios),
        no_plots=not should_make_plots,
        seed=args.seed,
        model=str(args.model),
        output=args.output,
        offline_path=args.data_path,
        offline_only=prefer_local,
        allow_local_fallback=allow_local_fallback,
        show=bool(args.show),
        details=bool(args.details),
    )


def _build_public_backtest_legacy_args(args: argparse.Namespace) -> argparse.Namespace:
    prefer_local, allow_local_fallback = _source_settings(str(args.source))
    return backtest_cli.build_backtest_args(
        tickers=_public_tickers_to_csv_arg(list(args.tickers)),
        lookback_days=int(args.lookback),
        holding_days=int(args.hold),
        rebalance_every=int(args.rebalance),
        top_k=int(args.top),
        model=str(args.model),
        scenarios=int(args.scenarios),
        seed=args.seed,
        offline_path=args.data_path,
        offline_only=prefer_local,
        allow_local_fallback=allow_local_fallback,
        output=args.output,
        details=bool(args.details),
    )


def _normalize_price_sources(
    price_sources: object,
) -> dict[str, dict[str, object]]:
    if not isinstance(price_sources, dict):
        return {}

    normalized: dict[str, dict[str, object]] = {}
    for ticker, payload in price_sources.items():
        if isinstance(ticker, str) and isinstance(payload, dict):
            normalized[ticker] = payload
    return normalized


def _price_source_label(source: dict[str, object]) -> str:
    kind = str(source.get("kind", ""))
    if kind == "live":
        return "live download"
    if kind == "cache":
        return "cached download"

    label = "bundled sample data" if bool(source.get("is_sample_data")) else "local CSV"
    if bool(source.get("used_fallback")):
        return f"{label} (fallback)"
    return label


def describe_price_sources(price_sources: object) -> str | None:
    normalized = _normalize_price_sources(price_sources)
    if not normalized:
        return None

    grouped: dict[str, list[str]] = {}
    for ticker in sorted(normalized):
        label = _price_source_label(normalized[ticker])
        grouped.setdefault(label, []).append(ticker)

    if len(normalized) == 1:
        label = _price_source_label(next(iter(normalized.values())))
        return f"Data source: {label}."

    if len(grouped) == 1:
        label, tickers = next(iter(grouped.items()))
        return f"Data source: {label} for {', '.join(tickers)}."

    parts = [f"{', '.join(tickers)} from {label}" for label, tickers in grouped.items()]
    return "Data source: " + "; ".join(parts) + "."


def _log_public_error(exc: Exception, *, command: str, args: argparse.Namespace) -> None:
    """Emit actionable, command-specific errors for terminal operators."""

    if isinstance(exc, PriceDataError):
        source = str(getattr(args, "source", "auto"))
        data_path = str(getattr(args, "data_path", "sample_data") or "sample_data")
        LOGGER.error(
            "Couldn't complete %s because price data could not be loaded: %s",
            command,
            exc,
        )

        if source == "online":
            LOGGER.error(
                "Try `--source auto` for local fallback, or `--source offline --data-path %s` "
                "for offline/CSV mode.",
                data_path,
            )
        else:
            LOGGER.error(
                "Tip: check `--data-path` points to a CSV file or a folder of <TICKER>.csv "
                "files, or use `--source offline --data-path sample_data` for bundled fixtures.",
            )
        return

    if isinstance(exc, ValueError):
        LOGGER.error("Invalid input for %s: %s", command, exc)
        LOGGER.error("Use `%s --help` to check valid flags and defaults.", command)
        return

    LOGGER.error("%s", exc)


def _log_no_simulation_output(
    *,
    args: argparse.Namespace,
    result: dict[str, Any],
) -> None:
    """Log a clear message when no ticker finished simulation."""

    report = result.get("report", {})
    errors = report.get("errors", []) if isinstance(report, dict) else []
    if errors:
        LOGGER.error(
            "No simulations were produced for %s.",
            ", ".join(str(ticker) for ticker in args.tickers),
        )
        for item in errors:
            if not isinstance(item, dict):
                continue
            ticker = str(item.get("ticker", "unknown"))
            message = str(item.get("error", "Unknown error"))
            LOGGER.error("  %s: %s", ticker, message)
        return

    LOGGER.error("No simulations were produced. Try loosening filters or check input history.")


def _price_source_details(price_sources: object) -> list[str]:
    normalized = _normalize_price_sources(price_sources)
    details: list[str] = []
    for ticker in sorted(normalized):
        source = normalized[ticker]
        path = source.get("path")
        label = _price_source_label(source)
        if isinstance(path, str) and path:
            details.append(f"{ticker}: {label} -> {path}")
        else:
            details.append(f"{ticker}: {label}")
    return details


def format_public_simulation_output(
    result: dict[str, Any],
    *,
    details: bool,
    output: str | None,
) -> str:
    report = result["report"]
    action_plan = report["action_plan"]
    lines = [
        f"Stance: {action_plan['stance']}",
        action_plan["headline"],
    ]
    source_summary = describe_price_sources(
        result.get("price_sources", report.get("price_sources"))
    )
    if source_summary:
        lines.append(source_summary)

    if action_plan["primary_pick"] is not None:
        pick = action_plan["primary_pick"]
        lines.append(
            f"Top idea: {pick['ticker']} at {pick['weight']:.1%} weight "
            f"(expected return {pick['expected_return']:.1%})."
        )
    if action_plan["avoid_list"]:
        lines.append(f"Avoid: {', '.join(action_plan['avoid_list'])}")
    if action_plan.get("cash_weight", 0.0) > 0:
        lines.append(f"Cash buffer: {action_plan['cash_weight']:.1%}")

    for item in report["errors"]:
        lines.append(f"Skipped {item['ticker']}: {item['error']}")

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
        detail_buffer = io.StringIO()
        with contextlib.redirect_stdout(detail_buffer):
            render_detailed_simulation_tables(
                summary_df,
                portfolio_summary,
                rankings,
                allocations,
            )
        detail_text = detail_buffer.getvalue().strip()
        if detail_text:
            lines.extend(["", detail_text])
        source_details = _price_source_details(
            result.get("price_sources", report.get("price_sources"))
        )
        if source_details:
            lines.extend(["", "Source details", *source_details])

    if output:
        lines.append(f"Saved outputs to {Path(output).expanduser()}")

    return "\n".join(lines).strip()


def _render_public_simulation_output(
    result: dict[str, Any],
    *,
    details: bool,
    output: str | None,
) -> None:
    print(format_public_simulation_output(result, details=details, output=output))


def format_public_backtest_output(
    result: dict[str, object],
    *,
    details: bool,
    output: str | None,
) -> str:
    summary = result["summary"]
    if not isinstance(summary, pd.Series):
        raise ValueError("summary output must be a pandas Series")

    lines = [
        "Strategy return: "
        f"{float(summary['strategy_total_return']):.1%} "
        f"({float(summary['strategy_annualized_return']):.1%} annualized)",
        f"Max drawdown: {float(summary['strategy_max_drawdown']):.1%}",
        f"vs equal weight: {float(summary['excess_return_vs_equal_weight']):.1%}",
        f"vs cash: {float(summary['excess_return_vs_cash']):.1%}",
    ]
    source_summary = describe_price_sources(result.get("price_sources"))
    if source_summary:
        lines.append(source_summary)

    if details:
        lines.append("")
        lines.append("Backtest summary")
        lines.append(
            summary.to_frame(name="value").to_string(
                float_format=lambda value: f"{value:0.4f}"
            )
        )
        source_details = _price_source_details(result.get("price_sources"))
        if source_details:
            lines.extend(["", "Source details", *source_details])

    if output:
        lines.append(f"Saved outputs to {Path(output).expanduser()}")

    return "\n".join(lines).strip()


def _render_public_backtest_output(
    result: dict[str, object],
    *,
    details: bool,
    output: str | None,
) -> None:
    print(format_public_backtest_output(result, details=details, output=output))


def execute_public_simulate(args: argparse.Namespace) -> dict[str, Any]:
    """Execute the simulate command without rendering text output."""

    legacy_args = _build_public_simulate_legacy_args(args)
    return run_simulation(legacy_args, render=False, display_plots=False)


def execute_public_backtest(args: argparse.Namespace) -> dict[str, object]:
    """Execute the backtest command without rendering text output."""

    legacy_args = _build_public_backtest_legacy_args(args)
    return backtest_cli.run(legacy_args, render=False)


def _execute_evaluation_run(run: EvaluationRun) -> dict[str, Any]:
    """Adapt one evaluation matrix cell to the public simulation interface."""

    return execute_public_simulate(
        argparse.Namespace(
            tickers=list(run.tickers),
            days=run.days,
            scenarios=run.scenarios,
            model=run.model,
            seed=run.seed,
            source=run.source_mode,
            data_path=str(run.data_path) if run.data_path is not None else None,
            output=None,
            show=False,
            details=False,
        )
    )


def execute_public_evaluate(args: argparse.Namespace) -> EvaluationReport:
    """Execute an evaluation set without rendering text output."""

    evaluation_set = load_evaluation_set(args.set_file)
    return evaluate_scenario_set(evaluation_set, _execute_evaluation_run)


def run_public_simulate(args: argparse.Namespace) -> dict[str, Any]:
    """Execute the simplified simulate command."""

    legacy_args = _build_public_simulate_legacy_args(args)
    result = execute_public_simulate(args)
    _render_public_simulation_output(
        result,
        details=bool(args.details),
        output=args.output,
    )
    maybe_show_simulation_plots(legacy_args, result)
    return result


def run_public_backtest(args: argparse.Namespace) -> dict[str, object]:
    """Execute the simplified backtest command."""

    result = execute_public_backtest(args)
    _render_public_backtest_output(
        result,
        details=bool(args.details),
        output=args.output,
    )
    return result


def run_public_evaluate(args: argparse.Namespace) -> EvaluationReport:
    """Execute and render a reproducible decision-stability evaluation."""

    report = execute_public_evaluate(args)
    print(format_evaluation_scorecard(report))
    if args.output:
        save_evaluation_report(report, args.output)
    return report


def main(argv: Optional[Iterable[str]] = None) -> int:
    """Entrypoint for the public ``monte-carlo`` command."""

    parser, args = _parse_public_args_with_parser(argv)
    if args.command is None:
        parser.print_help()
        print(
            "\nChoose `simulate` for current ideas, `backtest` for historical "
            "validation, or `evaluate` for decision stability."
        )
        return 1
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

    try:
        if args.command == "simulate":
            result = run_public_simulate(args)
            if result["summaries"].empty:
                _log_no_simulation_output(args=args, result=result)
                return 1
            return 0
        if args.command == "backtest":
            run_public_backtest(args)
            return 0
        if args.command == "evaluate":
            report = run_public_evaluate(args)
            return 1 if report.scorecard.failed_runs > 0 else 0
    except Exception as exc:
        _log_public_error(exc, command=str(args.command), args=args)
        return 2

    return 2


if __name__ == "__main__":
    raise SystemExit(main())
