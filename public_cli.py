"""Public CLI surface for the ``monte-carlo`` command."""

from __future__ import annotations

import argparse
import contextlib
import io
import logging
import os
import sys
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
    MAX_DAYS,
    MAX_PATH_CELLS,
    MAX_SCENARIOS,
    IntentionalDefaultsHelpFormatter,
    InputError,
    bounded_positive_int,
    seed_int,
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

EXIT_OK = 0
EXIT_FAILED = 1
EXIT_USAGE = 2
EXIT_INTERNAL = 3
EXIT_OUTPUT = 4

JSON_FLAG_HELP = (
    "Print one JSON document on stdout instead of text "
    "(schema: schemas/cli-result.v1.schema.json)."
)


class PublicUsageError(Exception):
    """A command line the parser rejected."""

    def __init__(self, parser: argparse.ArgumentParser, message: str) -> None:
        super().__init__(message)
        self.parser = parser
        self.message = message


class OutputWriteError(BaseException):
    """Stdout could not be written (closed pipe, full device, closed descriptor).

    Not an ``Exception`` so no error handler in the engine mistakes it for a
    failed computation; only ``main`` catches it.
    """


class _GuardedStdout:
    """Stdout that reports a failed write as :class:`OutputWriteError`."""

    def __init__(self, target: Any) -> None:
        self._target = target

    def write(self, text: str) -> int:
        if self._target is None:
            raise OutputWriteError("stdout is closed")
        try:
            return self._target.write(text)
        except (OSError, ValueError) as exc:
            raise OutputWriteError(str(exc)) from exc

    def flush(self) -> None:
        if self._target is None:
            return
        try:
            self._target.flush()
        except (OSError, ValueError) as exc:
            raise OutputWriteError(str(exc)) from exc

    def __getattr__(self, name: str) -> Any:
        return getattr(self._target, name)


def _discard_pending_stdout() -> None:
    """Point the stdout descriptor at the null device.

    After a failed write the interpreter would try the same write again while
    shutting down and print a traceback; this makes that last flush succeed.
    """

    try:
        descriptor = sys.stdout.fileno()
        null = os.open(os.devnull, os.O_WRONLY)
        os.dup2(null, descriptor)
        os.close(null)
    except Exception:
        pass


class _PublicArgumentParser(argparse.ArgumentParser):
    """Raise on bad usage so ``main`` can answer in the caller's format."""

    def error(self, message: str) -> None:  # type: ignore[override]
        raise PublicUsageError(self, message)


def _exit_with_usage(exc: PublicUsageError) -> None:
    """Report a usage error the way argparse does: usage on stderr, exit 2."""

    exc.parser.print_usage(sys.stderr)
    exc.parser.exit(EXIT_USAGE, f"{exc.parser.prog}: error: {exc.message}\n")


def build_public_parser() -> argparse.ArgumentParser:
    """Create the simplified public CLI parser."""

    parser = _PublicArgumentParser(
        prog="monte-carlo",
        description="Monte Carlo tools for current ideas and historical validation.",
        formatter_class=IntentionalDefaultsHelpFormatter,
        allow_abbrev=False,
    )
    parser.add_argument(
        "--json",
        dest="json_before_command",
        action="store_true",
        help=JSON_FLAG_HELP,
    )
    parser.add_argument(
        "--version",
        action="version",
        version=f"%(prog)s {package_version()}",
    )

    subparsers = parser.add_subparsers(dest="command")

    simulate_parser = subparsers.add_parser(
        "simulate",
        allow_abbrev=False,
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
        type=bounded_positive_int(MAX_DAYS),
        default=252,
        help="Trading days to simulate into the future.",
    )
    simulate_parser.add_argument(
        "--scenarios",
        type=bounded_positive_int(MAX_SCENARIOS),
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
        type=seed_int,
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

    simulate_parser.add_argument("--json", action="store_true", help=JSON_FLAG_HELP)

    backtest_parser = subparsers.add_parser(
        "backtest",
        allow_abbrev=False,
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
        default=None,
        help="Trading days between rebalances. Must equal --hold; defaults to it.",
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
        type=bounded_positive_int(MAX_SCENARIOS),
        default=1000,
        help="Number of Monte Carlo paths to run at each rebalance.",
    )
    backtest_parser.add_argument(
        "--seed",
        type=seed_int,
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

    backtest_parser.add_argument("--json", action="store_true", help=JSON_FLAG_HELP)

    evaluate_parser = subparsers.add_parser(
        "evaluate",
        allow_abbrev=False,
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

    evaluate_parser.add_argument("--json", action="store_true", help=JSON_FLAG_HELP)

    subparsers.add_parser(
        "schema",
        allow_abbrev=False,
        help="Print the JSON Schema of the --json document.",
    )

    return parser


def _parse_public_args_with_parser(
    argv: Optional[Iterable[str]] = None,
) -> tuple[argparse.ArgumentParser, argparse.Namespace]:
    parser = build_public_parser()
    args = parser.parse_args(list(argv) if argv is not None else None)
    if getattr(args, "command", None) in {"simulate", "backtest"} and not args.tickers:
        args.tickers = ["AAPL"]
    # --json may be written before or after the command; both mean the same.
    args.json = bool(getattr(args, "json", False) or args.json_before_command)
    if getattr(args, "command", None) == "simulate":
        cells = int(args.days) * int(args.scenarios)
        if cells > MAX_PATH_CELLS:
            raise PublicUsageError(
                parser,
                f"--days x --scenarios is {cells}; the most one run may hold is "
                f"{MAX_PATH_CELLS}. Lower one of them.",
            )
    return parser, args


def parse_public_args(argv: Optional[Iterable[str]] = None) -> argparse.Namespace:
    """Return parsed arguments for the public CLI."""

    try:
        _, args = _parse_public_args_with_parser(argv)
    except PublicUsageError as exc:
        _exit_with_usage(exc)
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
        rebalance_every=int(args.hold if args.rebalance is None else args.rebalance),
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
    """Execute an evaluation set without rendering text output.

    Tickers whose prices could not be loaded are listed on
    ``args.missing_price_data``: the manifest named data that does not exist,
    which is rejected input here as it is for simulate and backtest.
    """

    evaluation_set = load_evaluation_set(args.set_file)
    missing: list[str] = []

    def runner(run: EvaluationRun) -> dict[str, Any]:
        result = _execute_evaluation_run(run)
        report = result.get("report", {})
        for item in report.get("errors", []) if isinstance(report, dict) else []:
            if isinstance(item, dict) and item.get("kind") == "price_data":
                missing.append(f"{run.run_id}: {item.get('error')}")
        return result

    report = evaluate_scenario_set(evaluation_set, runner)
    args.missing_price_data = missing
    return report


def _prepare_output_directory(output: str | None) -> None:
    """Create ``--output`` up front so an unusable path is reported as input."""

    if not output:
        return
    try:
        Path(output).expanduser().mkdir(parents=True, exist_ok=True)
    except OSError as exc:
        raise InputError(f"--output {output!r} cannot be used as a directory: {exc}") from exc


def _evaluate_exit(args: argparse.Namespace, report: EvaluationReport) -> tuple[int, str, str]:
    """Exit code, error kind and message for a finished evaluation."""

    missing = list(getattr(args, "missing_price_data", []))
    if missing:
        return EXIT_USAGE, "price_data", "; ".join(missing)
    failed_runs = int(report.scorecard.failed_runs)
    if failed_runs > 0:
        message = f"{failed_runs} evaluation run(s) failed; see result.outcomes."
        return EXIT_FAILED, "failed_runs", message
    return EXIT_OK, "", ""


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


def _emit_json(document: dict[str, Any]) -> int:
    import cli_json

    print(cli_json.render_document(document))
    return int(document["exit_code"])


def _classify_exception(exc: Exception) -> tuple[int, str]:
    """Exit code and error kind for an exception a command raised.

    Price data that cannot be loaded, and input the validation code rejected
    (``InputError``), is the caller's to fix (2). Prices that loaded but are too
    short to produce anything is a run without a result (1). Everything else,
    including a ``ValueError`` or ``OSError`` from inside the engine, is a
    fault in this program (3).
    """

    if isinstance(exc, PriceDataError):
        return EXIT_USAGE, "price_data"
    if isinstance(exc, backtest_cli.InsufficientHistoryError):
        return EXIT_FAILED, "no_results"
    if isinstance(exc, InputError):
        return EXIT_USAGE, "invalid_input"
    return EXIT_INTERNAL, "internal"


def _empty_simulation_outcome(result: dict[str, Any]) -> tuple[int, str, str]:
    """Exit code, error kind and message for a simulate run with no summaries.

    When every requested ticker failed because its prices could not be loaded,
    the input named data that does not exist: the same condition, and the same
    exit code, as a backtest that cannot load its prices.
    """

    report = result.get("report", {})
    errors = report.get("errors", []) if isinstance(report, dict) else []
    kinds = {str(item.get("kind")) for item in errors if isinstance(item, dict)}
    if errors and kinds == {"price_data"}:
        message = "; ".join(str(item.get("error", "")) for item in errors)
        return EXIT_USAGE, "price_data", message
    return EXIT_FAILED, "no_results", "No ticker produced a simulation; see result.errors."


def _run_json_command(args: argparse.Namespace) -> int:
    """Run one command and print its result as a single JSON document."""

    # Imported here, not at module top: the web bridge imports this module and
    # never asks for JSON, so its import graph stays exactly as it was.
    import cli_json

    command = str(args.command)
    config: dict[str, Any] | None = None
    try:
        _prepare_output_directory(args.output)
        if command == "simulate":
            config = cli_json.simulate_config(args, seed_source=cli_json.resolve_seed(args))
            legacy_args = _build_public_simulate_legacy_args(args)
            result = execute_public_simulate(args)
            maybe_show_simulation_plots(legacy_args, result)
            if not result["summaries"].empty:
                return _emit_json(
                    cli_json.build_document(
                        command=command,
                        exit_code=EXIT_OK,
                        config=config,
                        result=cli_json.simulate_result(result),
                    )
                )
            _log_no_simulation_output(args=args, result=result)
            exit_code, kind, message = _empty_simulation_outcome(result)
            return _emit_json(
                cli_json.build_document(
                    command=command,
                    exit_code=exit_code,
                    config=config,
                    result=cli_json.simulate_result(result) if exit_code == EXIT_FAILED else None,
                    error_kind=kind,
                    error_message=message,
                )
            )
        if command == "backtest":
            config = cli_json.backtest_config(args, seed_source=cli_json.resolve_seed(args))
            return _emit_json(
                cli_json.build_document(
                    command=command,
                    exit_code=EXIT_OK,
                    config=config,
                    result=cli_json.backtest_result(execute_public_backtest(args)),
                )
            )
        report = execute_public_evaluate(args)
        if args.output:
            save_evaluation_report(report, args.output)
        report_dict = report.to_dict()
        exit_code, kind, message = _evaluate_exit(args, report)
        return _emit_json(
            cli_json.build_document(
                command=command,
                exit_code=exit_code,
                config=cli_json.evaluate_config(args, report_dict),
                result=None if exit_code == EXIT_USAGE else cli_json.evaluate_result(report_dict),
                error_kind=kind or None,
                error_message=message or None,
            )
        )
    except Exception as exc:
        _log_public_error(exc, command=command, args=args)
        exit_code, kind = _classify_exception(exc)
        return _emit_json(
            cli_json.build_document(
                command=command,
                exit_code=exit_code,
                config=config,
                error_kind=kind,
                error_message=str(exc) or type(exc).__name__,
            )
        )


def _requested_command(argv: list[str]) -> str | None:
    for token in argv:
        if token in {"simulate", "backtest", "evaluate"}:
            return token
    return None


def _json_requested(argv: list[str]) -> bool:
    """Whether a command line the parser rejected asked for JSON.

    Decided by argparse, not by looking for the text: ``--json`` after ``--``
    is a ticker, and an abbreviation is not the flag.
    """

    probe = argparse.ArgumentParser(add_help=False, allow_abbrev=False)
    probe.add_argument("--json", action="store_true")
    try:
        known, _ = probe.parse_known_args(argv)
    except SystemExit:
        return False
    return bool(known.json)


def _run(raw_argv: list[str]) -> int:
    try:
        parser, args = _parse_public_args_with_parser(raw_argv)
    except PublicUsageError as exc:
        if not _json_requested(raw_argv):
            _exit_with_usage(exc)
        import cli_json

        exc.parser.print_usage(sys.stderr)
        return _emit_json(
            cli_json.build_document(
                command=_requested_command(raw_argv),
                exit_code=EXIT_USAGE,
                error_kind="usage",
                error_message=exc.message,
            )
        )
    if args.command is None and args.json:
        import cli_json

        parser.print_usage(sys.stderr)
        return _emit_json(
            cli_json.build_document(
                command=None,
                exit_code=EXIT_USAGE,
                error_kind="usage",
                error_message="a command is required: simulate, backtest, evaluate, or schema",
            )
        )
    if args.command is None:
        parser.print_help()
        print(
            "\nChoose `simulate` for current ideas, `backtest` for historical "
            "validation, or `evaluate` for decision stability."
        )
        return 1
    if args.command == "schema":
        import cli_json

        print(cli_json.render_schema())
        return EXIT_OK
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

    if args.json:
        return _run_json_command(args)

    try:
        _prepare_output_directory(args.output)
        if args.command == "simulate":
            result = run_public_simulate(args)
            if result["summaries"].empty:
                _log_no_simulation_output(args=args, result=result)
                return _empty_simulation_outcome(result)[0]
            return EXIT_OK
        if args.command == "backtest":
            run_public_backtest(args)
            return EXIT_OK
        report = run_public_evaluate(args)
        exit_code, kind, message = _evaluate_exit(args, report)
        if kind == "price_data":
            LOGGER.error("Price data could not be loaded: %s", message)
        return exit_code
    except Exception as exc:
        _log_public_error(exc, command=str(args.command), args=args)
        return _classify_exception(exc)[0]


def main(argv: Optional[Iterable[str]] = None) -> int:
    """Entrypoint for the public ``monte-carlo`` command.

    Exit codes: 0 the command succeeded; 1 it ran but produced no usable result
    (prices too short, or an evaluation run failed); 2 the command line or its
    inputs were rejected, including price data that could not be loaded; 3 an
    internal error; 4 the output could not be written to stdout.
    """

    raw_argv = list(argv) if argv is not None else sys.argv[1:]
    guard = _GuardedStdout(sys.stdout)
    try:
        with contextlib.redirect_stdout(guard):  # type: ignore[type-var]
            try:
                return _run(raw_argv)
            finally:
                guard.flush()
    except OutputWriteError as exc:
        _discard_pending_stdout()
        try:
            print(f"monte-carlo: could not write to stdout: {exc}", file=sys.stderr)
        except Exception:
            pass
        return EXIT_OUTPUT


if __name__ == "__main__":
    raise SystemExit(main())
