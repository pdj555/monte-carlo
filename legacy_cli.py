"""Full deprecated simulation CLI preserved behind the thin ``cli.py`` facade."""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Any, Iterable, Optional

import pandas as pd

from cli_shared import (
    non_negative_int,
    package_version,
    positive_float,
    positive_int,
    ranking_display_columns,
)

LOGGER = logging.getLogger(__name__)


def _parser_error_if(
    parser: argparse.ArgumentParser,
    condition: bool,
    message: str,
) -> None:
    if condition:
        parser.error(message)


def _print_ticker_summary(*, ticker: str, summary: pd.Series, minimal: bool) -> None:
    if minimal:
        expected_return = float(summary.get("expected_return", 0.0))
        prob_up = float(summary.get("prob_above_current", 0.0))
        var95 = float(summary.get("value_at_risk_95_pct", 0.0))
        print(f"{ticker}: er={expected_return:.1%} up={prob_up:.1%} var95={var95:.1%}")
        return

    print(f"\nSummary for {ticker}")
    print(summary.to_frame(name="value").to_string(float_format=lambda v: f"{v:0.2f}"))


def _print_portfolio_summary(summary: pd.Series, *, minimal: bool) -> None:
    if minimal:
        expected_return = float(summary.get("expected_return", 0.0))
        prob_up = float(summary.get("prob_above_current", 0.0))
        print(f"PORTFOLIO: er={expected_return:.1%} up={prob_up:.1%}")
        return

    print("\nSummary for EQUAL_WEIGHT_PORTFOLIO")
    print(summary.to_frame(name="value").to_string(float_format=lambda v: f"{v:0.2f}"))


def build_parser() -> argparse.ArgumentParser:
    """Create the CLI argument parser."""

    parser = argparse.ArgumentParser(
        description="Run Monte Carlo simulations for one or more tickers.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--version",
        action="version",
        version=f"%(prog)s {package_version()}",
    )
    parser.add_argument(
        "--journal-file",
        type=str,
        help=(
            "Optional append-only JSONL decision journal. "
            "Each run writes a tamper-evident chained entry for audit history."
        ),
    )
    parser.add_argument(
        "--policy-file",
        type=str,
        help=(
            "Optional JSON policy contract with default guardrails/constraints. "
            "CLI flags override policy values when both are provided."
        ),
    )
    parser.add_argument(
        "--tickers",
        "--ticker",
        dest="tickers",
        default="AAPL",
        help="Comma-separated list of ticker symbols to simulate.",
    )
    parser.add_argument(
        "--days",
        type=positive_int,
        default=252,
        help="Number of future trading days to simulate.",
    )
    parser.add_argument(
        "--scenarios",
        type=positive_int,
        default=1000,
        help="Number of Monte Carlo scenarios to run per ticker.",
    )
    parser.add_argument(
        "--max-paths",
        type=non_negative_int,
        default=100,
        help="Maximum number of simulated paths to plot per ticker (0 = all).",
    )
    parser.add_argument(
        "--no-plots",
        action="store_true",
        help="Skip generating distribution/path plots (faster for large runs).",
    )
    parser.add_argument(
        "--dt",
        type=positive_float,
        default=1.0,
        help="Time increment for each simulation step (in trading days).",
    )
    parser.add_argument(
        "--block-size",
        type=positive_int,
        default=1,
        help=(
            "Bootstrap block size for historical model (1 = IID resampling). "
            "Use >1 to preserve short-term market regimes."
        ),
    )
    parser.add_argument(
        "--shock-probability",
        type=float,
        default=0.0,
        help=(
            "Probability (0-1) of a shock event at each simulated step. "
            "Use 0 for normal mode."
        ),
    )
    parser.add_argument(
        "--shock-return",
        type=float,
        default=-0.15,
        help="Simple return applied on shock days (e.g. -0.15 = -15%%).",
    )
    parser.add_argument(
        "--seed",
        type=non_negative_int,
        default=None,
        help="Random seed for reproducible results.",
    )
    parser.add_argument(
        "--model",
        choices=("historical", "gbm", "prediction_market"),
        default="historical",
        help=(
            "Simulation model: empirical historical bootstrap, geometric Brownian "
            "motion, or prediction-market probabilities."
        ),
    )
    parser.add_argument(
        "--fundamental-probability",
        type=float,
        default=None,
        help=(
            "Required for --model prediction_market. Fundamental truth prior "
            "for event probability (0-1)."
        ),
    )
    parser.add_argument(
        "--market-price",
        type=float,
        default=None,
        help=(
            "Optional current market-implied probability used as starting price "
            "for --model prediction_market (0-1)."
        ),
    )
    parser.add_argument(
        "--fundamental-certainty",
        type=positive_float,
        default=100.0,
        help=(
            "Pseudo-count confidence in --fundamental-probability for "
            "--model prediction_market (higher = tighter truth prior)."
        ),
    )
    parser.add_argument(
        "--prob-mean-reversion",
        type=float,
        default=0.2,
        help=(
            "Daily pull toward latent truth in --model prediction_market "
            "(0 = random walk around current level)."
        ),
    )
    parser.add_argument(
        "--prob-daily-volatility",
        type=float,
        default=0.03,
        help="Noise scale for --model prediction_market probability paths.",
    )
    parser.add_argument(
        "--start",
        type=str,
        help="Optional start date (YYYY-MM-DD) for historical price retrieval.",
    )
    parser.add_argument(
        "--end",
        type=str,
        help="Optional end date (YYYY-MM-DD) for historical price retrieval.",
    )
    parser.add_argument(
        "--output",
        type=str,
        help="Directory where plots are saved. Created if it does not exist.",
    )
    parser.add_argument(
        "--cache-dir",
        type=str,
        help="Directory used to cache downloaded price CSVs (keyed by ticker).",
    )
    parser.add_argument(
        "--refresh-cache",
        action="store_true",
        help="Ignore cached prices and attempt a fresh download.",
    )
    parser.add_argument(
        "--save-simulations",
        action="store_true",
        help="Save combined simulation paths to output directory as a gzip CSV.",
    )
    parser.add_argument(
        "--offline-path",
        type=str,
        help="Directory or CSV file to use for offline price data.",
    )
    parser.add_argument(
        "--offline-only",
        action="store_true",
        help="Use offline CSV data without attempting any network requests.",
    )
    parser.add_argument(
        "--no-show",
        dest="show",
        action="store_false",
        help="Skip displaying plots (useful for batch jobs and tests).",
    )
    parser.add_argument(
        "--ai-summary",
        action="store_true",
        help="Generate a natural-language summary using the OpenAI API (requires OPENAI_API_KEY).",
    )
    parser.add_argument(
        "--ai-model",
        type=str,
        default=None,
        help=(
            "OpenAI model name used when --ai-summary is enabled "
            "(defaults to OPENAI_MODEL or gpt-5.2)."
        ),
    )
    parser.add_argument(
        "--annual-cash-yield",
        type=float,
        default=0.04,
        help=(
            "Annualized cash benchmark yield used to compute excess-return metrics "
            "(e.g. 0.04 = 4%% per year)."
        ),
    )
    parser.add_argument(
        "--min-expected-return",
        type=float,
        default=0.0,
        help="Minimum expected return required to keep a ticker investable (e.g. 0.05 = 5%%).",
    )
    parser.add_argument(
        "--min-prob-up",
        type=float,
        default=0.5,
        help="Minimum probability that final price exceeds current price (0-1).",
    )
    parser.add_argument(
        "--portfolio-risk-budget-pct",
        type=float,
        default=0.02,
        help=(
            "Hard cap for blended portfolio 95%% VaR as a fraction of total capital; "
            "allocations are auto-scaled down to respect this budget."
        ),
    )
    parser.add_argument(
        "--max-var-95-pct",
        type=float,
        default=0.25,
        help="Maximum allowed 95%% VaR as a percent of current price (e.g. 0.20 = 20%%).",
    )
    parser.add_argument(
        "--max-drawdown-q95-pct",
        type=float,
        default=None,
        help=(
            "Optional cap on 95th percentile max drawdown (e.g. 0.30 = 30%%). "
            "Tickers above this are forced to AVOID."
        ),
    )
    parser.add_argument(
        "--target-return-pct",
        type=float,
        default=None,
        help=(
            "Optional return target as a fraction of current price "
            "(e.g. 0.1 = +10%%); enables prob_hit_target metrics."
        ),
    )
    parser.add_argument(
        "--max-loss-pct",
        type=float,
        default=None,
        help=(
            "Optional maximum acceptable loss as a fraction of current price "
            "(e.g. 0.12 = -12%%); enables prob_breach_max_loss metrics."
        ),
    )
    parser.add_argument(
        "--min-prob-hit-target",
        type=float,
        default=None,
        help="Optional guardrail: minimum probability of reaching --target-return-pct (0-1).",
    )
    parser.add_argument(
        "--max-prob-breach-loss",
        type=float,
        default=None,
        help="Optional guardrail: maximum allowed probability of breaching --max-loss-pct (0-1).",
    )
    parser.add_argument(
        "--capital",
        type=positive_float,
        default=None,
        help="Optional portfolio capital used to produce executable dollar/share sizing.",
    )
    parser.add_argument(
        "--allow-fractional-shares",
        action="store_true",
        help="Allow fractional shares when --capital is set.",
    )
    parser.add_argument(
        "--minimal",
        action="store_true",
        help=(
            "Print only compact, decision-first terminal output "
            "(suppresses verbose tables)."
        ),
    )
    parser.add_argument(
        "--strict",
        action="store_true",
        help="Return a non-zero exit code if any ticker fails.",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable verbose logging.",
    )
    parser.set_defaults(show=True)
    return parser


def parse_args(argv: Optional[Iterable[str]] = None) -> argparse.Namespace:
    """Return parsed CLI arguments."""

    parser = build_parser()
    raw_argv = list(argv) if argv is not None else None
    args = parser.parse_args(raw_argv)
    if args.policy_file:
        args = _apply_policy_file(args, parser=parser, argv=raw_argv)
    _parser_error_if(
        parser,
        not 0.0 <= float(args.shock_probability) <= 1.0,
        "--shock-probability must be between 0 and 1",
    )
    _parser_error_if(
        parser,
        float(args.shock_return) <= -1.0,
        "--shock-return must be greater than -1.0",
    )
    _parser_error_if(
        parser,
        args.max_loss_pct is not None and float(args.max_loss_pct) < 0,
        "--max-loss-pct must be non-negative",
    )
    _parser_error_if(
        parser,
        args.min_prob_hit_target is not None
        and not 0.0 <= float(args.min_prob_hit_target) <= 1.0,
        "--min-prob-hit-target must be between 0 and 1",
    )
    _parser_error_if(
        parser,
        args.max_prob_breach_loss is not None
        and not 0.0 <= float(args.max_prob_breach_loss) <= 1.0,
        "--max-prob-breach-loss must be between 0 and 1",
    )
    _parser_error_if(
        parser,
        args.min_prob_hit_target is not None and args.target_return_pct is None,
        "--min-prob-hit-target requires --target-return-pct",
    )
    _parser_error_if(
        parser,
        args.max_prob_breach_loss is not None and args.max_loss_pct is None,
        "--max-prob-breach-loss requires --max-loss-pct",
    )
    _parser_error_if(
        parser,
        float(args.portfolio_risk_budget_pct) < 0,
        "--portfolio-risk-budget-pct must be non-negative",
    )
    _parser_error_if(
        parser,
        float(args.annual_cash_yield) < 0,
        "--annual-cash-yield must be non-negative",
    )
    _parser_error_if(
        parser,
        float(args.prob_mean_reversion) < 0,
        "--prob-mean-reversion must be non-negative",
    )
    _parser_error_if(
        parser,
        float(args.prob_daily_volatility) < 0,
        "--prob-daily-volatility must be non-negative",
    )
    if args.model == "prediction_market":
        _parser_error_if(
            parser,
            args.fundamental_probability is None,
            "--fundamental-probability is required for --model prediction_market",
        )
        _parser_error_if(
            parser,
            not 0.0 < float(args.fundamental_probability) < 1.0,
            "--fundamental-probability must be strictly between 0 and 1",
        )
        _parser_error_if(
            parser,
            args.market_price is not None and not 0.0 < float(args.market_price) < 1.0,
            "--market-price must be strictly between 0 and 1",
        )
    return args


def _apply_policy_file(
    args: argparse.Namespace,
    *,
    parser: argparse.ArgumentParser,
    argv: Optional[list[str]],
) -> argparse.Namespace:
    """Apply JSON policy defaults to CLI args while respecting explicit flags."""

    policy_path = Path(args.policy_file).expanduser()
    try:
        payload = json.loads(policy_path.read_text(encoding="utf-8"))
    except FileNotFoundError:
        parser.error(f"--policy-file not found: {policy_path}")
    except json.JSONDecodeError as exc:
        parser.error(f"--policy-file is not valid JSON: {exc}")
    except OSError as exc:
        parser.error(f"Unable to read --policy-file: {exc}")

    if not isinstance(payload, dict):
        parser.error("--policy-file must contain a top-level JSON object")

    normalized: dict[str, object] = {}
    for key, value in payload.items():
        normalized_key = str(key).replace("-", "_")
        if not hasattr(args, normalized_key):
            parser.error(f"--policy-file contains unknown key: {key}")
        if normalized_key == "policy_file":
            continue
        normalized[normalized_key] = value

    provided_flags = set(argv or [])
    for key, value in normalized.items():
        option = f"--{key.replace('_', '-')}"
        if option in provided_flags:
            continue
        setattr(args, key, value)

    args.policy_file = str(policy_path)
    args.policy = normalized
    return args


def _render_legacy_simulation_output(result: dict[str, Any], args: argparse.Namespace) -> None:
    summary_df = result["summaries"]
    portfolio_summary = result["portfolio_summary"]
    report = result["report"]
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

    for ticker, row in summary_df.iterrows():
        _print_ticker_summary(
            ticker=str(ticker),
            summary=row,
            minimal=bool(args.minimal),
        )
        ai_text = report["results"].get(str(ticker), {}).get("ai_summary")
        if ai_text:
            print("\nAI summary")
            print(ai_text)

    if portfolio_summary is not None:
        _print_portfolio_summary(portfolio_summary, minimal=bool(args.minimal))

    if not rankings.empty and not args.minimal:
        print("\nTicker ranking")
        print(
            rankings.loc[:, ranking_display_columns(rankings)].to_string(
                float_format=lambda value: f"{value:0.3f}"
            )
        )

    if not allocations.empty and not args.minimal:
        print("\nSuggested allocation")
        print(
            allocations.loc[:, ["weight", "score", "value_at_risk_95_pct"]].to_string(
                float_format=lambda value: f"{value:0.3f}"
            )
        )

    action_plan = report["action_plan"]
    if args.minimal:
        print("\nPLAN")
    else:
        print("\nAction plan")
    print(f"- Stance: {action_plan['stance']}")
    print(f"- Headline: {action_plan['headline']}")
    if action_plan["primary_pick"] is not None:
        pick = action_plan["primary_pick"]
        print(
            "- Primary pick: "
            f"{pick['ticker']} (weight {pick['weight']:.1%}, score {pick['score']:.1f}, "
            f"expected return {pick['expected_return']:.1%})"
        )
    if action_plan["avoid_list"]:
        print(f"- Avoid: {', '.join(action_plan['avoid_list'])}")
    if action_plan.get("cash_weight", 0.0) > 0:
        print(f"- Cash buffer: {action_plan['cash_weight']:.1%}")


def run(
    args: argparse.Namespace,
    *,
    render: bool = True,
    display_plots: bool = True,
) -> dict[str, Any]:
    """Execute the legacy simulation workflow and return simulation artefacts."""

    from simulate_cli import run as run_simulation

    renderer = _render_legacy_simulation_output if render else None
    return run_simulation(
        args,
        render=render,
        display_plots=display_plots,
        renderer=renderer,
    )


def legacy_main(argv: Optional[Iterable[str]] = None) -> int:
    """Entrypoint used by the deprecated ``python cli.py`` wrapper."""

    print(
        "Deprecated: use `monte-carlo simulate ...` for the simplified CLI.",
        file=sys.stderr,
    )

    args = parse_args(argv)

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(levelname)s %(message)s",
    )

    try:
        result = run(args)
    except Exception as exc:
        if args.verbose:
            LOGGER.exception("Unhandled error")
        else:
            LOGGER.error("%s", exc)
        return 2
    if result["summaries"].empty:
        return 1
    if args.strict and result["report"]["errors"]:
        return 1
    return 0


if __name__ == "__main__":  # pragma: no cover - CLI invocation
    raise SystemExit(legacy_main())
