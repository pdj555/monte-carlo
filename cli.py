"""Command-line interface for running Monte Carlo stock simulations."""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import sys
import zlib
from datetime import datetime, timezone
from importlib import metadata
from pathlib import Path
from typing import Any, Iterable, Optional

import matplotlib.pyplot as plt
import pandas as pd

from ai import OpenAIConfigurationError, OpenAIRequestError, generate_ai_summary
from analysis import (
    summarize_equal_weight_portfolio,
    summarize_final_prices,
)
from data import PriceDataError, fetch_prices
from decision import (
    apply_risk_guards,
    build_action_plan,
    build_execution_plan,
    enforce_portfolio_risk_budget,
    rank_tickers,
    recommend_allocations,
)
from simulation import (
    estimate_gbm_parameters,
    simulate_gbm,
    simulate_prediction_market,
    simulate_prices,
)
from viz import plot_distribution, plot_paths

LOGGER = logging.getLogger(__name__)
_FALLBACK_VERSION = "0.1.0"


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


def _simulate_model(
    *,
    args: argparse.Namespace,
    returns: pd.Series,
    current_price: float,
    ticker_seed: int | None,
) -> pd.DataFrame:
    if args.model == "historical":
        return simulate_prices(
            returns,
            days=args.days,
            scenarios=args.scenarios,
            dt=args.dt,
            seed=ticker_seed,
            current_price=current_price,
            shock_probability=float(args.shock_probability),
            shock_return=float(args.shock_return),
            block_size=int(args.block_size),
        )
    if args.model == "gbm":
        mu, sigma = estimate_gbm_parameters(returns)
        return simulate_gbm(
            current_price=current_price,
            mu=mu,
            sigma=sigma,
            days=args.days,
            scenarios=args.scenarios,
            dt=args.dt,
            seed=ticker_seed,
            shock_probability=float(args.shock_probability),
            shock_return=float(args.shock_return),
        )
    return simulate_prediction_market(
        fundamental_probability=float(args.fundamental_probability),
        current_price=float(current_price),
        days=args.days,
        scenarios=args.scenarios,
        certainty=float(args.fundamental_certainty),
        mean_reversion=float(args.prob_mean_reversion),
        daily_volatility=float(args.prob_daily_volatility),
        dt=args.dt,
        seed=ticker_seed,
    )


def _save_outputs(
    *,
    output_dir: Path,
    summary_df: pd.DataFrame,
    report: dict[str, object],
    rankings: pd.DataFrame,
    allocations: pd.DataFrame,
    execution_plan: pd.DataFrame,
    action_plan: dict[str, object],
    combined: pd.DataFrame,
    save_simulations: bool,
) -> None:
    summary_df.to_csv(output_dir / "summaries.csv", float_format="%.6g")
    with (output_dir / "summaries.json").open("w", encoding="utf-8") as handle:
        json.dump(summary_df.to_dict(orient="index"), handle, indent=2)

    with (output_dir / "report.json").open("w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2)

    if not rankings.empty:
        rankings.to_csv(output_dir / "rankings.csv", float_format="%.6g")
    if not allocations.empty:
        allocations.to_csv(output_dir / "allocations.csv", float_format="%.6g")
    if not execution_plan.empty:
        execution_plan.to_csv(output_dir / "execution_plan.csv", float_format="%.6g")

    with (output_dir / "action_plan.md").open("w", encoding="utf-8") as handle:
        handle.write("# Action Plan\n\n")
        handle.write(f"- **Stance:** {action_plan['stance']}\n")
        handle.write(f"- **Headline:** {action_plan['headline']}\n")
        if action_plan["primary_pick"] is not None:
            pick = action_plan["primary_pick"]
            handle.write(
                "- **Primary pick:** "
                f"{pick['ticker']} (weight {pick['weight']:.1%}, score {pick['score']:.1f}, "
                f"expected return {pick['expected_return']:.1%})\n"
            )
        if action_plan["avoid_list"]:
            handle.write(f"- **Avoid:** {', '.join(action_plan['avoid_list'])}\n")
        if action_plan.get("cash_weight", 0.0) > 0:
            handle.write(f"- **Cash buffer:** {action_plan['cash_weight']:.1%}\n")

        if not execution_plan.empty:
            handle.write("\n## Execution Plan\n\n")
            handle.write(
                "| Ticker | Weight | Price | Target $ | Shares | Est. Cost | Cash Drift |\n"
            )
            handle.write("| --- | ---: | ---: | ---: | ---: | ---: | ---: |\n")
            for ticker, row in execution_plan.iterrows():
                handle.write(
                    (
                        f"| {ticker} | {row['weight']:.1%} | {row['price']:.2f} | "
                        f"{row['target_dollars']:.2f} | {row['shares']:.4f} | "
                        f"{row['est_cost']:.2f} | {row['cash_drift']:.2f} |\n"
                    )
                )

    if save_simulations and not combined.empty:
        combined.to_csv(output_dir / "simulations.csv.gz", compression="gzip")


def _package_version() -> str:
    try:
        return metadata.version("monte-carlo-sim")
    except metadata.PackageNotFoundError:
        return _FALLBACK_VERSION
    except Exception:
        return _FALLBACK_VERSION


def _positive_int(value: str) -> int:
    try:
        parsed = int(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("must be an integer") from exc
    if parsed <= 0:
        raise argparse.ArgumentTypeError("must be a positive integer")
    return parsed


def _non_negative_int(value: str) -> int:
    try:
        parsed = int(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("must be an integer") from exc
    if parsed < 0:
        raise argparse.ArgumentTypeError("must be a non-negative integer")
    return parsed


def _positive_float(value: str) -> float:
    try:
        parsed = float(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("must be a number") from exc
    if parsed <= 0:
        raise argparse.ArgumentTypeError("must be a positive number")
    return parsed


def build_parser() -> argparse.ArgumentParser:
    """Create the CLI argument parser."""

    parser = argparse.ArgumentParser(
        description="Run Monte Carlo simulations for one or more tickers.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--version",
        action="version",
        version=f"%(prog)s {_package_version()}",
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
        type=_positive_int,
        default=252,
        help="Number of future trading days to simulate.",
    )
    parser.add_argument(
        "--scenarios",
        type=_positive_int,
        default=1000,
        help="Number of Monte Carlo scenarios to run per ticker.",
    )
    parser.add_argument(
        "--max-paths",
        type=_non_negative_int,
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
        type=_positive_float,
        default=1.0,
        help="Time increment for each simulation step (in trading days).",
    )
    parser.add_argument(
        "--block-size",
        type=_positive_int,
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
        type=_non_negative_int,
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
        type=_positive_float,
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
        default="gpt-4o-mini",
        help="OpenAI model name used when --ai-summary is enabled.",
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
        type=_positive_float,
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


def build_public_parser() -> argparse.ArgumentParser:
    """Create the simplified public CLI parser."""

    from public_cli import build_public_parser as _build_public_parser

    return _build_public_parser()


def parse_public_args(argv: Optional[Iterable[str]] = None) -> argparse.Namespace:
    """Return parsed arguments for the public CLI."""

    from public_cli import parse_public_args as _parse_public_args

    return _parse_public_args(argv)


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


def _normalise_tickers(ticker_arg: str) -> list[str]:
    requested = [ticker.strip().upper() for ticker in ticker_arg.split(",") if ticker.strip()]
    if not requested:
        raise ValueError("No valid tickers were supplied. Provide at least one ticker symbol.")

    tickers: list[str] = []
    seen: set[str] = set()
    for ticker in requested:
        if ticker in seen:
            continue
        seen.add(ticker)
        tickers.append(ticker)
    return tickers


def _hash_payload(payload: dict[str, object]) -> str:
    """Return a deterministic SHA-256 hash for a JSON-compatible payload."""

    canonical = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def _last_journal_chain_hash(journal_path: Path) -> str | None:
    """Return the last chain hash from a JSONL journal, if available."""

    if not journal_path.exists():
        return None

    try:
        lines = journal_path.read_text(encoding="utf-8").splitlines()
    except OSError:
        return None

    for line in reversed(lines):
        line = line.strip()
        if not line:
            continue
        try:
            entry = json.loads(line)
        except json.JSONDecodeError:
            return None
        value = entry.get("chain_hash")
        return str(value) if isinstance(value, str) and value else None
    return None


def _append_journal_entry(
    *,
    journal_path: Path,
    report: dict[str, object],
    args: argparse.Namespace,
) -> dict[str, object]:
    """Append a tamper-evident run entry to the decision journal."""

    previous_chain_hash = _last_journal_chain_hash(journal_path)
    report_hash = _hash_payload(report)
    summary = {
        ticker: payload.get("summary", {})
        for ticker, payload in report.get("results", {}).items()
        if isinstance(payload, dict)
    }

    body = {
        "generated_at": report.get("generated_at"),
        "tickers": sorted(summary.keys()),
        "model": args.model,
        "days": int(args.days),
        "scenarios": int(args.scenarios),
        "portfolio_risk_budget_pct": float(args.portfolio_risk_budget_pct),
        "policy_crc32": report.get("policy_crc32"),
        "report_hash": report_hash,
        "previous_chain_hash": previous_chain_hash,
        "summaries": summary,
    }

    chain_hash_input = {
        "report_hash": report_hash,
        "previous_chain_hash": previous_chain_hash,
    }
    entry = {
        "schema_version": 1,
        **body,
        "chain_hash": _hash_payload(chain_hash_input),
    }
    journal_path.parent.mkdir(parents=True, exist_ok=True)
    with journal_path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(entry, sort_keys=True))
        handle.write("\n")
    return entry


def _ranking_display_columns(rankings: pd.DataFrame) -> list[str]:
    preferred = [
        "score",
        "expected_return",
        "prob_above_current",
        "value_at_risk_95_pct",
        "max_drawdown_q95",
        "recommendation",
        "guardrail_reasons",
    ]
    return [column for column in preferred if column in rankings.columns]


def _render_detailed_simulation_tables(
    summary_df: pd.DataFrame,
    portfolio_summary: pd.Series | None,
    rankings: pd.DataFrame,
    allocations: pd.DataFrame,
) -> None:
    for ticker, row in summary_df.iterrows():
        _print_ticker_summary(ticker=str(ticker), summary=row, minimal=False)

    if portfolio_summary is not None:
        _print_portfolio_summary(portfolio_summary, minimal=False)

    if not rankings.empty:
        print("\nTicker ranking")
        print(
            rankings.loc[:, _ranking_display_columns(rankings)].to_string(
                float_format=lambda value: f"{value:0.3f}"
            )
        )

    if not allocations.empty:
        print("\nSuggested allocation")
        print(
            allocations.loc[:, ["weight", "score", "value_at_risk_95_pct"]].to_string(
                float_format=lambda value: f"{value:0.3f}"
            )
        )


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
            rankings.loc[:, _ranking_display_columns(rankings)].to_string(
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


def _maybe_show_simulation_plots(args: argparse.Namespace, result: dict[str, Any]) -> None:
    if bool(args.show) and not bool(args.no_plots) and not result["simulations"].empty:
        plt.show()
        plt.close("all")


def run(
    args: argparse.Namespace,
    *,
    render: bool = True,
    display_plots: bool = True,
) -> dict[str, Any]:
    """Execute the legacy simulation workflow and return simulation artefacts."""

    tickers = _normalise_tickers(args.tickers)
    output_dir = Path(args.output).expanduser() if args.output else None
    offline_path = Path(args.offline_path).expanduser() if args.offline_path else None
    cache_dir = Path(args.cache_dir).expanduser() if args.cache_dir else None
    journal_file = Path(args.journal_file).expanduser() if args.journal_file else None

    if output_dir is not None:
        output_dir.mkdir(parents=True, exist_ok=True)

    if args.model == "gbm" and int(args.block_size) != 1:
        LOGGER.warning("--block-size only applies to --model historical; ignoring for gbm")

    if not args.show and not args.no_plots:
        try:
            plt.switch_backend("Agg")
        except Exception:
            pass

    combined_frames: list[pd.DataFrame] = []
    summaries: dict[str, pd.Series] = {}
    current_prices: dict[str, float] = {}
    artefacts: dict[str, dict[str, str]] = {}
    errors: list[dict[str, str]] = []
    ai_summaries: dict[str, str] = {}
    horizon_years = float(args.days) * float(args.dt) / 252.0
    benchmark_return_pct = (1.0 + float(args.annual_cash_yield)) ** horizon_years - 1.0

    for ticker in tickers:
        if args.model == "prediction_market":
            current_price = float(
                args.market_price
                if args.market_price is not None
                else args.fundamental_probability
            )
            returns = pd.Series(dtype=float)
        else:
            try:
                prices = fetch_prices(
                    ticker,
                    start=args.start,
                    end=args.end,
                    offline_path=offline_path,
                    prefer_local=args.offline_only,
                    allow_local_fallback=bool(getattr(args, "allow_local_fallback", True)),
                    cache_dir=cache_dir,
                    refresh_cache=args.refresh_cache,
                )
            except PriceDataError as exc:
                message = str(exc)
                LOGGER.warning("[%s] %s", ticker, message)
                errors.append({"ticker": ticker, "error": message})
                continue

            prices = prices.dropna()
            returns = prices.pct_change().dropna()
            if returns.empty:
                message = (
                    "Not enough return data was available to run a simulation. "
                    "Try a longer price history."
                )
                LOGGER.warning("[%s] %s", ticker, message)
                errors.append({"ticker": ticker, "error": message})
                continue

            current_price = float(prices.iloc[-1])
        current_prices[ticker] = current_price
        ticker_seed = (
            None
            if args.seed is None
            else int(args.seed) + zlib.adler32(ticker.encode("utf-8"))
        )
        sims = _simulate_model(
            args=args,
            returns=returns,
            current_price=current_price,
            ticker_seed=ticker_seed,
        )

        sims = sims.copy()
        sims.columns = pd.MultiIndex.from_product(
            [[ticker], sims.columns], names=["ticker", "scenario"]
        )
        combined_frames.append(sims)

        summary = summarize_final_prices(
            sims.xs(ticker, axis=1, level=0),
            current_price=current_price,
            target_return_pct=(
                None if args.target_return_pct is None else float(args.target_return_pct)
            ),
            max_loss_pct=(None if args.max_loss_pct is None else float(args.max_loss_pct)),
            benchmark_return_pct=benchmark_return_pct,
        )
        summaries[ticker] = summary

        if args.ai_summary:
            try:
                ai_text = generate_ai_summary(
                    ticker=ticker,
                    summary=summary,
                    simulation_model=args.model,
                    days=args.days,
                    scenarios=args.scenarios,
                    model=args.ai_model,
                )
            except (OpenAIConfigurationError, OpenAIRequestError) as exc:
                message = str(exc)
                LOGGER.error("[%s] %s", ticker, message)
                errors.append({"ticker": ticker, "error": message})
            else:
                ai_summaries[ticker] = ai_text

        if not args.no_plots:
            max_paths = None if args.max_paths == 0 else int(args.max_paths)
            fig_dist = plot_distribution(sims, ticker=ticker, current_price=current_price)
            fig_paths = plot_paths(
                sims,
                ticker=ticker,
                max_paths=max_paths,
                current_price=current_price,
            )

            if output_dir is not None:
                dist_name = f"{ticker}_distribution.png"
                paths_name = f"{ticker}_paths.png"
                fig_dist.savefig(output_dir / dist_name, bbox_inches="tight")
                fig_paths.savefig(output_dir / paths_name, bbox_inches="tight")
                artefacts[ticker] = {
                    "distribution_plot": dist_name,
                    "paths_plot": paths_name,
                }
                if ticker in ai_summaries:
                    ai_name = f"{ticker}_ai_summary.md"
                    (output_dir / ai_name).write_text(ai_summaries[ticker] + "\n", encoding="utf-8")
                    artefacts[ticker]["ai_summary"] = ai_name

            if not args.show:
                plt.close(fig_dist)
                plt.close(fig_paths)

    combined = pd.concat(combined_frames, axis=1) if combined_frames else pd.DataFrame()
    portfolio_summary: pd.Series | None = None
    if len(summaries) > 1 and not combined.empty:
        portfolio_summary = summarize_equal_weight_portfolio(
            combined,
            current_prices=current_prices,
            benchmark_return_pct=benchmark_return_pct,
        )

    summary_df = pd.DataFrame(summaries).T if summaries else pd.DataFrame()
    rankings = rank_tickers(summary_df) if not summary_df.empty else pd.DataFrame()
    rankings = (
        apply_risk_guards(
            rankings,
            min_expected_return=float(args.min_expected_return),
            min_prob_above_current=float(args.min_prob_up),
            max_value_at_risk_95_pct=float(args.max_var_95_pct),
            max_drawdown_q95=(
                None
                if args.max_drawdown_q95_pct is None
                else float(args.max_drawdown_q95_pct)
            ),
            min_prob_hit_target=(
                None if args.min_prob_hit_target is None else float(args.min_prob_hit_target)
            ),
            max_prob_breach_loss=(
                None
                if args.max_prob_breach_loss is None
                else float(args.max_prob_breach_loss)
            ),
        )
        if not rankings.empty
        else rankings
    )
    allocations = recommend_allocations(rankings) if not rankings.empty else pd.DataFrame()
    if not allocations.empty:
        allocations = enforce_portfolio_risk_budget(
            allocations,
            rankings,
            max_portfolio_var_95_pct=float(args.portfolio_risk_budget_pct),
        )
    action_plan = build_action_plan(rankings, allocations)
    execution_plan = pd.DataFrame()
    if args.capital is not None and not allocations.empty:
        execution_plan = build_execution_plan(
            allocations,
            current_prices=current_prices,
            capital=float(args.capital),
            allow_fractional_shares=bool(args.allow_fractional_shares),
        )

    policy = getattr(args, "policy", {})
    policy_crc32 = None
    if policy:
        policy_bytes = json.dumps(policy, sort_keys=True).encode("utf-8")
        policy_crc32 = f"{zlib.crc32(policy_bytes):08x}"

    report: dict[str, object] = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "args": {
            key: (str(value) if isinstance(value, Path) else value)
            for key, value in vars(args).items()
        },
        "results": {
            ticker: {
                "summary": summaries[ticker].to_dict(),
                "artefacts": artefacts.get(ticker, {}),
                "ai_summary": ai_summaries.get(ticker),
            }
            for ticker in summaries
        },
        "portfolio_summary": (
            portfolio_summary.to_dict() if portfolio_summary is not None else None
        ),
        "rankings": rankings.to_dict(orient="index") if not rankings.empty else {},
        "allocations": allocations.to_dict(orient="index") if not allocations.empty else {},
        "execution_plan": (
            execution_plan.to_dict(orient="index") if not execution_plan.empty else {}
        ),
        "portfolio_risk_budget_pct": float(args.portfolio_risk_budget_pct),
        "policy": policy,
        "policy_crc32": policy_crc32,
        "action_plan": action_plan,
        "errors": errors,
    }

    if journal_file is not None:
        journal_entry = _append_journal_entry(
            journal_path=journal_file,
            report=report,
            args=args,
        )
        report["journal"] = {
            "file": str(journal_file),
            "chain_hash": journal_entry["chain_hash"],
            "previous_chain_hash": journal_entry["previous_chain_hash"],
        }

    if output_dir is not None:
        _save_outputs(
            output_dir=output_dir,
            summary_df=summary_df,
            report=report,
            rankings=rankings,
            allocations=allocations,
            execution_plan=execution_plan,
            action_plan=action_plan,
            combined=combined,
            save_simulations=args.save_simulations,
        )

    result = {
        "simulations": combined,
        "summaries": summary_df,
        "portfolio_summary": portfolio_summary,
        "report": report,
    }

    if render:
        _render_legacy_simulation_output(result, args)
    if display_plots:
        _maybe_show_simulation_plots(args, result)

    return result


def run_public_simulate(args: argparse.Namespace) -> dict[str, Any]:
    """Execute the simplified simulate command."""

    from public_cli import run_public_simulate as _run_public_simulate

    return _run_public_simulate(args)


def run_public_backtest(args: argparse.Namespace) -> dict[str, pd.DataFrame | pd.Series]:
    """Execute the simplified backtest command."""

    from public_cli import run_public_backtest as _run_public_backtest

    return _run_public_backtest(args)


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


def main(argv: Optional[Iterable[str]] = None) -> int:
    """Entrypoint for the public ``monte-carlo`` command."""

    from public_cli import main as public_main

    return public_main(argv)


if __name__ == "__main__":  # pragma: no cover - CLI invocation
    raise SystemExit(legacy_main())
