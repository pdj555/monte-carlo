"""Command-line interface for running Monte Carlo stock simulations."""

from __future__ import annotations

import argparse
import json
import logging
import zlib
from datetime import datetime, timezone
from importlib import metadata
from pathlib import Path
from typing import Any, Iterable, Optional

import matplotlib.pyplot as plt
import pandas as pd

from ai import OpenAIConfigurationError, OpenAIRequestError, generate_ai_summary
from analysis import (
    apply_risk_guards,
    build_action_plan,
    rank_tickers,
    recommend_allocations,
    summarize_equal_weight_portfolio,
    summarize_final_prices,
)
from data import PriceDataError, fetch_prices
from simulation import estimate_gbm_parameters, simulate_gbm, simulate_prices
from viz import plot_distribution, plot_paths

LOGGER = logging.getLogger(__name__)
_FALLBACK_VERSION = "0.1.0"


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
        "--seed",
        type=_non_negative_int,
        default=None,
        help="Random seed for reproducible results.",
    )
    parser.add_argument(
        "--model",
        choices=("historical", "gbm"),
        default="historical",
        help="Simulation model: empirical distribution or geometric Brownian motion.",
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
        "--max-var-95-pct",
        type=float,
        default=0.25,
        help="Maximum allowed 95%% VaR as a percent of current price (e.g. 0.20 = 20%%).",
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
    return parser.parse_args(list(argv) if argv is not None else None)


def _normalise_tickers(ticker_arg: str) -> list[str]:
    requested = [ticker.strip().upper() for ticker in ticker_arg.split(",") if ticker.strip()]
    if not requested:
        raise ValueError("No valid tickers were supplied")

    tickers: list[str] = []
    seen: set[str] = set()
    for ticker in requested:
        if ticker in seen:
            continue
        seen.add(ticker)
        tickers.append(ticker)
    return tickers


def run(args: argparse.Namespace) -> dict[str, Any]:
    """Execute the CLI workflow and return simulation artefacts."""

    tickers = _normalise_tickers(args.tickers)
    output_dir = Path(args.output).expanduser() if args.output else None
    offline_path = Path(args.offline_path).expanduser() if args.offline_path else None
    cache_dir = Path(args.cache_dir).expanduser() if args.cache_dir else None

    if output_dir is not None:
        output_dir.mkdir(parents=True, exist_ok=True)

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

    for ticker in tickers:
        try:
            prices = fetch_prices(
                ticker,
                start=args.start,
                end=args.end,
                offline_path=offline_path,
                prefer_local=args.offline_only,
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
            message = "Not enough return data to run a simulation."
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
        if args.model == "historical":
            sims = simulate_prices(
                returns,
                days=args.days,
                scenarios=args.scenarios,
                dt=args.dt,
                seed=ticker_seed,
                current_price=current_price,
            )
        else:
            mu, sigma = estimate_gbm_parameters(returns)
            sims = simulate_gbm(
                current_price=current_price,
                mu=mu,
                sigma=sigma,
                days=args.days,
                scenarios=args.scenarios,
                dt=args.dt,
                seed=ticker_seed,
            )

        sims = sims.copy()
        sims.columns = pd.MultiIndex.from_product(
            [[ticker], sims.columns], names=["ticker", "scenario"]
        )
        combined_frames.append(sims)

        summary = summarize_final_prices(
            sims.xs(ticker, axis=1, level=0), current_price=current_price
        )
        summaries[ticker] = summary

        print(f"\nSummary for {ticker}")
        print(summary.to_frame(name="value").to_string(float_format=lambda v: f"{v:0.2f}"))

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
                print("\nAI summary")
                print(ai_text)

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
                    (output_dir / ai_name).write_text(
                        ai_summaries[ticker] + "\n", encoding="utf-8"
                    )
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
        )
        print("\nSummary for EQUAL_WEIGHT_PORTFOLIO")
        print(portfolio_summary.to_frame(name="value").to_string(float_format=lambda v: f"{v:0.2f}"))

    summary_df = pd.DataFrame(summaries).T if summaries else pd.DataFrame()
    rankings = rank_tickers(summary_df) if not summary_df.empty else pd.DataFrame()
    rankings = (
        apply_risk_guards(
            rankings,
            min_expected_return=float(args.min_expected_return),
            min_prob_above_current=float(args.min_prob_up),
            max_value_at_risk_95_pct=float(args.max_var_95_pct),
        )
        if not rankings.empty
        else rankings
    )
    allocations = recommend_allocations(rankings) if not rankings.empty else pd.DataFrame()
    action_plan = build_action_plan(rankings, allocations)

    if not rankings.empty:
        print("\nTicker ranking")
        display = rankings.loc[
            :,
            [
                "score",
                "expected_return",
                "prob_above_current",
                "value_at_risk_95_pct",
                "recommendation",
                "guardrail_reasons",
            ],
        ]
        print(display.to_string(float_format=lambda v: f"{v:0.3f}"))

    if not allocations.empty:
        print("\nSuggested allocation")
        print(
            allocations.loc[:, ["weight", "score", "value_at_risk_95_pct"]].to_string(
                float_format=lambda v: f"{v:0.3f}"
            )
        )

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
        "portfolio_summary": (portfolio_summary.to_dict() if portfolio_summary is not None else None),
        "rankings": rankings.to_dict(orient="index") if not rankings.empty else {},
        "allocations": allocations.to_dict(orient="index") if not allocations.empty else {},
        "action_plan": action_plan,
        "errors": errors,
    }

    if output_dir is not None:
        summary_df.to_csv(output_dir / "summaries.csv", float_format="%.6g")
        with (output_dir / "summaries.json").open("w", encoding="utf-8") as handle:
            json.dump(summary_df.to_dict(orient="index"), handle, indent=2)

        with (output_dir / "report.json").open("w", encoding="utf-8") as handle:
            json.dump(report, handle, indent=2)

        if not rankings.empty:
            rankings.to_csv(output_dir / "rankings.csv", float_format="%.6g")

        if not allocations.empty:
            allocations.to_csv(output_dir / "allocations.csv", float_format="%.6g")

        with (output_dir / "action_plan.md").open("w", encoding="utf-8") as handle:
            handle.write(f"# Action Plan\n\n")
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

        if args.save_simulations and not combined.empty:
            combined.to_csv(output_dir / "simulations.csv.gz", compression="gzip")

    if args.show and not args.no_plots and not combined.empty:
        plt.show()
        plt.close("all")

    return {"simulations": combined, "summaries": summary_df, "report": report}


def main(argv: Optional[Iterable[str]] = None) -> int:
    """Entrypoint used by the ``python cli.py`` command."""

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
    raise SystemExit(main())
