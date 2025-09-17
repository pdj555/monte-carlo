"""Command-line interface for running Monte Carlo stock simulations."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Iterable, Optional

import matplotlib.pyplot as plt
import pandas as pd

from analysis import summarize_final_prices
from data import PriceDataError, fetch_prices
from simulation import simulate_gbm, simulate_prices
from viz import plot_distribution, plot_paths


def build_parser() -> argparse.ArgumentParser:
    """Create the CLI argument parser."""

    parser = argparse.ArgumentParser(
        description="Run Monte Carlo simulations for one or more tickers.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
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
        type=int,
        default=252,
        help="Number of future trading days to simulate.",
    )
    parser.add_argument(
        "--scenarios",
        type=int,
        default=1000,
        help="Number of Monte Carlo scenarios to run per ticker.",
    )
    parser.add_argument(
        "--dt",
        type=float,
        default=1.0,
        help="Time increment for each simulation step.",
    )
    parser.add_argument(
        "--seed",
        type=int,
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
        "--output",
        type=str,
        help="Directory where plots are saved. Created if it does not exist.",
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
    parser.set_defaults(show=True)
    return parser


def parse_args(argv: Optional[Iterable[str]] = None) -> argparse.Namespace:
    """Return parsed CLI arguments."""

    parser = build_parser()
    return parser.parse_args(list(argv) if argv is not None else None)


def _normalise_tickers(ticker_arg: str) -> list[str]:
    tickers = [ticker.strip().upper() for ticker in ticker_arg.split(",") if ticker.strip()]
    if not tickers:
        raise ValueError("No valid tickers were supplied")
    return tickers


def run(args: argparse.Namespace) -> dict[str, pd.DataFrame]:
    """Execute the CLI workflow and return simulation artefacts."""

    tickers = _normalise_tickers(args.tickers)
    output_dir = Path(args.output).expanduser() if args.output else None
    offline_path = Path(args.offline_path).expanduser() if args.offline_path else None

    if output_dir is not None:
        output_dir.mkdir(parents=True, exist_ok=True)

    if not args.show:
        try:
            plt.switch_backend("Agg")
        except Exception:
            pass

    combined_frames: list[pd.DataFrame] = []
    summaries: dict[str, pd.Series] = {}

    for ticker in tickers:
        try:
            prices = fetch_prices(
                ticker,
                offline_path=offline_path,
                prefer_local=args.offline_only,
            )
        except PriceDataError as exc:
            print(f"[{ticker}] {exc}")
            continue

        prices = prices.dropna()
        returns = prices.pct_change().dropna()
        if returns.empty:
            print(f"[{ticker}] Not enough return data to run a simulation.")
            continue

        current_price = float(prices.iloc[-1])
        if args.model == "historical":
            sims = simulate_prices(
                returns,
                days=args.days,
                scenarios=args.scenarios,
                dt=args.dt,
                seed=args.seed,
                current_price=current_price,
            )
        else:
            mu = float(returns.mean())
            sigma = float(returns.std())
            sims = simulate_gbm(
                current_price=current_price,
                mu=mu,
                sigma=sigma,
                days=args.days,
                scenarios=args.scenarios,
                dt=args.dt,
                seed=args.seed,
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

        fig_dist = plot_distribution(sims, ticker=ticker)
        fig_paths = plot_paths(sims, ticker=ticker)

        if output_dir is not None:
            fig_dist.savefig(output_dir / f"{ticker}_distribution.png", bbox_inches="tight")
            fig_paths.savefig(output_dir / f"{ticker}_paths.png", bbox_inches="tight")

        if not args.show:
            plt.close(fig_dist)
            plt.close(fig_paths)

    combined = pd.concat(combined_frames, axis=1) if combined_frames else pd.DataFrame()
    summary_df = pd.DataFrame(summaries).T if summaries else pd.DataFrame()

    if args.show and not combined.empty:
        plt.show()
        plt.close("all")

    return {"simulations": combined, "summaries": summary_df}


def main(argv: Optional[Iterable[str]] = None) -> int:
    """Entrypoint used by the ``python cli.py`` command."""

    args = parse_args(argv)
    run(args)
    return 0


if __name__ == "__main__":  # pragma: no cover - CLI invocation
    raise SystemExit(main())
