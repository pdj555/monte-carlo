"""Deprecated single-ticker wrapper for the legacy Monte Carlo entrypoint."""

from __future__ import annotations

import argparse
import logging
import sys
from typing import Iterable, Optional

import cli
from simulate_cli import build_simulation_args


def parse_args(argv: Optional[Iterable[str]] = None) -> argparse.Namespace:
    """Return CLI options controlling the deprecated single-ticker wrapper."""

    parser = argparse.ArgumentParser(
        description="Run a single-ticker Monte Carlo simulation.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--ticker",
        default="AAPL",
        help="Stock ticker symbol to simulate.",
    )
    parser.add_argument(
        "--days",
        type=int,
        default=365,
        help="Number of future trading days.",
    )
    parser.add_argument(
        "--scenarios",
        type=int,
        default=10000,
        help="Number of simulated price paths.",
    )
    parser.add_argument(
        "--dt",
        type=float,
        default=1.0,
        help="Time increment for each step.",
    )
    return parser.parse_args(list(argv) if argv is not None else None)


def _build_legacy_args(args: argparse.Namespace) -> argparse.Namespace:
    return build_simulation_args(
        tickers=str(args.ticker),
        days=int(args.days),
        scenarios=int(args.scenarios),
        no_plots=False,
        dt=float(args.dt),
        show=True,
    )


def main(argv: Optional[Iterable[str]] = None) -> int:
    """Entrypoint for the deprecated ``python MonteCarlo.py`` command."""

    print(
        "Deprecated: use `monte-carlo simulate [TICKER ...]` for the simplified CLI. "
        "Add `--show` when you want plots on screen.",
        file=sys.stderr,
    )
    args = parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

    try:
        result = cli.run(_build_legacy_args(args))
    except Exception as exc:
        logging.getLogger(__name__).error("%s", exc)
        return 2
    return 0 if not result["summaries"].empty else 1


if __name__ == "__main__":
    raise SystemExit(main())
