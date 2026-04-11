"""Deprecated single-ticker wrapper for the legacy Monte Carlo entrypoint."""

from __future__ import annotations

import argparse
import logging
import sys
from typing import Iterable, Optional

import cli


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
    return argparse.Namespace(
        journal_file=None,
        policy_file=None,
        tickers=str(args.ticker),
        days=int(args.days),
        scenarios=int(args.scenarios),
        max_paths=100,
        no_plots=False,
        dt=float(args.dt),
        block_size=1,
        shock_probability=0.0,
        shock_return=-0.15,
        seed=None,
        model="historical",
        fundamental_probability=None,
        market_price=None,
        fundamental_certainty=100.0,
        prob_mean_reversion=0.2,
        prob_daily_volatility=0.03,
        start=None,
        end=None,
        output=None,
        cache_dir=None,
        refresh_cache=False,
        save_simulations=False,
        offline_path=None,
        offline_only=False,
        allow_local_fallback=True,
        show=True,
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
