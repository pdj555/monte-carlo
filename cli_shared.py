"""Shared helpers for the public and legacy CLI surfaces."""

from __future__ import annotations

import argparse
import re
from importlib import metadata

import pandas as pd

FALLBACK_VERSION = "0.1.0"

# Upper bounds the public CLI accepts. A run holds days x scenarios prices per
# ticker in memory, so the product is bounded as well as each factor.
MAX_SCENARIOS = 1_000_000
MAX_DAYS = 25_200
MAX_PATH_CELLS = 50_000_000
# The largest seed accepted. Every value up to it seeds numpy unchanged and is
# an exact integer in JSON readers that hold numbers as doubles.
MAX_SEED = 2**32 - 1


class InputError(ValueError):
    """Something the caller supplied was rejected: a setting, a file, a path.

    Only this (and price data that cannot be loaded) is the caller's to fix.
    Any other exception from the engine is treated as a fault in the program.
    """


class IntentionalDefaultsHelpFormatter(argparse.ArgumentDefaultsHelpFormatter):
    """Show only defaults that help a human choose the happy path."""

    def _get_help_string(self, action: argparse.Action) -> str:
        if action.default in (None, False, argparse.SUPPRESS):
            return action.help
        return super()._get_help_string(action)


def package_version() -> str:
    try:
        return metadata.version("monte-carlo-sim")
    except metadata.PackageNotFoundError:
        return FALLBACK_VERSION
    except Exception:
        return FALLBACK_VERSION


_PLAIN_INTEGER = re.compile(r"-?[0-9]+")


def _plain_int(value: str) -> int:
    """Parse an integer written as plain ASCII digits.

    ``int`` also takes ``4_2``, surrounding spaces, a leading ``+`` and digits
    from other scripts; none of those is what a caller typing a count means.
    """

    if not isinstance(value, str) or _PLAIN_INTEGER.fullmatch(value) is None:
        raise argparse.ArgumentTypeError("must be an integer")
    return int(value)


def positive_int(value: str) -> int:
    parsed = _plain_int(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("must be a positive integer")
    return parsed


def bounded_positive_int(maximum: int):
    """Return an argparse type for integers from 1 to ``maximum``."""

    def parse(value: str) -> int:
        parsed = positive_int(value)
        if parsed > maximum:
            raise argparse.ArgumentTypeError(f"must be at most {maximum}")
        return parsed

    return parse


def seed_int(value: str) -> int:
    parsed = non_negative_int(value)
    if parsed > MAX_SEED:
        raise argparse.ArgumentTypeError(f"must be at most {MAX_SEED}")
    return parsed


def non_negative_int(value: str) -> int:
    parsed = _plain_int(value)
    if parsed < 0:
        raise argparse.ArgumentTypeError("must be a non-negative integer")
    return parsed


def positive_float(value: str) -> float:
    try:
        parsed = float(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("must be a number") from exc
    if parsed <= 0:
        raise argparse.ArgumentTypeError("must be a positive number")
    return parsed


def ranking_display_columns(rankings: pd.DataFrame) -> list[str]:
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


def _print_ticker_summary(*, ticker: str, summary: pd.Series) -> None:
    print(f"\nSummary for {ticker}")
    print(summary.to_frame(name="value").to_string(float_format=lambda value: f"{value:0.2f}"))


def _print_portfolio_summary(summary: pd.Series) -> None:
    print("\nSummary for EQUAL_WEIGHT_PORTFOLIO")
    print(summary.to_frame(name="value").to_string(float_format=lambda value: f"{value:0.2f}"))


def render_detailed_simulation_tables(
    summary_df: pd.DataFrame,
    portfolio_summary: pd.Series | None,
    rankings: pd.DataFrame,
    allocations: pd.DataFrame,
) -> None:
    for ticker, row in summary_df.iterrows():
        _print_ticker_summary(ticker=str(ticker), summary=row)

    if portfolio_summary is not None:
        _print_portfolio_summary(portfolio_summary)

    if not rankings.empty:
        print("\nTicker ranking")
        print(
            rankings.loc[:, ranking_display_columns(rankings)].to_string(
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
