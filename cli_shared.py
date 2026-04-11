"""Shared helpers for the public and legacy CLI surfaces."""

from __future__ import annotations

import argparse
from importlib import metadata

import pandas as pd

FALLBACK_VERSION = "0.1.0"


def package_version() -> str:
    try:
        return metadata.version("monte-carlo-sim")
    except metadata.PackageNotFoundError:
        return FALLBACK_VERSION
    except Exception:
        return FALLBACK_VERSION


def positive_int(value: str) -> int:
    try:
        parsed = int(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("must be an integer") from exc
    if parsed <= 0:
        raise argparse.ArgumentTypeError("must be a positive integer")
    return parsed


def non_negative_int(value: str) -> int:
    try:
        parsed = int(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("must be an integer") from exc
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
