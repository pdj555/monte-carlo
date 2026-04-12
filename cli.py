"""Thin deprecated facade for legacy simulation and public CLI entrypoints."""

from __future__ import annotations

from legacy_cli import build_parser, legacy_main, parse_args, run
from public_cli import (
    build_public_parser,
    main,
    parse_public_args,
    run_public_backtest,
    run_public_simulate,
)

__all__ = [
    "build_parser",
    "build_public_parser",
    "legacy_main",
    "main",
    "parse_args",
    "parse_public_args",
    "run",
    "run_public_backtest",
    "run_public_simulate",
]


if __name__ == "__main__":  # pragma: no cover - CLI invocation
    raise SystemExit(legacy_main())
