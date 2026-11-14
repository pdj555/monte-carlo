"""Machine-readable output for the public ``monte-carlo`` CLI.

``--json`` makes a command print exactly one JSON document on stdout. The
document shape is fixed by ``schemas/cli-result.v1.schema.json``; logs and
warnings stay on stderr. Nothing in the document depends on the wall clock, so
the same configuration, seed, and price files give the same bytes.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import secrets
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import pandas as pd

from cli_schema import SCHEMA, SCHEMA_ID
from cli_shared import MAX_SEED, package_version

EXIT_OK = 0
EXIT_FAILED = 1
EXIT_USAGE = 2
EXIT_INTERNAL = 3

_STATUS_BY_EXIT_CODE = {
    EXIT_OK: "ok",
    EXIT_FAILED: "failed",
    EXIT_USAGE: "error",
    EXIT_INTERNAL: "error",
}

# Seeds drawn for the caller cover the whole range --seed accepts.
_GENERATED_SEED_BOUND = MAX_SEED + 1


def resolve_seed(args: argparse.Namespace) -> str:
    """Give ``args`` a concrete seed and return where it came from.

    A run without ``--seed`` is otherwise unrepeatable. Drawing the seed here
    and reporting it lets a caller replay the exact run.
    """

    if getattr(args, "seed", None) is not None:
        return "argument"
    args.seed = secrets.randbelow(_GENERATED_SEED_BOUND)
    return "generated"


def to_jsonable(value: Any) -> Any:
    """Convert engine output to plain JSON types; NaN and infinity become null."""

    if value is None or isinstance(value, (str, bool)):
        return value
    if isinstance(value, (np.bool_,)):
        return bool(value)
    if isinstance(value, (int, np.integer)):
        return int(value)
    if isinstance(value, (float, np.floating)):
        number = float(value)
        return number if math.isfinite(number) else None
    if isinstance(value, (pd.Timestamp,)):
        # Prices are daily, so a date says everything; keep the time only if there is one.
        return value.date().isoformat() if value == value.normalize() else value.isoformat()
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, pd.Series):
        return {str(key): to_jsonable(item) for key, item in value.items()}
    if isinstance(value, Mapping):
        return {str(key): to_jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [to_jsonable(item) for item in value]
    if value is pd.NaT:
        return None
    raise TypeError(f"cannot serialise {type(value).__name__} to JSON")


def _frame_records(frame: pd.DataFrame, *, index_name: str) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for index, row in frame.iterrows():
        record = {index_name: to_jsonable(index)}
        record.update({str(key): to_jsonable(item) for key, item in row.items()})
        records.append(record)
    return records


def _file_sha256(path: object) -> str | None:
    if not isinstance(path, str) or not path:
        return None
    candidate = Path(path).expanduser()
    try:
        return hashlib.sha256(candidate.read_bytes()).hexdigest()
    except OSError:
        return None


def _price_sources(price_sources: object) -> dict[str, dict[str, Any]]:
    """Price provenance per ticker, with a content hash for local files."""

    if not isinstance(price_sources, Mapping):
        return {}
    described: dict[str, dict[str, Any]] = {}
    for ticker in sorted(str(key) for key in price_sources):
        source = price_sources[ticker]
        if not isinstance(source, Mapping):
            continue
        path = source.get("path")
        described[ticker] = {
            "kind": str(source.get("kind", "")),
            # Absolute, like config.data_path, so the document reads the same
            # whichever directory the command was run from.
            "path": str(Path(path).expanduser().resolve()) if isinstance(path, str) else None,
            "sha256": _file_sha256(path),
            "used_fallback": bool(source.get("used_fallback", False)),
            "is_sample_data": bool(source.get("is_sample_data", False)),
        }
    return described


def _absolute_data_path(args: argparse.Namespace) -> str | None:
    """--data-path as an absolute path, so a replay works from any directory."""

    if args.data_path is None:
        return None
    return str(Path(args.data_path).expanduser().resolve())


def _shared_flags(args: argparse.Namespace) -> list[str]:
    flags = ["--model", str(args.model), "--scenarios", str(int(args.scenarios))]
    flags += ["--seed", str(int(args.seed)), "--source", str(args.source)]
    data_path = _absolute_data_path(args)
    if data_path is not None:
        flags += ["--data-path", data_path]
    return flags


def simulate_config(args: argparse.Namespace, *, seed_source: str) -> dict[str, Any]:
    tickers = [str(ticker) for ticker in args.tickers]
    return {
        "tickers": tickers,
        "days": int(args.days),
        "scenarios": int(args.scenarios),
        "model": str(args.model),
        "seed": int(args.seed),
        "seed_source": seed_source,
        "source": str(args.source),
        "data_path": _absolute_data_path(args),
        "reproduce": [
            "monte-carlo",
            "simulate",
            *tickers,
            "--days",
            str(int(args.days)),
            *_shared_flags(args),
            "--json",
        ],
    }


def backtest_config(args: argparse.Namespace, *, seed_source: str) -> dict[str, Any]:
    tickers = [str(ticker) for ticker in args.tickers]
    rebalance = int(args.hold if args.rebalance is None else args.rebalance)
    return {
        "tickers": tickers,
        "lookback": int(args.lookback),
        "hold": int(args.hold),
        "rebalance": rebalance,
        "top": int(args.top),
        "scenarios": int(args.scenarios),
        "model": str(args.model),
        "seed": int(args.seed),
        "seed_source": seed_source,
        "source": str(args.source),
        "data_path": _absolute_data_path(args),
        "reproduce": [
            "monte-carlo",
            "backtest",
            *tickers,
            "--lookback",
            str(int(args.lookback)),
            "--hold",
            str(int(args.hold)),
            "--rebalance",
            str(rebalance),
            "--top",
            str(int(args.top)),
            *_shared_flags(args),
            "--json",
        ],
    }


def evaluate_config(args: argparse.Namespace, report_dict: Mapping[str, Any]) -> dict[str, Any]:
    evaluation_set = report_dict.get("evaluation_set", {})
    # The path the manifest was actually read from, absolute, so the replay
    # works from any directory.
    resolved = evaluation_set.get("manifest_path")
    set_file = str(resolved or Path(args.set_file).expanduser().resolve())
    return {
        "set_file": set_file,
        "manifest_sha256": evaluation_set.get("manifest_sha256"),
        "seeds": to_jsonable(evaluation_set.get("seeds", [])),
        "reproduce": ["monte-carlo", "evaluate", set_file, "--json"],
    }


def simulate_result(result: Mapping[str, Any]) -> dict[str, Any]:
    report = result["report"]
    return {
        "tickers": {
            str(ticker): to_jsonable(payload["summary"])
            for ticker, payload in report["results"].items()
        },
        "portfolio_summary": to_jsonable(report["portfolio_summary"]),
        "allocation_portfolio_summary": to_jsonable(report["allocation_portfolio_summary"]),
        "rankings": to_jsonable(report["rankings"]),
        "allocations": to_jsonable(report["allocations"]),
        "action_plan": to_jsonable(report["action_plan"]),
        "price_sources": _price_sources(result.get("price_sources")),
        "errors": to_jsonable(report["errors"]),
    }


def backtest_result(result: Mapping[str, Any]) -> dict[str, Any]:
    rebalance_log = _frame_records(result["rebalance_log"], index_name="rebalance_date")
    for row in rebalance_log:
        # The engine keeps these two as CSV-friendly strings; give JSON readers
        # a list and an object instead of strings they would have to re-parse.
        tickers = row.get("selected_tickers")
        if isinstance(tickers, str):
            row["selected_tickers"] = [item for item in tickers.split(",") if item]
        weights = row.get("selected_weights")
        if isinstance(weights, str):
            row["selected_weights"] = json.loads(weights)
    summary = to_jsonable(result["summary"])
    # The engine keeps every summary value as a float; the period count is a count.
    summary["periods"] = int(summary["periods"])
    return {
        "summary": summary,
        "rebalance_log": rebalance_log,
        "equity_curve": _frame_records(result["equity_curve"], index_name="date"),
        "price_sources": _price_sources(result.get("price_sources")),
    }


def evaluate_result(report_dict: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "evaluation_set": to_jsonable(report_dict["evaluation_set"]),
        "runs": to_jsonable(report_dict["runs"]),
        "outcomes": to_jsonable(report_dict["outcomes"]),
        "scorecard": to_jsonable(report_dict["scorecard"]),
    }


def build_document(
    *,
    command: str | None,
    exit_code: int,
    config: Mapping[str, Any] | None = None,
    result: Mapping[str, Any] | None = None,
    error_kind: str | None = None,
    error_message: str | None = None,
) -> dict[str, Any]:
    """Assemble the one document a ``--json`` invocation prints."""

    error = None
    if error_kind is not None:
        error = {"kind": error_kind, "message": str(error_message or "")}
    return {
        "schema": SCHEMA_ID,
        "version": package_version(),
        "command": command,
        "status": _STATUS_BY_EXIT_CODE[exit_code],
        "exit_code": exit_code,
        "config": None if config is None else dict(config),
        "result": None if result is None else dict(result),
        "error": error,
    }


def render_document(document: Mapping[str, Any]) -> str:
    return json.dumps(document, indent=2, sort_keys=True, allow_nan=False)


def render_schema() -> str:
    return json.dumps(SCHEMA, indent=2)
