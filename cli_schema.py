"""JSON Schema for the document ``monte-carlo ... --json`` prints.

This module is the source; ``schemas/cli-result.v1.schema.json`` is the same
schema written out for readers of the repository (a test keeps them equal).
Keeping it in a module puts it in the installed package, so
``monte-carlo schema`` works without a source checkout.
"""

from typing import Any

from cli_shared import MAX_DAYS, MAX_SCENARIOS, MAX_SEED

SCHEMA_ID = "monte-carlo.cli-result.v1"

_NUMBER_OR_NULL = {"type": ["number", "null"]}
_PROBABILITY = {"type": "number", "minimum": 0, "maximum": 1}
_PROBABILITY_OR_NULL = {"type": ["number", "null"], "minimum": 0, "maximum": 1}
_SEED = {"type": "integer", "minimum": 0, "maximum": MAX_SEED}
_COUNT = {"type": "integer", "minimum": 0}
_STRINGS = {"type": "array", "items": {"type": "string"}}
# A date for daily prices; a date and time (ISO 8601) when the price file is intraday.
_DATE = {
    "type": "string",
    "pattern": (
        "^[0-9]{4}-[0-9]{2}-[0-9]{2}"
        "(T[0-9]{2}:[0-9]{2}:[0-9]{2}(\\.[0-9]+)?([+-][0-9]{2}:[0-9]{2})?)?$"
    ),
}
_SHA256 = "^[0-9a-f]{64}$"


def _closed(properties: dict[str, Any], optional: tuple[str, ...] = ()) -> dict[str, Any]:
    """An object with exactly these keys; ``optional`` ones may be absent."""

    return {
        "type": "object",
        "additionalProperties": False,
        "required": sorted(key for key in properties if key not in optional),
        "properties": properties,
    }


def _nullable(definition: str) -> dict[str, Any]:
    return {"oneOf": [{"type": "null"}, {"$ref": f"#/$defs/{definition}"}]}


def _when(conditions: dict[str, Any], then: dict[str, Any]) -> dict[str, Any]:
    return {
        "if": {
            "properties": {key: {"const": value} for key, value in conditions.items()},
            "required": sorted(conditions),
        },
        "then": then,
    }


_PROBABILITY_METRICS = (
    "prob_above_current",
    "prob_below_current",
    "prob_beat_benchmark",
    "prob_drawdown_10_pct",
    "prob_drawdown_20_pct",
    "prob_hit_target",
    "prob_breach_max_loss",
    "kelly_fraction",
)

_SUMMARY = {
    "type": "object",
    "required": [
        "mean",
        "median",
        "std",
        "q05",
        "q95",
        "expected_return",
        "prob_above_current",
        "value_at_risk_95_pct",
        "expected_shortfall_95_pct",
        "max_drawdown_q95",
    ],
    "properties": {name: _PROBABILITY_OR_NULL for name in _PROBABILITY_METRICS},
    "additionalProperties": _NUMBER_OR_NULL,
}

_PRICE_SOURCES = {
    "type": "object",
    "additionalProperties": _closed(
        {
            "kind": {"type": "string"},
            "path": {"type": ["string", "null"]},
            "sha256": {"type": ["string", "null"], "pattern": _SHA256},
            "used_fallback": {"type": "boolean"},
            "is_sample_data": {"type": "boolean"},
        }
    ),
}

_COMMON_CONFIG = {
    "tickers": {"type": "array", "items": {"type": "string"}, "minItems": 1},
    "scenarios": {"type": "integer", "minimum": 1, "maximum": MAX_SCENARIOS},
    "model": {"enum": ["historical", "gbm"]},
    "seed": _SEED,
    "seed_source": {"enum": ["argument", "generated"]},
    "source": {"enum": ["auto", "offline", "online"]},
    "data_path": {"type": ["string", "null"]},
    "reproduce": {
        "description": (
            "argv that replays the numbers. It leaves out --output, --show and "
            "--details, which change what is saved or shown, not what is computed. "
            "The replay prints the same result and the same config except "
            "seed_source, which reads argument because the seed is now given."
        ),
        "type": "array",
        "items": {"type": "string"},
        "minItems": 3,
    },
}

_POSITIVE = {"type": "integer", "minimum": 1}

_RANKING_ROW = _closed(
    {
        "score": {"type": "number"},
        "expected_return": {"type": "number"},
        "prob_above_current": _PROBABILITY,
        "value_at_risk_95_pct": {"type": "number"},
        "kelly_fraction": _PROBABILITY,
        "max_drawdown_q95": {"type": "number"},
        "expected_shortfall_95_pct": {"type": "number"},
        "expected_excess_return": {"type": "number"},
        "prob_beat_benchmark": _PROBABILITY,
        "prob_hit_target": _PROBABILITY,
        "prob_breach_max_loss": _PROBABILITY,
        "recommendation": {"enum": ["BUY", "WATCH", "AVOID"]},
        "guardrail_reasons": {"type": "string"},
    },
    optional=(
        "kelly_fraction",
        "max_drawdown_q95",
        "expected_shortfall_95_pct",
        "expected_excess_return",
        "prob_beat_benchmark",
        "prob_hit_target",
        "prob_breach_max_loss",
    ),
)

_ACTION_PLAN = _closed(
    {
        "stance": {"enum": ["RISK_ON", "SELECTIVE", "DEFENSIVE", "NO_TRADE"]},
        "headline": {"type": "string", "minLength": 1},
        "primary_pick": {
            "oneOf": [
                {"type": "null"},
                _closed(
                    {
                        "ticker": {"type": "string"},
                        "weight": _PROBABILITY,
                        "score": {"type": "number"},
                        "expected_return": {"type": "number"},
                        "prob_above_current": _PROBABILITY,
                        "value_at_risk_95_pct": {"type": "number"},
                    }
                ),
            ]
        },
        "focus_list": _STRINGS,
        "avoid_list": _STRINGS,
        "cash_weight": _PROBABILITY,
    }
)

_SIMULATE_RESULT = _closed(
    {
        "tickers": {"type": "object", "additionalProperties": {"$ref": "#/$defs/summary"}},
        "portfolio_summary": _nullable("summary"),
        "allocation_portfolio_summary": _nullable("summary"),
        "rankings": {"type": "object", "additionalProperties": _RANKING_ROW},
        "allocations": {
            "type": "object",
            "additionalProperties": _closed(
                {
                    "weight": _PROBABILITY,
                    "score": {"type": "number"},
                    "value_at_risk_95_pct": {"type": "number"},
                }
            ),
        },
        "action_plan": _ACTION_PLAN,
        "price_sources": {"$ref": "#/$defs/price_sources"},
        "errors": {
            "type": "array",
            "items": _closed(
                {
                    "ticker": {"type": "string"},
                    "error": {"type": "string"},
                    "kind": {"enum": ["price_data", "insufficient_history", "ai_summary"]},
                }
            ),
        },
    }
)

_BACKTEST_RESULT = _closed(
    {
        "summary": _closed(
            {
                "periods": _POSITIVE,
                "years": {"type": "number", "exclusiveMinimum": 0},
                "strategy_total_return": {"type": "number"},
                "strategy_annualized_return": _NUMBER_OR_NULL,
                "strategy_max_drawdown": _PROBABILITY,
                "strategy_win_rate": _PROBABILITY,
                "average_turnover": {"type": "number", "minimum": 0},
                "total_transaction_cost_drag": {"type": "number", "minimum": 0},
                "equal_weight_total_return": {"type": "number"},
                "cash_total_return": {"type": "number"},
                "excess_return_vs_equal_weight": {"type": "number"},
                "excess_return_vs_cash": {"type": "number"},
                "excess_return_t_stat": _NUMBER_OR_NULL,
            }
        ),
        "rebalance_log": {
            "type": "array",
            "minItems": 1,
            "items": _closed(
                {
                    "rebalance_date": _DATE,
                    "end_date": _DATE,
                    "holding_days": _POSITIVE,
                    "selection_count": _COUNT,
                    "selected_tickers": _STRINGS,
                    "selected_weights": {
                        "type": "object",
                        "additionalProperties": _PROBABILITY,
                    },
                    "cash_weight": _PROBABILITY,
                    "turnover": {"type": "number", "minimum": 0},
                    "transaction_cost_drag": {"type": "number", "minimum": 0},
                    "strategy_return_gross": {"type": "number"},
                    "strategy_return_net": {"type": "number"},
                    "equal_weight_return": {"type": "number"},
                    "cash_return": {"type": "number"},
                }
            ),
        },
        "equity_curve": {
            "type": "array",
            "minItems": 2,
            "items": _closed(
                {
                    "date": _DATE,
                    "strategy": {"type": "number"},
                    "equal_weight": {"type": "number"},
                    "cash": {"type": "number"},
                }
            ),
        },
        "price_sources": {"$ref": "#/$defs/price_sources"},
    }
)

_RUN_IDENTITY = {
    "run_id": {"type": "string", "minLength": 1},
    "universe_name": {"type": "string"},
    "model": {"enum": ["historical", "gbm"]},
    "seed": _SEED,
    "source_name": {"type": "string"},
    "source_mode": {"enum": ["auto", "offline", "online"]},
}

_EVALUATE_RESULT = _closed(
    {
        "evaluation_set": {
            "type": "object",
            "required": ["name", "days", "scenarios", "models", "seeds", "manifest_sha256"],
            "properties": {
                "seeds": {"type": "array", "items": _SEED},
                "manifest_sha256": {"type": "string", "pattern": _SHA256},
            },
        },
        "runs": {
            "type": "array",
            "items": _closed(
                {
                    **_RUN_IDENTITY,
                    "tickers": _STRINGS,
                    "data_path": {"type": ["string", "null"]},
                    "days": _POSITIVE,
                    "scenarios": _POSITIVE,
                }
            ),
        },
        "outcomes": {
            "type": "array",
            "items": _closed(
                {
                    **_RUN_IDENTITY,
                    "requested_tickers": _COUNT,
                    "completed_tickers": _COUNT,
                    "status": {"enum": ["completed", "failed"]},
                    "error": {"type": ["string", "null"]},
                    "stance": {"type": ["string", "null"]},
                    "top_pick": {"type": ["string", "null"]},
                    "rank_order": _STRINGS,
                    "guardrail_rejections": _COUNT,
                    "worst_var_95_pct": _NUMBER_OR_NULL,
                }
            ),
        },
        "scorecard": _closed(
            {
                "total_runs": _COUNT,
                "completed_runs": _COUNT,
                "failed_runs": _COUNT,
                "run_success_rate": _PROBABILITY,
                "ticker_success_rate": _PROBABILITY,
                "mean_rank_correlation": {"type": ["number", "null"], "minimum": -1, "maximum": 1},
                "top_pick_consistency": _PROBABILITY_OR_NULL,
                "guardrail_rejection_rate": _PROBABILITY,
                "no_trade_rate": _PROBABILITY,
                "worst_var_95_pct": _NUMBER_OR_NULL,
                "source_reliability": {
                    "type": "object",
                    "additionalProperties": _closed(
                        {
                            "total_runs": _COUNT,
                            "completed_runs": _COUNT,
                            "failed_runs": _COUNT,
                            "run_success_rate": _PROBABILITY,
                        }
                    ),
                },
            }
        ),
    }
)


def _command_rule(command: str) -> dict[str, Any]:
    return _when(
        {"command": command},
        {
            "properties": {
                "config": _nullable(f"{command}_config"),
                "result": _nullable(f"{command}_result"),
            }
        },
    )


def _result_rule(command: str, status: str, result: dict[str, Any]) -> dict[str, Any]:
    return _when(
        {"command": command, "status": status},
        {"properties": {"result": {"properties": result}}},
    )


def _result_type_rule(command: str, status: str, json_type: str) -> dict[str, Any]:
    return _when(
        {"command": command, "status": status},
        {"properties": {"result": {"type": json_type}}},
    )


def _config_type_rule(command: str, json_type: str) -> dict[str, Any]:
    return _when(
        {"command": command, "status": "ok"},
        {"properties": {"config": {"type": json_type}}},
    )


_HELP_RESULT = _closed(
    {
        "command": {"enum": ["simulate", "backtest", "evaluate", "schema", None]},
        "usage": {"type": "string", "minLength": 1},
        "help": {"type": "string", "minLength": 1},
        "commands": {"type": "array", "items": {"type": "string"}, "minItems": 1},
        "exit_codes": _closed({str(code): {"type": "string"} for code in range(5)}),
        "limits": _closed(
            {
                "max_scenarios": _POSITIVE,
                "max_days": _POSITIVE,
                "max_path_cells": _POSITIVE,
                "max_seed": _POSITIVE,
            }
        ),
    }
)


SCHEMA: dict[str, Any] = {
    "$schema": "https://json-schema.org/draft/2020-12/schema",
    "$id": "https://github.com/pdj555/monte-carlo/schemas/cli-result.v1.schema.json",
    "title": "monte-carlo --json result",
    "description": (
        "The single document `monte-carlo simulate|backtest|evaluate --json` prints on "
        "stdout. exit_code equals the process exit code: 0 ok; 1 the command ran but "
        "produced no usable result; 2 the command line or its inputs were rejected, "
        "including price data that could not be loaded; 3 an internal error. Exit 4 "
        "means the document could not be written to stdout, so no document carries it."
    ),
    "type": "object",
    "additionalProperties": False,
    "required": [
        "schema",
        "version",
        "command",
        "status",
        "exit_code",
        "config",
        "result",
        "error",
    ],
    "properties": {
        "schema": {"const": SCHEMA_ID},
        "version": {"type": "string", "minLength": 1},
        "command": {"enum": ["simulate", "backtest", "evaluate", "help", "version", None]},
        "status": {"enum": ["ok", "failed", "error"]},
        "exit_code": {"enum": [0, 1, 2, 3]},
        "config": {"type": ["object", "null"]},
        "result": {"type": ["object", "null"]},
        "error": {
            "oneOf": [
                {"type": "null"},
                _closed(
                    {
                        "kind": {
                            "enum": [
                                "usage",
                                "invalid_input",
                                "price_data",
                                "internal",
                                "no_results",
                                "failed_runs",
                            ]
                        },
                        "message": {"type": "string", "minLength": 1},
                    }
                ),
            ]
        },
    },
    "allOf": [
        _when(
            {"status": "ok"},
            {
                "properties": {
                    "exit_code": {"const": 0},
                    "error": {"type": "null"},
                    "result": {"type": "object"},
                    "command": {"type": "string"},
                }
            },
        ),
        _when(
            {"status": "failed"},
            {
                "properties": {
                    "exit_code": {"const": 1},
                    "error": {
                        "type": "object",
                        "properties": {"kind": {"enum": ["no_results", "failed_runs"]}},
                    },
                    "config": {"type": "object"},
                    "command": {"type": "string"},
                }
            },
        ),
        _result_type_rule("simulate", "failed", "object"),
        _result_type_rule("evaluate", "failed", "object"),
        # A backtest with too little history stops before it has anything to report.
        _result_type_rule("backtest", "failed", "null"),
        _when(
            {"status": "error"},
            {
                "properties": {
                    "exit_code": {"enum": [2, 3]},
                    "error": {"type": "object"},
                    "result": {"type": "null"},
                }
            },
        ),
        _when(
            {"exit_code": 3},
            {"properties": {"error": {"properties": {"kind": {"const": "internal"}}}}},
        ),
        _when(
            {"exit_code": 2},
            {
                "properties": {
                    "error": {
                        "properties": {"kind": {"enum": ["usage", "invalid_input", "price_data"]}}
                    }
                }
            },
        ),
        _command_rule("simulate"),
        _command_rule("backtest"),
        _command_rule("evaluate"),
        _config_type_rule("simulate", "object"),
        _config_type_rule("backtest", "object"),
        _config_type_rule("evaluate", "object"),
        # --json with --help or --version describes the program; nothing was run.
        _when(
            {"command": "help"},
            {"properties": {"config": {"type": "null"}, "result": {"$ref": "#/$defs/help_result"}}},
        ),
        _when(
            {"command": "version"},
            {
                "properties": {
                    "config": {"type": "null"},
                    "result": _closed({"version": {"type": "string", "minLength": 1}}),
                }
            },
        ),
        _result_rule("simulate", "ok", {"tickers": {"minProperties": 1}}),
        _result_rule("simulate", "failed", {"tickers": {"maxProperties": 0}}),
        _result_rule(
            "evaluate",
            "ok",
            {"scorecard": {"properties": {"failed_runs": {"const": 0}}}},
        ),
        _result_rule(
            "evaluate",
            "failed",
            {"scorecard": {"properties": {"failed_runs": {"minimum": 1}}}},
        ),
    ],
    "$defs": {
        "price_sources": _PRICE_SOURCES,
        "summary": _SUMMARY,
        "simulate_config": _closed(
            {**_COMMON_CONFIG, "days": {"type": "integer", "minimum": 1, "maximum": MAX_DAYS}}
        ),
        "backtest_config": _closed(
            {
                **_COMMON_CONFIG,
                "lookback": _POSITIVE,
                "hold": _POSITIVE,
                "rebalance": _POSITIVE,
                "top": _POSITIVE,
            }
        ),
        "evaluate_config": _closed(
            {
                "set_file": {"type": "string"},
                "manifest_sha256": {"type": "string", "pattern": _SHA256},
                "seeds": {"type": "array", "items": _SEED},
                "reproduce": _STRINGS,
            }
        ),
        "simulate_result": _SIMULATE_RESULT,
        "backtest_result": _BACKTEST_RESULT,
        "evaluate_result": _EVALUATE_RESULT,
        "help_result": _HELP_RESULT,
    },
}
