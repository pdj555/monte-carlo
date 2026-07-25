"""Versioned scenario-evaluation contracts and scorecard aggregation."""

from __future__ import annotations

import csv
import hashlib
import json
import math
import re
from collections import Counter
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable, Mapping


EVALUATION_SCHEMA_VERSION = 1
MAX_EVALUATION_RUNS = 100
SUPPORTED_MODELS = ("historical", "gbm")
SUPPORTED_SOURCE_MODES = ("auto", "offline", "online")

_TICKER_PATTERN = re.compile(r"^[A-Z0-9][A-Z0-9.^=-]{0,14}$")


class EvaluationConfigError(ValueError):
    """Raised when an evaluation manifest cannot be safely expanded."""


@dataclass(frozen=True)
class EvaluationUniverse:
    name: str
    tickers: tuple[str, ...]

    def to_dict(self) -> dict[str, object]:
        return {"name": self.name, "tickers": list(self.tickers)}


@dataclass(frozen=True)
class EvaluationSource:
    name: str
    mode: str
    data_path: Path | None = None

    def to_dict(self) -> dict[str, object]:
        return {
            "name": self.name,
            "mode": self.mode,
            "data_path": str(self.data_path) if self.data_path is not None else None,
        }


@dataclass(frozen=True)
class EvaluationSet:
    name: str
    days: int
    scenarios: int
    universes: tuple[EvaluationUniverse, ...]
    models: tuple[str, ...]
    seeds: tuple[int, ...]
    sources: tuple[EvaluationSource, ...]
    manifest_path: Path
    manifest_sha256: str

    def to_dict(self) -> dict[str, object]:
        return {
            "name": self.name,
            "days": self.days,
            "scenarios": self.scenarios,
            "universes": [universe.to_dict() for universe in self.universes],
            "models": list(self.models),
            "seeds": list(self.seeds),
            "sources": [source.to_dict() for source in self.sources],
            "manifest_path": str(self.manifest_path),
            "manifest_sha256": self.manifest_sha256,
        }


@dataclass(frozen=True)
class EvaluationRun:
    run_id: str
    universe_name: str
    tickers: tuple[str, ...]
    model: str
    seed: int
    source_name: str
    source_mode: str
    data_path: Path | None
    days: int
    scenarios: int

    def to_dict(self) -> dict[str, object]:
        return {
            "run_id": self.run_id,
            "universe_name": self.universe_name,
            "tickers": list(self.tickers),
            "model": self.model,
            "seed": self.seed,
            "source_name": self.source_name,
            "source_mode": self.source_mode,
            "data_path": str(self.data_path) if self.data_path is not None else None,
            "days": self.days,
            "scenarios": self.scenarios,
        }


@dataclass(frozen=True)
class EvaluationRunOutcome:
    run_id: str
    universe_name: str
    model: str
    seed: int
    source_name: str
    source_mode: str
    requested_tickers: int
    completed_tickers: int
    status: str
    error: str | None
    stance: str | None
    top_pick: str | None
    rank_order: tuple[str, ...]
    guardrail_rejections: int
    worst_var_95_pct: float | None

    def to_dict(self) -> dict[str, object]:
        return {
            "run_id": self.run_id,
            "universe_name": self.universe_name,
            "model": self.model,
            "seed": self.seed,
            "source_name": self.source_name,
            "source_mode": self.source_mode,
            "requested_tickers": self.requested_tickers,
            "completed_tickers": self.completed_tickers,
            "status": self.status,
            "error": self.error,
            "stance": self.stance,
            "top_pick": self.top_pick,
            "rank_order": list(self.rank_order),
            "guardrail_rejections": self.guardrail_rejections,
            "worst_var_95_pct": self.worst_var_95_pct,
        }


@dataclass(frozen=True)
class EvaluationScorecard:
    total_runs: int
    completed_runs: int
    failed_runs: int
    run_success_rate: float
    ticker_success_rate: float
    mean_rank_correlation: float | None
    top_pick_consistency: float | None
    guardrail_rejection_rate: float
    no_trade_rate: float
    worst_var_95_pct: float | None
    source_reliability: dict[str, dict[str, int | float]]

    def to_dict(self) -> dict[str, object]:
        return {
            "total_runs": self.total_runs,
            "completed_runs": self.completed_runs,
            "failed_runs": self.failed_runs,
            "run_success_rate": self.run_success_rate,
            "ticker_success_rate": self.ticker_success_rate,
            "mean_rank_correlation": self.mean_rank_correlation,
            "top_pick_consistency": self.top_pick_consistency,
            "guardrail_rejection_rate": self.guardrail_rejection_rate,
            "no_trade_rate": self.no_trade_rate,
            "worst_var_95_pct": self.worst_var_95_pct,
            "source_reliability": self.source_reliability,
        }


@dataclass(frozen=True)
class EvaluationReport:
    schema_version: int
    evaluation_set: EvaluationSet
    generated_at: str
    outcomes: tuple[EvaluationRunOutcome, ...]
    scorecard: EvaluationScorecard

    def to_dict(self) -> dict[str, object]:
        return {
            "schema_version": self.schema_version,
            "evaluation_set": self.evaluation_set.to_dict(),
            "generated_at": self.generated_at,
            "runs": [run.to_dict() for run in expand_evaluation_runs(self.evaluation_set)],
            "outcomes": [outcome.to_dict() for outcome in self.outcomes],
            "scorecard": self.scorecard.to_dict(),
        }


def _config_error(field: str, message: str, fix: str) -> None:
    raise EvaluationConfigError(f"{field}: {message}. {fix}")


def _strict_keys(value: object, allowed: set[str], field: str) -> Mapping[str, object]:
    if not isinstance(value, Mapping):
        _config_error(field, "must be an object", "Provide a JSON object")
    unknown = sorted(set(value) - allowed)
    if unknown:
        _config_error(
            field,
            f"contains unsupported key {unknown[0]!r}",
            "Remove unsupported keys",
        )
    return value


def _nonempty_string(value: object, field: str) -> str:
    if not isinstance(value, str) or not value.strip():
        _config_error(field, "must be a non-empty string", "Provide a non-empty string")
    return value.strip()


def _positive_int(value: object, field: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        _config_error(field, "must be a positive integer", "Provide an integer greater than zero")
    return value


def _required(mapping: Mapping[str, object], key: str, field: str) -> object:
    if key not in mapping:
        _config_error(field, "is required", "Provide this field")
    return mapping[key]


def _unique(values: tuple[object, ...], field: str) -> None:
    if len(set(values)) != len(values):
        _config_error(field, "contains duplicate values", "Use each value only once")


def load_evaluation_set(path: str | Path) -> EvaluationSet:
    """Load and strictly validate a version-one evaluation manifest."""

    manifest_path = Path(path).resolve()
    try:
        manifest_bytes = manifest_path.read_bytes()
    except OSError as exc:
        raise EvaluationConfigError(
            f"manifest_path: cannot read {manifest_path}. Provide a readable JSON manifest"
        ) from exc
    try:
        document = json.loads(manifest_bytes)
    except (TypeError, json.JSONDecodeError) as exc:
        raise EvaluationConfigError(
            "manifest: invalid JSON. Provide a valid JSON evaluation manifest"
        ) from exc

    top_level = _strict_keys(
        document,
        {
            "schema_version",
            "name",
            "days",
            "scenarios",
            "universes",
            "models",
            "seeds",
            "sources",
        },
        "manifest",
    )
    schema_version = _required(top_level, "schema_version", "schema_version")
    if (
        isinstance(schema_version, bool)
        or not isinstance(schema_version, int)
        or schema_version != EVALUATION_SCHEMA_VERSION
    ):
        _config_error(
            "schema_version",
            f"must be {EVALUATION_SCHEMA_VERSION}",
            f"Use schema_version {EVALUATION_SCHEMA_VERSION}",
        )

    name = _nonempty_string(_required(top_level, "name", "name"), "name")
    days = _positive_int(_required(top_level, "days", "days"), "days")
    scenarios = _positive_int(
        _required(top_level, "scenarios", "scenarios"), "scenarios"
    )

    raw_universes = _required(top_level, "universes", "universes")
    if not isinstance(raw_universes, list) or not raw_universes:
        _config_error("universes", "must be a non-empty array", "Provide at least one universe")
    universes: list[EvaluationUniverse] = []
    for index, raw_universe in enumerate(raw_universes):
        field = f"universes[{index}]"
        universe = _strict_keys(raw_universe, {"name", "tickers"}, field)
        universe_name = _nonempty_string(
            _required(universe, "name", f"{field}.name"), f"{field}.name"
        )
        raw_tickers = _required(universe, "tickers", f"{field}.tickers")
        if not isinstance(raw_tickers, list) or not raw_tickers:
            _config_error(
                f"{field}.tickers",
                "must be a non-empty array",
                "Provide at least one ticker",
            )
        tickers: list[str] = []
        for ticker_index, raw_ticker in enumerate(raw_tickers):
            ticker_field = f"{field}.tickers[{ticker_index}]"
            ticker = _nonempty_string(raw_ticker, ticker_field).upper()
            if not _TICKER_PATTERN.fullmatch(ticker):
                _config_error(
                    ticker_field,
                    "is not a supported ticker token",
                    "Use 1-15 uppercase letters, digits, or .^=- characters",
                )
            tickers.append(ticker)
        _unique(tuple(tickers), f"{field}.tickers")
        universes.append(EvaluationUniverse(universe_name, tuple(tickers)))
    _unique(tuple(universe.name for universe in universes), "universes.name")

    raw_models = _required(top_level, "models", "models")
    if not isinstance(raw_models, list) or not raw_models:
        _config_error("models", "must be a non-empty array", "Provide at least one model")
    models = tuple(_nonempty_string(value, "models") for value in raw_models)
    for model in models:
        if model not in SUPPORTED_MODELS:
            _config_error(
                "models",
                f"contains unsupported model {model!r}",
                f"Use one of {', '.join(SUPPORTED_MODELS)}",
            )
    _unique(models, "models")

    raw_seeds = _required(top_level, "seeds", "seeds")
    if not isinstance(raw_seeds, list) or not raw_seeds:
        _config_error("seeds", "must be a non-empty array", "Provide at least one seed")
    seeds: list[int] = []
    for index, raw_seed in enumerate(raw_seeds):
        if isinstance(raw_seed, bool) or not isinstance(raw_seed, int) or raw_seed < 0:
            _config_error(
                f"seeds[{index}]",
                "must be a non-negative integer",
                "Provide zero or a positive integer seed",
            )
        seeds.append(raw_seed)
    _unique(tuple(seeds), "seeds")

    raw_sources = _required(top_level, "sources", "sources")
    if not isinstance(raw_sources, list) or not raw_sources:
        _config_error("sources", "must be a non-empty array", "Provide at least one source")
    sources: list[EvaluationSource] = []
    for index, raw_source in enumerate(raw_sources):
        field = f"sources[{index}]"
        source = _strict_keys(raw_source, {"name", "mode", "data_path"}, field)
        source_name = _nonempty_string(_required(source, "name", f"{field}.name"), f"{field}.name")
        source_mode = _nonempty_string(_required(source, "mode", f"{field}.mode"), f"{field}.mode")
        if source_mode not in SUPPORTED_SOURCE_MODES:
            _config_error(
                f"{field}.mode",
                f"is unsupported mode {source_mode!r}",
                f"Use one of {', '.join(SUPPORTED_SOURCE_MODES)}",
            )
        raw_data_path = source.get("data_path")
        if source_mode == "offline" and raw_data_path is None:
            _config_error(
                f"{field}.data_path",
                "is required for offline sources",
                "Provide a local data_path",
            )
        if raw_data_path is not None and not isinstance(raw_data_path, str):
            _config_error(
                f"{field}.data_path",
                "must be a string when provided",
                "Provide a relative or absolute path string",
            )
        data_path = (
            (manifest_path.parent / raw_data_path).resolve()
            if isinstance(raw_data_path, str)
            else None
        )
        sources.append(EvaluationSource(source_name, source_mode, data_path))
    _unique(tuple(source.name for source in sources), "sources.name")

    count = len(universes) * len(models) * len(seeds) * len(sources)
    if count > MAX_EVALUATION_RUNS:
        _config_error(
            "runs",
            f"expands to {count}, above the {MAX_EVALUATION_RUNS}-run limit",
            "Reduce universes, models, seeds, or sources",
        )

    return EvaluationSet(
        name=name,
        days=days,
        scenarios=scenarios,
        universes=tuple(universes),
        models=models,
        seeds=tuple(seeds),
        sources=tuple(sources),
        manifest_path=manifest_path,
        manifest_sha256=hashlib.sha256(manifest_bytes).hexdigest(),
    )


def expand_evaluation_runs(evaluation_set: EvaluationSet) -> tuple[EvaluationRun, ...]:
    """Expand an evaluation matrix in manifest order."""

    runs: list[EvaluationRun] = []
    for universe in evaluation_set.universes:
        for model in evaluation_set.models:
            for seed in evaluation_set.seeds:
                for source in evaluation_set.sources:
                    run_id = f"{universe.name}/{model}/seed-{seed}/{source.name}"
                    runs.append(
                        EvaluationRun(
                            run_id=run_id,
                            universe_name=universe.name,
                            tickers=universe.tickers,
                            model=model,
                            seed=seed,
                            source_name=source.name,
                            source_mode=source.mode,
                            data_path=source.data_path,
                            days=evaluation_set.days,
                            scenarios=evaluation_set.scenarios,
                        )
                    )
    return tuple(runs)


def _mapping_at(value: object, key: str) -> Mapping[str, object]:
    if not isinstance(value, Mapping):
        return {}
    nested = value.get(key)
    return nested if isinstance(nested, Mapping) else {}


def _number(value: object) -> float | None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    number = float(value)
    return number if math.isfinite(number) else None


def _failed_outcome(run: EvaluationRun, error: str) -> EvaluationRunOutcome:
    return EvaluationRunOutcome(
        run_id=run.run_id,
        universe_name=run.universe_name,
        model=run.model,
        seed=run.seed,
        source_name=run.source_name,
        source_mode=run.source_mode,
        requested_tickers=len(run.tickers),
        completed_tickers=0,
        status="failed",
        error=error,
        stance=None,
        top_pick=None,
        rank_order=(),
        guardrail_rejections=0,
        worst_var_95_pct=None,
    )


def _format_simulation_errors(errors: object) -> str:
    if isinstance(errors, (list, tuple)):
        messages: list[str] = []
        for error in errors:
            if isinstance(error, Mapping):
                ticker = error.get("ticker")
                message = error.get("error")
                if ticker and message:
                    messages.append(f"{ticker}: {message}")
                    continue
            if str(error):
                messages.append(str(error))
        return "; ".join(messages)
    return str(errors) if errors else ""


def _normalise_outcome(run: EvaluationRun, result: Mapping[str, object]) -> EvaluationRunOutcome:
    report = _mapping_at(result, "report")
    rankings = report.get("rankings")
    errors = report.get("errors")
    if not isinstance(rankings, Mapping) or not rankings:
        message = _format_simulation_errors(errors) or "simulation returned no ranked tickers"
        return _failed_outcome(run, message or "simulation returned no ranked tickers")

    action_plan = _mapping_at(report, "action_plan")
    primary_pick = action_plan.get("primary_pick")
    top_pick = None
    if isinstance(primary_pick, Mapping) and isinstance(primary_pick.get("ticker"), str):
        top_pick = primary_pick["ticker"]
    stance = action_plan.get("stance")
    stance_value = stance if isinstance(stance, str) else None
    results = report.get("results")
    completed_tickers = len(results) if isinstance(results, Mapping) else 0
    rank_order = tuple(str(ticker) for ticker in rankings)
    guardrail_rejections = 0
    downside_values: list[float] = []
    for row in rankings.values():
        if not isinstance(row, Mapping):
            continue
        if row.get("recommendation") == "AVOID" and row.get("guardrail_reasons"):
            guardrail_rejections += 1
        downside = _number(row.get("value_at_risk_95_pct"))
        if downside is not None:
            downside_values.append(downside)
    return EvaluationRunOutcome(
        run_id=run.run_id,
        universe_name=run.universe_name,
        model=run.model,
        seed=run.seed,
        source_name=run.source_name,
        source_mode=run.source_mode,
        requested_tickers=len(run.tickers),
        completed_tickers=completed_tickers,
        status="completed",
        error=None,
        stance=stance_value,
        top_pick=top_pick,
        rank_order=rank_order,
        guardrail_rejections=guardrail_rejections,
        worst_var_95_pct=max(downside_values) if downside_values else None,
    )


def _rank_correlation(left: tuple[str, ...], right: tuple[str, ...]) -> float | None:
    common = [ticker for ticker in left if ticker in set(right)]
    if len(common) < 2:
        return None
    left_positions = {ticker: index + 1 for index, ticker in enumerate(left)}
    right_positions = {ticker: index + 1 for index, ticker in enumerate(right)}
    values_left = [left_positions[ticker] for ticker in common]
    values_right = [right_positions[ticker] for ticker in common]
    mean_left = sum(values_left) / len(values_left)
    mean_right = sum(values_right) / len(values_right)
    numerator = sum(
        (left_value - mean_left) * (right_value - mean_right)
        for left_value, right_value in zip(values_left, values_right)
    )
    denominator_left = sum((value - mean_left) ** 2 for value in values_left)
    denominator_right = sum((value - mean_right) ** 2 for value in values_right)
    denominator = math.sqrt(denominator_left * denominator_right)
    return numerator / denominator if denominator else None


def _scorecard(outcomes: tuple[EvaluationRunOutcome, ...]) -> EvaluationScorecard:
    total_runs = len(outcomes)
    completed = tuple(outcome for outcome in outcomes if outcome.status == "completed")
    completed_runs = len(completed)
    requested_tickers = sum(outcome.requested_tickers for outcome in outcomes)
    completed_tickers = sum(outcome.completed_tickers for outcome in completed)
    correlations: list[float] = []
    outcomes_by_universe: dict[str, list[EvaluationRunOutcome]] = {}
    for outcome in completed:
        outcomes_by_universe.setdefault(outcome.universe_name, []).append(outcome)
    for universe_outcomes in outcomes_by_universe.values():
        for index, left in enumerate(universe_outcomes):
            for right in universe_outcomes[index + 1 :]:
                correlation = _rank_correlation(left.rank_order, right.rank_order)
                if correlation is not None:
                    correlations.append(correlation)
    consistency: list[float] = []
    for universe_outcomes in outcomes_by_universe.values():
        picks = Counter(outcome.top_pick for outcome in universe_outcomes)
        consistency.append(max(picks.values()) / len(universe_outcomes))
    guardrails = sum(outcome.guardrail_rejections for outcome in completed)
    downside_values = [
        outcome.worst_var_95_pct
        for outcome in completed
        if outcome.worst_var_95_pct is not None
    ]
    source_reliability: dict[str, dict[str, int | float]] = {}
    for source_name in dict.fromkeys(outcome.source_name for outcome in outcomes):
        source_outcomes = [
            outcome for outcome in outcomes if outcome.source_name == source_name
        ]
        source_completed = sum(
            outcome.status == "completed" for outcome in source_outcomes
        )
        source_total = len(source_outcomes)
        source_reliability[source_name] = {
            "total_runs": source_total,
            "completed_runs": source_completed,
            "failed_runs": source_total - source_completed,
            "run_success_rate": source_completed / source_total if source_total else 0.0,
        }
    return EvaluationScorecard(
        total_runs=total_runs,
        completed_runs=completed_runs,
        failed_runs=total_runs - completed_runs,
        run_success_rate=completed_runs / total_runs if total_runs else 0.0,
        ticker_success_rate=(
            completed_tickers / requested_tickers if requested_tickers else 0.0
        ),
        mean_rank_correlation=(sum(correlations) / len(correlations) if correlations else None),
        top_pick_consistency=(sum(consistency) / len(consistency) if consistency else None),
        guardrail_rejection_rate=(
            guardrails / completed_tickers if completed_tickers else 0.0
        ),
        no_trade_rate=(
            sum(outcome.stance == "NO_TRADE" for outcome in completed) / completed_runs
            if completed_runs
            else 0.0
        ),
        worst_var_95_pct=max(downside_values) if downside_values else None,
        source_reliability=source_reliability,
    )


def evaluate_scenario_set(
    evaluation_set: EvaluationSet,
    runner: Callable[[EvaluationRun], Mapping[str, object]],
) -> EvaluationReport:
    """Run a deterministic matrix, isolating each simulation failure."""

    outcomes: list[EvaluationRunOutcome] = []
    for run in expand_evaluation_runs(evaluation_set):
        try:
            result = runner(run)
            if not isinstance(result, Mapping):
                raise TypeError("runner returned a non-mapping result")
            outcomes.append(_normalise_outcome(run, result))
        except Exception as exc:
            outcomes.append(_failed_outcome(run, str(exc) or type(exc).__name__))
    frozen_outcomes = tuple(outcomes)
    return EvaluationReport(
        schema_version=EVALUATION_SCHEMA_VERSION,
        evaluation_set=evaluation_set,
        generated_at=datetime.now(timezone.utc).isoformat(),
        outcomes=frozen_outcomes,
        scorecard=_scorecard(frozen_outcomes),
    )


def _percentage(value: float | None) -> str:
    return "n/a" if value is None else f"{value:.1%}"


def format_evaluation_scorecard(report: EvaluationReport) -> str:
    """Render the concise, operator-facing evaluation scorecard."""

    scorecard = report.scorecard
    lines = [
        f"Evaluation set: {report.evaluation_set.name}",
        (
            f"Runs: {scorecard.completed_runs}/{scorecard.total_runs} complete "
            f"({_percentage(scorecard.run_success_rate)})"
        ),
        f"Ticker coverage: {_percentage(scorecard.ticker_success_rate)}",
        f"Mean rank correlation: {_percentage(scorecard.mean_rank_correlation)}",
        f"Top-pick consistency: {_percentage(scorecard.top_pick_consistency)}",
        f"Guardrail rejections: {_percentage(scorecard.guardrail_rejection_rate)}",
        f"No-trade runs: {_percentage(scorecard.no_trade_rate)}",
        f"Worst observed 95% downside: {_percentage(scorecard.worst_var_95_pct)}",
        "Source reliability:",
    ]
    for source_name, reliability in scorecard.source_reliability.items():
        lines.append(
            f"  {source_name}: {reliability['completed_runs']}/"
            f"{reliability['total_runs']} complete "
            f"({_percentage(float(reliability['run_success_rate']))})"
        )
    return "\n".join(lines)


def save_evaluation_report(
    report: EvaluationReport,
    output_dir: str | Path,
) -> tuple[Path, Path, Path]:
    """Persist a scorecard, ordered run rows, and machine-readable report."""

    directory = Path(output_dir)
    directory.mkdir(parents=True, exist_ok=True)
    scorecard_path = directory / "scorecard.md"
    runs_path = directory / "runs.csv"
    report_path = directory / "report.json"
    scorecard_path.write_text(format_evaluation_scorecard(report) + "\n", encoding="utf-8")
    fieldnames = list(EvaluationRunOutcome.__dataclass_fields__)
    with runs_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for outcome in report.outcomes:
            row = outcome.to_dict()
            row["rank_order"] = " > ".join(outcome.rank_order)
            writer.writerow(row)
    with report_path.open("w", encoding="utf-8") as handle:
        json.dump(report.to_dict(), handle, indent=2)
        handle.write("\n")
    return scorecard_path, runs_path, report_path
