from __future__ import annotations

import csv
import json
from pathlib import Path

import pytest

from evaluation import (
    EvaluationConfigError,
    _rank_correlation,
    evaluate_scenario_set,
    expand_evaluation_runs,
    format_evaluation_scorecard,
    load_evaluation_set,
    save_evaluation_report,
)


def write_set(tmp_path: Path, payload: dict[str, object]) -> Path:
    path = tmp_path / "evaluation.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


def base_set() -> dict[str, object]:
    return {
        "schema_version": 1,
        "name": "sample-stability",
        "days": 20,
        "scenarios": 250,
        "universes": [{"name": "core", "tickers": ["AAPL", "MSFT"]}],
        "models": ["historical"],
        "seeds": [7, 42],
        "sources": [{"name": "bundled", "mode": "offline", "data_path": "prices"}],
    }


def write_two_seed_set(tmp_path: Path) -> Path:
    return write_set(tmp_path, base_set())


def simulation_result(
    *,
    rankings: list[tuple[str, float, str, str, float]],
    top_pick: str | None,
    stance: str,
    errors: list[str] | None = None,
) -> dict[str, object]:
    ranking_data = {
        ticker: {
            "score": score,
            "recommendation": recommendation,
            "guardrail_reasons": reasons,
            "value_at_risk_95_pct": downside,
        }
        for ticker, score, recommendation, reasons, downside in rankings
    }
    return {
        "report": {
            "rankings": ranking_data,
            "action_plan": {
                "stance": stance,
                "primary_pick": None if top_pick is None else {"ticker": top_pick},
            },
            "results": {ticker: {"summary": {}} for ticker in ranking_data},
            "errors": [] if errors is None else errors,
        },
        "price_sources": {
            ticker: {"mode": "offline", "detail": "fixture"}
            for ticker in ranking_data
        },
    }


def test_load_evaluation_set_resolves_paths_and_expands_in_manifest_order(tmp_path):
    set_path = write_set(
        tmp_path,
        {
            "schema_version": 1,
            "name": "core-stability",
            "days": 20,
            "scenarios": 250,
            "universes": [
                {"name": "tech", "tickers": ["aapl", "MSFT"]},
                {"name": "single", "tickers": ["JPM"]},
            ],
            "models": ["historical", "gbm"],
            "seeds": [7, 42],
            "sources": [
                {"name": "bundled", "mode": "offline", "data_path": "../prices"},
                {"name": "live", "mode": "online"},
            ],
        },
    )

    evaluation_set = load_evaluation_set(set_path)
    runs = expand_evaluation_runs(evaluation_set)

    assert len(runs) == 16
    assert evaluation_set.universes[0].tickers == ("AAPL", "MSFT")
    assert evaluation_set.sources[0].data_path == (tmp_path / "../prices").resolve()
    assert [
        (run.universe_name, run.model, run.seed, run.source_name)
        for run in runs[:4]
    ] == [
        ("tech", "historical", 7, "bundled"),
        ("tech", "historical", 7, "live"),
        ("tech", "historical", 42, "bundled"),
        ("tech", "historical", 42, "live"),
    ]


@pytest.mark.parametrize(
    ("mutate", "field"),
    [
        (lambda data: data.update({"unexpected": True}), "unexpected"),
        (lambda data: data.pop("schema_version"), "schema_version"),
        (lambda data: data.update({"schema_version": 2}), "schema_version"),
        (lambda data: data.update({"schema_version": 1.0}), "schema_version"),
        (lambda data: data.update({"models": ["historical", "historical"]}), "models"),
        (lambda data: data.update({"universes": []}), "universes"),
        (lambda data: data["universes"][0].update({"tickers": []}), "tickers"),
        (lambda data: data.update({"models": []}), "models"),
        (lambda data: data.update({"seeds": []}), "seeds"),
        (lambda data: data.update({"sources": []}), "sources"),
        (lambda data: data["universes"][0].update({"tickers": ["BAD TICKER"]}), "tickers"),
        (lambda data: data.update({"seeds": [-1]}), "seeds"),
        (lambda data: data.update({"seeds": [True]}), "seeds"),
        (lambda data: data.update({"models": ["prediction"]}), "models"),
        (lambda data: data["sources"][0].update({"mode": "invalid"}), "mode"),
        (lambda data: data["sources"][0].pop("data_path"), "data_path"),
        (lambda data: data.update({"days": 0}), "days"),
        (lambda data: data.update({"scenarios": False}), "scenarios"),
        (lambda data: data.update({"seeds": list(range(101))}), "runs"),
    ],
)
def test_load_evaluation_set_rejects_invalid_manifests(tmp_path, mutate, field):
    payload = base_set()
    mutate(payload)

    with pytest.raises(EvaluationConfigError) as exc:
        load_evaluation_set(write_set(tmp_path, payload))

    message = str(exc.value)
    assert field in message
    assert any(word in message for word in ("Provide", "Use", "Remove", "Reduce"))


def test_load_evaluation_set_rejects_names_that_can_collide_in_run_ids(tmp_path):
    payload = base_set()
    payload["universes"] = [
        {"name": "core", "tickers": ["AAPL"]},
        {"name": "core/historical/seed-7/bundled", "tickers": ["MSFT"]},
    ]
    payload["seeds"] = [7]
    payload["sources"] = [
        {
            "name": "bundled/historical/seed-7/local",
            "mode": "offline",
            "data_path": "prices",
        },
        {"name": "local", "mode": "offline", "data_path": "prices"},
    ]

    with pytest.raises(EvaluationConfigError) as exc:
        load_evaluation_set(write_set(tmp_path, payload))

    assert "name" in str(exc.value)
    assert "safe token" in str(exc.value)


def test_load_evaluation_set_normalizes_non_utf_manifest_errors(tmp_path):
    path = tmp_path / "evaluation.json"
    path.write_bytes(b'{"schema_version": 1, "name": "\xff"}')

    with pytest.raises(EvaluationConfigError) as exc:
        load_evaluation_set(path)

    assert str(exc.value) == (
        "manifest: invalid JSON. Provide a valid JSON evaluation manifest"
    )


def test_evaluate_scenario_set_reports_stability_reliability_and_downside(tmp_path):
    evaluation_set = load_evaluation_set(write_two_seed_set(tmp_path))

    def runner(run):
        if run.seed == 7:
            return simulation_result(
                rankings=[
                    ("AAPL", 12.0, "BUY", "", 0.08),
                    ("MSFT", 8.0, "AVOID", "expected_return<0.0%", 0.14),
                ],
                top_pick="AAPL",
                stance="SELECTIVE",
            )
        return simulation_result(
            rankings=[
                ("MSFT", 11.0, "BUY", "", 0.10),
                ("AAPL", 9.0, "BUY", "", 0.20),
            ],
            top_pick="MSFT",
            stance="RISK_ON",
        )

    report = evaluate_scenario_set(evaluation_set, runner)

    assert report.scorecard.run_success_rate == pytest.approx(1.0)
    assert report.scorecard.ticker_success_rate == pytest.approx(1.0)
    assert report.scorecard.mean_rank_correlation == pytest.approx(-1.0)
    assert report.scorecard.top_pick_consistency == pytest.approx(0.5)
    assert report.scorecard.guardrail_rejection_rate == pytest.approx(0.25)
    assert report.scorecard.no_trade_rate == pytest.approx(0.0)
    assert report.scorecard.worst_var_95_pct == pytest.approx(0.20)
    assert report.scorecard.source_reliability["bundled"]["completed_runs"] == 2


def test_evaluation_counts_defensive_hold_cash_as_no_trade(tmp_path):
    payload = base_set()
    payload["seeds"] = [7]
    evaluation_set = load_evaluation_set(write_set(tmp_path, payload))

    report = evaluate_scenario_set(
        evaluation_set,
        lambda run: simulation_result(
            rankings=[
                ("AAPL", -1.0, "AVOID", "expected_return<0.0%", 0.08),
                ("MSFT", -2.0, "AVOID", "expected_return<0.0%", 0.14),
            ],
            top_pick=None,
            stance="DEFENSIVE",
        ),
    )

    assert report.outcomes[0].status == "completed"
    assert report.outcomes[0].top_pick is None
    assert report.scorecard.no_trade_rate == pytest.approx(1.0)


def test_rank_correlation_hashes_a_large_right_universe_once():
    class CountingTicker(str):
        hash_calls = 0

        def __hash__(self):
            type(self).hash_calls += 1
            return super().__hash__()

    left = tuple(CountingTicker(f"T{index:03}") for index in range(250))
    right = tuple(reversed(left))

    correlation = _rank_correlation(left, right)

    assert correlation == pytest.approx(-1.0)
    assert CountingTicker.hash_calls < 5_000


def test_scorecard_source_reliability_is_recursively_immutable(tmp_path):
    evaluation_set = load_evaluation_set(write_two_seed_set(tmp_path))
    report = evaluate_scenario_set(
        evaluation_set,
        lambda run: simulation_result(
            rankings=[("AAPL", 1.0, "BUY", "", 0.08)],
            top_pick="AAPL",
            stance="SELECTIVE",
        ),
    )

    with pytest.raises(TypeError):
        report.scorecard.source_reliability["new"] = {}
    with pytest.raises(TypeError):
        report.scorecard.source_reliability["bundled"]["completed_runs"] = 0

    serialized = report.scorecard.to_dict()
    serialized["source_reliability"]["bundled"]["completed_runs"] = 0
    assert report.scorecard.source_reliability["bundled"]["completed_runs"] == 2


def test_evaluation_isolates_runner_errors_and_empty_rankings(tmp_path):
    evaluation_set = load_evaluation_set(write_two_seed_set(tmp_path))

    def runner(run):
        if run.seed == 7:
            raise RuntimeError("fixture exploded")
        return simulation_result(
            rankings=[],
            top_pick=None,
            stance="DEFENSIVE",
            errors=[
                {"ticker": "AAPL", "error": "no data"},
                {"ticker": "MSFT", "error": "bad history"},
            ],
        )

    report = evaluate_scenario_set(evaluation_set, runner)

    assert [outcome.status for outcome in report.outcomes] == ["failed", "failed"]
    assert report.outcomes[0].error == "fixture exploded"
    assert report.outcomes[1].error == "AAPL: no data; MSFT: bad history"
    assert report.scorecard.completed_runs == 0


def test_evaluation_rank_correlation_and_format_handle_unavailable_metrics(tmp_path):
    payload = base_set()
    payload["seeds"] = [7]
    evaluation_set = load_evaluation_set(write_set(tmp_path, payload))
    report = evaluate_scenario_set(
        evaluation_set,
        lambda run: simulation_result(
            rankings=[("AAPL", 1.0, "BUY", "", 0.08)],
            top_pick="AAPL",
            stance="SELECTIVE",
        ),
    )

    assert report.scorecard.mean_rank_correlation is None
    assert "Mean rank correlation: n/a" in format_evaluation_scorecard(report)


def test_save_evaluation_report_persists_scorecard_runs_and_report(tmp_path):
    evaluation_set = load_evaluation_set(write_two_seed_set(tmp_path))
    report = evaluate_scenario_set(
        evaluation_set,
        lambda run: simulation_result(
            rankings=[
                ("AAPL", 1.0, "BUY", "", 0.08),
                ("MSFT", 0.5, "AVOID", "risk", 0.14),
            ],
            top_pick="AAPL",
            stance="SELECTIVE",
        ),
    )

    scorecard_path, runs_path, report_path = save_evaluation_report(
        report, tmp_path / "nested" / "outputs"
    )

    assert [path.name for path in (scorecard_path, runs_path, report_path)] == [
        "scorecard.md",
        "runs.csv",
        "report.json",
    ]
    assert scorecard_path.read_text(encoding="utf-8") == (
        format_evaluation_scorecard(report) + "\n"
    )
    with report_path.open(encoding="utf-8") as handle:
        persisted = json.load(handle)
    assert persisted["evaluation_set"]["name"] == "sample-stability"
    assert persisted["scorecard"]["completed_runs"] == 2
    with runs_path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    assert len(rows) == 2
    assert rows[0]["rank_order"] == "AAPL > MSFT"
