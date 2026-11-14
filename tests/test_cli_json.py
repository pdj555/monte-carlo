"""The ``--json`` contract, checked the way a script would use it: as a subprocess."""

from __future__ import annotations

import copy
import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path

import pytest
from jsonschema import Draft202012Validator

REPO_ROOT = Path(__file__).resolve().parents[1]
SCHEMA_PATH = REPO_ROOT / "schemas" / "cli-result.v1.schema.json"
OFFLINE = ["--source", "offline", "--data-path", "sample_data"]
SIMULATE = ["simulate", "AAPL", "MSFT", *OFFLINE, "--days", "5", "--scenarios", "50"]
BACKTEST = [
    "backtest", "AAPL", "MSFT", *OFFLINE,
    "--lookback", "5", "--hold", "3", "--top", "1", "--scenarios", "10",
]


def _run(*args: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, str(REPO_ROOT / "public_cli.py"), *args],
        capture_output=True,
        check=False,
        cwd=REPO_ROOT,
        env={**os.environ, "MPLBACKEND": "Agg"},
        text=True,
    )


@pytest.fixture(scope="module")
def validator() -> Draft202012Validator:
    schema = json.loads(SCHEMA_PATH.read_text(encoding="utf-8"))
    Draft202012Validator.check_schema(schema)
    return Draft202012Validator(schema)


def _document(completed: subprocess.CompletedProcess[str], validator) -> dict:
    """Parse stdout as exactly one schema-valid document."""

    document = json.loads(completed.stdout)
    errors = sorted(validator.iter_errors(document), key=lambda error: list(error.path))
    assert not errors, [error.message for error in errors]
    assert document["exit_code"] == completed.returncode
    return document


@pytest.fixture(scope="module")
def seeded_simulate() -> subprocess.CompletedProcess[str]:
    return _run(*SIMULATE, "--seed", "7", "--json")


def test_simulate_json_is_one_valid_document_with_exit_zero(seeded_simulate, validator):
    assert seeded_simulate.returncode == 0, seeded_simulate.stderr
    document = _document(seeded_simulate, validator)

    assert document["status"] == "ok"
    assert document["command"] == "simulate"
    assert document["error"] is None
    assert set(document["result"]["tickers"]) == {"AAPL", "MSFT"}
    assert document["result"]["action_plan"]["stance"]


def test_simulate_json_records_seed_config_and_input_hashes(seeded_simulate, validator):
    document = _document(seeded_simulate, validator)
    config = document["config"]

    assert config["seed"] == 7
    assert config["seed_source"] == "argument"
    assert config["tickers"] == ["AAPL", "MSFT"]
    assert (config["days"], config["scenarios"], config["model"]) == (5, 50, "historical")
    assert (config["source"], config["data_path"]) == ("offline", "sample_data")
    for ticker, source in document["result"]["price_sources"].items():
        csv_bytes = (REPO_ROOT / "sample_data" / f"{ticker}.csv").read_bytes()
        assert source["sha256"] == hashlib.sha256(csv_bytes).hexdigest()


def test_simulate_json_same_seed_gives_identical_bytes(seeded_simulate):
    again = _run(*SIMULATE, "--seed", "7", "--json")

    assert again.returncode == 0
    assert again.stdout == seeded_simulate.stdout


def test_simulate_json_different_seed_gives_different_numbers(seeded_simulate, validator):
    other = _document(_run(*SIMULATE, "--seed", "8", "--json"), validator)
    base = _document(seeded_simulate, validator)

    assert other["config"]["seed"] == 8
    assert other["result"]["tickers"] != base["result"]["tickers"]


def test_unseeded_simulate_reports_a_seed_that_replays_the_run(validator):
    first = _document(_run(*SIMULATE, "--json"), validator)
    config = first["config"]

    assert config["seed_source"] == "generated"
    assert config["reproduce"][0] == "monte-carlo"
    replay = _document(_run(*config["reproduce"][1:]), validator)
    assert replay["config"]["seed"] == config["seed"]
    assert replay["config"]["seed_source"] == "argument"
    assert replay["result"] == first["result"]


def test_backtest_json_is_valid_and_repeatable(validator):
    first = _run(*BACKTEST, "--seed", "7", "--json")
    second = _run(*BACKTEST, "--seed", "7", "--json")

    assert first.returncode == 0, first.stderr
    document = _document(first, validator)
    assert second.stdout == first.stdout
    assert document["config"]["seed"] == 7
    assert document["config"]["rebalance"] == 3
    summary = document["result"]["summary"]
    assert summary["periods"] == 1
    # One period has no spread, so the t-statistic is undefined: null, not NaN.
    assert summary["excess_return_t_stat"] is None
    assert "NaN" not in first.stdout
    row = document["result"]["rebalance_log"][0]
    assert isinstance(row["selected_tickers"], list)
    assert isinstance(row["selected_weights"], dict)


def test_evaluate_json_is_valid_and_matches_the_manifest(validator):
    completed = _run("evaluate", "evaluation_sets/sample-stability.json", "--json")

    assert completed.returncode == 0, completed.stderr
    document = _document(completed, validator)
    manifest_bytes = (REPO_ROOT / "evaluation_sets" / "sample-stability.json").read_bytes()
    manifest = json.loads(manifest_bytes)
    assert document["config"]["manifest_sha256"] == hashlib.sha256(manifest_bytes).hexdigest()
    assert document["config"]["seeds"] == manifest["seeds"]
    scorecard = document["result"]["scorecard"]
    expected_runs = len(manifest["models"]) * len(manifest["seeds"])
    assert scorecard["total_runs"] == expected_runs == len(document["result"]["outcomes"])
    assert scorecard["failed_runs"] == 0


def test_usage_error_exits_two_with_a_json_error_document(validator):
    completed = _run("simulate", "AAPL", "--days", "not-a-number", "--json")

    assert completed.returncode == 2
    document = _document(completed, validator)
    assert document["status"] == "error"
    assert document["command"] == "simulate"
    assert document["error"]["kind"] == "usage"
    assert "--days" in document["error"]["message"]
    assert document["result"] is None


def test_usage_error_without_json_exits_two_and_keeps_stdout_empty():
    completed = _run("simulate", "AAPL", "--days", "not-a-number")

    assert completed.returncode == 2
    assert completed.stdout == ""
    assert "argument --days: must be an integer" in completed.stderr


def test_unknown_flag_with_json_exits_two(validator):
    completed = _run("backtest", "AAPL", "--json", "--no-such-flag")

    assert completed.returncode == 2
    assert _document(completed, validator)["error"]["kind"] == "usage"


def test_rejected_input_exits_two_with_a_json_error_document(tmp_path, validator):
    completed = _run("evaluate", str(tmp_path / "missing.json"), "--json")

    assert completed.returncode == 2
    document = _document(completed, validator)
    assert document["error"]["kind"] == "invalid_input"
    assert document["result"] is None


def test_simulate_with_no_usable_ticker_exits_one(validator):
    completed = _run("simulate", "ZZZZ", *OFFLINE, "--seed", "7", "--json")

    assert completed.returncode == 1
    document = _document(completed, validator)
    assert document["status"] == "failed"
    assert document["error"]["kind"] == "no_results"
    assert document["result"]["tickers"] == {}
    assert document["result"]["errors"][0]["ticker"] == "ZZZZ"
    assert document["config"]["seed"] == 7


def test_evaluate_with_failed_runs_exits_one(tmp_path, validator):
    manifest = json.loads(
        (REPO_ROOT / "evaluation_sets" / "sample-stability.json").read_text(encoding="utf-8")
    )
    empty_dir = tmp_path / "no-prices"
    empty_dir.mkdir()
    manifest["sources"][0]["data_path"] = str(empty_dir)
    set_file = tmp_path / "failing.json"
    set_file.write_text(json.dumps(manifest), encoding="utf-8")

    completed = _run("evaluate", str(set_file), "--json")

    assert completed.returncode == 1
    document = _document(completed, validator)
    assert document["status"] == "failed"
    assert document["error"]["kind"] == "failed_runs"
    assert document["result"]["scorecard"]["failed_runs"] == 6


def test_text_output_stays_the_default(seeded_simulate):
    completed = _run(*SIMULATE, "--seed", "7")

    assert completed.returncode == 0
    assert completed.stdout.startswith("Stance: ")
    with pytest.raises(json.JSONDecodeError):
        json.loads(completed.stdout)


@pytest.mark.parametrize(
    "mutate",
    [
        lambda doc: doc.update(exit_code=1),
        lambda doc: doc["config"].pop("seed"),
        lambda doc: doc["config"].update(seed=-1),
        lambda doc: doc["result"].pop("price_sources"),
        lambda doc: doc["result"]["tickers"]["AAPL"].update(expected_return="high"),
        lambda doc: doc.update(extra=True),
        lambda doc: doc.update(error={"kind": "usage", "message": "x"}),
    ],
)
def test_schema_rejects_documents_that_break_the_contract(seeded_simulate, validator, mutate):
    document = json.loads(seeded_simulate.stdout)
    broken = copy.deepcopy(document)
    mutate(broken)

    assert validator.is_valid(document)
    assert not validator.is_valid(broken)


def _agent_block_commands() -> list[list[str]]:
    text = (REPO_ROOT / "AGENTS.md").read_text(encoding="utf-8")
    section = text.split("## For agents", 1)[1].split("\n## ", 1)[0]
    block = section.split("```bash", 1)[1].split("```", 1)[0]
    return [
        line.split()[1:]
        for line in block.splitlines()
        if line.startswith("monte-carlo ") and "--json" in line
    ]


def test_agents_md_json_commands_run_as_written(validator):
    commands = _agent_block_commands()

    assert {command[0] for command in commands} == {"simulate", "backtest", "evaluate"}
    for command in commands:
        completed = _run(*command)
        assert completed.returncode == 0, completed.stderr
        document = _document(completed, validator)
        assert document["command"] == command[0]
        assert document["status"] == "ok"
