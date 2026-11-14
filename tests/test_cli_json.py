"""The ``--json`` contract, checked the way a script would use it: as a subprocess."""

from __future__ import annotations

import copy
import hashlib
import json
import math
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


GOLDEN_PATH = REPO_ROOT / "tests" / "golden" / "simulate-seed42.json"
CLI = [sys.executable, str(REPO_ROOT / "public_cli.py")]
ENV = {**os.environ, "MPLBACKEND": "Agg"}


def _run(*args: str, cwd: Path = REPO_ROOT) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [*CLI, *args], capture_output=True, check=False, cwd=cwd, env=ENV, text=True
    )


def _run_patched(patch: str, *args: str) -> subprocess.CompletedProcess[str]:
    """Run the CLI in a child process after applying ``patch`` to ``public_cli``."""

    code = f"import sys, public_cli\n{patch}\nsys.exit(public_cli.main(sys.argv[1:]))"
    return subprocess.run(
        [sys.executable, "-c", code, *args],
        capture_output=True,
        check=False,
        cwd=REPO_ROOT,
        env=ENV,
        text=True,
    )


def _differences(expected, actual, path: str = "$") -> list[str]:
    """Where two documents differ; floats may differ by rounding only."""

    if isinstance(expected, bool) or isinstance(actual, bool):
        return [] if expected is actual else [f"{path}: {expected!r} != {actual!r}"]
    if isinstance(expected, (int, float)) and isinstance(actual, (int, float)):
        close = math.isclose(expected, actual, rel_tol=1e-9, abs_tol=1e-12)
        return [] if close else [f"{path}: {expected!r} != {actual!r}"]
    if isinstance(expected, dict) and isinstance(actual, dict):
        found = [f"{path}: keys {sorted(set(expected) ^ set(actual))}"] * (
            set(expected) != set(actual)
        )
        for key in sorted(set(expected) & set(actual)):
            found += _differences(expected[key], actual[key], f"{path}.{key}")
        return found
    if isinstance(expected, list) and isinstance(actual, list) and len(expected) == len(actual):
        found = []
        for index, (left, right) in enumerate(zip(expected, actual)):
            found += _differences(left, right, f"{path}[{index}]")
        return found
    return [] if expected == actual else [f"{path}: {expected!r} != {actual!r}"]


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


def _one_row_prices(tmp_path: Path) -> Path:
    data_dir = tmp_path / "short"
    data_dir.mkdir()
    (data_dir / "AAPL.csv").write_text("Date,Close\n2024-01-02,185.0\n", encoding="utf-8")
    return data_dir


def test_simulate_with_prices_too_short_exits_one(tmp_path, validator):
    data_dir = _one_row_prices(tmp_path)
    completed = _run(
        "simulate", "AAPL", "--source", "offline", "--data-path", str(data_dir),
        "--seed", "7", "--json",
    )

    assert completed.returncode == 1
    document = _document(completed, validator)
    assert document["status"] == "failed"
    assert document["error"]["kind"] == "no_results"
    assert document["result"]["tickers"] == {}
    assert document["result"]["errors"][0]["kind"] == "insufficient_history"
    assert document["config"]["seed"] == 7


def test_backtest_with_prices_too_short_exits_one(validator):
    completed = _run("backtest", "AAPL", *OFFLINE, "--seed", "7", "--json")

    assert completed.returncode == 1
    document = _document(completed, validator)
    assert document["status"] == "failed"
    assert document["error"]["kind"] == "no_results"
    assert document["result"] is None
    assert _run("backtest", "AAPL", *OFFLINE, "--seed", "7").returncode == 1


@pytest.mark.parametrize("command", ["simulate", "backtest"])
@pytest.mark.parametrize(
    "arguments",
    [
        ["ZZZZ", "--source", "offline", "--data-path", "sample_data"],
        ["AAPL", "--source", "offline", "--data-path", "no-such-directory"],
    ],
    ids=["unknown-ticker", "missing-data-path"],
)
def test_missing_price_data_exits_two_in_both_commands(command, arguments, validator):
    completed = _run(command, *arguments, "--seed", "7", "--json")

    assert completed.returncode == 2
    document = _document(completed, validator)
    assert document["status"] == "error"
    assert document["error"]["kind"] == "price_data"
    assert document["result"] is None
    assert _run(command, *arguments, "--seed", "7").returncode == 2


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
        lambda doc: doc.update(exit_code=3),
        lambda doc: doc.update(version=""),
        lambda doc: doc["config"].pop("seed"),
        lambda doc: doc["config"].update(seed=-1),
        lambda doc: doc["config"].update(scenarios=10**12),
        lambda doc: doc["result"].pop("price_sources"),
        lambda doc: doc["result"].update(tickers={}),
        lambda doc: doc["result"]["tickers"]["AAPL"].update(expected_return="high"),
        lambda doc: doc["result"]["tickers"]["AAPL"].update(prob_above_current=1.5),
        lambda doc: doc["result"]["action_plan"].update(note="extra"),
        lambda doc: doc["result"]["action_plan"].update(stance="YOLO"),
        lambda doc: doc["result"]["action_plan"].update(cash_weight=2),
        lambda doc: doc["result"]["rankings"]["AAPL"].update(extra=1),
        lambda doc: doc["result"]["allocations"]["AAPL"].update(weight=1.2),
        lambda doc: doc.update(extra=True),
        lambda doc: doc.update(error={"kind": "usage", "message": "x"}),
    ],
)
def test_schema_rejects_simulate_documents_that_break_the_contract(
    seeded_simulate, validator, mutate
):
    document = json.loads(seeded_simulate.stdout)
    broken = copy.deepcopy(document)
    mutate(broken)

    assert validator.is_valid(document)
    assert not validator.is_valid(broken)


@pytest.mark.parametrize(
    "mutate",
    [
        lambda doc: doc["result"]["summary"].update(periods=1.5),
        lambda doc: doc["result"]["summary"].update(strategy_win_rate=1.1),
        lambda doc: doc["result"]["summary"].update(extra=1),
        lambda doc: doc["result"]["rebalance_log"][0].update(rebalance_date="yesterday"),
        lambda doc: doc["result"]["rebalance_log"][0].update(extra=1),
        lambda doc: doc["result"]["rebalance_log"][0].update(cash_weight=-0.1),
        lambda doc: doc["result"]["equity_curve"][0].update(date="2024-01-09T00:00:00"),
        lambda doc: doc["result"].update(rebalance_log=[]),
    ],
)
def test_schema_rejects_backtest_documents_that_break_the_contract(validator, mutate):
    document = _document(_run(*BACKTEST, "--seed", "7", "--json"), validator)
    broken = copy.deepcopy(document)
    mutate(broken)

    assert not validator.is_valid(broken)


@pytest.mark.parametrize(
    "mutate",
    [
        lambda doc: doc["result"]["scorecard"].update(failed_runs=1),
        lambda doc: doc["result"]["scorecard"].update(run_success_rate=1.01),
        lambda doc: doc["result"]["scorecard"].update(mean_rank_correlation=-2),
        lambda doc: doc["result"]["scorecard"].update(extra=1),
        lambda doc: doc["result"]["outcomes"][0].update(extra=1),
        lambda doc: doc["result"]["outcomes"][0].update(status="maybe"),
        lambda doc: doc.update(
            status="failed", exit_code=1, error={"kind": "failed_runs", "message": "x"}
        ),
    ],
)
def test_schema_rejects_evaluate_documents_that_break_the_contract(validator, mutate):
    completed = _run("evaluate", "evaluation_sets/sample-stability.json", "--json")
    document = _document(completed, validator)
    broken = copy.deepcopy(document)
    mutate(broken)

    assert not validator.is_valid(broken)


def test_backtest_json_counts_periods_as_an_integer_and_dates_as_dates(validator):
    completed = _run(*BACKTEST, "--seed", "7", "--json")
    document = _document(completed, validator)

    assert '"periods": 1,' in completed.stdout
    assert document["result"]["rebalance_log"][0]["rebalance_date"] == "2024-01-09"
    assert document["result"]["equity_curve"][-1]["date"] == "2024-01-12"


def test_scenarios_and_days_above_the_documented_maximum_exit_two(validator):
    too_many = _run("simulate", "AAPL", *OFFLINE, "--scenarios", "99999999999999", "--json")
    too_long = _run("simulate", "AAPL", *OFFLINE, "--days", "25201", "--json")
    too_big = _run(
        "simulate", "AAPL", *OFFLINE, "--days", "25200", "--scenarios", "1000000", "--json"
    )

    for completed, text in (
        (too_many, "at most 1000000"),
        (too_long, "at most 25200"),
        (too_big, "50000000"),
    ):
        assert completed.returncode == 2
        document = _document(completed, validator)
        assert document["error"]["kind"] == "usage"
        assert text in document["error"]["message"]
    backtest = _run("backtest", "AAPL", *OFFLINE, "--scenarios", "1000001", "--json")
    assert backtest.returncode == 2


@pytest.mark.parametrize(
    "patch",
    [
        "def boom(args): raise RuntimeError('engine fault')",
        "def boom(args): raise MemoryError('engine fault')",
    ],
    ids=["runtime-error", "memory-error"],
)
def test_unexpected_exception_exits_three_as_an_internal_error(patch, validator):
    patch += "\npublic_cli.execute_public_simulate = boom"
    arguments = ["simulate", "AAPL", *OFFLINE, "--seed", "7"]

    as_json = _run_patched(patch, *arguments, "--json")
    as_text = _run_patched(patch, *arguments)

    assert as_json.returncode == 3
    document = _document(as_json, validator)
    assert document["status"] == "error"
    assert document["error"] == {"kind": "internal", "message": "engine fault"}
    assert document["config"]["seed"] == 7
    assert "Traceback" not in as_json.stderr
    assert as_text.returncode == 3
    assert as_text.stdout == ""


def _run_with_closed_pipe(*args: str) -> tuple[int, str]:
    process = subprocess.Popen(
        [*CLI, *args], stdout=subprocess.PIPE, stderr=subprocess.PIPE, cwd=REPO_ROOT, env=ENV
    )
    process.stdout.close()
    stderr = process.stderr.read().decode()
    process.stderr.close()
    return process.wait(), stderr


@pytest.mark.parametrize("flags", [["--json"], []], ids=["json", "text"])
def test_reader_that_goes_away_exits_four_without_a_traceback(flags):
    returncode, stderr = _run_with_closed_pipe(*SIMULATE, "--seed", "7", *flags)

    assert returncode == 4
    assert "Traceback" not in stderr
    assert "Exception ignored" not in stderr
    assert "could not write to stdout" in stderr


@pytest.mark.skipif(not Path("/dev/full").exists(), reason="needs /dev/full")
@pytest.mark.parametrize(
    "arguments",
    [[*SIMULATE, "--seed", "7", "--json"], [*SIMULATE, "--seed", "7"], ["schema"], ["--help"]],
    ids=["json", "text", "schema", "help"],
)
def test_full_output_device_exits_four_without_a_traceback(arguments):
    with open("/dev/full", "w") as full:
        completed = subprocess.run(
            [*CLI, *arguments], stdout=full, stderr=subprocess.PIPE, cwd=REPO_ROOT, env=ENV,
            text=True, check=False,
        )

    assert completed.returncode == 4
    assert "Traceback" not in completed.stderr
    assert "Exception ignored" not in completed.stderr


def test_json_flag_is_accepted_before_the_command(seeded_simulate, validator):
    leading = _run("--json", *SIMULATE, "--seed", "7")

    assert leading.returncode == 0
    assert leading.stdout == seeded_simulate.stdout
    usage = _run("--json", "simulate", "AAPL", "--days", "x")
    assert usage.returncode == 2
    assert _document(usage, validator)["error"]["kind"] == "usage"


def test_json_mode_is_decided_by_the_parser_not_by_the_text():
    # An abbreviation is not the flag, and "--json" after "--" is a ticker name.
    abbreviated = _run(*SIMULATE, "--seed", "7", "--js")
    as_ticker = _run("simulate", *OFFLINE, "--days", "x", "--", "--json")

    assert abbreviated.returncode == 2
    assert abbreviated.stdout == ""
    assert "unrecognized arguments: --js" in abbreviated.stderr
    assert as_ticker.returncode == 2
    assert as_ticker.stdout == ""
    assert "argument --days: must be an integer" in as_ticker.stderr


def test_schema_command_prints_the_committed_schema_from_any_directory(tmp_path):
    completed = _run("schema", cwd=tmp_path)

    assert completed.returncode == 0
    assert json.loads(completed.stdout) == json.loads(SCHEMA_PATH.read_text(encoding="utf-8"))


def test_seeded_simulate_matches_the_committed_golden_result(validator):
    """Catches a dependency or engine change that moves the numbers."""

    golden = json.loads(GOLDEN_PATH.read_text(encoding="utf-8"))
    completed = _run(*golden["config"]["reproduce"][1:])
    document = _document(completed, validator)

    assert validator.is_valid(golden)
    for key in ("config", "result", "status", "exit_code", "command"):
        assert not _differences(golden[key], document[key], key)


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
