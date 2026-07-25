# Scenario-Set Evaluation Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a reproducible `monte-carlo evaluate` workflow that measures ranking stability, data-source reliability, and downside guardrail behavior across a bounded matrix of universes, models, seeds, and source modes.

**Architecture:** A new dependency-free `evaluation.py` module owns the versioned JSON contract, deterministic matrix expansion, run normalization, scorecard aggregation, and artifact persistence. `public_cli.py` remains the thin operator surface: it adapts each evaluation run to the existing simulation workflow, renders the scorecard, and returns CI-useful exit codes. A small bundled offline evaluation set proves the complete path without network access.

**Tech Stack:** Python 3.9+, standard-library dataclasses/JSON/CSV/statistics, existing pandas-backed simulation engine, argparse, pytest.

## Global Constraints

- Preserve the existing `monte-carlo simulate` and `monte-carlo backtest` contracts.
- Add exactly one public command: `monte-carlo evaluate SET_FILE [--output DIR]`.
- Evaluation files use `schema_version: 1` and expand to at most `100` runs.
- Supported models are exactly `historical` and `gbm`; supported source modes are exactly `auto`, `offline`, and `online`.
- Resolve relative `data_path` values against the evaluation file's directory, never the caller's current directory.
- Execute the existing public simulation engine through an injected runner; do not duplicate simulation or ranking logic.
- Preserve deterministic run ordering: universe, then model, then seed, then source, all in manifest order.
- Persist exactly `scorecard.md`, `runs.csv`, and `report.json` when `--output` is supplied.
- Return exit code `0` when every expanded run completes, `1` when any run fails, and `2` for invalid input or an unexpected command failure.
- Add no runtime dependency and make no network call in tests.
- Keep user-facing names and errors concise, specific, and actionable.

---

## File Structure

- `evaluation.py`: versioned evaluation contract, validation, matrix expansion, run normalization, aggregation, formatting, and persistence.
- `tests/test_evaluation.py`: focused contract, validation, aggregation, failure-isolation, and artifact tests using injected deterministic runners.
- `public_cli.py`: parser and adapter for the new public command only; the simulation engine stays unchanged.
- `evaluation_sets/sample-stability.json`: fast, deterministic, offline reference set using bundled AAPL/MSFT history.
- `tests/test_cli.py`: parser and real end-to-end command coverage.
- `tests/test_installation.py`: installed entrypoint discovery for `evaluate --help`.
- `tests/test_docs.py`: operator-document and reference-set coverage.
- `pyproject.toml`: package the new top-level module.
- `README.md`, `AGENTS.md`, `docs/output-guide.md`, `docs/improvements.md`: concise operator and maintainer guidance.

### Task 1: Evaluation Core

**Files:**
- Create: `evaluation.py`
- Create: `tests/test_evaluation.py`
- Modify: `pyproject.toml`

**Interfaces:**
- Produces: `EvaluationConfigError(ValueError)`.
- Produces: immutable `EvaluationUniverse`, `EvaluationSource`, `EvaluationSet`, and `EvaluationRun` dataclasses.
- Produces: immutable `EvaluationRunOutcome`, `EvaluationScorecard`, and `EvaluationReport` dataclasses, each with a JSON-safe `to_dict()` where persisted.
- Produces: `load_evaluation_set(path: str | Path) -> EvaluationSet`.
- Produces: `expand_evaluation_runs(evaluation_set: EvaluationSet) -> tuple[EvaluationRun, ...]`.
- Produces: `evaluate_scenario_set(evaluation_set: EvaluationSet, runner: Callable[[EvaluationRun], Mapping[str, object]]) -> EvaluationReport`.
- Produces: `format_evaluation_scorecard(report: EvaluationReport) -> str`.
- Produces: `save_evaluation_report(report: EvaluationReport, output_dir: str | Path) -> tuple[Path, Path, Path]` returning paths in scorecard/runs/report order.
- Consumes: the existing simulation result shape: `result["report"]["rankings"]`, `result["report"]["action_plan"]`, `result["report"]["results"]`, `result["report"]["errors"]`, and `result["price_sources"]`.

- [ ] **Step 1: Write contract and matrix-expansion tests**

Create `tests/test_evaluation.py` with helpers that write JSON manifests under `tmp_path`. Cover the exact contract below:

```python
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
```

Add parametrized invalid-manifest cases for: unknown top-level keys, missing/unsupported `schema_version`, duplicate names, empty tickers/models/seeds/sources, invalid ticker tokens, negative or boolean seeds, unsupported models/source modes, offline sources without `data_path`, non-positive days/scenarios, and a Cartesian product above `100` runs. Assert each `EvaluationConfigError` names the offending field and tells the operator how to fix it.

- [ ] **Step 2: Run the contract tests and verify RED**

Run: `MPLBACKEND=Agg pytest -q tests/test_evaluation.py`

Expected: FAIL during collection because `evaluation` does not exist.

- [ ] **Step 3: Implement the versioned contract and deterministic expansion**

In `evaluation.py`, define these constants and dataclass fields exactly:

```python
EVALUATION_SCHEMA_VERSION = 1
MAX_EVALUATION_RUNS = 100
SUPPORTED_MODELS = ("historical", "gbm")
SUPPORTED_SOURCE_MODES = ("auto", "offline", "online")

@dataclass(frozen=True)
class EvaluationUniverse:
    name: str
    tickers: tuple[str, ...]

@dataclass(frozen=True)
class EvaluationSource:
    name: str
    mode: str
    data_path: Path | None = None

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
```

Use a strict allowed-key check at every object level. Reject booleans where integers are required. Normalize tickers to uppercase, de-duplicate only by rejecting duplicates, and accept ticker tokens only when they match `^[A-Z0-9][A-Z0-9.^=-]{0,14}$`. Hash the exact manifest bytes with SHA-256. Construct `run_id` as `<universe>/<model>/seed-<seed>/<source>` and enforce the `100`-run cap before returning the set.

- [ ] **Step 4: Run the contract tests and verify GREEN**

Run: `MPLBACKEND=Agg pytest -q tests/test_evaluation.py`

Expected: all contract and expansion tests pass with pristine output.

- [ ] **Step 5: Write failing aggregation, isolation, and artifact tests**

Add deterministic fake runners that return the same mapping shape as `simulate_cli.run`. The completed result helper must include ordered rankings, action-plan stance/top pick, completed ticker results, source provenance, and optional guardrail reasons. Cover:

```python
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
```

Also prove: one runner exception becomes a failed outcome without aborting later runs; empty rankings become a failed outcome with the simulation errors joined into an actionable message; source reliability is grouped by the manifest's source name; rank correlation is `None` when fewer than two comparable completed runs exist; `format_evaluation_scorecard` renders `n/a` for unavailable metrics; and artifact persistence creates exactly the three required files with parseable JSON/CSV and the same displayed scorecard.

- [ ] **Step 6: Run the new aggregation tests and verify RED**

Run: `MPLBACKEND=Agg pytest -q tests/test_evaluation.py`

Expected: FAIL because report aggregation and persistence interfaces are not implemented.

- [ ] **Step 7: Implement run normalization, scorecard aggregation, and persistence**

Implement immutable outcomes and reports with these fields:

```python
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

@dataclass(frozen=True)
class EvaluationReport:
    schema_version: int
    evaluation_set: EvaluationSet
    generated_at: str
    outcomes: tuple[EvaluationRunOutcome, ...]
    scorecard: EvaluationScorecard
```

Treat a run as completed only when rankings are a non-empty mapping. Count completed tickers from `report.results`, guardrail rejections from rows whose recommendation is `AVOID` and whose `guardrail_reasons` is non-empty, and the worst downside from `value_at_risk_95_pct`. Compute mean rank correlation only across outcome pairs from the same universe with at least two common ranked tickers, using a local Pearson calculation over their rank positions. Compute top-pick consistency per universe and average those universe ratios so large matrices do not drown out small universes. Catch `Exception` around each injected runner call, preserve the exception message in that outcome, and continue the deterministic matrix.

Format the scorecard as:

```text
Evaluation set: sample-stability
Runs: 6/6 complete (100.0%)
Ticker coverage: 100.0%
Mean rank correlation: 92.5%
Top-pick consistency: 83.3%
Guardrail rejections: 16.7%
No-trade runs: 0.0%
Worst observed 95% downside: 14.2%
Source reliability:
  bundled: 6/6 complete (100.0%)
```

`report.json` must contain the manifest name/path/hash, normalized matrix, complete outcomes, and scorecard. `runs.csv` must have one row per outcome in run order and join `rank_order` with ` > `. `scorecard.md` must be the formatted scorecard plus one trailing newline. Create the output directory with `parents=True, exist_ok=True`.

- [ ] **Step 8: Package the module and verify the task**

Add `"evaluation"` to `tool.setuptools.py-modules` in `pyproject.toml`.

Run:

```bash
MPLBACKEND=Agg pytest -q tests/test_evaluation.py
flake8 evaluation.py tests/test_evaluation.py
MPLBACKEND=Agg pytest -q
```

Expected: all commands pass; the full suite remains at least `136 passed` with no new failures.

- [ ] **Step 9: Commit Task 1**

```bash
git add evaluation.py tests/test_evaluation.py pyproject.toml
git commit -m "Add scenario evaluation core"
```

### Task 2: Public CLI and Operator Workflow

**Files:**
- Create: `evaluation_sets/sample-stability.json`
- Modify: `public_cli.py`
- Modify: `tests/test_cli.py`
- Modify: `tests/test_installation.py`
- Modify: `tests/test_docs.py`
- Modify: `README.md`
- Modify: `AGENTS.md`
- Modify: `docs/output-guide.md`
- Modify: `docs/improvements.md`

**Interfaces:**
- Consumes: all Task 1 interfaces exactly as declared above.
- Produces: public `monte-carlo evaluate SET_FILE [--output DIR]` command.
- Produces: `execute_public_evaluate(args: argparse.Namespace) -> EvaluationReport` for programmatic callers.
- Produces: `run_public_evaluate(args: argparse.Namespace) -> EvaluationReport` for formatted CLI execution.
- Produces: `evaluation_sets/sample-stability.json`, a six-run offline set: one universe × two models × three seeds × one source.

- [ ] **Step 1: Write failing parser and end-to-end CLI tests**

In `tests/test_cli.py`, add parser coverage and a real offline command test:

```python
def test_public_parser_accepts_evaluation_set_and_output():
    args = parse_public_args(
        ["evaluate", "evaluation_sets/sample-stability.json", "--output", "results/eval"]
    )

    assert args.command == "evaluate"
    assert args.set_file == "evaluation_sets/sample-stability.json"
    assert args.output == "results/eval"


def test_public_evaluate_runs_reference_set_and_writes_auditable_outputs(tmp_path, capsys):
    output = tmp_path / "evaluation"

    exit_code = main(
        [
            "evaluate",
            "evaluation_sets/sample-stability.json",
            "--output",
            str(output),
        ]
    )

    rendered = capsys.readouterr().out
    assert exit_code == 0
    assert "Evaluation set: sample-stability" in rendered
    assert "Runs: 6/6 complete (100.0%)" in rendered
    assert {path.name for path in output.iterdir()} == {
        "scorecard.md",
        "runs.csv",
        "report.json",
    }
```

Add a focused monkeypatched test proving `main(["evaluate", ...])` returns `1` when the report scorecard has failed runs, and a malformed manifest test proving the error is actionable and returns `2`.

- [ ] **Step 2: Run the CLI tests and verify RED**

Run: `MPLBACKEND=Agg pytest -q tests/test_cli.py -k evaluate`

Expected: FAIL because `evaluate` is not a recognized command.

- [ ] **Step 3: Add the thin CLI adapter**

Import Task 1 interfaces into `public_cli.py`. Add the parser:

```python
evaluate_parser = subparsers.add_parser(
    "evaluate",
    help="Test decision stability across a reproducible scenario set.",
    formatter_class=IntentionalDefaultsHelpFormatter,
)
evaluate_parser.add_argument(
    "set_file",
    help="Versioned JSON evaluation-set file.",
)
evaluate_parser.add_argument(
    "--output",
    default=None,
    help="Directory for scorecard.md, runs.csv, and report.json.",
)
```

Adapt each `EvaluationRun` to the existing `execute_public_simulate` function with an `argparse.Namespace` containing: `tickers`, `days`, `scenarios`, `model`, `seed`, `source`, `data_path`, `output=None`, `show=False`, and `details=False`. Do not call private simulation functions or render per-run output. `run_public_evaluate` prints the formatted scorecard once and saves artifacts only when `args.output` is present. In `main`, return `1` when `report.scorecard.failed_runs > 0`; let the existing command-level exception path return `2` for invalid manifests or unexpected failures.

- [ ] **Step 4: Add the deterministic reference set**

Create `evaluation_sets/sample-stability.json` with this exact content:

```json
{
  "schema_version": 1,
  "name": "sample-stability",
  "days": 5,
  "scenarios": 100,
  "universes": [
    {
      "name": "large-cap-tech",
      "tickers": ["AAPL", "MSFT"]
    }
  ],
  "models": ["historical", "gbm"],
  "seeds": [7, 42, 99],
  "sources": [
    {
      "name": "bundled",
      "mode": "offline",
      "data_path": "../sample_data"
    }
  ]
}
```

- [ ] **Step 5: Run focused CLI tests and verify GREEN**

Run: `MPLBACKEND=Agg pytest -q tests/test_cli.py -k evaluate`

Expected: all evaluate tests pass with pristine output.

- [ ] **Step 6: Add installed-entrypoint and documentation tests**

Update `tests/test_installation.py` so the help loop includes `["evaluate", "--help"]` and the no-command hint includes `evaluate`. Update `tests/test_docs.py` to require:

- `monte-carlo evaluate evaluation_sets/sample-stability.json --output results/evaluation` in `README.md`.
- `scorecard.md`, `runs.csv`, and the evaluation `report.json` explanation in `docs/output-guide.md`.
- `evaluation.py` and `evaluation_sets/` in `AGENTS.md`.
- A real `load_evaluation_set("evaluation_sets/sample-stability.json")` assertion that expands to six runs and resolves its source path to the repository's `sample_data` directory.

- [ ] **Step 7: Run the documentation tests and verify RED**

Run: `MPLBACKEND=Agg pytest -q tests/test_installation.py tests/test_docs.py`

Expected: FAIL until the operator documentation and help text describe the new workflow.

- [ ] **Step 8: Document the minimum operator path and close the shipped backlog item**

Add one evaluate example to the README CLI block and one sentence explaining that it gates a decision across seeds/models before capital is risked. Add an `Evaluate` section to `docs/output-guide.md` with this output tree:

```text
results/evaluation/
  scorecard.md
  runs.csv
  report.json
```

Explain that `scorecard.md` is the first read, `runs.csv` isolates unstable or failed matrix cells, and `report.json` preserves the manifest hash, normalized matrix, outcomes, and source reliability. Update `docs/improvements.md` by replacing “Add scenario-set evaluations” with a short `Shipped foundation` entry and leave the typed bridge contract and provenance-extension opportunities as the only current opportunities. Add the two new paths to the `AGENTS.md` structure list. Keep all copy factual and compact.

- [ ] **Step 9: Verify all Task 2 surfaces**

Run:

```bash
MPLBACKEND=Agg pytest -q tests/test_cli.py tests/test_installation.py tests/test_docs.py tests/test_evaluation.py
flake8 public_cli.py evaluation.py tests/test_cli.py tests/test_evaluation.py tests/test_docs.py tests/test_installation.py
MPLBACKEND=Agg python -m public_cli evaluate evaluation_sets/sample-stability.json --output /tmp/monte-carlo-evaluation-smoke
```

Expected: tests and lint pass; the smoke command prints `Runs: 6/6 complete (100.0%)` and writes exactly the three documented artifacts.

- [ ] **Step 10: Commit Task 2**

```bash
git add public_cli.py evaluation_sets/sample-stability.json tests/test_cli.py tests/test_installation.py tests/test_docs.py README.md AGENTS.md docs/output-guide.md docs/improvements.md
git commit -m "Expose reproducible scenario evaluations"
```

## Plan Self-Review

- Spec coverage: Task 1 owns the versioned contract, bounded deterministic expansion, aggregation, failure isolation, and persistence. Task 2 owns the public command, reference set, CI-useful exits, packaging integration, and operator guidance.
- Placeholder scan: every implementation step names exact interfaces, values, file paths, commands, and expected behavior; no deferred implementation markers remain.
- Type consistency: Task 2 consumes the exact `EvaluationRun` and `EvaluationReport` interfaces produced by Task 1. The manifest field names, output filenames, run limits, supported values, and exit codes match across both tasks.
