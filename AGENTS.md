# Repository Guidelines

## Project Structure & Module Organization

- `app/`: Next.js App Router surface and `/api/run` route.
- `components/workbench/`: browser UI (controls, results, layout).
- `lib/types.ts`: TypeScript contracts for the workbench API.
- `lib/python-bridge.ts`: Node spawn bridge to `ui_bridge.py`.
- `ui_state.py`: presentation state builder shared by the browser bridge.
- `ui_bridge.py`: JSON bridge used by the Next.js API route.
- `web_entrypoint.py`: local `monte-carlo-ui` launcher for the Next.js workbench.
- `public_cli.py`: public CLI implementation for `monte-carlo simulate|backtest|evaluate`.
- `evaluation.py`: versioned scenario-evaluation contracts and scorecard aggregation.
- `evaluation_sets/`: versioned, bounded scenario manifests for offline evaluation.
- `simulate_cli.py`: shared simulation workflow used by public and legacy entrypoints.
- `cli_shared.py`: shared parser and rendering helpers for CLI surfaces.
- `backtest.py`: walk-forward engine plus deprecated backtest wrapper.
- `simulation.py`: vectorized simulation engines (`simulate_prices`, `simulate_gbm`).
- `data.py`: price retrieval via `yfinance`, plus offline CSV fallback.
- `analysis.py`: summary statistics for simulated final prices.
- `decision.py`: ranking, guardrails, allocation, and action-plan logic.
- `viz.py`: plotting helpers that return `matplotlib` `Figure`s.
- `tests/`: `pytest` suite.
- `docs/`: project standards and operator references.

## Build, Test, and Development Commands

```bash
python3 -m pip install -e .
npm install

# Browser workbench
npm run dev

# Run simulations
monte-carlo simulate AAPL MSFT --days 252 --scenarios 5000 --model gbm --seed 42 --output results

# Walk-forward validation
monte-carlo backtest AAPL MSFT --lookback 60 --hold 20 --rebalance 20 --model gbm --scenarios 1000 --seed 42

# Offline mode
monte-carlo simulate AAPL --source offline --data-path sample_data

# Tests
pytest -q
npm run typecheck
npm run build
```

Tip: set `MPLBACKEND=Agg` in headless environments and leave `--show` off unless you want plots on screen.

## For agents

Run from a source checkout. Every command below is offline and uses `sample_data/`.

```bash
python3 -m pip install -e '.[dev]'

monte-carlo simulate AAPL MSFT --source offline --data-path sample_data --days 5 --scenarios 100 --seed 42 --json
monte-carlo backtest AAPL MSFT --source offline --data-path sample_data --lookback 5 --hold 3 --top 1 --scenarios 100 --seed 42 --json
monte-carlo evaluate evaluation_sets/sample-stability.json --json

pytest -q
flake8 .
ruff check .
```

- `--json` prints one JSON document on stdout; logs go to stderr. It may be written before or after the command. `monte-carlo --json` with no command exits `2` with a usage document. With `--json`, `--help` and `--version` print a document too (`command` is `help` or `version`), and `schema --json` is accepted. `monte-carlo schema` prints the document's JSON Schema, also from an installed package; the same schema is committed as `schemas/cli-result.v1.schema.json`.
- Exit codes, the same with and without `--json`; `exit_code` in the document equals the process exit code:

  | Code | Meaning |
  | --- | --- |
  | `0` | ok |
  | `1` | ran, but no usable result: prices loaded but are too short to simulate or to form one backtest period, or an evaluation run failed for a reason other than missing price data |
  | `2` | the command line or its inputs were rejected, including price data that could not be loaded (unknown ticker, missing `--data-path`, no `Close` column, a duplicate date, or a price that is missing, not a number, not finite, zero or negative) in `simulate`, `backtest`, or any `evaluate` run; the document has `error` and no `result` |
  | `3` | internal error (`error.kind` is `internal`): any exception that is not a rejected input, including a `ValueError` or `OSError` from inside the engine |
  | `4` | stdout could not be written (reader closed the pipe, device full, descriptor closed); one line on stderr, no document |

- `simulate` skips a ticker whose prices cannot be loaded and reports it in `result.errors`; it exits `2` only when no ticker loaded. `backtest` needs every ticker and exits `2` if any is missing.
- Limits: `--scenarios` at most 1000000, `--days` at most 25200, `--days` x `--scenarios` at most 50000000 for `simulate`, and `--seed` from 0 to 4294967295. Integers are plain ASCII digits (`42`, not `4_2`, `+42` or `42.0`). An `evaluate` manifest is held to the same limits. Above them the exit code is `2`.
- `config` holds the seed and every setting of the run, with `data_path` made absolute. Without `--seed`, one is drawn and reported (`seed_source: "generated"`). Same seed, settings, and price files give the same bytes; `result.price_sources` carries each local file's absolute path and SHA-256.
- `config.reproduce` is the argv that replays the numbers from any directory. It leaves out `--output`, `--show`, and `--details`, which change what is saved or shown, not what is computed. The replay prints the same `result` and the same `config` except `seed_source`, which reads `argument` because the seed is now given.
- CI installs `requirements.lock.txt` first (Python 3.13 and 3.14) and compares a seeded run with `tests/golden/simulate-seed42.json`; do the same to match its numbers exactly.
- Dates in a backtest are `YYYY-MM-DD` for daily prices and ISO 8601 date-times (`2024-01-02T14:30:00`) when the price file is intraday.
- A character the terminal cannot encode (an accented ticker under an ASCII locale) is written as an escape such as `\xc5`, not treated as a write failure.

Behavior changes against `main`, which apply without `--json` too:

- `simulate` where no ticker's prices load: exit `1` -> `2`.
- `backtest` with too little history for one period: exit `2` -> `1`, and stderr says "No usable result".
- An exception from inside the engine, `ValueError` included: exit `2` -> `3`, and stderr says "Internal error".
- `evaluate` where a run names a ticker with no price data: exit `1` -> `2`.
- New upper bounds on `--seed`, `--days`, `--scenarios` (and the same in an `evaluate` manifest); integers must be plain ASCII digits.
- Price files are checked strictly. A row whose price is missing, not a number, not finite, zero or negative, a duplicate date, or a file without a `Close` (or `Adj Close`) column is rejected; such rows used to be dropped silently, and a lone numeric column used to be taken as the price. In `simulate` that ticker is now skipped and listed in the errors, even next to a good one; `backtest` stops.
- Each entry of `errors` in the saved `report.json` gains a `kind`.
- Option abbreviations are rejected: `--scen 100` used to mean `--scenarios 100` and is now a usage error. Write options in full.
- A closed or full stdout exits `4` instead of ending in a traceback.
- Otherwise the text output of a successful run is the same as on `main`. `monte-carlo` with no subcommand still prints help and exits `1`.

## Coding Style & Naming Conventions

- Follow `docs/constitution.md`: keep changes small, readable, and well-tested.
- Python style: PEP 8, 4-space indentation, `snake_case` for functions/vars.
- TypeScript style: strict types, small components, server work in route handlers or server modules.
- Prefer type hints and clear docstrings in Python modules.
- Keep simulation code vectorized and validate inputs early with actionable errors.

## Testing Guidelines

- Use `pytest` for Python and `npm run typecheck` / `npm run build` for Next.js.
- New features should include deterministic tests for happy paths and edge cases.
- Avoid live network calls in tests; use offline CSV fixtures.

## Commit & Pull Request Guidelines

- Use short, imperative commit subjects.
- PRs should include behavior summary, validation commands, and screenshots when UI output changes.
- Do not commit generated artifacts, `.next/`, `node_modules/`, plots, or local agent runtime files.
