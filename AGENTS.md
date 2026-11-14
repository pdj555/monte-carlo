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
```

- `--json` prints one JSON document on stdout; logs go to stderr. It may be written before or after the command, and is not accepted abbreviated. `monte-carlo schema` prints the document's JSON Schema, also from an installed package; the same schema is committed as `schemas/cli-result.v1.schema.json`.
- Exit codes, the same with and without `--json`; `exit_code` in the document equals the process exit code:

  | Code | Meaning |
  | --- | --- |
  | `0` | ok |
  | `1` | ran, but no usable result: prices loaded but are too short to simulate or to form one backtest period, or an evaluation run failed |
  | `2` | the command line or its inputs were rejected, including price data that could not be loaded (unknown ticker, missing `--data-path`); the document has `error` and no `result` |
  | `3` | internal error (`error.kind` is `internal`) |
  | `4` | stdout could not be written (reader closed the pipe, device full, descriptor closed); one line on stderr, no document |

- `simulate` skips a ticker whose prices cannot be loaded and reports it in `result.errors`; it exits `2` only when no ticker loaded. `backtest` needs every ticker and exits `2` if any is missing.
- Limits: `--scenarios` at most 1000000, `--days` at most 25200, and `--days` x `--scenarios` at most 50000000 for `simulate`. Above them the exit code is `2`.
- `config` holds the seed and every setting of the run. Without `--seed`, one is drawn and reported (`seed_source: "generated"`). Same seed, settings, and price files give the same bytes; `result.price_sources` carries each local file's SHA-256.
- `config.reproduce` is the argv that replays the numbers. It leaves out `--output`, `--show`, and `--details`, which change what is saved or shown, not what is computed.
- CI installs `requirements.lock.txt` first (Python 3.9 to 3.12) and compares a seeded run with `tests/golden/simulate-seed42.json`; do the same to match its numbers exactly.
- Without `--json` the text output is unchanged. `monte-carlo` with no subcommand prints help and exits `1`.

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
