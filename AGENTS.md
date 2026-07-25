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
