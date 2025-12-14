# Repository Guidelines

## Project Structure & Module Organization

- `cli.py`: primary entrypoint for multi-ticker runs, model selection (`historical`/`gbm`), and saving plots.
- `MonteCarlo.py`: legacy single-ticker script (kept for backwards compatibility).
- `simulation.py`: vectorized simulation engines (`simulate_prices`, `simulate_gbm`).
- `data.py`: price retrieval via `yfinance`, plus offline CSV fallback (`sample_data/<TICKER>.csv`).
- `analysis.py`: summary statistics for simulated final prices.
- `viz.py`: plotting helpers (returns `matplotlib` `Figure`s).
- `tests/`: `pytest` suite (note: `tests/conftest.py` adds repo root to `sys.path`).
- `docs/`: project standards (`docs/constitution.md`) and improvement notes.

## Build, Test, and Development Commands

```bash
python -m pip install -r requirements.txt

# Run simulations (network required unless offline-only)
python cli.py --tickers AAPL,MSFT --days 252 --scenarios 5000 --model gbm --seed 42 --output results

# Headless/offline mode (recommended for CI)
python cli.py --offline-only --offline-path sample_data --no-show

# Tests
pytest
pytest tests/test_simulation.py
```

Tip: to avoid GUI pop-ups in headless environments, set `MPLBACKEND=Agg` or use `--no-show`.

## Coding Style & Naming Conventions

- Follow `docs/constitution.md`: keep changes small, readable, and well-tested.
- Python style: PEP 8, 4-space indentation, `snake_case` for functions/vars, `PascalCase` for classes.
- Prefer type hints and clear docstrings (most modules use `from __future__ import annotations`).
- Keep simulation code vectorized (NumPy/Pandas) and validate inputs early with actionable errors (`ValueError`, `PriceDataError`).

## Testing Guidelines

- Use `pytest`; new features should include tests for happy paths and edge cases.
- Tests should be deterministic: pass explicit `seed` values and avoid live network calls (use offline CSVs).

## Commit & Pull Request Guidelines

- Git history trends toward short, imperative commit subjects (e.g., “Add…”, “Fix…”); keep messages specific and scoped.
- PRs should include: summary of behavior change, how you validated it (e.g., `pytest` output), and screenshots when plot output changes.
- Don’t commit generated artefacts (plots should go under an `--output` directory like `results/`).
