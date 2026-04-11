# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Monte Carlo stock price simulation toolkit using historical data from Yahoo Finance. The codebase provides vectorized simulation engines for both historical bootstrap and geometric Brownian motion (GBM) models, along with analytics and visualization utilities.

## Development Commands

### Running Simulations

**Public CLI:**
```bash
monte-carlo simulate AAPL MSFT --days 252 --scenarios 5000 --model gbm --output ./results --seed 42
```

```bash
monte-carlo backtest AAPL MSFT --lookback 60 --hold 20 --rebalance 20 --model gbm --scenarios 1000 --seed 42
```

Key CLI options:
- `--model historical` (default) or `--model gbm` for geometric Brownian motion
- `--source offline` to use local CSV data only
- `--source auto` to try live data first, then local CSV files
- `--data-path DIR` to specify a custom CSV directory or file
- `--show` to display plots on screen
- `--seed N` for reproducible simulations
- `--details` for full tables and secondary metrics

### Testing

Run all tests:
```bash
pytest
```

Run specific test file:
```bash
pytest tests/test_simulation.py
```

Run tests with verbose output:
```bash
pytest -v
```

### Environment Setup

Install the package and console script:
```bash
python3 -m pip install -e .
```

For headless environments (no GUI):
```bash
export MPLBACKEND=Agg
```

## Architecture

### Module Structure

The codebase is organized into focused modules with clear separation of concerns:

**Core Simulation Engine (`simulation.py`):**
- `simulate_prices()` - Historical bootstrap model using empirical drift and volatility
- `simulate_gbm()` - Geometric Brownian motion with explicit mu/sigma parameters
- Both functions return DataFrames with shape `(days, scenarios)` and support reproducible seeding
- Internal helper `_as_float()` handles both scalar floats and pandas Series for current_price

**Data Fetching (`data.py`):**
- `fetch_prices()` retrieves closing prices via yfinance with automatic retry logic
- Falls back to local CSV files in `sample_data/` when network requests fail
- Use `prefer_local=True` to skip network entirely (controlled by `--source offline` in the public CLI)
- Raises `PriceDataError` on failures
- CSV files must have `Date` and `Close` columns

**Analytics (`analysis.py`):**
- `summarize_final_prices()` computes statistics on the final row of simulation results
- Returns mean, median, std, min, max, quantiles (default: 5%, 25%, 75%, 95%)
- When `current_price` is provided, adds expected_return, prob_above_current, value_at_risk_95

**Visualization (`viz.py`):**
- `plot_distribution()` creates histogram + KDE of final prices
- `plot_paths()` plots simulated price trajectories over time
- Both functions return `matplotlib.Figure` objects for flexible display/saving
- Support MultiIndex columns for multi-ticker DataFrames

**Command-Line Interfaces:**
- `monte-carlo` - Public entrypoint with `simulate` and `backtest` subcommands
- `public_cli.py` - Public CLI parser and command runners
  - `build_public_parser()` defines the public argparse structure
  - `run_public_simulate(args)` executes the public simulation surface
  - `main()` is the installed `monte-carlo` entrypoint
- `simulate_cli.py` - Shared simulation workflow used by both CLI surfaces
- `cli_shared.py` - Shared version, validation, and detailed-rendering helpers
- `cli.py` - Deprecated compatibility facade for `python cli.py ...`
  - `legacy_main()` preserves the old wrapper flow
- `backtest.py` - Walk-forward engine plus deprecated wrapper entrypoint
- `MonteCarlo.py` - Deprecated single-ticker compatibility wrapper

### Data Flow

1. **Fetch** historical prices via `data.fetch_prices()` → returns pd.Series indexed by date
2. **Transform** to returns: `prices.pct_change().dropna()`
3. **Simulate** using either:
   - `simulate_prices(returns, days, scenarios, current_price)` - historical model
   - `simulate_gbm(current_price, mu, sigma, days, scenarios)` - GBM model
4. **Analyze** with `summarize_final_prices(sims, current_price)`
5. **Visualize** with `plot_distribution()` and `plot_paths()`

### MultiIndex Convention

The CLI creates simulation DataFrames with MultiIndex columns:
- Level 0: ticker symbol (e.g., "AAPL", "MSFT")
- Level 1: scenario number (1 to scenarios)

Extract single ticker with: `df.xs("AAPL", axis=1, level=0)`

### Testing Strategy

Tests use pytest fixtures defined in `tests/conftest.py`:
- Tests cover simulation edge cases (empty data, invalid parameters)
- CLI tests verify argument parsing and output structure
- Mock data used to avoid network dependencies during testing
