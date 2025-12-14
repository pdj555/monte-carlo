# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Monte Carlo stock price simulation toolkit using historical data from Yahoo Finance. The codebase provides vectorized simulation engines for both historical bootstrap and geometric Brownian motion (GBM) models, along with analytics and visualization utilities.

## Development Commands

### Running Simulations

**Legacy single-ticker workflow:**
```bash
python MonteCarlo.py --ticker AAPL --days 252 --scenarios 1000
```

**Advanced CLI with multi-ticker support:**
```bash
python cli.py --tickers AAPL,MSFT --days 252 --scenarios 5000 --model gbm --output ./results --seed 42
```

Key CLI options:
- `--model historical` (default) or `--model gbm` for geometric Brownian motion
- `--offline-only` to skip network requests and use local CSV data
- `--offline-path DIR` to specify custom CSV directory
- `--no-show` to skip displaying plots (useful for headless/CI environments)
- `--seed N` for reproducible simulations

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

Install dependencies:
```bash
pip install -r requirements.txt
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
- Use `prefer_local=True` to skip network entirely (controlled by `--offline-only` CLI flag)
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
- `MonteCarlo.py` - Simple single-ticker script, good for quick tests
- `cli.py` - Full-featured interface with multi-ticker, model selection, output management
  - `build_parser()` defines argparse structure
  - `run(args)` executes the workflow and returns `{"simulations": DataFrame, "summaries": DataFrame}`
  - Can be imported as a library for programmatic use

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
