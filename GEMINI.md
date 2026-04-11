# Monte Carlo Stock Simulation

## Project Overview

This project is a Python toolkit for two jobs:

1. simulate forward outcomes for current ideas
2. backtest the decision process on historical data

The public interface is the `monte-carlo` command with `simulate` and `backtest` subcommands.

## Architecture & Key Files

The project is structured into focused modules:

* **`cli.py`**: Public CLI implementation for `monte-carlo simulate|backtest`, plus deprecated simulation wrapper helpers.
* **`backtest.py`**: Walk-forward validation engine and deprecated backtest wrapper.
* **`MonteCarlo.py`**: Deprecated single-ticker compatibility wrapper.
* **`simulation.py`**: Core simulation logic (`simulate_prices` for historical bootstrap and `simulate_gbm` for GBM).
* **`data.py`**: Data retrieval via `yfinance` and local CSV parsing for offline mode.
* **`analysis.py`**: Statistical summaries such as VaR, expected return, and quantiles.
* **`viz.py`**: Plot helpers for distributions, price paths, and equity curves.
* **`tests/`**: Automated test suite.

## Setup & Dependencies

The project requires Python 3.9+.

### Installation

```bash
python3 -m pip install -e .
```

## Usage

### Public CLI

**Simulate current opportunities:**
```bash
monte-carlo simulate AAPL MSFT --days 252 --scenarios 5000 --model gbm --seed 42 --output results
```

**Backtest the process:**
```bash
monte-carlo backtest AAPL MSFT --lookback 60 --hold 20 --rebalance 20 --model gbm --scenarios 1000 --seed 42 --output results/backtest
```

**Offline usage:**
```bash
monte-carlo simulate AAPL --source offline --data-path sample_data
```

**Key flags:**
* `--model`: `historical` (default) or `gbm`.
* `--source`: `auto` (default), `offline`, or `online`.
* `--data-path`: Directory or CSV file for local price data.
* `--show`: Display plots on screen.
* `--details`: Print tables and secondary metrics.
* `--output`: Directory to save generated reports and plots.

### Legacy Wrappers

`python cli.py`, `python backtest.py`, and `python MonteCarlo.py` still work as deprecated compatibility wrappers. Prefer `monte-carlo simulate|backtest`.

## Testing

The project uses `pytest` for testing.

**Run all tests:**
```bash
pytest
```

**Run specific tests:**
```bash
pytest tests/test_simulation.py
```

## Development Conventions

* **Type Hinting:** Code uses Python type hints extensively.
* **Data Structures:** `pandas.DataFrame` is the primary data structure for simulation results, often using MultiIndex columns for multi-ticker runs.
* **Plotting:** Use `MPLBACKEND=Agg` in headless environments and opt into `--show` only when interactive plots are desired.
* **Error Handling:** Custom exceptions like `PriceDataError` are used to manage data fetching issues with actionable messages.
