# Monte Carlo Stock Simulation

## Project Overview

This project is a Python-based toolkit for running Monte Carlo simulations on stock prices. It supports both **Historical Bootstrap** (resampling actual past returns) and **Geometric Brownian Motion (GBM)** models. It is designed to be modular, offering reusable components for data fetching, simulation, analysis, and visualization.

## Architecture & Key Files

The project is structured into focused modules:

*   **`cli.py`**: The primary entry point for the application. It handles argument parsing, orchestration of the simulation workflow (fetch -> simulate -> analyze -> plot), and output management.
*   **`MonteCarlo.py`**: A simpler, legacy entry point for single-ticker simulations.
*   **`simulation.py`**: Contains the core simulation logic (`simulate_prices` for historical, `simulate_gbm` for GBM). It uses vectorized operations (via `numpy` and `pandas`) for efficiency.
*   **`data.py`**: Handles data retrieval using `yfinance` for live data and local CSV parsing for offline mode. It includes error handling and retry logic.
*   **`analysis.py`**: Provides statistical summaries of simulation results (VaR, expected return, quantiles).
*   **`viz.py`**: Helper functions for generating plots (histograms, KDEs, price paths) using `matplotlib` and `seaborn`.
*   **`tests/`**: Contains the automated test suite.

## Setup & Dependencies

The project requires Python 3.9+.

### Installation

1.  **Install Dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

    Key libraries: `yfinance` (data), `pandas`/`numpy` (computation), `matplotlib`/`seaborn` (plotting), `pytest` (testing).

## Usage

### Command-Line Interface (Recommended)

The `cli.py` script is the most robust way to interact with the tool.

**Basic Run:**
```bash
python cli.py --tickers AAPL --days 252 --scenarios 1000
```

**Advanced Usage:**
```bash
python cli.py --tickers "AAPL,MSFT" \
              --days 252 \
              --scenarios 5000 \
              --model gbm \
              --seed 42 \
              --output ./results
```

**Key Flags:**
*   `--tickers`: Comma-separated list of symbols (e.g., "AAPL,GOOG").
*   `--model`: `historical` (default) or `gbm`.
*   `--offline-only`: Force usage of local CSV data (no network calls).
*   `--no-show`: Suppress interactive plot windows (useful for scripts/CI).
*   `--output`: Directory to save generated plot images.

### Legacy Script

For simple, single-ticker tests:
```bash
python MonteCarlo.py --ticker AAPL --days 252
```

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

*   **Type Hinting:** Code uses Python type hints extensively.
*   **Data Structures:** `pandas.DataFrame` is the primary data structure for simulation results, often using MultiIndex columns for multi-ticker runs.
*   **Plotting:** `matplotlib` backend handling is built-in (switches to 'Agg' if `--no-show` is used) to support headless environments.
*   **Error Handling:** Custom exceptions like `PriceDataError` (in `data.py`) are used to manage data fetching issues.
