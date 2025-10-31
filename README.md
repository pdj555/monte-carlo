# Monte Carlo Stock Price Simulation

[![CI](https://github.com/pdj555/monte-carlo/actions/workflows/ci.yml/badge.svg)](https://github.com/pdj555/monte-carlo/actions/workflows/ci.yml)
[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

This project provides reusable building blocks for running Monte Carlo stock price simulations. Historical data is retrieved from Yahoo! Finance (via `yfinance`) and fed into vectorised simulation routines and plotting helpers.

## Table of Contents
- [Prerequisites](#prerequisites)
- [Installation](#installation)
- [Quick Start](#quick-start)
- [Command-Line Interface](#command-line-interface)
- [Offline Data](#offline-data)
- [Testing](#testing)
- [License](#license)

## Prerequisites

Python 3.9+ is recommended. Install the required packages using the bundled `requirements.txt`:

```bash
pip install -r requirements.txt
```

The key runtime dependencies are:

- `yfinance` for historical prices.
- `pandas` and `numpy` for data wrangling and vectorised simulations.
- `matplotlib` and `seaborn` for visualisations.
- `pytest` for the automated test-suite.

## Installation

### From PyPI (Recommended)

```bash
pip install monte-carlo-stock-sim
```

### From Source

1. Clone or download the repository to your local machine.
2. Install the dependencies listed above.

```bash
git clone https://github.com/pdj555/monte-carlo.git
cd monte-carlo
pip install -r requirements.txt
```

## Quick Start

Run the original single-ticker workflow:

```bash
python MonteCarlo.py --ticker AAPL --days 252 --scenarios 1000
```

The script fetches prices, generates simulations, prints key statistics for the final price distribution and opens histogram/path plots.

## Command-Line Interface

`cli.py` offers an expanded workflow capable of processing multiple tickers, choosing between the historical bootstrap model and geometric Brownian motion, and saving artefacts to disk.

```bash
python cli.py --tickers AAPL,MSFT --days 252 --scenarios 5000 --model gbm \
  --output ./results --seed 42
```

Notable options:

| Flag | Description |
| ---- | ----------- |
| `--tickers` | Comma separated list of tickers to simulate. |
| `--model` | `historical` (default) or `gbm` for geometric Brownian motion. |
| `--seed` | Fix the random seed to obtain reproducible paths. |
| `--output` | Directory where distribution and path plots are saved. |
| `--no-show` | Skip displaying plots (useful on servers/CI). |
| `--offline-path` | Directory or CSV file used when offline data is required. |
| `--offline-only` | Never hit the network—expect local CSV data. |

Both the CLI and the legacy script output a statistical summary including mean, median, quantiles, expected return and 95% value-at-risk for the simulated final prices.

## Advanced Risk Analytics

The `analysis` module provides comprehensive risk metrics:

```python
from analysis import calculate_risk_metrics
from simulation import simulate_prices
import pandas as pd

# Run simulation
returns = pd.Series([0.01, -0.02, 0.03, 0.04])
sims = simulate_prices(returns, days=252, scenarios=1000, current_price=100, seed=42)

# Calculate advanced risk metrics
metrics = calculate_risk_metrics(sims, current_price=100, risk_free_rate=0.02)
print(metrics)
```

Available metrics include:
- **Sharpe Ratio**: Risk-adjusted return measure
- **Sortino Ratio**: Downside risk-adjusted return
- **Maximum Drawdown**: Largest peak-to-trough decline
- **Value at Risk (VaR)**: Potential loss at 90%, 95%, and 99% confidence levels
- **Conditional VaR (CVaR)**: Expected loss beyond VaR threshold

## Offline Data

When internet access is restricted, pass `--offline-only` so `cli.py` and the underlying `fetch_prices` helper use CSV data exclusively. CSV files should contain `Date` and `Close` columns. By default the repository looks for `sample_data/<TICKER>.csv` but you can point to your own directory via `--offline-path /path/to/csvs`.

To avoid GUI pop-ups in headless environments set `MPLBACKEND=Agg` before running the commands or pass `--no-show` to the CLI.

## Testing

Execute the automated tests with `pytest`:

```bash
pytest
```

The tests exercise the simulation routines, summary statistics and CLI entry points to guard against regressions.

## License

This project is licensed under the [MIT License](LICENSE).

## Documentation

- [Quick Reference Guide](docs/quick_reference.md) - API reference and examples
- [Strategic Plan](docs/strategic_plan.md) - Roadmap and future improvements
- [Constitution](docs/constitution.md) - Project principles and guidelines

## Contributing

Contributions are welcome! Please see our [Constitution](docs/constitution.md) for guidelines.

## Demo

Try out the comprehensive demo:
```bash
python demo.py --ticker AAPL
```
