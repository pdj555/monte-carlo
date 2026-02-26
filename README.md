# Monte Carlo Stock Price Simulation

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

1. Clone or download the repository to your local machine.
2. Install the dependencies listed above.

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
| `--max-paths` | Cap how many scenarios are drawn in the path plot (0 = all). |
| `--no-plots` | Skip generating plots (faster for large runs). |
| `--no-show` | Skip displaying plots (useful on servers/CI). |
| `--offline-path` | Directory or CSV file used when offline data is required. |
| `--offline-only` | Never hit the network—expect local CSV data. |
| `--strict` | Exit non-zero if any ticker fails. |

Both the CLI and the legacy script output a statistical summary including mean, median, quantiles, expected return and 95% value-at-risk for the simulated final prices. When you pass multiple tickers to `cli.py`, it now also emits an **equal-weight portfolio** summary so you can judge basket-level upside/downside instead of isolated symbols.

The CLI now also prints a concise **Action plan** section (stance, primary pick, and avoid list) and saves the same guidance to `action_plan.md` in the output directory. This gives a decision-oriented readout rather than raw metrics only.

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
