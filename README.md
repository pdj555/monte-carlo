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
| `--block-size` | Consecutive return block length for historical bootstrap (use `>1` for regime-aware simulations). |
| `--seed` | Fix the random seed to obtain reproducible paths. |
| `--output` | Directory where distribution and path plots are saved. |
| `--max-paths` | Cap how many scenarios are drawn in the path plot (0 = all). |
| `--no-plots` | Skip generating plots (faster for large runs). |
| `--no-show` | Skip displaying plots (useful on servers/CI). |
| `--offline-path` | Directory or CSV file used when offline data is required. |
| `--offline-only` | Never hit the network—expect local CSV data. |
| `--strict` | Exit non-zero if any ticker fails. |
| `--min-expected-return` | Hard floor for expected return (below this becomes `AVOID`). |
| `--min-prob-up` | Minimum probability of finishing above current price. |
| `--max-var-95-pct` | Maximum tolerated 95% VaR (as % of current price). |
| `--portfolio-risk-budget-pct` | Hard cap on blended portfolio 95% VaR; auto-scales positions and leaves excess in cash. |
| `--max-drawdown-q95-pct` | Optional cap on path max drawdown (95th percentile) to hard-filter fragile setups. |
| `--target-return-pct` | Optional upside target (as return %) and report probability of hitting it (`prob_hit_target`). |
| `--max-loss-pct` | Optional maximum acceptable loss and report breach probability (`prob_breach_max_loss`). |
| `--min-prob-hit-target` | Optional hard floor for `prob_hit_target` (requires `--target-return-pct`). |
| `--max-prob-breach-loss` | Optional hard cap for `prob_breach_max_loss` (requires `--max-loss-pct`). |
| `--shock-probability` | Inject rare stress events per step (0-1) to pressure-test fragility. |
| `--shock-return` | Return applied on stress events (e.g. `-0.2` for a 20% drop). |
| `--capital` | Optional total capital used to emit execution-ready dollar/share sizing. |
| `--allow-fractional-shares` | Allow non-integer share sizing when `--capital` is set. |
| `--annual-cash-yield` | Benchmark annual cash yield used for excess-return metrics (`expected_excess_return`, `prob_beat_benchmark`). |

Both the CLI and the legacy script output a statistical summary including mean, median, quantiles, expected return and 95% value-at-risk for the simulated final prices. The CLI now also reports benchmark-aware performance (`benchmark_return_pct`, `expected_excess_return`, and `prob_beat_benchmark`) so each ticker is judged against a cash baseline rather than raw return alone. The CLI now also reports **path-risk drawdown metrics** (`max_drawdown_mean`, `max_drawdown_q95`, and probability of breaching 10%/20% drawdowns), which are more decision-useful than endpoint-only stats. When you pass multiple tickers to `cli.py`, it also emits an **equal-weight portfolio** summary so you can judge basket-level upside/downside instead of isolated symbols.

The CLI now also prints a concise **Action plan** section (stance, primary pick, and avoid list) and saves the same guidance to `action_plan.md` in the output directory. This gives a decision-oriented readout rather than raw metrics only.
When `--capital` is provided, the CLI also produces an **Execution plan** with target dollars, estimated shares, and cash drift per ticker so output maps directly to tradable orders (saved as `execution_plan.csv`).
The allocator is now **risk-budgeted by default**: set `--portfolio-risk-budget-pct` (default `0.02`) and the tool automatically scales total exposure so the blended portfolio 95% VaR stays under your loss budget, with the remainder explicitly left in cash.
It now also estimates **payoff asymmetry** (`avg_upside_pct`, `avg_downside_pct`, `payoff_ratio`) and a capped **Kelly fraction** signal per ticker, then uses that conviction signal to tilt ranking/allocation toward setups with stronger risk-reward geometry.
It now supports an explicit **goal/risk contract**: set an upside target and maximum loss, then enforce probability constraints directly with guardrails (`prob_hit_target`, `prob_breach_max_loss`) so output maps to concrete decision thresholds instead of vague optimism.
You also get **path-aware execution metrics** (`prob_touch_target`, `prob_touch_max_loss`, `median_days_to_target`, `median_days_to_max_loss`) so you can judge whether a setup tends to hit your take-profit/stop-loss *at any point* during the run, not only at final day.

For decisive execution, use guardrails to enforce your risk policy directly in ranking/allocation output, e.g.:

```bash
python cli.py --tickers AAPL,MSFT,TSLA --model gbm --days 252 --scenarios 5000 \
  --min-expected-return 0.08 --min-prob-up 0.55 --max-var-95-pct 0.18 --no-show
```

For more realistic historical-mode paths, use **block bootstrap** to preserve short-term volatility regimes instead of shuffling each day independently:

```bash
python cli.py --tickers AAPL,MSFT --model historical --block-size 5 \
  --days 252 --scenarios 5000 --offline-only --offline-path sample_data --no-show
```

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


### Stress-test mode

Use stress mode to move from optimistic backtests to brutal downside realism:

```bash
python cli.py --tickers AAPL,MSFT --days 252 --scenarios 5000 --model gbm \
  --shock-probability 0.02 --shock-return -0.2 --no-show
```

This overlays rare shock events on every simulated path and exposes which tickers still survive when tail risk actually happens.
