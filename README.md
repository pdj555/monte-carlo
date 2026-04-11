# Monte Carlo Decision Engine

This project does two things:

1. simulate forward outcomes for current ideas
2. backtest the decision process on historical data

There are two entrypoints:

- `monte-carlo simulate` for current opportunities
- `monte-carlo backtest` for walk-forward validation
- `monte-carlo-ui` for a lean browser UI

## Install

Python 3.9+ is required.

CLI install:

```bash
python3 -m pip install -e .
```

Browser UI install:

```bash
python3 -m pip install -e .[ui]
```

Quick start in the browser:

```bash
monte-carlo-ui
```

Then open [http://127.0.0.1:8000](http://127.0.0.1:8000). The app starts with
the bundled AAPL demo so you land on a real decision instead of an empty page.

Quick start in the CLI:

```bash
monte-carlo --help
```

## Workflow 1: simulate

Use `simulate` when you want a decision-first view of one or more tickers.

```bash
monte-carlo simulate AAPL MSFT \
  --days 252 \
  --scenarios 5000 \
  --model gbm \
  --seed 42 \
  --output results
```

### Offline example

```bash
monte-carlo simulate AAPL \
  --source offline \
  --data-path sample_data
```

### What it prints

- stance and headline
- top idea
- avoid list when guardrails fail
- cash buffer when conviction is low

Add `--details` when you want tables and secondary metrics.

## Workflow 2: backtest

Use `backtest` when you want to validate the process instead of trusting the forecast.

```bash
monte-carlo backtest AAPL MSFT \
  --lookback 60 \
  --hold 20 \
  --rebalance 20 \
  --model gbm \
  --scenarios 1000 \
  --seed 42 \
  --output results/backtest
```

### Offline example

The bundled `sample_data` history is intentionally short, so use a short
walk-forward window for the offline example.

```bash
monte-carlo backtest AAPL \
  --source offline \
  --data-path sample_data \
  --lookback 5 \
  --hold 3 \
  --rebalance 3 \
  --top 1 \
  --scenarios 10
```

### What it prints

- strategy return
- annualized return
- max drawdown
- excess return vs equal weight
- excess return vs cash

Add `--details` for the full metric table.

## Data Sources

Use `--source` to pick how prices are loaded:

- `auto` tries live downloads first, then local CSV files
- `offline` uses local CSV files only
- `online` uses live downloads only

Local CSVs should include `Date` and `Close` columns. By default the repo looks in:

```text
sample_data/<TICKER>.csv
```

Use `--data-path` to point at a custom directory or a single CSV file.

## Browser UI

The web UI keeps the happy path tiny:

- choose `Simulate` or `Backtest`
- enter one or more tickers
- pick `Demo sample`, `Live first`, or `Local CSV`

`Demo sample` opens instantly and stays deterministic. `Live first` starts
online and falls back to local CSVs. `Local CSV` accepts a single file or a
directory of `<TICKER>.csv` files with `Date` and `Close` columns.

For headless CLI environments:

```bash
export MPLBACKEND=Agg
```

## Migration Note

Legacy script entrypoints remain available as deprecated compatibility wrappers.

## Architecture

- `app.py` - lean Flask UI for the browser surface and local `monte-carlo-ui` entrypoint
- `public_cli.py` - public `monte-carlo simulate|backtest` parser and runners
- `simulate_cli.py` - shared simulation workflow used by public and legacy entrypoints
- `cli_shared.py` - shared parser and rendering helpers for CLI surfaces
- `cli.py` - deprecated simulation wrapper and compatibility facade
- `backtest.py` - walk-forward engine plus legacy wrapper
- `simulation.py` - vectorized simulation engines
- `analysis.py` - summary statistics
- `decision.py` - ranking, guardrails, and allocation rules
- `data.py` - Yahoo Finance fetch plus offline CSV fallback
- `viz.py` - distribution, path, and equity-curve plots

## Testing

```bash
uv run --with pytest pytest -q
```

Targeted examples:

```bash
uv run --with pytest pytest tests/test_cli.py -q
uv run --with pytest pytest tests/test_backtest.py -q
```

## License

MIT. See [LICENSE](LICENSE).
