# Monte Carlo Decision Engine

This project does two things:

1. simulate forward outcomes for current ideas
2. backtest the decision process on historical data

The public interface is one command:

- `monte-carlo simulate` for current opportunities
- `monte-carlo backtest` for walk-forward validation

## Install

Python 3.9+ is required.

```bash
python3 -m pip install -e .
```

After install, use the packaged command:

```bash
monte-carlo --help
```

For headless environments:

```bash
export MPLBACKEND=Agg
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

```bash
monte-carlo backtest AAPL \
  --source offline \
  --data-path sample_data
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

## Migration Note

Legacy script entrypoints remain available as deprecated compatibility wrappers.

## Architecture

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
