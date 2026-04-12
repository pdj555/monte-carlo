# Monte Carlo Decision Engine

This project does two things:

1. simulate forward outcomes for current ideas
2. backtest the decision process on historical data

The main CLI has two commands:

- `monte-carlo simulate` for current opportunities
- `monte-carlo backtest` for walk-forward validation

Optional browser entrypoint:

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
monte-carlo simulate AAPL MSFT \
  --source offline \
  --data-path sample_data
```

### What it prints

- stance and headline
- data source
- top idea
- avoid list when guardrails fail
- cash buffer when conviction is low

Add `--details` when you want tables and secondary metrics.

### How to read the result

- `Stance` is the posture: lean in, selective, defensive, or stand aside.
- `Data source` tells you whether the run used live prices, local CSVs, or a fallback.
- `Top idea` is the first name to inspect; the suggested weight is a sizing hint, not an order.
- `Avoid` and `Cash buffer` are guardrails. Treat them as a signal to pass or keep more capital idle.

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

### How to read the result

- `Strategy return` is the outcome of the full rebalance process.
- `Annualized return` lets you compare runs with different window lengths.
- `Max drawdown` is the deepest peak-to-trough loss; smaller is easier to hold.
- `vs equal weight` asks whether the process beat a simple own-everything baseline.
- `vs cash` asks whether taking market risk paid for itself.
- Saved backtest folders also include `price_sources.json` so the origin of the prices survives the run.

## Data Sources

Use `--source` to pick how prices are loaded:

- `auto` tries live downloads first, then local CSV files
- `offline` uses local CSV files only
- `online` uses live downloads only

Run results tell you which source actually supplied the prices, so `auto`
stays honest when it falls back.

Local CSVs should include `Date` and `Close` columns. By default the repo looks in:

```text
sample_data/<TICKER>.csv
```

The bundled `sample_data` directory includes `AAPL.csv` and `MSFT.csv` so the
offline path stays deterministic out of the box.

Use `--data-path` to point at a custom directory or a single CSV file.

## Saved outputs

Use `--output` when the result needs to survive the terminal. The quickest
walkthrough of every saved artifact lives in
[docs/output-guide.md](docs/output-guide.md).

## Browser UI

The web UI keeps the happy path tiny:

- choose `Simulate` or `Backtest`
- enter one or more tickers
- pick `Demo sample`, `Try live data`, or `Local CSV`

`Demo sample` opens instantly and stays deterministic. `Try live data` starts
online and falls back to local CSVs. `Local CSV` accepts a single file or a
directory of `<TICKER>.csv` files with `Date` and `Close` columns.

For headless CLI environments:

```bash
export MPLBACKEND=Agg
```

## Migration Note

Legacy script entrypoints still work during migration, but each one now has one
obvious replacement:

- `python cli.py ...` -> `monte-carlo simulate ...`
- `python backtest.py ...` -> `monte-carlo backtest ...`
- `python MonteCarlo.py --ticker AAPL` -> `monte-carlo simulate AAPL --show`

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
