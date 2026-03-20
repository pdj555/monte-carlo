# Monte Carlo Decision Engine

This repository is for one job:

1. **simulate forward outcomes**
2. **validate the decision process with walk-forward backtests**

That is the whole product.

The codebase is intentionally focused on decision-grade output:

- scenario distributions
- downside metrics
- ranking and allocation rules
- historical validation against equal-weight and cash baselines

## Why this exists

Raw Monte Carlo charts are not enough.

If a system recommends capital allocation, it should also answer:

> would this process have worked historically after turnover and costs?

This project now supports both sides of that question:

- `cli.py` for **forward-looking simulation**
- `backtest.py` for **historical walk-forward validation**

## Install

Python 3.9+ is required.

```bash
uv pip install -r requirements.txt
```

For headless environments:

```bash
export MPLBACKEND=Agg
```

## Workflow 1: forward simulation

Use `cli.py` when you want to rank current opportunities from simulated future paths.

### Offline example

```bash
python cli.py \
  --tickers AAPL \
  --days 60 \
  --scenarios 2000 \
  --model historical \
  --offline-only \
  --offline-path sample_data \
  --no-show
```

### What it produces

- per-ticker summary statistics
- expected return and probability of finishing above current price
- downside metrics:
  - VaR
  - expected shortfall
  - drawdown risk
- ranking and allocation suggestions
- equal-weight portfolio summary when multiple tickers are supplied

### High-value flags

| Flag | Purpose |
| --- | --- |
| `--model historical` / `--model gbm` | choose the simulation engine |
| `--seed` | deterministic runs |
| `--offline-only` | disable network and use local CSVs |
| `--output` | save plots and reports |
| `--min-expected-return` | hard floor for investable names |
| `--min-prob-up` | require upside probability |
| `--max-var-95-pct` | cap downside risk |
| `--portfolio-risk-budget-pct` | hard cap on blended portfolio VaR |
| `--capital` | emit executable dollar/share sizing |

## Workflow 2: walk-forward validation

Use `backtest.py` when you want proof instead of optimism.

It repeatedly:

1. looks back at trailing history
2. simulates the next holding window
3. ranks and allocates capital
4. measures what actually happened
5. compounds the result against benchmarks

### Offline example

```bash
python backtest.py \
  --tickers AAPL \
  --lookback-days 60 \
  --holding-days 20 \
  --rebalance-every 20 \
  --top-k 1 \
  --model gbm \
  --scenarios 1000 \
  --offline-only \
  --offline-path sample_data \
  --output results/backtest
```

### What it produces

- `backtest_summary.csv`
- `rebalance_log.csv`
- `equity_curve.csv`
- `equity_curve.png`

### Core metrics

- strategy total return
- strategy annualized return
- strategy max drawdown
- win rate by rebalance period
- average turnover
- transaction cost drag
- excess return vs equal-weight benchmark
- excess return vs cash

## Offline data

Offline mode expects CSV files with `Date` and `Close` columns.

By default the project looks for:

```text
sample_data/<TICKER>.csv
```

You can also point both CLIs at a custom directory:

```bash
--offline-path /path/to/csvs
```

## Architecture

- `simulation.py` — historical bootstrap, GBM, prediction-market simulation
- `analysis.py` — pure path/statistical summaries
- `decision.py` — ranking, guardrails, allocation, execution sizing
- `backtest.py` — walk-forward validation engine
- `viz.py` — path, distribution, and equity-curve plots
- `data.py` — Yahoo Finance fetch plus offline CSV fallback
- `cli.py` — forward simulation entrypoint

## Advanced features

These are secondary. They exist, but they are not the product center:

- optional OpenAI-written narrative summaries
- prediction-market probability simulation
- policy-file driven guardrails
- tamper-evident decision journals
- stress overlays via shock events

## Testing

Run the test suite with:

```bash
uv run --with pytest pytest -q
```

Prefer targeted runs while iterating, for example:

```bash
uv run --with pytest pytest tests/test_backtest.py -q
uv run --with pytest pytest tests/test_cli.py -q
```

## License

MIT. See [LICENSE](LICENSE).
