# Output Guide

Use `--output` when the result needs to survive the terminal.

## Simulate

Example folder:

```text
results/
  action_plan.md
  allocations.csv
  rankings.csv
  report.json
  summaries.csv
  summaries.json
  AAPL_distribution.png
  AAPL_paths.png
  MSFT_distribution.png
  MSFT_paths.png
```

Open these first:

- `action_plan.md` is the shortest human-readable decision.
- `report.json` is the full machine-readable record, including inputs, rankings,
  allocations, path-aware allocation portfolio risk, errors, and `price_sources`.
- `rankings.csv` is the scored list of tickers after the model and guardrails.
- `allocations.csv` is the suggested weight for the names that survived. The
  matching `allocation_portfolio_summary` in `report.json` measures the
  simulated portfolio after cash buffers and risk-budget scaling.
- `summaries.csv` and `summaries.json` are the per-ticker simulation stats.
- `*_distribution.png` and `*_paths.png` are the fastest visual check for shape,
  spread, and downside.

## Backtest

Example folder:

```text
results/backtest/
  backtest_summary.csv
  rebalance_log.csv
  equity_curve.csv
  equity_curve.png
  price_sources.json
```

Open these first:

- `backtest_summary.csv` is the quickest scorecard for the process.
- `rebalance_log.csv` shows each rebalance date, selected tickers, weights,
  turnover, and transaction-cost drag.
- `equity_curve.csv` shows strategy, equal weight, and cash through time.
- `equity_curve.png` is the fastest read on path shape and drawdown.
- `price_sources.json` records where the history came from so live, offline,
  and fallback runs stay auditable after the terminal closes.

## Evaluate

Example folder:

```text
results/evaluation/
  scorecard.md
  runs.csv
  report.json
```

Open `scorecard.md` first. Use `runs.csv` to isolate unstable or failed matrix
cells. `report.json` preserves the manifest hash, normalized matrix, outcomes,
and source reliability for later audit.

## Good Handoff

- Send `action_plan.md` or `backtest_summary.csv` when someone only needs the
  decision.
- Send `report.json` or `rebalance_log.csv` when someone needs to audit the run.
- Keep the whole folder when the origin of the price data matters.
