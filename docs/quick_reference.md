# Quick Reference Guide

This document provides quick examples and reference information for using the Monte Carlo simulation package.

## Table of Contents
- [Installation](#installation)
- [Basic Usage](#basic-usage)
- [Advanced Features](#advanced-features)
- [API Reference](#api-reference)
- [Examples](#examples)

## Installation

### From PyPI (when published)
```bash
pip install monte-carlo-stock-sim
```

### From Source
```bash
git clone https://github.com/pdj555/monte-carlo.git
cd monte-carlo
pip install -r requirements.txt
```

## Basic Usage

### Command Line Interface

Run a basic simulation:
```bash
python cli.py --tickers AAPL --days 252 --scenarios 1000
```

Run multiple tickers with GBM model:
```bash
python cli.py --tickers AAPL,MSFT,GOOGL --days 252 --scenarios 5000 --model gbm --seed 42
```

Save results to disk:
```bash
python cli.py --tickers AAPL --output ./results --no-show
```

Use offline data:
```bash
python cli.py --tickers AAPL --offline-only --offline-path ./sample_data
```

### Python API

Basic simulation:
```python
from data import fetch_prices
from simulation import simulate_prices
from analysis import summarize_final_prices
from viz import plot_distribution

# Fetch historical data
prices = fetch_prices("AAPL")
returns = prices.pct_change().dropna()
current_price = float(prices.iloc[-1])

# Run simulation
sims = simulate_prices(
    returns,
    days=252,
    scenarios=1000,
    current_price=current_price,
    seed=42
)

# Analyze results
summary = summarize_final_prices(sims, current_price=current_price)
print(summary)

# Visualize
plot_distribution(sims, ticker="AAPL")
```

## Advanced Features

### Risk Metrics

Calculate comprehensive risk metrics:
```python
from analysis import calculate_risk_metrics

metrics = calculate_risk_metrics(
    sims,
    current_price=100.0,
    risk_free_rate=0.02,
    periods_per_year=252
)

print(f"Sharpe Ratio: {metrics['sharpe_ratio']:.4f}")
print(f"Sortino Ratio: {metrics['sortino_ratio']:.4f}")
print(f"Max Drawdown: {metrics['max_drawdown']:.2%}")
print(f"VaR (95%): ${metrics['var_95']:.2f}")
print(f"CVaR (95%): ${metrics['cvar_95']:.2f}")
```

### Individual Risk Calculations

```python
from analysis import (
    calculate_sharpe_ratio,
    calculate_sortino_ratio,
    calculate_max_drawdown
)

sharpe = calculate_sharpe_ratio(sims, risk_free_rate=0.02)
sortino = calculate_sortino_ratio(sims, target_return=0.0)
dd = calculate_max_drawdown(sims)
```

### Geometric Brownian Motion

```python
from simulation import simulate_gbm

# Calculate parameters from historical data
mu = float(returns.mean())
sigma = float(returns.std())

# Run GBM simulation
gbm_sims = simulate_gbm(
    current_price=100.0,
    mu=mu,
    sigma=sigma,
    days=252,
    scenarios=1000,
    seed=42
)
```

### Multi-Ticker Analysis

```python
from cli import run, parse_args

# Run simulations for multiple tickers
args = parse_args(["--tickers", "AAPL,MSFT,GOOGL", "--scenarios", "5000"])
results = run(args)

simulations = results["simulations"]  # Combined DataFrame
summaries = results["summaries"]      # Summary statistics DataFrame
```

## API Reference

### Data Module

**fetch_prices(ticker, start=None, end=None, offline_path=None, prefer_local=False)**
- Fetch historical price data from Yahoo Finance or local CSV
- Returns: pandas.Series of closing prices

### Simulation Module

**simulate_prices(returns, days, scenarios, dt=1.0, seed=None, current_price=None)**
- Historical bootstrap Monte Carlo simulation
- Returns: pandas.DataFrame of shape (days, scenarios)

**simulate_gbm(current_price, mu, sigma, days, scenarios, dt=1.0, seed=None)**
- Geometric Brownian Motion simulation
- Returns: pandas.DataFrame of shape (days, scenarios)

### Analysis Module

**summarize_final_prices(df, current_price=None, quantiles=None)**
- Calculate summary statistics for final prices
- Returns: pandas.Series with mean, median, std, quantiles, etc.

**calculate_risk_metrics(df, current_price=None, risk_free_rate=0.0, periods_per_year=252)**
- Calculate comprehensive risk metrics
- Returns: pandas.Series with Sharpe, Sortino, VaR, CVaR, etc.

**calculate_sharpe_ratio(df, risk_free_rate=0.0, periods_per_year=252)**
- Calculate Sharpe ratio
- Returns: float

**calculate_sortino_ratio(df, target_return=0.0, periods_per_year=252)**
- Calculate Sortino ratio (downside risk-adjusted)
- Returns: float

**calculate_max_drawdown(df)**
- Calculate maximum drawdown statistics
- Returns: dict with max, avg, and median drawdown

### Visualization Module

**plot_distribution(df, ticker, title=None, palette="tab10")**
- Plot histogram of final prices
- Returns: matplotlib.figure.Figure

**plot_paths(df, ticker, title=None, palette="tab10")**
- Plot simulated price paths
- Returns: matplotlib.figure.Figure

## Examples

### Example 1: Basic Simulation and Analysis

```python
from data import fetch_prices
from simulation import simulate_prices
from analysis import summarize_final_prices
import matplotlib.pyplot as plt

# Fetch data
prices = fetch_prices("AAPL")
returns = prices.pct_change().dropna()
current = float(prices.iloc[-1])

# Simulate
sims = simulate_prices(returns, days=252, scenarios=1000, current_price=current, seed=42)

# Analyze
summary = summarize_final_prices(sims, current_price=current)
print(summary)
```

### Example 2: Compare Models

```python
from simulation import simulate_prices, simulate_gbm

# Historical bootstrap
hist_sims = simulate_prices(returns, days=252, scenarios=1000, current_price=100, seed=42)

# GBM
mu, sigma = float(returns.mean()), float(returns.std())
gbm_sims = simulate_gbm(current_price=100, mu=mu, sigma=sigma, days=252, scenarios=1000, seed=42)

# Compare risk metrics
from analysis import calculate_risk_metrics
hist_risk = calculate_risk_metrics(hist_sims, current_price=100)
gbm_risk = calculate_risk_metrics(gbm_sims, current_price=100)

import pandas as pd
comparison = pd.DataFrame({"Historical": hist_risk, "GBM": gbm_risk})
print(comparison)
```

### Example 3: Custom Quantiles and Analysis

```python
from analysis import summarize_final_prices

# Custom quantiles for more detailed distribution
custom_quantiles = [0.01, 0.05, 0.10, 0.25, 0.50, 0.75, 0.90, 0.95, 0.99]
summary = summarize_final_prices(sims, current_price=100, quantiles=custom_quantiles)
print(summary)
```

### Example 4: Batch Processing Multiple Tickers

```python
from data import fetch_prices
from simulation import simulate_prices
from analysis import calculate_risk_metrics
import pandas as pd

tickers = ["AAPL", "MSFT", "GOOGL", "AMZN"]
risk_metrics = {}

for ticker in tickers:
    try:
        prices = fetch_prices(ticker)
        returns = prices.pct_change().dropna()
        current = float(prices.iloc[-1])
        
        sims = simulate_prices(returns, days=252, scenarios=1000, current_price=current, seed=42)
        metrics = calculate_risk_metrics(sims, current_price=current)
        risk_metrics[ticker] = metrics
    except Exception as e:
        print(f"Error processing {ticker}: {e}")

# Combine into DataFrame
results_df = pd.DataFrame(risk_metrics).T
print(results_df)
```

### Example 5: Saving Results

```python
from pathlib import Path
from viz import plot_distribution, plot_paths

# Create output directory
output_dir = Path("./results")
output_dir.mkdir(exist_ok=True)

# Save plots
fig_dist = plot_distribution(sims, ticker="AAPL")
fig_dist.savefig(output_dir / "distribution.png", dpi=300, bbox_inches='tight')

fig_paths = plot_paths(sims, ticker="AAPL")
fig_paths.savefig(output_dir / "paths.png", dpi=300, bbox_inches='tight')

# Save data
sims.to_csv(output_dir / "simulations.csv")
summary.to_csv(output_dir / "summary.csv")
```

## Environment Variables

- `MPLBACKEND=Agg`: Use non-interactive backend (useful for servers/CI)

## Testing

Run the test suite:
```bash
pytest
```

Run with coverage:
```bash
pytest --cov=. --cov-report=html
```

## Demo Script

Run the comprehensive demo:
```bash
python demo.py --ticker AAPL
```

Run with offline data:
```bash
python demo.py --ticker AAPL --offline
```

## Tips and Best Practices

1. **Reproducibility**: Always set a `seed` parameter for reproducible results
2. **Sample Size**: Use at least 1,000 scenarios for reliable statistics, 10,000+ for production
3. **Time Horizon**: 252 days represents 1 trading year
4. **Model Selection**: Use historical bootstrap for fat tails and skewness, GBM for theoretical consistency
5. **Risk Metrics**: CVaR (Expected Shortfall) is generally preferred over VaR for risk management
6. **Performance**: For large simulations, consider using smaller time steps or fewer scenarios initially

## Common Issues

### Issue: Network timeout when fetching data
**Solution**: Use `--offline-only` flag or increase timeout in `data.py`

### Issue: Memory error with large simulations
**Solution**: Reduce scenarios or use chunked processing

### Issue: Plots not displaying
**Solution**: Ensure you're not in a headless environment, or use `--no-show` and save to disk

### Issue: Import errors
**Solution**: Ensure all dependencies are installed: `pip install -r requirements.txt`

## Further Reading

- [Strategic Plan](docs/strategic_plan.md) - Roadmap for future development
- [Constitution](docs/constitution.md) - Project principles and guidelines
- [Improvements](docs/improvements.md) - Detailed improvement proposals
- [README](README.md) - Main project documentation
