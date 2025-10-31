#!/usr/bin/env python3
"""Demo script showcasing advanced features of the Monte Carlo simulation package.

This script demonstrates:
1. Running simulations with both historical and GBM models
2. Calculating advanced risk metrics (Sharpe, Sortino, max drawdown, VaR, CVaR)
3. Generating comprehensive visualizations
4. Comparing multiple simulation scenarios
"""

from __future__ import annotations

import matplotlib.pyplot as plt
import pandas as pd

from analysis import (
    calculate_risk_metrics,
    calculate_sharpe_ratio,
    calculate_sortino_ratio,
    calculate_max_drawdown,
    summarize_final_prices,
)
from data import fetch_prices, PriceDataError
from simulation import simulate_prices, simulate_gbm
from viz import plot_distribution, plot_paths

# Use consistent plotting style
plt.style.use("ggplot")


def run_demo(ticker: str = "AAPL", offline: bool = False) -> None:
    """Run a comprehensive demo of the Monte Carlo simulation features.
    
    Parameters
    ----------
    ticker : str
        Stock ticker symbol to simulate (default: AAPL)
    offline : bool
        If True, use offline sample data instead of fetching from the internet
    """
    
    print(f"{'=' * 80}")
    print(f"Monte Carlo Stock Price Simulation Demo - {ticker}")
    print(f"{'=' * 80}\n")

    # Step 1: Fetch historical data
    print("Step 1: Fetching historical price data...")
    try:
        if offline:
            prices = fetch_prices(ticker, prefer_local=True)
        else:
            prices = fetch_prices(ticker)
        print(f"✓ Loaded {len(prices)} days of historical data")
        print(f"  Current price: ${prices.iloc[-1]:.2f}")
        print(f"  Date range: {prices.index[0].date()} to {prices.index[-1].date()}\n")
    except PriceDataError as exc:
        print(f"✗ Error fetching data: {exc}")
        print("  Tip: Try running with offline=True to use sample data\n")
        return

    current_price = float(prices.iloc[-1])
    returns = prices.pct_change().dropna()

    # Step 2: Run Historical Bootstrap Simulation
    print("Step 2: Running Historical Bootstrap simulation...")
    print("  Parameters: 252 days, 1000 scenarios, seed=42")
    
    hist_sims = simulate_prices(
        returns,
        days=252,
        scenarios=1000,
        dt=1.0,
        seed=42,
        current_price=current_price,
    )
    print(f"✓ Generated simulation matrix: {hist_sims.shape}\n")

    # Step 3: Run Geometric Brownian Motion Simulation
    print("Step 3: Running Geometric Brownian Motion (GBM) simulation...")
    mu = float(returns.mean())
    sigma = float(returns.std())
    print(f"  Estimated parameters: μ={mu:.6f}, σ={sigma:.6f}")
    
    gbm_sims = simulate_gbm(
        current_price=current_price,
        mu=mu,
        sigma=sigma,
        days=252,
        scenarios=1000,
        dt=1.0,
        seed=42,
    )
    print(f"✓ Generated simulation matrix: {gbm_sims.shape}\n")

    # Step 4: Calculate Basic Summary Statistics
    print("Step 4: Summary statistics for final prices...")
    print("\nHistorical Bootstrap Model:")
    print("-" * 80)
    hist_summary = summarize_final_prices(hist_sims, current_price=current_price)
    print(hist_summary.to_frame(name="value").to_string(float_format=lambda v: f"{v:0.2f}"))
    
    print("\n\nGeometric Brownian Motion Model:")
    print("-" * 80)
    gbm_summary = summarize_final_prices(gbm_sims, current_price=current_price)
    print(gbm_summary.to_frame(name="value").to_string(float_format=lambda v: f"{v:0.2f}"))
    print()

    # Step 5: Calculate Advanced Risk Metrics
    print("Step 5: Advanced risk metrics...")
    print("\nHistorical Bootstrap Model:")
    print("-" * 80)
    hist_risk = calculate_risk_metrics(
        hist_sims,
        current_price=current_price,
        risk_free_rate=0.02,
        periods_per_year=252,
    )
    print(hist_risk.to_frame(name="value").to_string(float_format=lambda v: f"{v:0.4f}"))
    
    print("\n\nGeometric Brownian Motion Model:")
    print("-" * 80)
    gbm_risk = calculate_risk_metrics(
        gbm_sims,
        current_price=current_price,
        risk_free_rate=0.02,
        periods_per_year=252,
    )
    print(gbm_risk.to_frame(name="value").to_string(float_format=lambda v: f"{v:0.4f}"))
    print()

    # Step 6: Highlight Key Risk Metrics
    print("Step 6: Key risk metric comparisons...")
    print("-" * 80)
    
    comparison = pd.DataFrame({
        "Historical": [
            hist_risk["sharpe_ratio"],
            hist_risk["sortino_ratio"],
            hist_risk["max_drawdown"],
            hist_risk["var_95"],
            hist_risk["cvar_95"],
        ],
        "GBM": [
            gbm_risk["sharpe_ratio"],
            gbm_risk["sortino_ratio"],
            gbm_risk["max_drawdown"],
            gbm_risk["var_95"],
            gbm_risk["cvar_95"],
        ],
    }, index=["Sharpe Ratio", "Sortino Ratio", "Max Drawdown", "VaR (95%)", "CVaR (95%)"])
    
    print(comparison.to_string(float_format=lambda v: f"{v:0.4f}"))
    print()

    # Step 7: Generate Visualizations
    print("Step 7: Generating visualizations...")
    
    # Create a figure with 4 subplots
    fig = plt.figure(figsize=(16, 12))
    
    # Historical Bootstrap Distribution
    ax1 = plt.subplot(2, 2, 1)
    hist_sims.iloc[-1].hist(bins=50, ax=ax1, alpha=0.7, edgecolor='black')
    ax1.axvline(current_price, color='red', linestyle='--', linewidth=2, label='Current Price')
    ax1.axvline(hist_sims.iloc[-1].median(), color='green', linestyle='--', linewidth=2, label='Median')
    ax1.set_xlabel('Final Price ($)')
    ax1.set_ylabel('Frequency')
    ax1.set_title(f'{ticker} - Historical Bootstrap Distribution')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # GBM Distribution
    ax2 = plt.subplot(2, 2, 2)
    gbm_sims.iloc[-1].hist(bins=50, ax=ax2, alpha=0.7, color='orange', edgecolor='black')
    ax2.axvline(current_price, color='red', linestyle='--', linewidth=2, label='Current Price')
    ax2.axvline(gbm_sims.iloc[-1].median(), color='green', linestyle='--', linewidth=2, label='Median')
    ax2.set_xlabel('Final Price ($)')
    ax2.set_ylabel('Frequency')
    ax2.set_title(f'{ticker} - GBM Distribution')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # Historical Bootstrap Paths (sample)
    ax3 = plt.subplot(2, 2, 3)
    sample_paths = hist_sims.iloc[:, :50]  # Show 50 random paths
    sample_paths.plot(ax=ax3, legend=False, alpha=0.3, linewidth=0.8)
    ax3.set_xlabel('Trading Days')
    ax3.set_ylabel('Price ($)')
    ax3.set_title(f'{ticker} - Historical Bootstrap Simulated Paths (50 scenarios)')
    ax3.grid(True, alpha=0.3)
    
    # GBM Paths (sample)
    ax4 = plt.subplot(2, 2, 4)
    sample_paths = gbm_sims.iloc[:, :50]  # Show 50 random paths
    sample_paths.plot(ax=ax4, legend=False, alpha=0.3, linewidth=0.8, color='orange')
    ax4.set_xlabel('Trading Days')
    ax4.set_ylabel('Price ($)')
    ax4.set_title(f'{ticker} - GBM Simulated Paths (50 scenarios)')
    ax4.grid(True, alpha=0.3)
    
    plt.tight_layout()
    print("✓ Visualization complete")
    print("\nDisplaying plots...")
    plt.show()

    # Step 8: Summary
    print("\n" + "=" * 80)
    print("Demo Summary")
    print("=" * 80)
    print(f"✓ Successfully simulated {ticker} stock prices")
    print(f"✓ Compared Historical Bootstrap vs GBM models")
    print(f"✓ Calculated advanced risk metrics (Sharpe, Sortino, VaR, CVaR)")
    print(f"✓ Generated comprehensive visualizations")
    print("\nKey Insights:")
    print(f"  • Expected return (Historical): {hist_summary['expected_return']:.2%}")
    print(f"  • Expected return (GBM): {gbm_summary['expected_return']:.2%}")
    print(f"  • Probability of profit (Historical): {hist_summary['prob_above_current']:.2%}")
    print(f"  • Probability of profit (GBM): {gbm_summary['prob_above_current']:.2%}")
    print(f"  • Sharpe ratio (Historical): {hist_risk['sharpe_ratio']:.4f}")
    print(f"  • Sharpe ratio (GBM): {gbm_risk['sharpe_ratio']:.4f}")
    print("=" * 80 + "\n")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Demo of Monte Carlo simulation advanced features",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--ticker",
        default="AAPL",
        help="Stock ticker symbol to simulate",
    )
    parser.add_argument(
        "--offline",
        action="store_true",
        help="Use offline sample data instead of fetching from the internet",
    )
    
    args = parser.parse_args()
    
    try:
        run_demo(ticker=args.ticker, offline=args.offline)
    except KeyboardInterrupt:
        print("\n\nDemo interrupted by user.")
    except Exception as exc:
        print(f"\n✗ Demo failed with error: {exc}")
        import traceback
        traceback.print_exc()
