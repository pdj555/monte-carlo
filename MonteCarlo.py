"""Run Monte Carlo stock price simulations from the command line.

This script is intentionally small. It simply wires together the helper
modules that handle data retrieval, running the actual simulations and
visualising the results. The command-line defaults mirror the old behaviour
so it can be executed without any arguments.
"""

from __future__ import annotations

import argparse
import matplotlib.pyplot as plt

# Core functionality lives in these modules so they can be reused elsewhere
from data import fetch_prices, PriceDataError
from simulation import simulate_prices
from viz import plot_distribution, plot_paths

# Use a consistent aesthetic for plots
plt.style.use("ggplot")


def parse_args() -> argparse.Namespace:
    """Return CLI options controlling the simulation."""
    # The parser mirrors options previously set as module constants
    parser = argparse.ArgumentParser(
        description="Run a Monte Carlo stock price simulation."
    )
    parser.add_argument(
        "--ticker",
        default="AAPL",
        help="Stock ticker symbol to simulate (default: %(default)s)",
    )
    parser.add_argument(
        "--days",
        type=int,
        default=365,
        help="Number of future trading days (default: %(default)s)",
    )
    parser.add_argument(
        "--scenarios",
        type=int,
        default=10000,
        help="Number of simulated price paths (default: %(default)s)",
    )
    parser.add_argument(
        "--dt",
        type=float,
        default=1.0,
        help="Time increment for each step (default: %(default)s)",
    )
    return parser.parse_args()


def main() -> None:
    """Fetch data, run simulations, and plot the results."""
    args = parse_args()

    # 1. Pull historical data then derive daily returns
    try:
        prices = fetch_prices(args.ticker)
    except PriceDataError as exc:
        print(f"Error fetching prices: {exc}")
        return

    returns = prices.pct_change().dropna()

    # 2. Produce a matrix of simulated future prices
    sims = simulate_prices(
        returns, days=args.days, scenarios=args.scenarios, dt=args.dt
    )

    # 3. Display a histogram of where each scenario ends up
    plot_distribution(sims, ticker=args.ticker)
    plt.show()

    # 4. Visualise a subset of simulated paths over time
    plot_paths(sims, ticker=args.ticker)
    plt.show()


if __name__ == "__main__":
    main()
