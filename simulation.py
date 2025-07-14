from __future__ import annotations

import numpy as np
import pandas as pd
from typing import Optional


def simulate_prices(
    returns: pd.Series,
    days: int,
    scenarios: int,
    dt: float = 1.0,
    seed: Optional[int] = None,
    current_price: Optional[float] = None,
) -> pd.DataFrame:
    """Vectorized Monte Carlo simulation of future prices.

    Parameters
    ----------
    returns : pd.Series
        Historical daily returns. The index may be dates or integers.
    days : int
        Number of future days to simulate.
    scenarios : int
        Number of simulated price paths.
    dt : float, optional
        Time increment for each step. ``1.0`` corresponds to daily steps.
    seed : int, optional
        Random seed for reproducible simulations.
    current_price : float, optional
        Current stock price to use as starting point. If None, returns
        cumulative returns starting from 1.0.

    Returns
    -------
    pandas.DataFrame
        Simulated future prices with shape ``(days, scenarios)``.
    """
    if days <= 0 or scenarios <= 0:
        raise ValueError("days and scenarios must be positive integers")

    rng = np.random.default_rng(seed)

    # Calculate drift and volatility from historical returns
    drift = returns.mean()
    volatility = returns.std()

    # Random shocks for each time step and scenario
    shocks = rng.standard_normal(size=(days, scenarios))

    # Daily returns based on drift and volatility
    # Extract scalar values to avoid pandas Series index alignment issues
    rets = drift.iloc[0] * dt + volatility.iloc[0] * np.sqrt(dt) * shocks

    # Cumulative product to get compounded returns
    cumulative = np.cumprod(1 + rets, axis=0)

    # Convert to actual stock prices if current_price is provided
    if current_price is not None:
        # Extract scalar value if current_price is a pandas Series
        if hasattr(current_price, 'iloc'):
            price_scalar = current_price.iloc[0]
        else:
            price_scalar = current_price
        cumulative = cumulative * price_scalar

    return pd.DataFrame(cumulative)
