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

    Returns
    -------
    pandas.DataFrame
        Simulated cumulative returns with shape ``(days, scenarios)``.
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

    return pd.DataFrame(cumulative)
