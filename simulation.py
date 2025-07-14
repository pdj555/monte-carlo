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
    current_price : float or pd.Series, optional
        Current stock price to use as starting point. If None, returns
        cumulative returns starting from 1.0 (normalized). If provided,
        the function returns absolute price paths starting from this value.
        Can be a scalar float or a pandas Series (in which case the first
        value will be used).

    Returns
    -------
    pandas.DataFrame
        Simulated future prices with shape ``(days, scenarios)``.
        If current_price is None, returns cumulative returns starting from 1.0.
        If current_price is provided, returns absolute price paths starting
        from the given price.
        
    Examples
    --------
    >>> import pandas as pd
    >>> import numpy as np
    >>> returns = pd.Series(np.random.randn(100) * 0.02)  # 2% daily volatility
    >>> 
    >>> # Simulate normalized returns (starting from 1.0)
    >>> sims = simulate_prices(returns, days=30, scenarios=1000)
    >>> 
    >>> # Simulate absolute prices (starting from current price)
    >>> sims = simulate_prices(returns, days=30, scenarios=1000, current_price=150.0)
    """
    if days <= 0 or scenarios <= 0:
        raise ValueError("days and scenarios must be positive integers")

    rng = np.random.default_rng(seed)

    # Calculate drift and volatility from historical returns
    drift_val = returns.mean()
    volatility_val = returns.std()
    
    # Convert to float, handling both scalar and Series cases
    drift = float(drift_val.iloc[0] if hasattr(drift_val, 'iloc') else drift_val)
    volatility = float(volatility_val.iloc[0] if hasattr(volatility_val, 'iloc') else volatility_val)
    
    # Random shocks for each time step and scenario
    shocks = rng.standard_normal(size=(days, scenarios))

    # Daily returns based on drift and volatility
    # drift and volatility are scalar values, not Series
    rets = drift * dt + volatility * np.sqrt(dt) * shocks

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
