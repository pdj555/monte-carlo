"""Monte Carlo simulation helpers."""

from __future__ import annotations

from typing import Optional

import numpy as np
import pandas as pd

__all__ = ["estimate_gbm_parameters", "simulate_prices", "simulate_gbm"]


def _as_float(value: float | pd.Series) -> float:
    """Return a scalar float from ``value`` which may be a pandas object."""

    if hasattr(value, "iloc"):
        scalar = value.iloc[-1]
    else:
        scalar = value
    return float(scalar)


def estimate_gbm_parameters(returns: pd.Series) -> tuple[float, float]:
    """Estimate GBM drift and volatility from simple returns.

    The returned parameters match the standard GBM formulation used by
    :func:`simulate_gbm` where log returns have mean
    ``(mu - 0.5*sigma**2) * dt`` and standard deviation
    ``sigma * sqrt(dt)``.
    """

    if returns.empty:
        raise ValueError("returns must contain at least one observation")

    cleaned = pd.to_numeric(returns, errors="coerce").dropna()
    if cleaned.empty:
        raise ValueError("returns must contain at least one numeric observation")

    values = cleaned.to_numpy(dtype=float)
    if not np.isfinite(values).all():
        raise ValueError("returns must contain only finite values")
    if (values <= -1.0).any():
        raise ValueError("returns must be greater than -1.0")

    log_returns = np.log1p(values)
    sigma = float(log_returns.std(ddof=1))
    mu = float(log_returns.mean() + 0.5 * sigma**2)
    return mu, sigma


def simulate_prices(
    returns: pd.Series,
    days: int,
    scenarios: int,
    dt: float = 1.0,
    seed: Optional[int] = None,
    current_price: Optional[float | pd.Series] = None,
) -> pd.DataFrame:
    """Vectorized Monte Carlo simulation of future prices using historical data.

    The default approach bootstraps (resamples) historical returns with
    replacement to preserve the empirical distribution (fat tails, skew,
    etc.). Providing ``current_price`` converts the normalised paths into
    absolute price levels.

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
    current_price : float or pandas.Series, optional
        Starting price for the simulated paths. If omitted the output is
        normalised to start at ``1.0``.

    Returns
    -------
    pandas.DataFrame
        Simulated future prices with shape ``(days, scenarios)``.

    Examples
    --------
    >>> import pandas as pd
    >>> import numpy as np
    >>> returns = pd.Series(np.random.randn(100) * 0.02)
    >>> simulate_prices(returns, days=30, scenarios=500).head()
    """

    if days <= 0 or scenarios <= 0:
        raise ValueError("days and scenarios must be positive integers")
    if dt <= 0:
        raise ValueError("dt must be positive")
    if returns.empty:
        raise ValueError("returns must contain at least one observation")

    cleaned = pd.to_numeric(returns, errors="coerce").dropna()
    if cleaned.empty:
        raise ValueError("returns must contain at least one numeric observation")

    values = cleaned.to_numpy(dtype=float)
    if not np.isfinite(values).all():
        raise ValueError("returns must contain only finite values")
    if (values <= -1.0).any():
        raise ValueError("returns must be greater than -1.0")

    rng = np.random.default_rng(seed)
    sample_indices = rng.integers(0, values.shape[0], size=(days, scenarios))
    sampled_returns = values[sample_indices]

    if dt != 1.0:
        sampled_returns = np.expm1(np.log1p(sampled_returns) * dt)

    cumulative = np.cumprod(1 + sampled_returns, axis=0)

    if current_price is not None:
        price_scalar = _as_float(current_price)
        if price_scalar <= 0:
            raise ValueError("current_price must be positive")
        cumulative = cumulative * price_scalar

    index = pd.RangeIndex(start=1, stop=days + 1, name="day")
    columns = pd.RangeIndex(start=1, stop=scenarios + 1, name="scenario")
    return pd.DataFrame(cumulative, index=index, columns=columns)


def simulate_gbm(
    *,
    current_price: float | pd.Series,
    mu: float,
    sigma: float,
    days: int,
    scenarios: int,
    dt: float = 1.0,
    seed: Optional[int] = None,
) -> pd.DataFrame:
    """Simulate price paths using the geometric Brownian motion model.

    Parameters
    ----------
    current_price : float or pandas.Series
        Starting price for the simulation. If a Series is passed the final
        value is used.
    mu : float
        Drift term of the GBM process per time unit.
    sigma : float
        Volatility per time unit.
    days : int
        Number of future days to simulate.
    scenarios : int
        Number of simulated price paths.
    dt : float, optional
        Time increment for each step. ``1.0`` corresponds to daily steps.
    seed : int, optional
        Random seed for reproducibility.

    Returns
    -------
    pandas.DataFrame
        Simulated price paths following the GBM dynamics.

    Notes
    -----
    The simulated log returns follow ``mu - 0.5 * sigma**2`` drift with
    ``sigma`` volatility. This mirrors the standard GBM formulation used in
    option pricing and many risk models.
    """

    if days <= 0 or scenarios <= 0:
        raise ValueError("days and scenarios must be positive integers")
    if dt <= 0:
        raise ValueError("dt must be positive")
    if sigma < 0:
        raise ValueError("sigma must be non-negative")

    start_price = _as_float(current_price)
    if start_price <= 0:
        raise ValueError("current_price must be positive")

    rng = np.random.default_rng(seed)
    shocks = rng.standard_normal(size=(days, scenarios))

    drift = (mu - 0.5 * sigma**2) * dt
    diffusion = sigma * np.sqrt(dt) * shocks
    log_returns = drift + diffusion
    cumulative = np.exp(np.cumsum(log_returns, axis=0)) * start_price

    index = pd.RangeIndex(start=1, stop=days + 1, name="day")
    columns = pd.RangeIndex(start=1, stop=scenarios + 1, name="scenario")
    return pd.DataFrame(cumulative, index=index, columns=columns)
