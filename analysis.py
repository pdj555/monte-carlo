"""High-level analytics for simulated price paths."""

from __future__ import annotations

from typing import Sequence

import numpy as np
import pandas as pd

_DEFAULT_QUANTILES: tuple[float, ...] = (0.05, 0.25, 0.75, 0.95)


def summarize_final_prices(
    df: pd.DataFrame,
    *,
    current_price: float | None = None,
    quantiles: Sequence[float] | None = None,
) -> pd.Series:
    """Return summary statistics for the final simulated prices.

    Parameters
    ----------
    df : pandas.DataFrame
        Simulation results where the final row represents the final price of
        each scenario.
    current_price : float, optional
        Observed current price. When provided the summary includes expected
        return, probability of finishing above the current price and the 95%
        value at risk.
    quantiles : sequence of float, optional
        Additional quantiles to report. Values must be between ``0`` and ``1``.

    Returns
    -------
    pandas.Series
        Series containing descriptive statistics for the simulated prices.
    """

    if df.empty:
        raise ValueError("df must contain simulation results")

    final_prices = df.iloc[-1]
    summary = {
        "mean": float(final_prices.mean()),
        "median": float(final_prices.median()),
        "std": float(final_prices.std(ddof=1)),
        "min": float(final_prices.min()),
        "max": float(final_prices.max()),
    }

    quantiles = tuple(_DEFAULT_QUANTILES if quantiles is None else quantiles)
    for q in quantiles:
        if not 0 <= q <= 1:
            raise ValueError("quantiles must lie between 0 and 1")
        key = f"q{int(q * 100):02d}"
        summary[key] = float(final_prices.quantile(q))

    if current_price is not None:
        if current_price <= 0:
            raise ValueError("current_price must be positive when provided")
        summary["expected_return"] = float(final_prices.mean() / current_price - 1)
        summary["prob_above_current"] = float((final_prices > current_price).mean())
        summary["value_at_risk_95"] = float(
            current_price - final_prices.quantile(0.05)
        )
        summary["cvar_95"] = float(
            current_price - final_prices[final_prices <= final_prices.quantile(0.05)].mean()
        )

    return pd.Series(summary)


def calculate_sharpe_ratio(
    df: pd.DataFrame,
    *,
    risk_free_rate: float = 0.0,
    periods_per_year: int = 252,
) -> float:
    """Calculate the Sharpe ratio for simulated price paths.

    Parameters
    ----------
    df : pandas.DataFrame
        Simulation results where rows represent days and columns represent
        scenarios.
    risk_free_rate : float, optional
        Annual risk-free rate. Default is 0.0.
    periods_per_year : int, optional
        Number of trading periods per year. Default is 252 (trading days).

    Returns
    -------
    float
        Sharpe ratio averaged across all scenarios.

    Examples
    --------
    >>> import pandas as pd
    >>> import numpy as np
    >>> from simulation import simulate_prices
    >>> returns = pd.Series(np.random.randn(100) * 0.02)
    >>> sims = simulate_prices(returns, days=252, scenarios=100, seed=42)
    >>> sharpe = calculate_sharpe_ratio(sims)
    """
    if df.empty:
        raise ValueError("df must contain simulation results")

    # Calculate returns for each scenario
    returns = df.pct_change().dropna()
    
    # Average returns across scenarios
    mean_return = float(returns.mean().mean())
    std_return = float(returns.std().mean())
    
    if std_return == 0:
        return 0.0
    
    # Annualize
    annual_return = mean_return * periods_per_year
    annual_std = std_return * np.sqrt(periods_per_year)
    
    sharpe = (annual_return - risk_free_rate) / annual_std
    return float(sharpe)


def calculate_sortino_ratio(
    df: pd.DataFrame,
    *,
    target_return: float = 0.0,
    periods_per_year: int = 252,
) -> float:
    """Calculate the Sortino ratio for simulated price paths.

    The Sortino ratio is similar to the Sharpe ratio but only considers
    downside volatility, making it more appropriate for asymmetric returns.

    Parameters
    ----------
    df : pandas.DataFrame
        Simulation results where rows represent days and columns represent
        scenarios.
    target_return : float, optional
        Target or minimum acceptable return (annualized). Default is 0.0.
    periods_per_year : int, optional
        Number of trading periods per year. Default is 252 (trading days).

    Returns
    -------
    float
        Sortino ratio averaged across all scenarios.

    Examples
    --------
    >>> import pandas as pd
    >>> import numpy as np
    >>> from simulation import simulate_prices
    >>> returns = pd.Series(np.random.randn(100) * 0.02)
    >>> sims = simulate_prices(returns, days=252, scenarios=100, seed=42)
    >>> sortino = calculate_sortino_ratio(sims)
    """
    if df.empty:
        raise ValueError("df must contain simulation results")

    # Calculate returns for each scenario
    returns = df.pct_change().dropna()
    
    # Average returns across scenarios
    mean_return = float(returns.mean().mean())
    
    # Calculate downside deviation (only negative returns)
    downside_returns = returns.copy()
    downside_returns[downside_returns > 0] = 0
    downside_std = float(downside_returns.std().mean())
    
    if downside_std == 0:
        return 0.0
    
    # Annualize
    annual_return = mean_return * periods_per_year
    annual_downside_std = downside_std * np.sqrt(periods_per_year)
    
    sortino = (annual_return - target_return) / annual_downside_std
    return float(sortino)


def calculate_max_drawdown(df: pd.DataFrame) -> dict[str, float]:
    """Calculate the maximum drawdown for simulated price paths.

    Parameters
    ----------
    df : pandas.DataFrame
        Simulation results where rows represent days and columns represent
        scenarios.

    Returns
    -------
    dict
        Dictionary containing 'max_drawdown' (worst case), 'avg_drawdown' (average
        across scenarios), and 'median_drawdown'.

    Examples
    --------
    >>> import pandas as pd
    >>> import numpy as np
    >>> from simulation import simulate_prices
    >>> returns = pd.Series(np.random.randn(100) * 0.02)
    >>> sims = simulate_prices(returns, days=252, scenarios=100, seed=42)
    >>> dd = calculate_max_drawdown(sims)
    >>> print(f"Max drawdown: {dd['max_drawdown']:.2%}")
    """
    if df.empty:
        raise ValueError("df must contain simulation results")

    drawdowns = []
    
    for col in df.columns:
        prices = df[col]
        # Calculate running maximum
        running_max = prices.expanding().max()
        # Calculate drawdown from peak
        drawdown = (prices - running_max) / running_max
        # Get maximum drawdown (most negative)
        max_dd = float(drawdown.min())
        drawdowns.append(max_dd)
    
    return {
        "max_drawdown": float(np.min(drawdowns)),  # Worst case
        "avg_drawdown": float(np.mean(drawdowns)),  # Average
        "median_drawdown": float(np.median(drawdowns)),  # Median
    }


def calculate_risk_metrics(
    df: pd.DataFrame,
    *,
    current_price: float | None = None,
    risk_free_rate: float = 0.0,
    periods_per_year: int = 252,
) -> pd.Series:
    """Calculate comprehensive risk metrics for simulated price paths.

    Combines all available risk metrics into a single comprehensive summary.

    Parameters
    ----------
    df : pandas.DataFrame
        Simulation results where rows represent days and columns represent
        scenarios.
    current_price : float, optional
        Current/starting price for calculating VaR and CVaR.
    risk_free_rate : float, optional
        Annual risk-free rate for Sharpe ratio calculation. Default is 0.0.
    periods_per_year : int, optional
        Number of trading periods per year. Default is 252 (trading days).

    Returns
    -------
    pandas.Series
        Series containing comprehensive risk metrics including Sharpe ratio,
        Sortino ratio, maximum drawdown, VaR, and CVaR.

    Examples
    --------
    >>> import pandas as pd
    >>> import numpy as np
    >>> from simulation import simulate_prices
    >>> returns = pd.Series(np.random.randn(100) * 0.02)
    >>> sims = simulate_prices(returns, days=252, scenarios=100, current_price=100, seed=42)
    >>> metrics = calculate_risk_metrics(sims, current_price=100)
    >>> print(metrics)
    """
    if df.empty:
        raise ValueError("df must contain simulation results")

    metrics = {}
    
    # Sharpe and Sortino ratios
    metrics["sharpe_ratio"] = calculate_sharpe_ratio(
        df, risk_free_rate=risk_free_rate, periods_per_year=periods_per_year
    )
    metrics["sortino_ratio"] = calculate_sortino_ratio(
        df, periods_per_year=periods_per_year
    )
    
    # Drawdown metrics
    dd_metrics = calculate_max_drawdown(df)
    metrics.update(dd_metrics)
    
    # VaR and CVaR if current price is provided
    if current_price is not None:
        if current_price <= 0:
            raise ValueError("current_price must be positive when provided")
        
        final_prices = df.iloc[-1]
        
        # VaR at multiple confidence levels
        for conf in [0.90, 0.95, 0.99]:
            q = 1 - conf
            var = float(current_price - final_prices.quantile(q))
            metrics[f"var_{int(conf * 100)}"] = var
            
            # CVaR (Expected Shortfall)
            cvar = float(
                current_price - final_prices[final_prices <= final_prices.quantile(q)].mean()
            )
            metrics[f"cvar_{int(conf * 100)}"] = cvar
    
    return pd.Series(metrics)



__all__ = ["summarize_final_prices", "calculate_risk_metrics", "calculate_sharpe_ratio", "calculate_sortino_ratio", "calculate_max_drawdown"]
