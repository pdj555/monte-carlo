from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from simulation import estimate_gbm_parameters, simulate_gbm, simulate_prices


def test_simulate_prices_shape_and_seed():
    rng = np.random.default_rng(0)
    returns = pd.Series(rng.normal(loc=0.001, scale=0.02, size=100))

    sims_a = simulate_prices(returns, days=30, scenarios=50, seed=123)
    sims_b = simulate_prices(returns, days=30, scenarios=50, seed=123)

    assert sims_a.shape == (30, 50)
    assert sims_a.equals(sims_b)


def test_simulate_prices_with_current_price():
    returns = pd.Series([0.01, -0.02, 0.015, 0.03])
    baseline = simulate_prices(returns, days=10, scenarios=5, seed=7)
    sims = simulate_prices(returns, days=10, scenarios=5, seed=7, current_price=200.0)

    # Scaling by the current price should simply scale the normalised paths.
    np.testing.assert_allclose(sims.values, baseline.values * 200.0, rtol=1e-6)


def test_simulate_gbm_reproducible():
    paths_a = simulate_gbm(
        current_price=150.0,
        mu=0.001,
        sigma=0.02,
        days=20,
        scenarios=10,
        dt=1.0,
        seed=99,
    )
    paths_b = simulate_gbm(
        current_price=150.0,
        mu=0.001,
        sigma=0.02,
        days=20,
        scenarios=10,
        dt=1.0,
        seed=99,
    )

    assert paths_a.equals(paths_b)
    assert paths_a.shape == (20, 10)


def test_estimate_gbm_parameters_produces_valid_inputs():
    returns = pd.Series([0.01, -0.005, 0.02, 0.0, 0.015])
    mu, sigma = estimate_gbm_parameters(returns)

    assert np.isfinite(mu)
    assert sigma >= 0.0

    paths = simulate_gbm(
        current_price=100.0,
        mu=mu,
        sigma=sigma,
        days=5,
        scenarios=3,
        seed=123,
    )
    assert paths.shape == (5, 3)


def test_simulate_prices_rejects_invalid_dt():
    returns = pd.Series([0.01, -0.02, 0.015])
    with pytest.raises(ValueError):
        simulate_prices(returns, days=10, scenarios=5, dt=0.0)


def test_shock_overlay_reduces_terminal_prices():
    returns = pd.Series([0.01, 0.0, -0.01, 0.02, 0.005])

    baseline = simulate_prices(returns, days=15, scenarios=200, seed=42)
    shocked = simulate_prices(
        returns,
        days=15,
        scenarios=200,
        seed=42,
        shock_probability=0.2,
        shock_return=-0.2,
    )

    assert shocked.iloc[-1].mean() < baseline.iloc[-1].mean()


def test_simulate_gbm_rejects_invalid_shock_probability():
    with pytest.raises(ValueError):
        simulate_gbm(
            current_price=100.0,
            mu=0.001,
            sigma=0.02,
            days=5,
            scenarios=10,
            shock_probability=1.5,
        )
