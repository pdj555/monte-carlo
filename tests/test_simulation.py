from __future__ import annotations

import numpy as np
import pandas as pd

from simulation import simulate_gbm, simulate_prices


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
