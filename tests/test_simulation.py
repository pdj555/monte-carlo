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


def test_estimate_gbm_parameters_requires_two_observations():
    with pytest.raises(ValueError, match="at least two observations"):
        estimate_gbm_parameters(pd.Series([0.01]))


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


def test_simulate_prices_block_bootstrap_is_reproducible():
    returns = pd.Series([0.01, -0.02, 0.03, -0.01, 0.02])

    sims_a = simulate_prices(returns, days=12, scenarios=8, seed=42, block_size=3)
    sims_b = simulate_prices(returns, days=12, scenarios=8, seed=42, block_size=3)

    assert sims_a.equals(sims_b)
    assert sims_a.shape == (12, 8)


def test_simulate_prices_rejects_invalid_block_size():
    returns = pd.Series([0.01, -0.02, 0.015])
    with pytest.raises(ValueError):
        simulate_prices(returns, days=10, scenarios=5, block_size=0)


def test_simulate_prediction_market_is_bounded_and_reproducible():
    from simulation import simulate_prediction_market

    sims_a = simulate_prediction_market(
        fundamental_probability=0.62,
        current_price=0.55,
        days=30,
        scenarios=200,
        certainty=120.0,
        mean_reversion=0.25,
        daily_volatility=0.04,
        seed=123,
    )
    sims_b = simulate_prediction_market(
        fundamental_probability=0.62,
        current_price=0.55,
        days=30,
        scenarios=200,
        certainty=120.0,
        mean_reversion=0.25,
        daily_volatility=0.04,
        seed=123,
    )

    assert sims_a.equals(sims_b)
    assert sims_a.shape == (30, 200)
    assert sims_a.to_numpy().min() > 0.0
    assert sims_a.to_numpy().max() < 1.0


def test_simulate_prediction_market_first_row_is_first_future_step():
    from simulation import simulate_prediction_market

    sims = simulate_prediction_market(
        fundamental_probability=0.8,
        current_price=0.2,
        days=3,
        scenarios=2,
        certainty=10_000.0,
        mean_reversion=1.0,
        daily_volatility=0.0,
        seed=1,
    )

    assert sims.iloc[0].to_numpy() == pytest.approx(sims.iloc[1].to_numpy())
    assert not np.allclose(sims.iloc[0].to_numpy(), 0.2)


def test_simulate_prediction_market_rejects_out_of_range_probability():
    from simulation import simulate_prediction_market

    with pytest.raises(ValueError):
        simulate_prediction_market(fundamental_probability=1.0, days=10, scenarios=5)


def test_simulate_prices_scales_volatility_with_sqrt_dt():
    rng = np.random.default_rng(3)
    returns = pd.Series(rng.normal(loc=0.0005, scale=0.02, size=2000))
    hist = np.log1p(returns.to_numpy())

    sims = simulate_prices(returns, days=200, scenarios=2000, dt=0.25, seed=5)
    steps = np.diff(np.log(np.vstack([np.ones(2000), sims.to_numpy()])), axis=0)

    # A quarter-day step carries half the daily volatility and a quarter of the drift.
    assert steps.std() == pytest.approx(0.5 * hist.std(), rel=0.02)
    assert steps.mean() == pytest.approx(0.25 * hist.mean(), abs=2e-4)


def test_historical_and_gbm_models_agree_on_dt_scaling():
    rng = np.random.default_rng(4)
    returns = pd.Series(rng.normal(loc=0.0, scale=0.015, size=3000))
    mu, sigma = estimate_gbm_parameters(returns)
    kwargs = {"days": 100, "scenarios": 3000, "dt": 0.1, "seed": 9}

    bootstrap = simulate_prices(returns, current_price=1.0, **kwargs)
    gbm = simulate_gbm(current_price=1.0, mu=mu, sigma=sigma, **kwargs)

    def step_vol(paths: pd.DataFrame) -> float:
        logs = np.log(np.vstack([np.ones(paths.shape[1]), paths.to_numpy()]))
        return float(np.diff(logs, axis=0).std())

    assert step_vol(bootstrap) == pytest.approx(step_vol(gbm), rel=0.03)


def _correlated_returns(rho: float, n: int = 1000, seed: int = 7) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    z = rng.multivariate_normal([0.0, 0.0], [[1.0, rho], [rho, 1.0]], size=n) * 0.02
    return pd.DataFrame({"A": np.expm1(z[:, 0]), "B": np.expm1(z[:, 1])})


def test_joint_bootstrap_with_one_ticker_matches_the_single_ticker_path():
    from simulation import simulate_joint_prices

    returns = _correlated_returns(0.5)["A"]
    common = dict(days=20, scenarios=40, seed=3, block_size=5)
    single = simulate_prices(returns, current_price=10.0, **common)
    joint = simulate_joint_prices(returns.to_frame("A"), current_prices={"A": 10.0}, **common)
    np.testing.assert_allclose(joint.xs("A", axis=1, level="ticker").to_numpy(), single.to_numpy())


@pytest.mark.parametrize("model", ["historical", "gbm"])
def test_joint_simulation_keeps_the_correlation_in_the_history(model):
    from simulation import simulate_joint_gbm, simulate_joint_prices

    returns = _correlated_returns(0.8)
    simulate = simulate_joint_prices if model == "historical" else simulate_joint_gbm
    paths = simulate(returns, days=1, scenarios=20000, seed=1)
    day_one = np.log(paths.iloc[0].unstack("ticker"))
    simulated = np.corrcoef(day_one["A"], day_one["B"])[0, 1]
    historical = np.corrcoef(np.log1p(returns.to_numpy()), rowvar=False)[0, 1]
    assert simulated == pytest.approx(historical, abs=0.02)


@pytest.mark.parametrize("cash_return", [0.0, 0.01, -0.01])
def test_joint_gbm_keeps_volatile_correlations_with_constant_asset(cash_return):
    from simulation import simulate_joint_gbm

    returns = _correlated_returns(0.8)
    returns.insert(1, "CASH", cash_return)
    prices = {"A": 100.0, "CASH": 50.0, "B": 200.0}
    paths = simulate_joint_gbm(
        returns, days=5, scenarios=10000, dt=0.25, seed=1, current_prices=prices
    )
    first = np.log(paths.iloc[0].unstack("ticker").div(pd.Series(prices)))
    historical = np.log1p(returns[["A", "B"]]).corr().iloc[0, 1]
    assert first[["A", "B"]].corr().iloc[0, 1] == pytest.approx(historical, abs=0.02)
    assert np.isfinite(paths.to_numpy()).all()

    expected = 50.0 * (1.0 + cash_return) ** (np.arange(1, 6) * 0.25)
    np.testing.assert_allclose(paths["CASH"].iloc[:, 0], expected, rtol=1e-14)
    assert (np.ptp(paths["CASH"].to_numpy(), axis=1) == 0.0).all()


def test_joint_gbm_cash_does_not_create_extra_diversification():
    from analysis import summarize_equal_weight_portfolio
    from simulation import simulate_joint_gbm

    returns = _correlated_returns(0.8)
    common = dict(days=20, scenarios=10000, seed=42)
    pair_prices = {"A": 100.0, "B": 100.0}
    pair = simulate_joint_gbm(returns, current_prices=pair_prices, **common)
    pair_risk = summarize_equal_weight_portfolio(pair, current_prices=pair_prices)

    returns.insert(1, "CASH", 0.0)
    prices = {**pair_prices, "CASH": 100.0}
    with_cash = simulate_joint_gbm(returns, current_prices=prices, **common)
    cash_risk = summarize_equal_weight_portfolio(with_cash, current_prices=prices)
    # One third cash scales risk by 2/3; it must not decorrelate the stocks as well.
    # The two draws use different RNG dimensions, so allow Monte Carlo sampling error.
    for metric in ("value_at_risk_95_pct", "expected_shortfall_95_pct"):
        assert cash_risk[metric] == pytest.approx(pair_risk[metric] * 2.0 / 3.0, rel=0.06)


@pytest.mark.parametrize("rho", [-1.0, 1.0])
def test_joint_gbm_handles_singular_correlations_with_constant_asset(rho):
    from simulation import simulate_joint_gbm

    logs = np.linspace(-0.02, 0.02, 100)
    returns = pd.DataFrame({"A": np.expm1(logs), "CASH": 0.0, "B": np.expm1(rho * logs)})
    paths = simulate_joint_gbm(returns, days=3, scenarios=2000, seed=2)
    first = np.log(paths.iloc[0].unstack("ticker"))
    assert np.isfinite(paths.to_numpy()).all()
    assert first[["A", "B"]].corr().iloc[0, 1] == pytest.approx(rho, abs=1e-8)
    np.testing.assert_array_equal(paths["CASH"].to_numpy(), 1.0)


@pytest.mark.parametrize("volatile", [False, True])
def test_joint_gbm_handles_at_most_one_volatile_asset(volatile):
    from simulation import simulate_joint_gbm

    returns = pd.DataFrame({"CASH": [0.01] * 1000, "FLAT": [0.0] * 1000})
    if volatile:
        returns["A"] = _correlated_returns(0.0)["A"]
    paths = simulate_joint_gbm(returns, days=3, scenarios=100, seed=4)
    assert np.isfinite(paths.to_numpy()).all()
    expected = np.broadcast_to((1.01 ** np.arange(1, 4))[:, None], (3, 100))
    np.testing.assert_allclose(paths["CASH"].to_numpy(), expected, rtol=1e-14)
    np.testing.assert_array_equal(paths["FLAT"].to_numpy(), 1.0)
    if volatile:
        assert paths["A"].iloc[0].std() > 0.0


def test_identical_histories_give_identical_paths():
    from simulation import simulate_joint_prices

    one = _correlated_returns(0.0)["A"]
    twins = pd.DataFrame({"A": one, "B": one})
    paths = simulate_joint_prices(twins, days=10, scenarios=100, seed=0)
    np.testing.assert_allclose(paths["A"].to_numpy(), paths["B"].to_numpy())


def test_joint_shock_hits_every_ticker_on_the_same_day():
    from simulation import simulate_joint_prices

    flat = pd.DataFrame({"A": [0.0] * 10, "B": [0.0] * 10})
    paths = simulate_joint_prices(
        flat, days=50, scenarios=200, seed=2, shock_probability=0.1, shock_return=-0.5
    )
    np.testing.assert_allclose(paths["A"].to_numpy(), paths["B"].to_numpy())
    assert (paths["A"].iloc[-1] < 1.0).any()


@pytest.mark.parametrize("model", ["historical", "gbm"])
def test_joint_simulation_uses_only_dates_every_ticker_traded(model):
    from simulation import simulate_joint_gbm, simulate_joint_prices

    simulate = simulate_joint_prices if model == "historical" else simulate_joint_gbm
    returns = pd.DataFrame({"A": [0.01, 0.02, None, 0.03], "B": [None, 0.01, 0.02, 0.04]})
    paths = simulate(returns, days=3, scenarios=5, seed=0)
    trimmed = simulate(returns.dropna(), days=3, scenarios=5, seed=0)
    pd.testing.assert_frame_equal(paths, trimmed)


@pytest.mark.parametrize("model", ["historical", "gbm"])
@pytest.mark.parametrize("overlap", [0, 1])
def test_joint_simulation_rejects_insufficient_overlapping_history(model, overlap):
    from simulation import simulate_joint_gbm, simulate_joint_prices

    simulate = simulate_joint_prices if model == "historical" else simulate_joint_gbm
    returns = pd.DataFrame({"A": [0.01, 0.02, None], "B": [None, None, 0.03]})
    if overlap:
        returns.loc[0, "B"] = 0.01
    with pytest.raises(ValueError, match="at least two dates"):
        simulate(returns, days=3, scenarios=5)


def test_joint_portfolio_var_matches_the_closed_form_where_independent_draws_understated_it():
    """50/50 portfolio of two assets with correlation 0.8, 20-day 95% VaR.

    Drawing each asset's future on its own priced the pair as uncorrelated: about 10% VaR.
    Drawing them together gives about 13%, which is what the normal closed form says.
    """
    import zlib

    from analysis import summarize_equal_weight_portfolio
    from simulation import simulate_joint_prices

    returns = _correlated_returns(0.8)
    prices = {"A": 100.0, "B": 100.0}
    joint = simulate_joint_prices(returns, days=20, scenarios=20000, seed=42, current_prices=prices)
    frames = []
    for ticker in ("A", "B"):
        seed = 42 + zlib.adler32(ticker.encode())
        alone = simulate_prices(
            returns[ticker], days=20, scenarios=20000, seed=seed, current_price=100.0
        )
        alone.columns = pd.MultiIndex.from_product(
            [[ticker], alone.columns], names=["ticker", "scenario"]
        )
        frames.append(alone)
    separate = pd.concat(frames, axis=1)

    log_returns = np.log1p(returns.to_numpy())
    weights = np.array([0.5, 0.5])
    mean = log_returns.mean(axis=0) @ weights * 20
    spread = np.sqrt(weights @ np.cov(log_returns, rowvar=False) @ weights * 20)
    closed_form = 1.0 - np.exp(mean - 1.6449 * spread)

    var = "value_at_risk_95_pct"
    joint_var = summarize_equal_weight_portfolio(joint, current_prices=prices)[var]
    separate_var = summarize_equal_weight_portfolio(separate, current_prices=prices)[var]
    assert joint_var == pytest.approx(closed_form, rel=0.05)
    assert separate_var < 0.8 * closed_form
