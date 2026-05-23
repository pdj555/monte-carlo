from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from viz import plot_distribution, plot_equity_curve, plot_paths  # noqa: E402


def test_plot_paths_limits_drawn_paths():
    rng = np.random.default_rng(0)
    days = 20
    scenarios = 250
    shocks = rng.normal(loc=0.0, scale=0.02, size=(days, scenarios))
    prices = 100.0 * np.exp(np.cumsum(shocks, axis=0))
    df = pd.DataFrame(prices)

    fig = plot_paths(df, ticker="AAPL", max_paths=10, current_price=100.0)
    ax = fig.axes[0]

    assert len(ax.lines) <= 12  # 10 paths + mean + current price
    plt.close(fig)


def test_plot_paths_legend_omits_individual_scenarios():
    rng = np.random.default_rng(1)
    days = 20
    scenarios = 120
    shocks = rng.normal(loc=0.0, scale=0.02, size=(days, scenarios))
    prices = 100.0 * np.exp(np.cumsum(shocks, axis=0))
    df = pd.DataFrame(prices)

    fig = plot_paths(df, ticker="AAPL", max_paths=10)
    legend = fig.axes[0].get_legend()

    assert legend is not None
    labels = [text.get_text() for text in legend.get_texts()]
    assert labels == ["5-95% band", "Mean"]
    plt.close(fig)


def test_plot_distribution_rejects_invalid_current_price():
    df = pd.DataFrame([[100.0, 101.0, 99.5]])
    with pytest.raises(ValueError):
        plot_distribution(df, ticker="AAPL", current_price=0.0)


def test_plot_equity_curve_returns_figure():
    equity_curve = pd.DataFrame(
        {
            "strategy": [1.0, 1.05, 1.10],
            "equal_weight": [1.0, 1.02, 1.03],
            "cash": [1.0, 1.001, 1.002],
        },
        index=pd.date_range("2024-01-01", periods=3, freq="D"),
    )

    fig = plot_equity_curve(equity_curve)

    assert isinstance(fig, plt.Figure)
    assert len(fig.axes[0].lines) == 3
    plt.close(fig)
