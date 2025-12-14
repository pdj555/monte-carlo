from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from viz import plot_distribution, plot_paths  # noqa: E402


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


def test_plot_distribution_rejects_invalid_current_price():
    df = pd.DataFrame([[100.0, 101.0, 99.5]])
    with pytest.raises(ValueError):
        plot_distribution(df, ticker="AAPL", current_price=0.0)
