from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from analysis import summarize_final_prices


def test_summarize_final_prices_reports_key_metrics():
    base = np.linspace(100, 120, 50)
    df = pd.DataFrame({i: base * (1 + 0.01 * i) for i in range(1, 6)})

    summary = summarize_final_prices(df, current_price=100.0, quantiles=(0.1, 0.9))

    assert {"mean", "median", "std", "expected_return", "value_at_risk_95"} <= set(summary.index)
    assert summary["prob_above_current"] > 0.0
    assert summary["q10"] <= summary["q90"]


def test_summarize_final_prices_requires_data():
    with pytest.raises(ValueError):
        summarize_final_prices(pd.DataFrame())
