"""Programmatic SDK for Monte Carlo simulation toolkit.

This module provides a single-import, typed, composable API designed for
both human and agent consumption. Every operation returns structured data
that can be serialised to JSON, piped between processes, or consumed
directly by AI agents.

Example
-------
>>> from sdk import MonteCarloSDK
>>> sdk = MonteCarloSDK(offline_only=True)
>>> result = sdk.analyze("AAPL", days=60, scenarios=500, seed=42)
>>> print(result.summary["expected_return"])
"""

from __future__ import annotations

import concurrent.futures
import json
import zlib
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Sequence

import pandas as pd

from analysis import summarize_final_prices
from data import PriceDataError, fetch_prices
from decision import (
    apply_risk_guards,
    build_action_plan,
    build_execution_plan,
    enforce_portfolio_risk_budget,
    rank_tickers,
    recommend_allocations,
)
from simulation import estimate_gbm_parameters, simulate_gbm, simulate_prices

__all__ = [
    "MonteCarloSDK",
    "TickerResult",
    "PortfolioResult",
    "ScreenResult",
]


# ---------------------------------------------------------------------------
# Result dataclasses -- structured, serialisable output
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class TickerResult:
    """Result of analysing a single ticker."""

    ticker: str
    current_price: float
    summary: dict[str, float]
    model: str
    days: int
    scenarios: int

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def to_json(self, **kwargs: Any) -> str:
        return json.dumps(self.to_dict(), **kwargs)


@dataclass(frozen=True)
class PortfolioResult:
    """Result of a multi-ticker portfolio analysis."""

    tickers: list[str]
    ticker_results: dict[str, TickerResult]
    rankings: dict[str, dict[str, float]]
    allocations: dict[str, dict[str, float]]
    action_plan: dict[str, Any]
    portfolio_summary: dict[str, float] | None = None
    execution_plan: dict[str, dict[str, float]] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        return d

    def to_json(self, **kwargs: Any) -> str:
        return json.dumps(self.to_dict(), **kwargs)


@dataclass(frozen=True)
class ScreenResult:
    """Result of screening multiple tickers for opportunities."""

    buy: list[str]
    watch: list[str]
    avoid: list[str]
    rankings: dict[str, dict[str, float]]
    top_pick: str | None
    headline: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def to_json(self, **kwargs: Any) -> str:
        return json.dumps(self.to_dict(), **kwargs)


# ---------------------------------------------------------------------------
# SDK
# ---------------------------------------------------------------------------


class MonteCarloSDK:
    """High-level, agent-friendly API for Monte Carlo simulation.

    This is the primary programmatic interface. It wraps the underlying
    modules into a composable, typed API that returns structured results
    suitable for agent consumption or JSON serialisation.

    Parameters
    ----------
    offline_only : bool
        Skip network requests and use local CSV data exclusively.
    offline_path : str or Path, optional
        Directory containing offline CSV files.
    cache_dir : str or Path, optional
        Directory for caching downloaded price data.
    """

    def __init__(
        self,
        *,
        offline_only: bool = False,
        offline_path: str | Path | None = None,
        cache_dir: str | Path | None = None,
    ) -> None:
        self._offline_only = offline_only
        self._offline_path = Path(offline_path) if offline_path else None
        self._cache_dir = Path(cache_dir) if cache_dir else None

    # -- Core analysis -------------------------------------------------------

    def analyze(
        self,
        ticker: str,
        *,
        days: int = 252,
        scenarios: int = 1000,
        model: str = "historical",
        seed: int | None = None,
        start: str | None = None,
        end: str | None = None,
        target_return_pct: float | None = None,
        max_loss_pct: float | None = None,
        benchmark_return_pct: float | None = None,
    ) -> TickerResult:
        """Run a full simulation and analysis for a single ticker.

        Returns a structured :class:`TickerResult` with all metrics.
        """

        prices = fetch_prices(
            ticker,
            start=start,
            end=end,
            offline_path=self._offline_path,
            prefer_local=self._offline_only,
            cache_dir=self._cache_dir,
        )
        returns = prices.pct_change().dropna()
        current_price = float(prices.iloc[-1])

        sims = self._simulate(
            model=model,
            returns=returns,
            current_price=current_price,
            days=days,
            scenarios=scenarios,
            seed=seed,
        )

        summary = summarize_final_prices(
            sims,
            current_price=current_price,
            target_return_pct=target_return_pct,
            max_loss_pct=max_loss_pct,
            benchmark_return_pct=benchmark_return_pct,
        )

        return TickerResult(
            ticker=ticker.upper(),
            current_price=current_price,
            summary={str(k): float(v) for k, v in summary.to_dict().items()},
            model=model,
            days=days,
            scenarios=scenarios,
        )

    def analyze_many(
        self,
        tickers: Sequence[str],
        *,
        days: int = 252,
        scenarios: int = 1000,
        model: str = "historical",
        seed: int | None = None,
        start: str | None = None,
        end: str | None = None,
        max_workers: int | None = None,
        benchmark_return_pct: float | None = None,
    ) -> dict[str, TickerResult | PriceDataError]:
        """Analyze multiple tickers concurrently.

        Returns a dict mapping ticker -> TickerResult (or PriceDataError on failure).
        Failed tickers do not block successful ones.
        """

        def _analyze_one(t: str) -> TickerResult:
            t_seed = (
                None if seed is None
                else int(seed) + zlib.adler32(t.upper().encode("utf-8"))
            )
            return self.analyze(
                t,
                days=days,
                scenarios=scenarios,
                model=model,
                seed=t_seed,
                start=start,
                end=end,
                benchmark_return_pct=benchmark_return_pct,
            )

        results: dict[str, TickerResult | PriceDataError] = {}
        with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as pool:
            futures = {pool.submit(_analyze_one, t): t for t in tickers}
            for future in concurrent.futures.as_completed(futures):
                ticker = futures[future]
                try:
                    results[ticker.upper()] = future.result()
                except PriceDataError as exc:
                    results[ticker.upper()] = exc

        return results

    # -- Portfolio-level analysis ---------------------------------------------

    def portfolio(
        self,
        tickers: Sequence[str],
        *,
        days: int = 252,
        scenarios: int = 1000,
        model: str = "historical",
        seed: int | None = None,
        start: str | None = None,
        end: str | None = None,
        capital: float | None = None,
        allow_fractional_shares: bool = False,
        min_expected_return: float = 0.0,
        min_prob_above_current: float = 0.5,
        max_var_95_pct: float = 0.25,
        max_drawdown_q95: float | None = None,
        portfolio_risk_budget_pct: float = 0.02,
        annual_cash_yield: float = 0.04,
    ) -> PortfolioResult:
        """Run a full portfolio analysis: simulate, rank, allocate, plan.

        This is the primary entry point for agent-driven portfolio construction.
        """

        horizon_years = float(days) / 252.0
        benchmark_return_pct = (1.0 + annual_cash_yield) ** horizon_years - 1.0

        raw = self.analyze_many(
            tickers,
            days=days,
            scenarios=scenarios,
            model=model,
            seed=seed,
            start=start,
            end=end,
            benchmark_return_pct=benchmark_return_pct,
        )

        ticker_results: dict[str, TickerResult] = {}
        for t, r in raw.items():
            if isinstance(r, TickerResult):
                ticker_results[t] = r

        if not ticker_results:
            return PortfolioResult(
                tickers=[t.upper() for t in tickers],
                ticker_results={},
                rankings={},
                allocations={},
                action_plan=build_action_plan(pd.DataFrame(), pd.DataFrame()),
            )

        summary_df = pd.DataFrame({
            t: r.summary for t, r in ticker_results.items()
        }).T
        current_prices = {t: r.current_price for t, r in ticker_results.items()}

        rankings = rank_tickers(summary_df)
        rankings = apply_risk_guards(
            rankings,
            min_expected_return=min_expected_return,
            min_prob_above_current=min_prob_above_current,
            max_value_at_risk_95_pct=max_var_95_pct,
            max_drawdown_q95=max_drawdown_q95,
        )
        allocations = recommend_allocations(rankings) if not rankings.empty else pd.DataFrame()
        if not allocations.empty:
            allocations = enforce_portfolio_risk_budget(
                allocations, rankings,
                max_portfolio_var_95_pct=portfolio_risk_budget_pct,
            )
        action_plan = build_action_plan(rankings, allocations)

        exec_plan: dict[str, dict[str, float]] = {}
        if capital is not None and not allocations.empty:
            ep = build_execution_plan(
                allocations,
                current_prices=current_prices,
                capital=capital,
                allow_fractional_shares=allow_fractional_shares,
            )
            exec_plan = ep.to_dict(orient="index")

        return PortfolioResult(
            tickers=[t.upper() for t in tickers],
            ticker_results=ticker_results,
            rankings=rankings.to_dict(orient="index") if not rankings.empty else {},
            allocations=allocations.to_dict(orient="index") if not allocations.empty else {},
            action_plan=action_plan,
            execution_plan=exec_plan,
        )

    # -- Screening -----------------------------------------------------------

    def screen(
        self,
        tickers: Sequence[str],
        *,
        days: int = 252,
        scenarios: int = 1000,
        model: str = "historical",
        seed: int | None = None,
        min_expected_return: float = 0.0,
        min_prob_above_current: float = 0.5,
        max_var_95_pct: float = 0.25,
        annual_cash_yield: float = 0.04,
    ) -> ScreenResult:
        """Screen tickers and categorise into BUY / WATCH / AVOID.

        This is designed for agent-driven stock screening workflows where
        the agent needs a quick, structured answer.
        """

        result = self.portfolio(
            tickers,
            days=days,
            scenarios=scenarios,
            model=model,
            seed=seed,
            min_expected_return=min_expected_return,
            min_prob_above_current=min_prob_above_current,
            max_var_95_pct=max_var_95_pct,
            annual_cash_yield=annual_cash_yield,
        )

        buy, watch, avoid = [], [], []
        for t, data in result.rankings.items():
            rec = data.get("recommendation", "WATCH")
            if rec == "BUY":
                buy.append(t)
            elif rec == "AVOID":
                avoid.append(t)
            else:
                watch.append(t)

        return ScreenResult(
            buy=buy,
            watch=watch,
            avoid=avoid,
            rankings=result.rankings,
            top_pick=buy[0] if buy else None,
            headline=result.action_plan.get("headline", "No opportunities found."),
        )

    # -- Comparison ----------------------------------------------------------

    def compare(
        self,
        tickers: Sequence[str],
        *,
        days: int = 252,
        scenarios: int = 1000,
        model: str = "historical",
        seed: int | None = None,
    ) -> dict[str, Any]:
        """Compare tickers head-to-head with a compact summary.

        Returns a dict suitable for direct JSON serialisation or agent
        consumption with key comparative metrics.
        """

        raw = self.analyze_many(
            tickers, days=days, scenarios=scenarios, model=model, seed=seed,
        )

        comparison: dict[str, Any] = {}
        for t, r in sorted(raw.items()):
            if isinstance(r, TickerResult):
                comparison[t] = {
                    "current_price": r.current_price,
                    "expected_return": r.summary.get("expected_return"),
                    "prob_above_current": r.summary.get("prob_above_current"),
                    "value_at_risk_95_pct": r.summary.get("value_at_risk_95_pct"),
                    "max_drawdown_mean": r.summary.get("max_drawdown_mean"),
                    "kelly_fraction": r.summary.get("kelly_fraction"),
                }
            else:
                comparison[t] = {"error": str(r)}

        return {
            "days": days,
            "scenarios": scenarios,
            "model": model,
            "tickers": comparison,
        }

    # -- Internal helpers ----------------------------------------------------

    @staticmethod
    def _simulate(
        *,
        model: str,
        returns: pd.Series,
        current_price: float,
        days: int,
        scenarios: int,
        seed: int | None,
    ) -> pd.DataFrame:
        if model == "gbm":
            mu, sigma = estimate_gbm_parameters(returns)
            return simulate_gbm(
                current_price=current_price,
                mu=mu,
                sigma=sigma,
                days=days,
                scenarios=scenarios,
                seed=seed,
            )
        return simulate_prices(
            returns,
            days=days,
            scenarios=scenarios,
            current_price=current_price,
            seed=seed,
        )
