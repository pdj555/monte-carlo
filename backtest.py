"""Walk-forward validation for simulation-driven portfolio decisions."""

from __future__ import annotations

import argparse
import json
import logging
import sys
import zlib
from pathlib import Path
from typing import Iterable, Optional

import matplotlib.pyplot as plt
import pandas as pd

from analysis import summarize_final_prices
from data import PriceDataError, fetch_prices
from decision import (
    apply_risk_guards,
    enforce_portfolio_risk_budget,
    rank_tickers,
    recommend_allocations,
)
from simulation import estimate_gbm_parameters, simulate_gbm, simulate_prices
from viz import plot_equity_curve

LOGGER = logging.getLogger(__name__)

BACKTEST_ARG_DEFAULTS: dict[str, object] = {
    "tickers": "AAPL",
    "lookback_days": 60,
    "holding_days": 20,
    "rebalance_every": 20,
    "top_k": 1,
    "model": "historical",
    "scenarios": 1000,
    "seed": None,
    "start": None,
    "end": None,
    "offline_path": None,
    "offline_only": False,
    "allow_local_fallback": True,
    "output": None,
    "transaction_cost_bps": 10.0,
    "annual_cash_yield": 0.04,
    "min_expected_return": 0.0,
    "min_prob_up": 0.5,
    "max_var_95_pct": 0.25,
    "max_drawdown_q95_pct": None,
    "portfolio_risk_budget_pct": 0.02,
    "verbose": False,
    "details": False,
}


def build_backtest_args(**overrides: object) -> argparse.Namespace:
    """Return a full backtest namespace with public-surface defaults."""

    values = dict(BACKTEST_ARG_DEFAULTS)
    values.update(overrides)
    return argparse.Namespace(**values)


def _positive_int(value: str) -> int:
    parsed = int(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("must be a positive integer")
    return parsed


def _non_negative_float(value: str) -> float:
    parsed = float(value)
    if parsed < 0:
        raise argparse.ArgumentTypeError("must be a non-negative number")
    return parsed


def _normalise_tickers(ticker_arg: str) -> list[str]:
    tickers = [ticker.strip().upper() for ticker in ticker_arg.split(",") if ticker.strip()]
    if not tickers:
        raise ValueError("No valid tickers were supplied. Provide at least one ticker symbol.")
    return list(dict.fromkeys(tickers))


def _simulate_window(
    *,
    model: str,
    returns: pd.Series,
    current_price: float,
    days: int,
    scenarios: int,
    seed: int | None,
) -> pd.DataFrame:
    if model == "historical":
        return simulate_prices(
            returns,
            days=days,
            scenarios=scenarios,
            current_price=current_price,
            seed=seed,
        )

    mu, sigma = estimate_gbm_parameters(returns)
    return simulate_gbm(
        current_price=current_price,
        mu=mu,
        sigma=sigma,
        days=days,
        scenarios=scenarios,
        seed=seed,
    )


def _load_price_history(
    *,
    tickers: list[str],
    start: str | None,
    end: str | None,
    offline_path: Path | None,
    offline_only: bool,
    allow_local_fallback: bool,
) -> pd.DataFrame:
    price_frames: list[pd.Series] = []
    for ticker in tickers:
        prices = fetch_prices(
            ticker,
            start=start,
            end=end,
            offline_path=offline_path,
            prefer_local=offline_only,
            allow_local_fallback=allow_local_fallback,
        ).rename(ticker)
        price_frames.append(prices)

    combined = pd.concat(price_frames, axis=1, join="inner").dropna()
    if combined.empty:
        raise ValueError(
            "No overlapping price history was available for the requested tickers. "
            "Try a different date range or use tickers with shared history."
        )
    return combined.sort_index()


def _select_backtest_allocations(
    rankings: pd.DataFrame,
    *,
    top_k: int,
    portfolio_risk_budget_pct: float,
) -> pd.DataFrame:
    allocations = recommend_allocations(rankings)
    if allocations.empty:
        return allocations

    selected = allocations.head(top_k).copy()
    selected["weight"] = selected["weight"] / float(selected["weight"].sum())
    return enforce_portfolio_risk_budget(
        selected,
        rankings,
        max_portfolio_var_95_pct=portfolio_risk_budget_pct,
    )


def _annualized_return(final_equity: float, years: float) -> float:
    if years <= 0:
        return 0.0
    return float(final_equity ** (1.0 / years) - 1.0)


def _max_drawdown(equity: pd.Series) -> float:
    running_peak = equity.cummax()
    drawdown = 1.0 - equity.div(running_peak)
    return float(drawdown.max())


def summarize_backtest(
    *,
    equity_curve: pd.DataFrame,
    rebalance_log: pd.DataFrame,
) -> pd.Series:
    """Return a concise performance summary for a walk-forward backtest."""

    if equity_curve.empty:
        raise ValueError("equity_curve must not be empty")
    if rebalance_log.empty:
        raise ValueError("rebalance_log must not be empty")

    years = float(rebalance_log["holding_days"].sum()) / 252.0
    strategy_final = float(equity_curve["strategy"].iloc[-1])
    equal_weight_final = float(equity_curve["equal_weight"].iloc[-1])
    cash_final = float(equity_curve["cash"].iloc[-1])

    summary = {
        "periods": float(len(rebalance_log)),
        "years": years,
        "strategy_total_return": strategy_final - 1.0,
        "strategy_annualized_return": _annualized_return(strategy_final, years),
        "strategy_max_drawdown": _max_drawdown(equity_curve["strategy"]),
        "strategy_win_rate": float((rebalance_log["strategy_return_net"] > 0).mean()),
        "average_turnover": float(rebalance_log["turnover"].mean()),
        "total_transaction_cost_drag": float(rebalance_log["transaction_cost_drag"].sum()),
        "equal_weight_total_return": equal_weight_final - 1.0,
        "cash_total_return": cash_final - 1.0,
        "excess_return_vs_equal_weight": (strategy_final - 1.0) - (equal_weight_final - 1.0),
        "excess_return_vs_cash": (strategy_final - 1.0) - (cash_final - 1.0),
    }
    return pd.Series(summary)


def run_walk_forward_backtest(
    *,
    tickers: list[str],
    lookback_days: int,
    holding_days: int,
    rebalance_every: int,
    top_k: int,
    model: str,
    scenarios: int,
    seed: int | None,
    start: str | None = None,
    end: str | None = None,
    offline_path: Path | None = None,
    offline_only: bool = False,
    allow_local_fallback: bool = True,
    min_expected_return: float = 0.0,
    min_prob_up: float = 0.5,
    max_var_95_pct: float = 0.25,
    max_drawdown_q95_pct: float | None = None,
    portfolio_risk_budget_pct: float = 0.02,
    transaction_cost_bps: float = 0.0,
    annual_cash_yield: float = 0.04,
) -> dict[str, pd.DataFrame | pd.Series]:
    """Run a deterministic walk-forward backtest over aligned price history."""

    if lookback_days <= 0 or holding_days <= 0 or rebalance_every <= 0 or top_k <= 0:
        raise ValueError(
            "lookback_days, holding_days, rebalance_every, and top_k must be positive. "
            "Use values greater than zero."
        )

    price_history = _load_price_history(
        tickers=tickers,
        start=start,
        end=end,
        offline_path=offline_path,
        offline_only=offline_only,
        allow_local_fallback=allow_local_fallback,
    )

    rebalance_positions = list(
        range(lookback_days, len(price_history) - holding_days, rebalance_every)
    )
    if not rebalance_positions:
        raise ValueError(
            "Not enough price history for the requested walk-forward configuration. "
            "Try smaller lookback/hold settings or provide a longer history."
        )

    cash_period_return = (1.0 + annual_cash_yield) ** (holding_days / 252.0) - 1.0
    previous_weights = pd.Series(dtype=float)
    rebalance_rows: list[dict[str, object]] = []
    equity_rows = [
        {
            "date": price_history.index[rebalance_positions[0]],
            "strategy": 1.0,
            "equal_weight": 1.0,
            "cash": 1.0,
        }
    ]
    strategy_equity = 1.0
    equal_weight_equity = 1.0
    cash_equity = 1.0

    for rebalance_idx in rebalance_positions:
        window = price_history.iloc[rebalance_idx - lookback_days : rebalance_idx + 1]
        current_prices = window.iloc[-1]
        period_end = price_history.iloc[rebalance_idx + holding_days]
        realized_returns = period_end / current_prices - 1.0

        ticker_summaries: dict[str, pd.Series] = {}
        for ticker in tickers:
            returns = window[ticker].pct_change().dropna()
            ticker_seed = (
                None
                if seed is None
                else int(seed) + zlib.adler32(f"{rebalance_idx}:{ticker}".encode("utf-8"))
            )
            simulations = _simulate_window(
                model=model,
                returns=returns,
                current_price=float(current_prices[ticker]),
                days=holding_days,
                scenarios=scenarios,
                seed=ticker_seed,
            )
            ticker_summaries[ticker] = summarize_final_prices(
                simulations,
                current_price=float(current_prices[ticker]),
                benchmark_return_pct=cash_period_return,
            )

        summary_df = pd.DataFrame(ticker_summaries).T
        rankings = rank_tickers(summary_df)
        rankings = apply_risk_guards(
            rankings,
            min_expected_return=min_expected_return,
            min_prob_above_current=min_prob_up,
            max_value_at_risk_95_pct=max_var_95_pct,
            max_drawdown_q95=max_drawdown_q95_pct,
        )
        allocations = _select_backtest_allocations(
            rankings,
            top_k=top_k,
            portfolio_risk_budget_pct=portfolio_risk_budget_pct,
        )

        current_weights = allocations["weight"] if not allocations.empty else pd.Series(dtype=float)
        turnover_index = current_weights.index.union(previous_weights.index)
        turnover = float(
            (
                current_weights.reindex(turnover_index, fill_value=0.0)
                - previous_weights.reindex(turnover_index, fill_value=0.0)
            )
            .abs()
            .sum()
        )
        transaction_cost_drag = turnover * transaction_cost_bps / 10_000.0

        invested_return = float(
            current_weights.mul(realized_returns.reindex(current_weights.index)).sum()
        )
        cash_weight = max(0.0, 1.0 - float(current_weights.sum()))
        strategy_return_gross = invested_return + cash_weight * cash_period_return
        strategy_return_net = strategy_return_gross - transaction_cost_drag
        equal_weight_return = float(realized_returns.mean())

        strategy_equity *= 1.0 + strategy_return_net
        equal_weight_equity *= 1.0 + equal_weight_return
        cash_equity *= 1.0 + cash_period_return

        rebalance_rows.append(
            {
                "rebalance_date": price_history.index[rebalance_idx],
                "end_date": price_history.index[rebalance_idx + holding_days],
                "holding_days": holding_days,
                "selection_count": int(len(current_weights)),
                "selected_tickers": ",".join(current_weights.index.tolist()),
                "selected_weights": json.dumps(
                    {ticker: float(weight) for ticker, weight in current_weights.items()},
                    sort_keys=True,
                ),
                "cash_weight": cash_weight,
                "turnover": turnover,
                "transaction_cost_drag": transaction_cost_drag,
                "strategy_return_gross": strategy_return_gross,
                "strategy_return_net": strategy_return_net,
                "equal_weight_return": equal_weight_return,
                "cash_return": cash_period_return,
            }
        )
        equity_rows.append(
            {
                "date": price_history.index[rebalance_idx + holding_days],
                "strategy": strategy_equity,
                "equal_weight": equal_weight_equity,
                "cash": cash_equity,
            }
        )
        previous_weights = current_weights.copy()

    rebalance_log = pd.DataFrame(rebalance_rows).set_index("rebalance_date")
    equity_curve = pd.DataFrame(equity_rows).set_index("date")
    summary = summarize_backtest(equity_curve=equity_curve, rebalance_log=rebalance_log)
    return {
        "summary": summary,
        "rebalance_log": rebalance_log,
        "equity_curve": equity_curve,
    }


def _save_outputs(
    *,
    output_dir: Path,
    result: dict[str, pd.DataFrame | pd.Series],
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    summary = result["summary"]
    rebalance_log = result["rebalance_log"]
    equity_curve = result["equity_curve"]

    if not isinstance(summary, pd.Series):
        raise ValueError("summary output must be a pandas Series")
    if not isinstance(rebalance_log, pd.DataFrame):
        raise ValueError("rebalance_log output must be a pandas DataFrame")
    if not isinstance(equity_curve, pd.DataFrame):
        raise ValueError("equity_curve output must be a pandas DataFrame")

    summary.to_frame(name="value").to_csv(output_dir / "backtest_summary.csv", float_format="%.6g")
    rebalance_log.to_csv(output_dir / "rebalance_log.csv", float_format="%.6g")
    equity_curve.to_csv(output_dir / "equity_curve.csv", float_format="%.6g")

    fig = plot_equity_curve(equity_curve)
    fig.savefig(output_dir / "equity_curve.png", bbox_inches="tight")
    plt.close(fig)


def build_parser() -> argparse.ArgumentParser:
    """Create the walk-forward backtest CLI parser."""

    parser = argparse.ArgumentParser(
        description="Validate a simulation-driven portfolio process with walk-forward backtesting.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--tickers", default="AAPL", help="Comma-separated tickers to evaluate.")
    parser.add_argument("--lookback-days", type=_positive_int, default=60)
    parser.add_argument("--holding-days", type=_positive_int, default=20)
    parser.add_argument("--rebalance-every", type=_positive_int, default=20)
    parser.add_argument("--top-k", type=_positive_int, default=2)
    parser.add_argument("--model", choices=("historical", "gbm"), default="historical")
    parser.add_argument("--scenarios", type=_positive_int, default=1000)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--start", type=str, default=None)
    parser.add_argument("--end", type=str, default=None)
    parser.add_argument("--offline-path", type=str, default=None)
    parser.add_argument("--offline-only", action="store_true")
    parser.add_argument("--output", type=str, default=None)
    parser.add_argument("--transaction-cost-bps", type=_non_negative_float, default=10.0)
    parser.add_argument("--annual-cash-yield", type=_non_negative_float, default=0.04)
    parser.add_argument("--min-expected-return", type=float, default=0.0)
    parser.add_argument("--min-prob-up", type=float, default=0.5)
    parser.add_argument("--max-var-95-pct", type=_non_negative_float, default=0.25)
    parser.add_argument("--max-drawdown-q95-pct", type=_non_negative_float, default=None)
    parser.add_argument("--portfolio-risk-budget-pct", type=_non_negative_float, default=0.02)
    parser.add_argument("--verbose", action="store_true")
    return parser


def parse_args(argv: Optional[Iterable[str]] = None) -> argparse.Namespace:
    """Return parsed backtest CLI arguments."""

    parser = build_parser()
    args = parser.parse_args(list(argv) if argv is not None else None)
    if args.seed is not None and args.seed < 0:
        parser.error("--seed must be non-negative")
    if not 0.0 <= float(args.min_prob_up) <= 1.0:
        parser.error("--min-prob-up must be between 0 and 1")
    return args


def _render_backtest_output(result: dict[str, pd.DataFrame | pd.Series]) -> None:
    summary = result["summary"]
    if not isinstance(summary, pd.Series):
        raise ValueError("summary output must be a pandas Series")
    print(summary.to_frame(name="value").to_string(float_format=lambda value: f"{value:0.4f}"))


def run(
    args: argparse.Namespace,
    *,
    render: bool = True,
) -> dict[str, pd.DataFrame | pd.Series]:
    """Execute the walk-forward backtest workflow."""

    tickers = _normalise_tickers(args.tickers)
    offline_path = Path(args.offline_path).expanduser() if args.offline_path else None
    result = run_walk_forward_backtest(
        tickers=tickers,
        lookback_days=int(args.lookback_days),
        holding_days=int(args.holding_days),
        rebalance_every=int(args.rebalance_every),
        top_k=int(args.top_k),
        model=str(args.model),
        scenarios=int(args.scenarios),
        seed=args.seed,
        start=args.start,
        end=args.end,
        offline_path=offline_path,
        offline_only=bool(args.offline_only),
        allow_local_fallback=bool(getattr(args, "allow_local_fallback", True)),
        min_expected_return=float(args.min_expected_return),
        min_prob_up=float(args.min_prob_up),
        max_var_95_pct=float(args.max_var_95_pct),
        max_drawdown_q95_pct=(
            None
            if args.max_drawdown_q95_pct is None
            else float(args.max_drawdown_q95_pct)
        ),
        portfolio_risk_budget_pct=float(args.portfolio_risk_budget_pct),
        transaction_cost_bps=float(args.transaction_cost_bps),
        annual_cash_yield=float(args.annual_cash_yield),
    )

    summary = result["summary"]
    if not isinstance(summary, pd.Series):
        raise ValueError("summary output must be a pandas Series")

    if render:
        _render_backtest_output(result)

    if args.output:
        _save_outputs(output_dir=Path(args.output).expanduser(), result=result)

    return result


def main(
    argv: Optional[Iterable[str]] = None,
    *,
    show_deprecation: bool = True,
) -> int:
    """Entrypoint for ``python backtest.py``."""

    if show_deprecation:
        print(
            "Deprecated: use `monte-carlo backtest ...` for the simplified CLI.",
            file=sys.stderr,
        )

    args = parse_args(argv)
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(levelname)s %(message)s",
    )

    try:
        run(args)
    except (OSError, PriceDataError, ValueError) as exc:
        LOGGER.error("%s", exc)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
