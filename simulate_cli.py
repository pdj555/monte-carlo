"""Simulation workflow shared by the public CLI and legacy wrapper."""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import zlib
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

import matplotlib.pyplot as plt
import pandas as pd

from ai import OpenAIConfigurationError, OpenAIRequestError, generate_ai_summary
from analysis import (
    summarize_equal_weight_portfolio,
    summarize_final_prices,
    summarize_weighted_portfolio,
)
from data import PriceDataError, fetch_prices, get_price_source
from decision import (
    apply_risk_guards,
    build_action_plan,
    build_execution_plan,
    enforce_portfolio_risk_budget,
    rank_tickers,
    recommend_allocations,
)
from simulation import (
    estimate_gbm_parameters,
    simulate_gbm,
    simulate_prediction_market,
    simulate_prices,
)
from viz import plot_distribution, plot_paths

LOGGER = logging.getLogger(__name__)

SIMULATION_ARG_DEFAULTS: dict[str, object] = {
    "journal_file": None,
    "policy_file": None,
    "tickers": "AAPL",
    "days": 252,
    "scenarios": 1000,
    "max_paths": 100,
    "no_plots": False,
    "dt": 1.0,
    "block_size": 1,
    "shock_probability": 0.0,
    "shock_return": -0.15,
    "seed": None,
    "model": "historical",
    "fundamental_probability": None,
    "market_price": None,
    "fundamental_certainty": 100.0,
    "prob_mean_reversion": 0.2,
    "prob_daily_volatility": 0.03,
    "start": None,
    "end": None,
    "output": None,
    "cache_dir": None,
    "refresh_cache": False,
    "save_simulations": False,
    "offline_path": None,
    "offline_only": False,
    "allow_local_fallback": True,
    "show": True,
    "ai_summary": False,
    "ai_model": None,
    "annual_cash_yield": 0.04,
    "min_expected_return": 0.0,
    "min_prob_up": 0.5,
    "portfolio_risk_budget_pct": 0.02,
    "max_var_95_pct": 0.25,
    "max_drawdown_q95_pct": None,
    "target_return_pct": None,
    "max_loss_pct": None,
    "min_prob_hit_target": None,
    "max_prob_breach_loss": None,
    "capital": None,
    "allow_fractional_shares": False,
    "minimal": False,
    "strict": False,
    "verbose": False,
    "details": False,
}


def build_simulation_args(**overrides: object) -> argparse.Namespace:
    """Return a full simulation namespace with legacy-compatible defaults."""

    values = dict(SIMULATION_ARG_DEFAULTS)
    values.update(overrides)
    return argparse.Namespace(**values)


def _normalise_tickers(ticker_arg: str) -> list[str]:
    requested = [ticker.strip().upper() for ticker in ticker_arg.split(",") if ticker.strip()]
    if not requested:
        raise ValueError("No valid tickers were supplied. Provide at least one ticker symbol.")

    tickers: list[str] = []
    seen: set[str] = set()
    for ticker in requested:
        if ticker in seen:
            continue
        seen.add(ticker)
        tickers.append(ticker)
    return tickers


def _simulate_model(
    *,
    args: argparse.Namespace,
    returns: pd.Series,
    current_price: float,
    ticker_seed: int | None,
) -> pd.DataFrame:
    if args.model == "historical":
        return simulate_prices(
            returns,
            days=args.days,
            scenarios=args.scenarios,
            dt=args.dt,
            seed=ticker_seed,
            current_price=current_price,
            shock_probability=float(args.shock_probability),
            shock_return=float(args.shock_return),
            block_size=int(args.block_size),
        )
    if args.model == "gbm":
        mu, sigma = estimate_gbm_parameters(returns)
        return simulate_gbm(
            current_price=current_price,
            mu=mu,
            sigma=sigma,
            days=args.days,
            scenarios=args.scenarios,
            dt=args.dt,
            seed=ticker_seed,
            shock_probability=float(args.shock_probability),
            shock_return=float(args.shock_return),
        )
    return simulate_prediction_market(
        fundamental_probability=float(args.fundamental_probability),
        current_price=float(current_price),
        days=args.days,
        scenarios=args.scenarios,
        certainty=float(args.fundamental_certainty),
        mean_reversion=float(args.prob_mean_reversion),
        daily_volatility=float(args.prob_daily_volatility),
        dt=args.dt,
        seed=ticker_seed,
    )


def _save_outputs(
    *,
    output_dir: Path,
    summary_df: pd.DataFrame,
    report: dict[str, object],
    rankings: pd.DataFrame,
    allocations: pd.DataFrame,
    execution_plan: pd.DataFrame,
    action_plan: dict[str, object],
    combined: pd.DataFrame,
    save_simulations: bool,
) -> None:
    summary_df.to_csv(output_dir / "summaries.csv", float_format="%.6g")
    with (output_dir / "summaries.json").open("w", encoding="utf-8") as handle:
        json.dump(summary_df.to_dict(orient="index"), handle, indent=2)

    with (output_dir / "report.json").open("w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2)

    if not rankings.empty:
        rankings.to_csv(output_dir / "rankings.csv", float_format="%.6g")
    if not allocations.empty:
        allocations.to_csv(output_dir / "allocations.csv", float_format="%.6g")
    if not execution_plan.empty:
        execution_plan.to_csv(output_dir / "execution_plan.csv", float_format="%.6g")

    with (output_dir / "action_plan.md").open("w", encoding="utf-8") as handle:
        handle.write("# Action Plan\n\n")
        handle.write(f"- **Stance:** {action_plan['stance']}\n")
        handle.write(f"- **Headline:** {action_plan['headline']}\n")
        if action_plan["primary_pick"] is not None:
            pick = action_plan["primary_pick"]
            handle.write(
                "- **Primary pick:** "
                f"{pick['ticker']} (weight {pick['weight']:.1%}, score {pick['score']:.1f}, "
                f"expected return {pick['expected_return']:.1%})\n"
            )
        if action_plan["avoid_list"]:
            handle.write(f"- **Avoid:** {', '.join(action_plan['avoid_list'])}\n")
        if action_plan.get("cash_weight", 0.0) > 0:
            handle.write(f"- **Cash buffer:** {action_plan['cash_weight']:.1%}\n")

        if not execution_plan.empty:
            handle.write("\n## Execution Plan\n\n")
            handle.write(
                "| Ticker | Weight | Price | Target $ | Shares | Est. Cost | Cash Drift |\n"
            )
            handle.write("| --- | ---: | ---: | ---: | ---: | ---: | ---: |\n")
            for ticker, row in execution_plan.iterrows():
                handle.write(
                    (
                        f"| {ticker} | {row['weight']:.1%} | {row['price']:.2f} | "
                        f"{row['target_dollars']:.2f} | {row['shares']:.4f} | "
                        f"{row['est_cost']:.2f} | {row['cash_drift']:.2f} |\n"
                    )
                )

    if save_simulations and not combined.empty:
        combined.to_csv(output_dir / "simulations.csv.gz", compression="gzip")


def _hash_payload(payload: dict[str, object]) -> str:
    canonical = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def _last_journal_chain_hash(journal_path: Path) -> str | None:
    if not journal_path.exists():
        return None

    try:
        lines = journal_path.read_text(encoding="utf-8").splitlines()
    except OSError:
        return None

    for line in reversed(lines):
        line = line.strip()
        if not line:
            continue
        try:
            entry = json.loads(line)
        except json.JSONDecodeError:
            return None
        value = entry.get("chain_hash")
        return str(value) if isinstance(value, str) and value else None
    return None


def _append_journal_entry(
    *,
    journal_path: Path,
    report: dict[str, object],
    args: argparse.Namespace,
) -> dict[str, object]:
    previous_chain_hash = _last_journal_chain_hash(journal_path)
    report_hash = _hash_payload(report)
    summary = {
        ticker: payload.get("summary", {})
        for ticker, payload in report.get("results", {}).items()
        if isinstance(payload, dict)
    }

    body = {
        "generated_at": report.get("generated_at"),
        "tickers": sorted(summary.keys()),
        "model": args.model,
        "days": int(args.days),
        "scenarios": int(args.scenarios),
        "portfolio_risk_budget_pct": float(args.portfolio_risk_budget_pct),
        "policy_crc32": report.get("policy_crc32"),
        "report_hash": report_hash,
        "previous_chain_hash": previous_chain_hash,
        "summaries": summary,
    }

    chain_hash_input = {
        "report_hash": report_hash,
        "previous_chain_hash": previous_chain_hash,
    }
    entry = {
        "schema_version": 1,
        **body,
        "chain_hash": _hash_payload(chain_hash_input),
    }
    journal_path.parent.mkdir(parents=True, exist_ok=True)
    with journal_path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(entry, sort_keys=True))
        handle.write("\n")
    return entry


def maybe_show_simulation_plots(args: argparse.Namespace, result: dict[str, Any]) -> None:
    if bool(args.show) and not bool(args.no_plots) and not result["simulations"].empty:
        plt.show()
        plt.close("all")


def run(
    args: argparse.Namespace,
    *,
    render: bool = True,
    display_plots: bool = True,
    renderer: Callable[[dict[str, Any], argparse.Namespace], None] | None = None,
) -> dict[str, Any]:
    """Execute the simulation workflow and return simulation artefacts."""

    tickers = _normalise_tickers(args.tickers)
    output_dir = Path(args.output).expanduser() if args.output else None
    offline_path = Path(args.offline_path).expanduser() if args.offline_path else None
    cache_dir = Path(args.cache_dir).expanduser() if args.cache_dir else None
    journal_file = Path(args.journal_file).expanduser() if args.journal_file else None

    if output_dir is not None:
        output_dir.mkdir(parents=True, exist_ok=True)

    if args.model == "gbm" and int(args.block_size) != 1:
        LOGGER.warning("--block-size only applies to --model historical; ignoring for gbm")

    if not args.show and not args.no_plots:
        try:
            plt.switch_backend("Agg")
        except Exception:
            pass

    combined_frames: list[pd.DataFrame] = []
    summaries: dict[str, pd.Series] = {}
    current_prices: dict[str, float] = {}
    price_sources: dict[str, dict[str, object]] = {}
    artefacts: dict[str, dict[str, str]] = {}
    errors: list[dict[str, str]] = []
    ai_summaries: dict[str, str] = {}
    horizon_years = float(args.days) * float(args.dt) / 252.0
    benchmark_return_pct = (1.0 + float(args.annual_cash_yield)) ** horizon_years - 1.0

    for ticker in tickers:
        if args.model == "prediction_market":
            current_price = float(
                args.market_price
                if args.market_price is not None
                else args.fundamental_probability
            )
            returns = pd.Series(dtype=float)
        else:
            try:
                prices = fetch_prices(
                    ticker,
                    start=args.start,
                    end=args.end,
                    offline_path=offline_path,
                    prefer_local=args.offline_only,
                    allow_local_fallback=bool(getattr(args, "allow_local_fallback", True)),
                    cache_dir=cache_dir,
                    refresh_cache=args.refresh_cache,
                )
            except PriceDataError as exc:
                message = str(exc)
                LOGGER.warning("[%s] %s", ticker, message)
                errors.append({"ticker": ticker, "error": message})
                continue

            source_info = get_price_source(prices)
            if source_info is not None:
                price_sources[ticker] = source_info
            prices = prices.dropna()
            returns = prices.pct_change().dropna()
            if returns.empty:
                message = (
                    "Not enough return data was available to run a simulation. "
                    "Try a longer price history."
                )
                LOGGER.warning("[%s] %s", ticker, message)
                errors.append({"ticker": ticker, "error": message})
                continue

            current_price = float(prices.iloc[-1])
        current_prices[ticker] = current_price
        ticker_seed = (
            None
            if args.seed is None
            else int(args.seed) + zlib.adler32(ticker.encode("utf-8"))
        )
        sims = _simulate_model(
            args=args,
            returns=returns,
            current_price=current_price,
            ticker_seed=ticker_seed,
        )

        sims = sims.copy()
        sims.columns = pd.MultiIndex.from_product(
            [[ticker], sims.columns], names=["ticker", "scenario"]
        )
        combined_frames.append(sims)

        summary = summarize_final_prices(
            sims.xs(ticker, axis=1, level=0),
            current_price=current_price,
            target_return_pct=(
                None if args.target_return_pct is None else float(args.target_return_pct)
            ),
            max_loss_pct=(None if args.max_loss_pct is None else float(args.max_loss_pct)),
            benchmark_return_pct=benchmark_return_pct,
        )
        summaries[ticker] = summary

        if args.ai_summary:
            try:
                ai_text = generate_ai_summary(
                    ticker=ticker,
                    summary=summary,
                    simulation_model=args.model,
                    days=args.days,
                    scenarios=args.scenarios,
                    model=args.ai_model,
                )
            except (OpenAIConfigurationError, OpenAIRequestError) as exc:
                message = str(exc)
                LOGGER.error("[%s] %s", ticker, message)
                errors.append({"ticker": ticker, "error": message})
            else:
                ai_summaries[ticker] = ai_text

        if not args.no_plots:
            max_paths = None if args.max_paths == 0 else int(args.max_paths)
            fig_dist = plot_distribution(sims, ticker=ticker, current_price=current_price)
            fig_paths = plot_paths(
                sims,
                ticker=ticker,
                max_paths=max_paths,
                current_price=current_price,
            )

            if output_dir is not None:
                dist_name = f"{ticker}_distribution.png"
                paths_name = f"{ticker}_paths.png"
                fig_dist.savefig(output_dir / dist_name, bbox_inches="tight")
                fig_paths.savefig(output_dir / paths_name, bbox_inches="tight")
                artefacts[ticker] = {
                    "distribution_plot": dist_name,
                    "paths_plot": paths_name,
                }
                if ticker in ai_summaries:
                    ai_name = f"{ticker}_ai_summary.md"
                    (output_dir / ai_name).write_text(ai_summaries[ticker] + "\n", encoding="utf-8")
                    artefacts[ticker]["ai_summary"] = ai_name

            if not args.show:
                plt.close(fig_dist)
                plt.close(fig_paths)

    combined = pd.concat(combined_frames, axis=1) if combined_frames else pd.DataFrame()
    portfolio_summary: pd.Series | None = None
    if len(summaries) > 1 and not combined.empty:
        portfolio_summary = summarize_equal_weight_portfolio(
            combined,
            current_prices=current_prices,
            benchmark_return_pct=benchmark_return_pct,
        )

    summary_df = pd.DataFrame(summaries).T if summaries else pd.DataFrame()
    rankings = rank_tickers(summary_df) if not summary_df.empty else pd.DataFrame()
    rankings = (
        apply_risk_guards(
            rankings,
            min_expected_return=float(args.min_expected_return),
            min_prob_above_current=float(args.min_prob_up),
            max_value_at_risk_95_pct=float(args.max_var_95_pct),
            max_drawdown_q95=(
                None
                if args.max_drawdown_q95_pct is None
                else float(args.max_drawdown_q95_pct)
            ),
            min_prob_hit_target=(
                None if args.min_prob_hit_target is None else float(args.min_prob_hit_target)
            ),
            max_prob_breach_loss=(
                None
                if args.max_prob_breach_loss is None
                else float(args.max_prob_breach_loss)
            ),
        )
        if not rankings.empty
        else rankings
    )
    allocations = recommend_allocations(rankings) if not rankings.empty else pd.DataFrame()
    allocation_portfolio_summary: pd.Series | None = None
    if not allocations.empty and not combined.empty:
        allocation_portfolio_summary = summarize_weighted_portfolio(
            combined,
            current_prices=current_prices,
            weights=allocations["weight"],
            benchmark_return_pct=benchmark_return_pct,
        )
        allocations = enforce_portfolio_risk_budget(
            allocations,
            rankings,
            max_portfolio_var_95_pct=float(args.portfolio_risk_budget_pct),
            portfolio_var_95_pct=float(
                allocation_portfolio_summary.get("value_at_risk_95_pct", 0.0)
            ),
        )
        allocation_portfolio_summary = summarize_weighted_portfolio(
            combined,
            current_prices=current_prices,
            weights=allocations["weight"],
            benchmark_return_pct=benchmark_return_pct,
        )
    elif not allocations.empty:
        allocations = enforce_portfolio_risk_budget(
            allocations,
            rankings,
            max_portfolio_var_95_pct=float(args.portfolio_risk_budget_pct),
        )
    action_plan = build_action_plan(rankings, allocations)
    execution_plan = pd.DataFrame()
    if args.capital is not None and not allocations.empty:
        execution_plan = build_execution_plan(
            allocations,
            current_prices=current_prices,
            capital=float(args.capital),
            allow_fractional_shares=bool(args.allow_fractional_shares),
        )

    policy = getattr(args, "policy", {})
    policy_crc32 = None
    if policy:
        policy_bytes = json.dumps(policy, sort_keys=True).encode("utf-8")
        policy_crc32 = f"{zlib.crc32(policy_bytes):08x}"

    report: dict[str, object] = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "args": {
            key: (str(value) if isinstance(value, Path) else value)
            for key, value in vars(args).items()
        },
        "results": {
            ticker: {
                "summary": summaries[ticker].to_dict(),
                "artefacts": artefacts.get(ticker, {}),
                "ai_summary": ai_summaries.get(ticker),
            }
            for ticker in summaries
        },
        "portfolio_summary": (
            portfolio_summary.to_dict() if portfolio_summary is not None else None
        ),
        "allocation_portfolio_summary": (
            allocation_portfolio_summary.to_dict()
            if allocation_portfolio_summary is not None
            else None
        ),
        "rankings": rankings.to_dict(orient="index") if not rankings.empty else {},
        "allocations": allocations.to_dict(orient="index") if not allocations.empty else {},
        "execution_plan": (
            execution_plan.to_dict(orient="index") if not execution_plan.empty else {}
        ),
        "portfolio_risk_budget_pct": float(args.portfolio_risk_budget_pct),
        "price_sources": price_sources,
        "policy": policy,
        "policy_crc32": policy_crc32,
        "action_plan": action_plan,
        "errors": errors,
    }

    if journal_file is not None:
        journal_entry = _append_journal_entry(
            journal_path=journal_file,
            report=report,
            args=args,
        )
        report["journal"] = {
            "file": str(journal_file),
            "chain_hash": journal_entry["chain_hash"],
            "previous_chain_hash": journal_entry["previous_chain_hash"],
        }

    if output_dir is not None:
        _save_outputs(
            output_dir=output_dir,
            summary_df=summary_df,
            report=report,
            rankings=rankings,
            allocations=allocations,
            execution_plan=execution_plan,
            action_plan=action_plan,
            combined=combined,
            save_simulations=args.save_simulations,
        )

    result = {
        "simulations": combined,
        "summaries": summary_df,
        "portfolio_summary": portfolio_summary,
        "price_sources": price_sources,
        "report": report,
    }

    if render:
        if renderer is None:
            raise ValueError("renderer is required when render=True")
        renderer(result, args)
    if display_plots:
        maybe_show_simulation_plots(args, result)

    return result
