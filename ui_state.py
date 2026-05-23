"""Presentation state for the Monte Carlo browser UI."""

from __future__ import annotations

import io
import os
from dataclasses import dataclass
from pathlib import Path

os.environ.setdefault("MPLBACKEND", "Agg")

import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402

from public_cli import (  # noqa: E402
    describe_price_sources,
    execute_public_backtest,
    execute_public_simulate,
    format_public_backtest_output,
    format_public_simulation_output,
    parse_public_args,
)
from viz import plot_equity_curve, plot_paths  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parent
SAMPLE_DATA_DIR = REPO_ROOT / "sample_data"
DEFAULT_TICKERS = "AAPL"
DEMO_SEED = "42"

CHOICES = {
    "job": ("simulate", "backtest"),
    "source": ("auto", "online", "demo", "local"),
}
SOURCE_NOTES = {
    "auto": "Live Yahoo Finance prices with CSV fallback.",
    "online": "Live Yahoo Finance prices only.",
    "demo": "Bundled sample. Fast and offline.",
    "local": (
        "Use one CSV, or a folder of <TICKER>.csv files, with Date and Close "
        "columns."
    ),
}
STANCE_LABELS = {
    "RISK_ON": "Lean in",
    "SELECTIVE": "Selective",
    "DEFENSIVE": "Defensive",
    "NO_TRADE": "Stand aside",
}


@dataclass(frozen=True)
class UIRequest:
    """Normalized browser request values."""

    job: str = "simulate"
    tickers: str = DEFAULT_TICKERS
    source: str = "demo"
    data_path: str | None = None

    @property
    def ticker_label(self) -> str:
        return ", ".join(_ticker_tokens(self.tickers))


@dataclass(frozen=True)
class Metric:
    label: str
    value: str


@dataclass(frozen=True)
class PageState:
    request: UIRequest
    source_note: str
    eyebrow: str
    title: str
    summary: str
    notes: tuple[str, ...] = ()
    metrics: tuple[Metric, ...] = ()
    chart_svg: str | None = None
    chart_alt: str = ""
    details_text: str = ""
    error: str | None = None


def _coerce_choice(raw: str | None, *, group: str, default: str) -> str:
    value = (raw or "").strip().lower()
    return value if value in CHOICES[group] else default


def _ticker_tokens(raw: str) -> list[str]:
    tokens = [
        token.strip().upper()
        for token in raw.replace(",", " ").split()
        if token.strip()
    ]
    return tokens or [DEFAULT_TICKERS]


def _normalise_request(
    *,
    job: str | None = None,
    tickers: str | None = None,
    source: str | None = None,
    data_path: str | None = None,
) -> UIRequest:
    clean_tickers = " ".join(_ticker_tokens(tickers or DEFAULT_TICKERS))
    clean_path = (data_path or "").strip() or None
    return UIRequest(
        job=_coerce_choice(job, group="job", default="simulate"),
        tickers=clean_tickers,
        source=_coerce_choice(source, group="source", default="demo"),
        data_path=clean_path,
    )


def request_from_form(form: object) -> UIRequest:
    getter = getattr(form, "get", lambda *_args, **_kwargs: None)
    return _normalise_request(
        job=getter("job"),
        tickers=getter("tickers"),
        source=getter("source"),
        data_path=getter("data_path"),
    )


def request_from_payload(payload: dict[str, object]) -> UIRequest:
    raw_path = payload.get("data_path")
    if raw_path is None:
        raw_path = payload.get("dataPath")
    data_path = str(raw_path).strip() if raw_path else None
    return _normalise_request(
        job=str(payload.get("job") or "simulate"),
        tickers=str(payload.get("tickers") or DEFAULT_TICKERS),
        source=str(payload.get("source") or "auto"),
        data_path=data_path,
    )


def validate_request(ui_request: UIRequest) -> str | None:
    if ui_request.source == "local":
        if ui_request.data_path is None:
            return "Choose a CSV file or folder before running CSV."
        if not Path(ui_request.data_path).expanduser().exists():
            return "That path was not found. Choose a CSV file or folder that exists."
    return None


def _uses_sample_data(ui_request: UIRequest) -> bool:
    if ui_request.source == "demo":
        return True
    if ui_request.source != "local" or not ui_request.data_path:
        return False
    try:
        return Path(ui_request.data_path).expanduser().resolve().is_relative_to(
            SAMPLE_DATA_DIR.resolve()
        )
    except OSError:
        return False


def _ui_backtest_short_argv() -> list[str]:
    return [
        "--lookback",
        "5",
        "--hold",
        "3",
        "--rebalance",
        "3",
        "--top",
        "1",
        "--scenarios",
        "10",
    ]


def _ui_backtest_live_argv() -> list[str]:
    return [
        "--lookback",
        "60",
        "--hold",
        "20",
        "--rebalance",
        "20",
        "--top",
        "1",
        "--scenarios",
        "100",
        "--seed",
        DEMO_SEED,
    ]


def build_public_argv(ui_request: UIRequest) -> list[str]:
    argv = [ui_request.job, *_ticker_tokens(ui_request.tickers)]

    if ui_request.source == "demo":
        argv.extend(
            [
                "--source",
                "offline",
                "--data-path",
                str(SAMPLE_DATA_DIR),
                "--seed",
                DEMO_SEED,
            ]
        )
        if ui_request.job == "simulate":
            argv.extend(["--days", "20"])
    elif ui_request.source == "local":
        argv.extend(
            [
                "--source",
                "offline",
                "--data-path",
                str(Path(ui_request.data_path or "").expanduser()),
            ]
        )
    elif ui_request.source == "online":
        argv.extend(["--source", "online"])
    else:
        argv.extend(
            [
                "--source",
                "auto",
                "--data-path",
                str(SAMPLE_DATA_DIR),
            ]
        )

    if ui_request.job == "backtest":
        if _uses_sample_data(ui_request):
            argv.extend(_ui_backtest_short_argv())
            if "--seed" not in argv:
                argv.extend(["--seed", DEMO_SEED])
        else:
            argv.extend(_ui_backtest_live_argv())

    return argv


def _format_pct(value: float, *, signed: bool = False) -> str:
    return f"{value:+.1%}" if signed else f"{value:.1%}"


# Chart palette aligned with the DisTrO-style workbench.
_CHART_INK = "#1769ff"
_CHART_MUTED = "#74a7ff"
_CHART_HAIRLINE = "#74a7ff"
_CHART_LINE = "#1769ff"


def _apply_chart_style(fig: plt.Figure) -> None:
    fig.patch.set_alpha(0.0)
    for ax in fig.get_axes():
        ax.set_facecolor("none")
        for spine_name, spine in ax.spines.items():
            if spine_name in ("top", "right"):
                spine.set_visible(False)
            else:
                spine.set_color(_CHART_HAIRLINE)
                spine.set_linewidth(0.8)
        ax.tick_params(
            colors=_CHART_MUTED,
            labelsize=9,
            length=4,
            width=0.6,
        )
        for label in ax.get_xticklabels() + ax.get_yticklabels():
            label.set_color(_CHART_MUTED)
            label.set_fontfamily("monospace")
        ax.grid(True, which="major", color=_CHART_HAIRLINE, linewidth=0.5, alpha=0.6)
        ax.set_axisbelow(True)
        if ax.get_title():
            ax.set_title(
                ax.get_title(),
                color=_CHART_INK,
                fontsize=12,
                pad=14,
                loc="left",
                fontweight="normal",
            )
        if ax.get_xlabel():
            ax.set_xlabel(ax.get_xlabel(), color=_CHART_MUTED, fontsize=10)
        if ax.get_ylabel():
            ax.set_ylabel(ax.get_ylabel(), color=_CHART_MUTED, fontsize=10)
        for line in ax.get_lines():
            lw = float(line.get_linewidth() or 1.0)
            line.set_color(_CHART_INK if lw >= 1.8 else _CHART_LINE)
            line.set_linewidth(max(lw, 0.6))
            line.set_alpha(1.0 if lw >= 1.8 else 0.28)


def _encode_figure_svg(fig: plt.Figure) -> str:
    _apply_chart_style(fig)
    fig.set_size_inches(9.6, 4.8)
    fig.tight_layout()
    buffer = io.StringIO()
    fig.savefig(
        buffer,
        format="svg",
        bbox_inches="tight",
        transparent=True,
        metadata={"Date": None},
    )
    plt.close(fig)
    raw = buffer.getvalue()
    marker = raw.find("<svg")
    return raw[marker:] if marker != -1 else raw


def _simulate_chart_payload(result: dict[str, object]) -> tuple[str | None, str]:
    report = result["report"]
    if not isinstance(report, dict):
        return None, ""
    simulations = result["simulations"]
    if not isinstance(simulations, pd.DataFrame) or simulations.empty:
        return None, ""

    action_plan = report.get("action_plan", {})
    ticker = None
    if isinstance(action_plan, dict):
        pick = action_plan.get("primary_pick")
        if isinstance(pick, dict):
            raw_ticker = pick.get("ticker")
            if isinstance(raw_ticker, str) and raw_ticker:
                ticker = raw_ticker

    if ticker is None:
        report_rankings = report.get("rankings", {})
        if isinstance(report_rankings, dict) and report_rankings:
            ticker = str(next(iter(report_rankings.keys())))

    if ticker is None:
        return None, ""

    fig = plot_paths(
        simulations,
        ticker=ticker,
        title=f"{ticker} · simulated paths",
        max_paths=50,
    )
    for ax in fig.get_axes():
        ax.set_title("")
        legend = ax.get_legend()
        if legend is not None:
            legend.remove()
    return _encode_figure_svg(fig), f"Simulated price paths for {ticker}"


def _backtest_chart_payload(result: dict[str, object]) -> tuple[str | None, str]:
    equity_curve = result["equity_curve"]
    if not isinstance(equity_curve, pd.DataFrame) or equity_curve.empty:
        return None, ""

    fig = plot_equity_curve(equity_curve, title="Backtest · equity curve")
    for ax in fig.get_axes():
        ax.set_title("")
        legend = ax.get_legend()
        if legend is not None:
            legend.remove()
    return _encode_figure_svg(fig), "Backtest equity curve"


def _build_simulation_state(
    ui_request: UIRequest,
    result: dict[str, object],
    details_text: str,
) -> PageState:
    report = result["report"]
    if not isinstance(report, dict):
        raise ValueError("Simulation report was not returned in the expected format.")

    action_plan = report.get("action_plan", {})
    if not isinstance(action_plan, dict):
        raise ValueError(
            "Simulation action plan was not returned in the expected format."
        )

    rankings = pd.DataFrame.from_dict(report.get("rankings", {}), orient="index")
    notes: list[str] = []
    pick = action_plan.get("primary_pick")
    if isinstance(pick, dict):
        notes.append(
            f"Top idea: {pick['ticker']} at {float(pick['weight']):.1%} weight."
        )
    avoid_list = action_plan.get("avoid_list", [])
    if isinstance(avoid_list, list) and avoid_list:
        notes.append(f"Avoid for now: {', '.join(str(item) for item in avoid_list)}.")
    cash_weight = float(action_plan.get("cash_weight", 0.0))
    if cash_weight > 0:
        notes.append(f"Hold {cash_weight:.1%} in cash while conviction stays muted.")

    errors = report.get("errors", [])
    if isinstance(errors, list):
        for item in errors:
            if isinstance(item, dict):
                ticker = item.get("ticker")
                error = item.get("error")
                if ticker and error:
                    notes.append(f"Skipped {ticker}: {error}")

    metrics: list[Metric] = []
    if not rankings.empty:
        top_ticker = str(rankings.index[0])
        top_row = rankings.iloc[0]
        metrics = [
            Metric("Top ticker", top_ticker),
            Metric("Expected return", _format_pct(float(top_row["expected_return"]))),
            Metric("Chance of gain", _format_pct(float(top_row["prob_above_current"]))),
            Metric("95% downside", _format_pct(float(top_row["value_at_risk_95_pct"]))),
        ]

    chart_svg, chart_alt = _simulate_chart_payload(result)
    stance = str(action_plan.get("stance", "Decision ready"))
    title = STANCE_LABELS.get(stance, stance.replace("_", " ").title())
    summary = str(action_plan.get("headline", "A fresh read is ready."))
    source_note = describe_price_sources(result.get("price_sources")) or SOURCE_NOTES[
        ui_request.source
    ]

    if result["summaries"].empty and notes:
        title = "No decision yet"
        summary = notes[0]

    return PageState(
        request=ui_request,
        source_note=source_note,
        eyebrow="Simulation",
        title=title,
        summary=summary,
        notes=tuple(notes),
        metrics=tuple(metrics),
        chart_svg=chart_svg,
        chart_alt=chart_alt,
        details_text=details_text,
    )


def _backtest_headline(summary: pd.Series) -> str:
    excess = float(summary["excess_return_vs_equal_weight"])
    if excess > 0.002:
        return f"Beat equal weight by {_format_pct(excess)}."
    if excess < -0.002:
        return f"Trailed equal weight by {_format_pct(abs(excess))}."
    return "Finished close to equal weight."


def _build_backtest_state(
    ui_request: UIRequest,
    result: dict[str, object],
    details_text: str,
) -> PageState:
    summary = result["summary"]
    rebalance_log = result["rebalance_log"]
    if not isinstance(summary, pd.Series):
        raise ValueError("Backtest summary was not returned in the expected format.")

    cash_comparison = _format_pct(
        float(summary["excess_return_vs_cash"]),
        signed=True,
    )
    notes = [f"Cash comparison: {cash_comparison}."]
    if isinstance(rebalance_log, pd.DataFrame) and not rebalance_log.empty:
        notes.append(f"Rebalances completed: {len(rebalance_log)}.")

    metrics = (
        Metric("Strategy return", _format_pct(float(summary["strategy_total_return"]))),
        Metric("Annualized", _format_pct(float(summary["strategy_annualized_return"]))),
        Metric("Max drawdown", _format_pct(float(summary["strategy_max_drawdown"]))),
        Metric(
            "Vs equal weight",
            _format_pct(float(summary["excess_return_vs_equal_weight"]), signed=True),
        ),
    )
    chart_svg, chart_alt = _backtest_chart_payload(result)
    source_note = describe_price_sources(result.get("price_sources")) or SOURCE_NOTES[
        ui_request.source
    ]

    return PageState(
        request=ui_request,
        source_note=source_note,
        eyebrow="Backtest",
        title=_backtest_headline(summary),
        summary=(
            f"Strategy return {_format_pct(float(summary['strategy_total_return']))}, "
            f"annualized {_format_pct(float(summary['strategy_annualized_return']))}, "
            f"max drawdown {_format_pct(float(summary['strategy_max_drawdown']))}."
        ),
        notes=tuple(notes),
        metrics=metrics,
        chart_svg=chart_svg,
        chart_alt=chart_alt,
        details_text=details_text,
    )


def _friendly_runtime_message(ui_request: UIRequest, raw_message: str) -> str:
    if ui_request.source in {"auto", "online"} and "price history" not in raw_message.lower():
        if "download" in raw_message.lower() or "network" in raw_message.lower():
            return "Live prices weren't available. Try again or switch to Sample or CSV."
    return raw_message


def _error_state(
    ui_request: UIRequest,
    summary: str,
    *,
    details_text: str | None = None,
) -> PageState:
    return PageState(
        request=ui_request,
        source_note=SOURCE_NOTES[ui_request.source],
        eyebrow="Needs attention",
        title="Couldn't finish that run.",
        summary=summary,
        details_text=details_text or summary,
        error=summary,
    )


def create_page_state(ui_request: UIRequest) -> PageState:
    validation_error = validate_request(ui_request)
    if validation_error is not None:
        return _error_state(ui_request, validation_error)

    argv = build_public_argv(ui_request)
    args = parse_public_args(argv)

    try:
        if ui_request.job == "simulate":
            result = execute_public_simulate(args)
            details_text = format_public_simulation_output(
                result,
                details=True,
                output=args.output,
            )
            return _build_simulation_state(ui_request, result, details_text)

        result = execute_public_backtest(args)
        details_text = format_public_backtest_output(
            result,
            details=True,
            output=args.output,
        )
        return _build_backtest_state(ui_request, result, details_text)
    except Exception as exc:
        raw_message = str(exc)
        return _error_state(
            ui_request,
            _friendly_runtime_message(ui_request, raw_message),
            details_text=raw_message,
        )


def build_default_state() -> PageState:
    return create_page_state(_normalise_request(source="demo"))
