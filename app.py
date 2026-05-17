"""Lean web UI for the Monte Carlo decision engine."""

from __future__ import annotations

import base64
import io
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

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

try:
    from flask import (  # noqa: E402
        Flask,
        Response,
        render_template_string,
        request as flask_request,
        send_from_directory,
    )
except ModuleNotFoundError as exc:  # pragma: no cover - covered through entrypoint tests
    Flask = None  # type: ignore[assignment]
    Response = Any  # type: ignore[misc,assignment]
    render_template_string = None  # type: ignore[assignment]
    flask_request = None  # type: ignore[assignment]
    send_from_directory = None  # type: ignore[assignment]
    _FLASK_IMPORT_ERROR: ModuleNotFoundError | None = exc
else:
    _FLASK_IMPORT_ERROR = None

REPO_ROOT = Path(__file__).resolve().parent
PUBLIC_DIR = REPO_ROOT / "public"
SAMPLE_DATA_DIR = REPO_ROOT / "sample_data"
DEFAULT_TICKERS = "AAPL"
DEMO_SEED = "42"
CHOICES = {
    "job": ("simulate", "backtest"),
    "source": ("demo", "auto", "local"),
}
SOURCE_NOTES = {
    "demo": "Starts with the bundled sample so the first decision is immediate.",
    "auto": "Tries live prices, then falls back to local CSVs.",
    "local": (
        "Use one CSV, or a folder of <TICKER>.csv files, "
        "each with Date and Close columns."
    ),
}
STANCE_LABELS = {
    "RISK_ON": "Lean in",
    "SELECTIVE": "Selective",
    "DEFENSIVE": "Defensive",
    "NO_TRADE": "Stand aside",
}
FLASK_INSTALL_HINT = (
    "Browser UI needs the optional UI extra. "
    "Install `python3 -m pip install -e .[ui]`."
)

PAGE_TEMPLATE = """
<!doctype html>
<html lang="en">
  <head>
    <meta charset="utf-8">
    <meta name="viewport" content="width=device-width, initial-scale=1">
    <meta name="color-scheme" content="dark">
    <meta name="description"
      content="Monte Carlo — a decision engine for current ideas and historical backtests.">
    <title>Monte Carlo — Decision engine</title>
    <link rel="preconnect" href="https://fonts.googleapis.com">
    <link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
    <link rel="stylesheet" href="/styles.css">
  </head>
  <body>
    <main class="shell">
      <header class="masthead">
        <div>
          <p class="eyebrow">Monte&nbsp;Carlo</p>
          <h1>Simulate current ideas, or backtest history.</h1>
          <p class="lede">
            Start with the sample. Switch to live prices or your own CSVs when you're ready.
          </p>
        </div>
        <p class="masthead-note">The sample gives you a first decision in a second.</p>
      </header>

      <form class="controls" method="post" data-ui-form>
        <div class="group group-job">
          <fieldset>
            <legend>Job</legend>
            <div class="choice-row">
              {% for value, label in job_options %}
                <label class="choice">
                  <input
                    type="radio"
                    name="job"
                    value="{{ value }}"
                    {% if state.request.job == value %}checked{% endif %}
                  >
                  <span>{{ label }}</span>
                </label>
              {% endfor %}
            </div>
          </fieldset>
        </div>

        <label class="group group-tickers">
          <span>Tickers</span>
          <input
            type="text"
            name="tickers"
            value="{{ state.request.tickers }}"
            placeholder="AAPL MSFT"
          >
        </label>

        <div class="group group-source">
          <fieldset>
            <legend>Source</legend>
            <div class="choice-row" data-source-picker>
              {% for value, label in source_options %}
                <label class="choice">
                  <input
                    type="radio"
                    name="source"
                    value="{{ value }}"
                    data-note="{{ source_notes[value] }}"
                    {% if state.request.source == value %}checked{% endif %}
                  >
                  <span>{{ label }}</span>
                </label>
              {% endfor %}
            </div>
          </fieldset>
        </div>

        <div class="group group-actions">
          <span>Run</span>
          <div class="actions">
            <button type="submit" data-run-button>Run</button>
          </div>
        </div>

        <p class="source-note" data-source-note>{{ state.source_note }}</p>

        <label
          class="group group-path"
          data-local-path
          {% if state.request.source != "local" %}hidden{% endif %}
        >
          <span>CSV file or folder</span>
          <input
            type="text"
            name="data_path"
            value="{{ state.request.data_path or '' }}"
            placeholder="/Users/you/data or /Users/you/AAPL.csv"
          >
        </label>
      </form>

      <section class="result" aria-live="polite">
        {% if state.error %}
          <div class="alert">{{ state.error }}</div>
        {% endif %}

        <div class="headline">
          <p class="eyebrow">{{ state.eyebrow }}</p>
          <h2>{{ state.title }}</h2>
          <p class="summary">{{ state.summary }}</p>
        </div>

        {% if state.notes %}
          <ul class="notes">
            {% for note in state.notes %}
              <li>{{ note }}</li>
            {% endfor %}
          </ul>
        {% endif %}

        {% if state.metrics %}
          <section class="metrics">
            {% for metric in state.metrics %}
              <article class="metric">
                <div class="metric-label">{{ metric.label }}</div>
                <div class="metric-value">{{ metric.value }}</div>
              </article>
            {% endfor %}
          </section>
        {% endif %}

        {% if state.chart_svg %}
          <figure class="chart">
            <div class="chart-figure" role="img" aria-label="{{ state.chart_alt }}">
              {{ state.chart_svg|safe }}
            </div>
          </figure>
        {% elif state.chart_data_url %}
          <figure class="chart">
            <img class="chart-figure" src="{{ state.chart_data_url }}" alt="{{ state.chart_alt }}">
          </figure>
        {% endif %}

        {% if state.details_text %}
          <details>
            <summary>Terminal output</summary>
            <pre>{{ state.details_text }}</pre>
          </details>
        {% endif %}
      </section>

      <footer class="signature">
        <span>Monte&nbsp;Carlo · decision engine</span>
        <a href="https://github.com/pdj555/monte-carlo" rel="noopener">source · github</a>
      </footer>
    </main>

    <script>
      const form = document.querySelector("[data-ui-form]");
      const localPath = document.querySelector("[data-local-path]");
      const runButton = document.querySelector("[data-run-button]");
      const sourceNote = document.querySelector("[data-source-note]");

      function syncSource() {
        const selected = form.querySelector("input[name='source']:checked");
        localPath.hidden = !selected || selected.value !== "local";
        if (selected && sourceNote) {
          sourceNote.textContent = selected.dataset.note || "";
        }
      }

      form.addEventListener("change", syncSource);
      form.addEventListener("submit", () => {
        runButton.textContent = "Running...";
        runButton.disabled = true;
      });

      syncSource();
    </script>
  </body>
</html>
"""


@dataclass(frozen=True)
class UIRequest:
    job: str = "simulate"
    tickers: str = DEFAULT_TICKERS
    source: str = "demo"
    data_path: str | None = None


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
    chart_data_url: str | None = None
    chart_alt: str = ""
    details_text: str = ""
    error: str | None = None


def _coerce_choice(raw: str | None, *, group: str, default: str) -> str:
    value = (raw or "").strip().lower()
    return value if value in CHOICES[group] else default


def _ticker_tokens(raw: str) -> list[str]:
    tokens = [token.strip().upper() for token in raw.replace(",", " ").split() if token.strip()]
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


def validate_request(ui_request: UIRequest) -> str | None:
    if ui_request.source == "local":
        if ui_request.data_path is None:
            return "Choose a CSV file or folder before running Local CSV."
        if not Path(ui_request.data_path).expanduser().exists():
            return "That path was not found. Choose a CSV file or folder that exists."
    return None


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
    else:
        argv.extend(["--source", "auto"])

    if ui_request.job == "backtest" and ui_request.source == "demo":
        argv.extend(
            [
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
        )
    return argv


def _format_pct(value: float, *, signed: bool = False) -> str:
    return f"{value:+.1%}" if signed else f"{value:.1%}"


# Editorial palette — must stay aligned with public/styles.css.
_CHART_PAPER = "#f4efe6"
_CHART_MUTED = "#8d877c"
_CHART_HAIRLINE = "#2a2724"
_CHART_GOLD = "#d4a373"
_CHART_SAGE = "#8fb59a"


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
                color=_CHART_PAPER,
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
            if line.get_color() in {"C0", "#1f77b4", "tab:blue", "blue"}:
                line.set_color(_CHART_GOLD)
            line.set_linewidth(max(line.get_linewidth(), 0.9))


def _encode_figure_svg(fig: plt.Figure) -> str:
    _apply_chart_style(fig)
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
    # Strip XML/doctype prologue so the SVG inlines cleanly inside the page.
    marker = raw.find("<svg")
    return raw[marker:] if marker != -1 else raw


def _encode_figure_data_url(fig: plt.Figure) -> str:
    _apply_chart_style(fig)
    buffer = io.BytesIO()
    fig.savefig(
        buffer,
        format="png",
        bbox_inches="tight",
        transparent=True,
        dpi=180,
    )
    plt.close(fig)
    return "data:image/png;base64," + base64.b64encode(buffer.getvalue()).decode("ascii")


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
    return _encode_figure_svg(fig), f"Simulated price paths for {ticker}"


def _backtest_chart_payload(result: dict[str, object]) -> tuple[str | None, str]:
    equity_curve = result["equity_curve"]
    if not isinstance(equity_curve, pd.DataFrame) or equity_curve.empty:
        return None, ""

    fig = plot_equity_curve(equity_curve, title="Backtest · equity curve")
    return _encode_figure_svg(fig), "Backtest equity curve"


def _build_simulation_state(
    ui_request: UIRequest,
    result: dict[str, object],
    details_text: str,
) -> PageState:
    report = result["report"]
    if not isinstance(report, dict):
        raise ValueError(
            "Simulation report was not returned in the expected format."
        )

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
            Metric(
                "Expected return",
                _format_pct(float(top_row["expected_return"])),
            ),
            Metric(
                "Chance of gain",
                _format_pct(float(top_row["prob_above_current"])),
            ),
            Metric(
                "95% downside",
                _format_pct(float(top_row["value_at_risk_95_pct"])),
            ),
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
        eyebrow="Simulate",
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
        raise ValueError(
            "Backtest summary was not returned in the expected format."
        )

    cash_comparison = _format_pct(
        float(summary["excess_return_vs_cash"]),
        signed=True,
    )
    notes = [f"Cash comparison: {cash_comparison}."]
    if isinstance(rebalance_log, pd.DataFrame) and not rebalance_log.empty:
        notes.append(f"Rebalances completed: {len(rebalance_log)}.")

    metrics = (
        Metric(
            "Strategy return",
            _format_pct(float(summary["strategy_total_return"])),
        ),
        Metric(
            "Annualized",
            _format_pct(float(summary["strategy_annualized_return"])),
        ),
        Metric(
            "Max drawdown",
            _format_pct(float(summary["strategy_max_drawdown"])),
        ),
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
    if ui_request.source == "auto":
        return "Live prices weren’t available. Try again or switch to Demo sample or Local CSV."
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
        title="Couldn’t finish that run.",
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
    return create_page_state(_normalise_request())


app = Flask(__name__) if Flask is not None else None


if app is not None:

    @app.get("/styles.css")
    def styles_css() -> Response:
        return send_from_directory(PUBLIC_DIR, "styles.css", mimetype="text/css")

    @app.get("/healthz")
    def healthz() -> tuple[str, int]:
        return "ok", 200

    @app.get("/favicon.ico")
    def favicon() -> Response:
        return Response(status=204)

    @app.route("/", methods=["GET", "POST"])
    def index() -> str:
        if flask_request.method == "POST":
            ui_request = request_from_form(flask_request.form)
        else:
            ui_request = _normalise_request()
        state = create_page_state(ui_request)
        return render_template_string(
            PAGE_TEMPLATE,
            job_options=(("simulate", "Simulate"), ("backtest", "Backtest")),
            source_options=(
                ("demo", "Demo sample"),
                ("auto", "Try live data"),
                ("local", "Local CSV"),
            ),
            source_notes=SOURCE_NOTES,
            state=state,
        )


def main() -> int:
    if app is None:
        print(FLASK_INSTALL_HINT, file=sys.stderr)
        return 2

    app.run(
        host=os.environ.get("HOST", "127.0.0.1"),
        port=int(os.environ.get("PORT", "8000")),
        debug=False,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
