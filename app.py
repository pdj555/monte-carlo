"""Lean web UI for the Monte Carlo decision engine."""

from __future__ import annotations

import base64
import contextlib
import io
import os
from dataclasses import dataclass
from pathlib import Path

os.environ.setdefault("MPLBACKEND", "Agg")

import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402
from flask import Flask, Response, render_template_string, request as flask_request  # noqa: E402

from public_cli import parse_public_args, run_public_backtest, run_public_simulate  # noqa: E402
from viz import plot_equity_curve, plot_paths  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parent
SAMPLE_DATA_DIR = REPO_ROOT / "sample_data"
DEFAULT_TICKERS = "AAPL"
DEMO_SEED = "42"
CHOICES = {
    "job": ("simulate", "backtest"),
    "source": ("demo", "auto", "local"),
}
SOURCE_NOTES = {
    "demo": "Starts with the bundled sample so the first decision is immediate.",
    "auto": "Starts with live prices, then falls back to local CSVs.",
    "local": "Use a CSV file or folder with Date and Close columns.",
}
STANCE_LABELS = {
    "RISK_ON": "Lean in",
    "SELECTIVE": "Selective",
    "DEFENSIVE": "Defensive",
    "NO_TRADE": "Stand aside",
}

APP_CSS = """
:root {
  --ink: #171716;
  --muted: #5f5d59;
  --line: rgba(23, 23, 22, 0.12);
  --panel: rgba(255, 255, 255, 0.76);
  --accent: #2b6c58;
  --accent-soft: #dceee7;
  --warm-soft: #f2e3dc;
  --alert-ink: #7a3d23;
  --alert-bg: #f8eeea;
}

* {
  box-sizing: border-box;
}

[hidden] {
  display: none !important;
}

html {
  background:
    linear-gradient(180deg, rgba(255, 255, 255, 0.88), rgba(255, 255, 255, 0.96)),
    linear-gradient(
      135deg,
      rgba(220, 238, 231, 0.85),
      rgba(242, 227, 220, 0.8) 58%,
      rgba(248, 248, 245, 0.96)
    );
}

body {
  margin: 0;
  min-height: 100vh;
  color: var(--ink);
  font-family:
    Inter,
    ui-sans-serif,
    system-ui,
    -apple-system,
    BlinkMacSystemFont,
    "Segoe UI",
    sans-serif;
}

.shell {
  width: min(1120px, calc(100% - 32px));
  margin: 0 auto;
  padding: 32px 0 48px;
}

.masthead {
  display: flex;
  justify-content: space-between;
  align-items: flex-start;
  gap: 24px;
  padding-bottom: 24px;
  border-bottom: 1px solid var(--line);
}

.eyebrow {
  margin: 0 0 10px;
  color: var(--accent);
  font-size: 12px;
  font-weight: 600;
  letter-spacing: 0.08em;
  text-transform: uppercase;
}

h1 {
  margin: 0;
  max-width: 10ch;
  font-size: 42px;
  line-height: 1.02;
  font-weight: 700;
}

.lede {
  margin: 14px 0 0;
  max-width: 52ch;
  color: var(--muted);
  font-size: 16px;
  line-height: 1.55;
}

.masthead-note {
  margin: 2px 0 0;
  max-width: 18ch;
  color: var(--muted);
  font-size: 14px;
  line-height: 1.45;
  text-align: right;
}

.controls {
  display: grid;
  grid-template-columns: repeat(12, minmax(0, 1fr));
  gap: 18px 16px;
  align-items: end;
  padding: 24px 0 28px;
  border-bottom: 1px solid var(--line);
}

.group {
  display: grid;
  gap: 10px;
}

.group span,
.group legend {
  padding: 0;
  color: var(--muted);
  font-size: 13px;
  font-weight: 600;
}

.group fieldset {
  margin: 0;
  padding: 0;
  border: 0;
}

.group-job {
  grid-column: span 3;
}

.group-tickers {
  grid-column: span 4;
}

.group-source {
  grid-column: span 3;
}

.group-actions {
  grid-column: span 2;
  justify-items: end;
}

.group-path {
  grid-column: 1 / -1;
}

.choice-row {
  display: flex;
  flex-wrap: wrap;
  gap: 8px;
}

.choice {
  position: relative;
}

.choice input {
  position: absolute;
  inset: 0;
  opacity: 0;
  pointer-events: none;
}

.choice span {
  display: inline-flex;
  align-items: center;
  justify-content: center;
  min-height: 42px;
  padding: 0 14px;
  border: 1px solid var(--line);
  border-radius: 8px;
  background: rgba(255, 255, 255, 0.62);
  color: var(--ink);
  font-size: 14px;
  transition: border-color 140ms ease, background-color 140ms ease, color 140ms ease;
}

.choice input:checked + span {
  border-color: var(--ink);
  background: var(--ink);
  color: #fff;
}

input[type="text"] {
  width: 100%;
  min-height: 46px;
  border: 1px solid var(--line);
  border-radius: 8px;
  padding: 0 14px;
  background: rgba(255, 255, 255, 0.72);
  color: var(--ink);
  font: inherit;
}

input[type="text"]:focus-visible,
button:focus-visible,
.choice input:focus-visible + span {
  outline: 2px solid rgba(43, 108, 88, 0.35);
  outline-offset: 2px;
}

.source-note {
  grid-column: 1 / -1;
  margin: -6px 0 0;
  color: var(--muted);
  font-size: 14px;
  line-height: 1.45;
}

.actions {
  display: flex;
  align-items: center;
  gap: 14px;
}

button {
  min-height: 46px;
  border: 0;
  border-radius: 8px;
  padding: 0 18px;
  background: var(--ink);
  color: #fff;
  font: inherit;
  font-weight: 600;
  cursor: pointer;
}

button[disabled] {
  cursor: wait;
  opacity: 0.82;
}

.result {
  display: grid;
  gap: 20px;
  padding-top: 28px;
}

.headline {
  display: grid;
  gap: 10px;
}

.headline h2 {
  margin: 0;
  max-width: 18ch;
  font-size: 34px;
  line-height: 1.05;
  font-weight: 700;
}

.summary {
  margin: 0;
  max-width: 58ch;
  color: var(--muted);
  font-size: 16px;
  line-height: 1.55;
}

.notes {
  display: grid;
  gap: 10px;
  margin: 0;
  padding: 0;
  list-style: none;
}

.notes li {
  position: relative;
  padding-left: 16px;
  color: var(--muted);
  font-size: 15px;
  line-height: 1.5;
}

.notes li::before {
  content: "";
  position: absolute;
  left: 0;
  top: 0.54em;
  width: 7px;
  height: 7px;
  border-radius: 999px;
  background: var(--accent);
}

.metrics {
  display: grid;
  grid-template-columns: repeat(auto-fit, minmax(170px, 1fr));
  gap: 12px;
}

.metric {
  padding: 16px;
  border: 1px solid var(--line);
  border-radius: 8px;
  background: var(--panel);
  backdrop-filter: blur(8px);
}

.metric-label {
  color: var(--muted);
  font-size: 13px;
  line-height: 1.35;
}

.metric-value {
  margin-top: 8px;
  font-size: 28px;
  line-height: 1.05;
  font-weight: 650;
}

.chart {
  margin: 0;
  overflow: hidden;
  border: 1px solid var(--line);
  border-radius: 8px;
  background: rgba(255, 255, 255, 0.82);
}

.chart img {
  display: block;
  width: 100%;
  height: auto;
}

.alert {
  padding: 14px 16px;
  border: 1px solid rgba(122, 61, 35, 0.18);
  border-radius: 8px;
  background: var(--alert-bg);
  color: var(--alert-ink);
  font-size: 15px;
  line-height: 1.5;
}

details {
  border-top: 1px solid var(--line);
  padding-top: 14px;
}

details summary {
  cursor: pointer;
  font-weight: 600;
}

pre {
  margin: 14px 0 0;
  padding: 18px;
  overflow: auto;
  border-radius: 8px;
  background: #181817;
  color: #f5f5f0;
  font-size: 13px;
  line-height: 1.55;
  white-space: pre-wrap;
  word-break: break-word;
}

@media (max-width: 900px) {
  .masthead {
    display: grid;
  }

  .masthead-note {
    text-align: left;
  }

  .controls {
    grid-template-columns: repeat(6, minmax(0, 1fr));
  }

  .group-job,
  .group-tickers,
  .group-source,
  .group-actions {
    grid-column: span 3;
  }
}

@media (max-width: 640px) {
  .shell {
    width: min(100% - 24px, 100%);
    padding-top: 24px;
  }

  h1 {
    font-size: 32px;
  }

  .headline h2 {
    font-size: 28px;
  }

  .controls {
    grid-template-columns: 1fr;
  }

  .group-job,
  .group-tickers,
  .group-source,
  .group-actions,
  .group-path {
    grid-column: 1;
  }

  .group-actions {
    justify-items: stretch;
  }

  .actions {
    justify-content: space-between;
  }
}
"""

PAGE_TEMPLATE = """
<!doctype html>
<html lang="en">
  <head>
    <meta charset="utf-8">
    <meta name="viewport" content="width=device-width, initial-scale=1">
    <title>Monte Carlo</title>
    <link rel="stylesheet" href="/app.css">
  </head>
  <body>
    <main class="shell">
      <header class="masthead">
        <div>
          <p class="eyebrow">Monte Carlo</p>
          <h1>Current idea or historical test.</h1>
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
          <span>Local CSV path</span>
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

        {% if state.chart_data_url %}
          <figure class="chart">
            <img src="{{ state.chart_data_url }}" alt="{{ state.chart_alt }}">
          </figure>
        {% endif %}

        {% if state.details_text %}
          <details>
            <summary>Terminal output</summary>
            <pre>{{ state.details_text }}</pre>
          </details>
        {% endif %}
      </section>
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
    chart_data_url: str | None = None
    chart_alt: str = ""
    details_text: str = ""
    error: str | None = None


app = Flask(__name__)


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

    argv.append("--details")
    return argv


def _format_pct(value: float, *, signed: bool = False) -> str:
    return f"{value:+.1%}" if signed else f"{value:.1%}"


def _encode_figure(fig: plt.Figure) -> str:
    buffer = io.BytesIO()
    fig.savefig(buffer, format="png", bbox_inches="tight")
    plt.close(fig)
    return "data:image/png;base64," + base64.b64encode(buffer.getvalue()).decode("ascii")


def _combined_output(
    output_buffer: io.StringIO,
    error_buffer: io.StringIO,
) -> str:
    return "\n\n".join(
        part
        for part in (output_buffer.getvalue().strip(), error_buffer.getvalue().strip())
        if part
    ).strip()


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
        title=f"{ticker} simulated paths",
        max_paths=50,
    )
    return _encode_figure(fig), f"Simulated price paths for {ticker}"


def _backtest_chart_payload(result: dict[str, object]) -> tuple[str | None, str]:
    equity_curve = result["equity_curve"]
    if not isinstance(equity_curve, pd.DataFrame) or equity_curve.empty:
        return None, ""

    fig = plot_equity_curve(equity_curve, title="Backtest equity curve")
    return _encode_figure(fig), "Backtest equity curve"


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

    chart_data_url, chart_alt = _simulate_chart_payload(result)
    stance = str(action_plan.get("stance", "Decision ready"))
    title = STANCE_LABELS.get(stance, stance.replace("_", " ").title())
    summary = str(action_plan.get("headline", "A fresh read is ready."))

    if result["summaries"].empty and notes:
        title = "No decision yet"
        summary = notes[0]

    return PageState(
        request=ui_request,
        source_note=SOURCE_NOTES[ui_request.source],
        eyebrow="Simulate",
        title=title,
        summary=summary,
        notes=tuple(notes),
        metrics=tuple(metrics),
        chart_data_url=chart_data_url,
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
    chart_data_url, chart_alt = _backtest_chart_payload(result)

    return PageState(
        request=ui_request,
        source_note=SOURCE_NOTES[ui_request.source],
        eyebrow="Backtest",
        title=_backtest_headline(summary),
        summary=(
            f"Strategy return {_format_pct(float(summary['strategy_total_return']))}, "
            f"annualized {_format_pct(float(summary['strategy_annualized_return']))}, "
            f"max drawdown {_format_pct(float(summary['strategy_max_drawdown']))}."
        ),
        notes=tuple(notes),
        metrics=metrics,
        chart_data_url=chart_data_url,
        chart_alt=chart_alt,
        details_text=details_text,
    )


def _error_state(ui_request: UIRequest, message: str) -> PageState:
    return PageState(
        request=ui_request,
        source_note=SOURCE_NOTES[ui_request.source],
        eyebrow="Needs attention",
        title="Couldn’t finish that run.",
        summary=message,
        details_text=message,
        error=message,
    )


def create_page_state(ui_request: UIRequest) -> PageState:
    validation_error = validate_request(ui_request)
    if validation_error is not None:
        return _error_state(ui_request, validation_error)

    argv = build_public_argv(ui_request)
    args = parse_public_args(argv)
    output_buffer = io.StringIO()
    error_buffer = io.StringIO()

    try:
        with contextlib.redirect_stdout(output_buffer), contextlib.redirect_stderr(error_buffer):
            if ui_request.job == "simulate":
                result = run_public_simulate(args)
                details_text = _combined_output(output_buffer, error_buffer)
                return _build_simulation_state(ui_request, result, details_text)

            result = run_public_backtest(args)
            details_text = _combined_output(output_buffer, error_buffer)
            return _build_backtest_state(ui_request, result, details_text)
    except Exception as exc:
        return _error_state(ui_request, str(exc))


def build_default_state() -> PageState:
    return create_page_state(_normalise_request())


@app.get("/app.css")
def app_css() -> Response:
    return Response(APP_CSS, mimetype="text/css")


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
            ("auto", "Live first"),
            ("local", "Local CSV"),
        ),
        source_notes=SOURCE_NOTES,
        state=state,
    )


def main() -> int:
    app.run(
        host=os.environ.get("HOST", "127.0.0.1"),
        port=int(os.environ.get("PORT", "8000")),
        debug=False,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
