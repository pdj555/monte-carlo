"use client";

import type { WorkbenchPayload } from "../../lib/types";

type Props = {
  state: WorkbenchPayload;
  isPending: boolean;
};

export function RunResults({ state, isPending }: Props) {
  return (
    <div className={`dashboard${isPending ? " is-loading" : ""}`} aria-busy={isPending} aria-live="polite">
      {state.error ? <div className="banner">{state.error}</div> : null}

      <section className="frame frame-chart">
        <h2 className="frame-label">Paths</h2>
        {state.chartSvg ? (
          <figure className="chart">
            <div
              aria-label={state.chartAlt}
              dangerouslySetInnerHTML={{ __html: state.chartSvg }}
              role="img"
            />
          </figure>
        ) : (
          <div className="chart chart-empty">{isPending ? "Running…" : "—"}</div>
        )}
      </section>

      {state.metrics.length > 0 ? (
        <section className="frame frame-metrics">
          <h2 className="frame-label">Metrics</h2>
          <div className="metric-row">
            {state.metrics.map((metric) => (
              <p className="metric-line" key={metric.label}>
                <span className="metric-label">{metric.label}</span>
                <span className="metric-value">{metric.value}</span>
              </p>
            ))}
          </div>
        </section>
      ) : null}

      <section className="frame frame-result">
        <h2 className="frame-label">Read</h2>
        <p className="result-copy">{state.summary}</p>
        {state.notes.length > 0 ? (
          <ul className="note-list">
            {state.notes.map((note) => (
              <li key={note}>{note}</li>
            ))}
          </ul>
        ) : null}
      </section>

      {state.detailsText ? (
        <details className="frame frame-log">
          <summary>Log</summary>
          <pre>{state.detailsText}</pre>
        </details>
      ) : null}
    </div>
  );
}
