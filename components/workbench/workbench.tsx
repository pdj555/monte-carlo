"use client";

import { useEffect, useState, useTransition, type FormEvent } from "react";

import type { WorkbenchPayload, WorkbenchRequest } from "../../lib/types";
import { jobOptions, sourceOptions } from "./constants";
import { RunResults } from "./results";

type Props = {
  initialState: WorkbenchPayload;
};

function normalizeRequest(payload: WorkbenchPayload): WorkbenchRequest {
  return {
    job: payload.request.job,
    tickers: payload.request.tickers,
    source: payload.request.source,
    dataPath: payload.request.data_path ?? null,
  };
}

function readStoredTheme(): "light" | "dark" {
  if (typeof window === "undefined") {
    return "light";
  }
  const stored = window.localStorage.getItem("mc-theme");
  if (stored === "light" || stored === "dark") {
    return stored;
  }
  return window.matchMedia("(prefers-color-scheme: dark)").matches ? "dark" : "light";
}

export function Workbench({ initialState }: Props) {
  const [state, setState] = useState(initialState);
  const [request, setRequest] = useState<WorkbenchRequest>(() => normalizeRequest(initialState));
  const [isPending, startTransition] = useTransition();
  const [theme, setTheme] = useState<"light" | "dark">("light");

  useEffect(() => {
    setTheme(readStoredTheme());
  }, []);

  useEffect(() => {
    document.documentElement.dataset.theme = theme;
    window.localStorage.setItem("mc-theme", theme);
    const meta = document.querySelector('meta[name="theme-color"]');
    if (meta) {
      meta.setAttribute("content", theme === "dark" ? "#05080c" : "#ffffff");
    }
  }, [theme]);

  function updateRequest(patch: Partial<WorkbenchRequest>) {
    setRequest((current) => ({ ...current, ...patch }));
  }

  function selectSource(source: WorkbenchRequest["source"]) {
    updateRequest({
      source,
      dataPath:
        source === "local" && !request.dataPath?.trim() ? "sample_data" : request.dataPath,
    });
  }

  const canRun = request.source !== "local" || Boolean(request.dataPath?.trim());

  function run() {
    startTransition(async () => {
      try {
        const response = await fetch("/api/run", {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify(request),
        });
        const payload = (await response.json()) as WorkbenchPayload;
        if (!response.ok && !payload.error) {
          payload.error = "Run failed. Check the log for details.";
        }
        setState(payload);
        setRequest(normalizeRequest(payload));
      } catch {
        setState((current) => ({
          ...current,
          error: "Could not reach the simulation engine.",
          summary: "Could not reach the simulation engine.",
          chartSvg: "",
          metrics: [],
        }));
      }
    });
  }

  function onSubmit(event: FormEvent<HTMLFormElement>) {
    event.preventDefault();
    if (!canRun || isPending) {
      return;
    }
    run();
  }

  const sourceTag = sourceOptions.find((option) => option.value === request.source)?.label ?? request.source;
  const provenance = state.sourceNote.replace(/^Data source:\s*/i, "").replace(/\.$/, "");

  return (
    <div className="shell">
      <header className="topbar">
        <div className="topbar-left">
          <span className="pill">{request.job}</span>
          <span className="pill">{sourceTag.toLowerCase()}</span>
          <button
            className="pill pill-button"
            onClick={() => setTheme((current) => (current === "light" ? "dark" : "light"))}
            type="button"
          >
            {theme === "light" ? "dark" : "light"}
          </button>
        </div>
        <a className="topbar-link" href="https://github.com/pdj555/monte-carlo" rel="noopener noreferrer">
          github
        </a>
      </header>

      <h1 className="display-title" aria-label="Monte Carlo">
        <span>Monte</span>
        <span>Carlo</span>
      </h1>

      <section className="frame frame-config" aria-label="Run configuration">
        <h2 className="frame-label">Run</h2>
        <form className="config-row" onSubmit={onSubmit}>
          <div className="option-row" role="group" aria-label="Job">
            {jobOptions.map((option) => (
              <button
                aria-pressed={request.job === option.value}
                className={request.job === option.value ? "is-active" : ""}
                key={option.value}
                onClick={() => updateRequest({ job: option.value })}
                type="button"
              >
                {option.label}
              </button>
            ))}
          </div>

          <input
            aria-label="Tickers"
            autoComplete="off"
            className="ticker-input"
            onChange={(event) => updateRequest({ tickers: event.target.value })}
            placeholder="AAPL MSFT"
            spellCheck={false}
            type="text"
            value={request.tickers}
          />

          <div className="option-row" role="group" aria-label="Data source">
            {sourceOptions.map((option) => (
              <button
                aria-pressed={request.source === option.value}
                className={request.source === option.value ? "is-active" : ""}
                key={option.value}
                onClick={() => selectSource(option.value)}
                type="button"
              >
                {option.label}
              </button>
            ))}
          </div>

          {request.source === "local" ? (
            <input
              aria-label="CSV path"
              autoComplete="off"
              className="path-input"
              onChange={(event) => updateRequest({ dataPath: event.target.value })}
              placeholder="sample_data"
              spellCheck={false}
              type="text"
              value={request.dataPath ?? ""}
            />
          ) : null}

          <button className="run-button" disabled={isPending || !canRun} type="submit">
            {isPending ? "Running…" : "Run"}
          </button>
        </form>
      </section>

      <section className="status-line" aria-live="polite">
        <span>{state.eyebrow}</span>
        <span>{state.title}</span>
        <span>{state.request.tickers.replaceAll(" ", " · ")}</span>
        {provenance ? <span>{provenance}</span> : null}
      </section>

      <RunResults isPending={isPending} state={state} />

      <footer className="site-footer">python · provenance · walk-forward</footer>
    </div>
  );
}
