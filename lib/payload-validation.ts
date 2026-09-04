import type { DataSource, RunMode, WorkbenchPayload } from "./types";

const runModes: ReadonlySet<string> = new Set(["simulate", "backtest"]);
const dataSources: ReadonlySet<string> = new Set(["auto", "online", "demo", "local"]);

function isRecord(value: unknown): value is Record<string, unknown> {
  return typeof value === "object" && value !== null && !Array.isArray(value);
}

function isStringArray(value: unknown): value is string[] {
  return Array.isArray(value) && value.every((item) => typeof item === "string");
}

function isRunMode(value: unknown): value is RunMode {
  return typeof value === "string" && runModes.has(value);
}

function isDataSource(value: unknown): value is DataSource {
  return typeof value === "string" && dataSources.has(value);
}

/** Validate untrusted output from the Python engine before exposing it to the UI. */
export function isWorkbenchPayload(value: unknown): value is WorkbenchPayload {
  if (!isRecord(value) || !isRecord(value.request)) {
    return false;
  }

  const request = value.request;
  const validRequest =
    isRunMode(request.job) &&
    typeof request.tickers === "string" &&
    isDataSource(request.source) &&
    (request.data_path === undefined ||
      request.data_path === null ||
      typeof request.data_path === "string");
  const validMetrics =
    Array.isArray(value.metrics) &&
    value.metrics.every(
      (metric) =>
        isRecord(metric) &&
        typeof metric.label === "string" &&
        typeof metric.value === "string",
    );

  return (
    validRequest &&
    typeof value.sourceNote === "string" &&
    typeof value.eyebrow === "string" &&
    typeof value.title === "string" &&
    typeof value.summary === "string" &&
    isStringArray(value.notes) &&
    validMetrics &&
    typeof value.chartSvg === "string" &&
    typeof value.chartAlt === "string" &&
    typeof value.detailsText === "string" &&
    (value.error === null || typeof value.error === "string")
  );
}
