import { spawn } from "node:child_process";
import { existsSync } from "node:fs";
import path from "node:path";

import type { WorkbenchPayload, WorkbenchRequest } from "./types";
import { DEFAULT_REQUEST } from "./types";

const RUN_TIMEOUT_MS = 60_000;

const fallbackPayload: WorkbenchPayload = {
  request: {
    job: "simulate",
    tickers: "AAPL",
    source: "auto",
    data_path: null,
  },
  sourceNote: "Python engine did not return a run.",
  eyebrow: "Unavailable",
  title: "Engine offline",
  summary: "Install the Python package, then reload.",
  notes: [],
  metrics: [],
  chartSvg: "",
  chartAlt: "",
  detailsText: "Run `python3 -m pip install -e .` before starting Next.js.",
  error: "Python engine did not return a run.",
};

function resolveProjectRoot(): string {
  return process.env.MONTE_CARLO_ROOT ?? process.cwd();
}

function resolvePython(projectRoot: string): string {
  if (process.env.PYTHON) {
    return process.env.PYTHON;
  }
  const venvPython = path.join(projectRoot, ".venv/bin/python3");
  if (existsSync(venvPython)) {
    return venvPython;
  }
  return "python3";
}

function normalizeRequest(input?: Partial<WorkbenchRequest>): WorkbenchRequest {
  const job = input?.job === "backtest" ? "backtest" : "simulate";
  const source =
    input?.source === "auto" ||
    input?.source === "online" ||
    input?.source === "demo" ||
    input?.source === "local"
      ? input.source
      : DEFAULT_REQUEST.source;

  return {
    job,
    source,
    tickers: input?.tickers?.trim() || DEFAULT_REQUEST.tickers,
    dataPath: input?.dataPath?.trim() || null,
  };
}

function toPythonPayload(request: WorkbenchRequest): string {
  return JSON.stringify({
    job: request.job,
    tickers: request.tickers,
    source: request.source,
    data_path: request.dataPath,
  });
}

export async function runWorkbench(
  input?: Partial<WorkbenchRequest>,
): Promise<WorkbenchPayload> {
  const request = normalizeRequest(input);
  const projectRoot = resolveProjectRoot();
  const bridgeScript = path.join(projectRoot, "ui_bridge.py");
  const python = resolvePython(projectRoot);
  const child = spawn(python, [bridgeScript], {
    cwd: projectRoot,
    env: {
      ...process.env,
      MPLBACKEND: "Agg",
      PYTHONPATH: projectRoot,
    },
    stdio: ["pipe", "pipe", "pipe"],
  });

  const timeout = setTimeout(() => child.kill("SIGTERM"), RUN_TIMEOUT_MS);
  const stdout: Buffer[] = [];
  const stderr: Buffer[] = [];

  child.stdout.on("data", (chunk: Buffer) => stdout.push(chunk));
  child.stderr.on("data", (chunk: Buffer) => stderr.push(chunk));
  child.stdin.end(toPythonPayload(request));

  const exitCode = await new Promise<number | null>((resolve) => {
    child.on("close", resolve);
    child.on("error", () => resolve(1));
  });
  clearTimeout(timeout);

  if (exitCode !== 0) {
    const stderrText = Buffer.concat(stderr).toString("utf8");
    return {
      ...fallbackPayload,
      request: {
        job: request.job,
        tickers: request.tickers,
        source: request.source,
        data_path: request.dataPath,
      },
      detailsText: stderrText || fallbackPayload.detailsText,
      error: stderrText.trim() || fallbackPayload.error,
    };
  }

  try {
    return JSON.parse(Buffer.concat(stdout).toString("utf8")) as WorkbenchPayload;
  } catch {
    return {
      ...fallbackPayload,
      detailsText: Buffer.concat(stdout).toString("utf8") || fallbackPayload.detailsText,
    };
  }
}
