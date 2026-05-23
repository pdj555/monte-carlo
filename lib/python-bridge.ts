import { spawn } from "node:child_process";
import { existsSync } from "node:fs";
import path from "node:path";

import type { WorkbenchPayload, WorkbenchRequest } from "./types";
import { DEFAULT_REQUEST } from "./types";

const RUN_TIMEOUT_MS = 60_000;
const VERCEL_ENGINE_PATH = "/api/engine";

export type RunWorkbenchOptions = {
  origin?: string;
};

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

function buildPythonPath(projectRoot: string): string {
  const segments = [projectRoot];
  const vendorRoot = path.join(projectRoot, "python_packages");
  if (existsSync(vendorRoot)) {
    segments.push(vendorRoot);
  }
  if (process.env.PYTHONPATH) {
    segments.push(process.env.PYTHONPATH);
  }
  return segments.join(path.delimiter);
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

function engineFailure(
  request: WorkbenchRequest,
  message: string,
  detailsText: string,
): WorkbenchPayload {
  return {
    ...fallbackPayload,
    request: {
      job: request.job,
      tickers: request.tickers,
      source: request.source,
      data_path: request.dataPath,
    },
    summary: message,
    detailsText,
    error: message,
  };
}

function resolveEngineUrl(origin?: string): string {
  if (origin) {
    return new URL(VERCEL_ENGINE_PATH, origin).toString();
  }
  const host = process.env.VERCEL_URL;
  if (host) {
    return `https://${host}${VERCEL_ENGINE_PATH}`;
  }
  return `http://127.0.0.1:3000${VERCEL_ENGINE_PATH}`;
}

async function runViaVercelEngine(
  request: WorkbenchRequest,
  options?: RunWorkbenchOptions,
): Promise<WorkbenchPayload> {
  const url = resolveEngineUrl(options?.origin);
  const headers: Record<string, string> = { "Content-Type": "application/json" };
  const bypass = process.env.VERCEL_AUTOMATION_BYPASS_SECRET;
  if (bypass) {
    headers["x-vercel-protection-bypass"] = bypass;
  }

  let response: Response;
  try {
    response = await fetch(url, {
      method: "POST",
      headers,
      body: toPythonPayload(request),
    });
  } catch (error) {
    const message = error instanceof Error ? error.message : "Engine request failed.";
    return engineFailure(request, message, `url: ${url}`);
  }

  const text = await response.text();
  if (!response.ok) {
    return engineFailure(
      request,
      `Engine HTTP ${response.status}.`,
      text || `url: ${url}`,
    );
  }

  try {
    return JSON.parse(text) as WorkbenchPayload;
  } catch {
    return engineFailure(request, "Python returned invalid JSON.", text);
  }
}

async function runViaLocalSpawn(request: WorkbenchRequest): Promise<WorkbenchPayload> {
  const projectRoot = resolveProjectRoot();
  const bridgeScript = path.join(projectRoot, "ui_bridge.py");
  const python = resolvePython(projectRoot);

  if (!existsSync(bridgeScript)) {
    return engineFailure(
      request,
      "Python bridge script was not deployed.",
      `Missing ${bridgeScript}. Ensure engine files are included in the serverless bundle.`,
    );
  }

  const child = spawn(python, [bridgeScript], {
    cwd: projectRoot,
    env: {
      ...process.env,
      MPLBACKEND: "Agg",
      PYTHONPATH: buildPythonPath(projectRoot),
    },
    stdio: ["pipe", "pipe", "pipe"],
  });

  const timeout = setTimeout(() => child.kill("SIGTERM"), RUN_TIMEOUT_MS);
  const stdout: Buffer[] = [];
  const stderr: Buffer[] = [];
  let spawnErrorMessage: string | null = null;

  child.stdout.on("data", (chunk: Buffer) => stdout.push(chunk));
  child.stderr.on("data", (chunk: Buffer) => stderr.push(chunk));
  child.stdin.end(toPythonPayload(request));

  const exitCode = await new Promise<number | null>((resolve) => {
    child.on("close", resolve);
    child.on("error", (error) => {
      spawnErrorMessage = error.message;
      resolve(1);
    });
  });
  clearTimeout(timeout);

  const stderrText = Buffer.concat(stderr).toString("utf8").trim();
  const stdoutText = Buffer.concat(stdout).toString("utf8").trim();

  if (spawnErrorMessage !== null) {
    return engineFailure(
      request,
      spawnErrorMessage,
      [stderrText, `python: ${python}`, `bridge: ${bridgeScript}`].filter(Boolean).join("\n"),
    );
  }

  if (exitCode !== 0) {
    const message = stderrText || `Python exited with code ${exitCode ?? "unknown"}.`;
    return engineFailure(request, message, stderrText || stdoutText || fallbackPayload.detailsText);
  }

  try {
    return JSON.parse(stdoutText) as WorkbenchPayload;
  } catch {
    return engineFailure(
      request,
      "Python returned invalid JSON.",
      stdoutText || stderrText || fallbackPayload.detailsText,
    );
  }
}

export async function runWorkbench(
  input?: Partial<WorkbenchRequest>,
  options?: RunWorkbenchOptions,
): Promise<WorkbenchPayload> {
  const request = normalizeRequest(input);
  if (process.env.VERCEL === "1") {
    return runViaVercelEngine(request, options);
  }
  return runViaLocalSpawn(request);
}
