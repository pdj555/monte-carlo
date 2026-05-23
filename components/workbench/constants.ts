import type { DataSource, RunMode } from "../../lib/types";

export const jobOptions: Array<{ value: RunMode; label: string }> = [
  { value: "simulate", label: "Simulate" },
  { value: "backtest", label: "Backtest" },
];

export const sourceOptions: Array<{ value: DataSource; label: string }> = [
  { value: "auto", label: "Live" },
  { value: "online", label: "Strict" },
  { value: "demo", label: "Sample" },
  { value: "local", label: "CSV" },
];
