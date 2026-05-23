export type RunMode = "simulate" | "backtest";
export type DataSource = "auto" | "online" | "demo" | "local";

export type WorkbenchRequest = {
  job: RunMode;
  tickers: string;
  source: DataSource;
  dataPath?: string | null;
};

export type Metric = {
  label: string;
  value: string;
};

export type WorkbenchPayload = {
  request: {
    job: RunMode;
    tickers: string;
    source: DataSource;
    data_path?: string | null;
  };
  sourceNote: string;
  eyebrow: string;
  title: string;
  summary: string;
  notes: string[];
  metrics: Metric[];
  chartSvg: string;
  chartAlt: string;
  detailsText: string;
  error: string | null;
};

export const DEFAULT_REQUEST: WorkbenchRequest = {
  job: "simulate",
  tickers: "AAPL",
  source: "auto",
  dataPath: null,
};
