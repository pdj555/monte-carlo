/** Paths bundled into Vercel/Next serverless output for the Python engine. */
export const engineTraceIncludes = [
  "./analysis.py",
  "./backtest.py",
  "./cli_shared.py",
  "./cli.py",
  "./data.py",
  "./decision.py",
  "./legacy_cli.py",
  "./public_cli.py",
  "./simulate_cli.py",
  "./simulation.py",
  "./ui_bridge.py",
  "./ui_state.py",
  "./viz.py",
  "./sample_data/**/*",
  "./python_packages/**/*",
  "./.venv/**/*",
] as const;

export const engineTraceRoutes = ["/", "/api/run"] as const;
