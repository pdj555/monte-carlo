# Improvement Backlog

This backlog tracks follow-on work for the public CLI, Python engine, and
Next.js workbench. It no longer treats deprecated scripts as the primary
interface.

## Live Interface

- Install the Python engine with `python3 -m pip install -e .`
- Install the browser surface with `npm install`
- Run simulations with `monte-carlo simulate [TICKER ...]`
- Run walk-forward validation with `monte-carlo backtest [TICKER ...]`
- Run reproducible decision-stability checks with `monte-carlo evaluate SET_FILE`
- Run the browser workbench with `npm run dev`
- Keep `python cli.py`, `python backtest.py`, and `python MonteCarlo.py` only
  as deprecated compatibility wrappers during migration

## Shipped Foundation

Scenario-set evaluations now run a bounded, versioned matrix across universes,
models, seeds, and sources. The public command produces an auditable scorecard.

## Current Opportunities

### 1. Make the bridge contract typed end to end

The Next.js UI currently calls a compact JSON bridge over `ui_bridge.py`. The
next refinement is to generate a shared schema for `WorkbenchPayload`, validate
the Python response at the route boundary, and keep the UI resilient when a
field is missing.

### 2. Extend provenance checks when new artifacts appear

Simulation reports and backtest folders persist source provenance. Future
export formats should keep the same audit trail and land with regression
coverage so offline, cached, and live runs stay auditable after the terminal
closes.
