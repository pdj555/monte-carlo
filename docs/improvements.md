# Improvement Backlog

This document tracks follow-on work for the current CLI and package layout.
It no longer describes the retired pre-subcommand scripts as the primary
interface.

## Live Interface

- Install locally with `python3 -m pip install -e .`
- Run simulations with `monte-carlo simulate [TICKER ...]`
- Run walk-forward validation with `monte-carlo backtest [TICKER ...]`
- Keep `python cli.py`, `python backtest.py`, and `python MonteCarlo.py` only
  as deprecated compatibility wrappers during migration

## Current Opportunities

### 1. Keep sample-data smoke tests broad

The bundled offline path now covers `AAPL.csv` and `MSFT.csv`. The next useful
follow-on would be keeping docs, install smoke tests, and browser checks aligned
with those fixtures whenever the sample data changes so the deterministic path
stays trustworthy.

### 2. Keep the browser UI modular

The current browser UI is still easy to use, but `app.py` now owns CSS, HTML,
request parsing, and page-state builders in one file. The next UI feature
should be the trigger to split rendering and state helpers so the happy path
stays easy to change.

### 3. Extend provenance checks when new artefacts appear

Simulation reports and backtest folders now persist source provenance. Future
export formats should keep the same audit trail and land with regression
coverage so offline, cached, and live runs stay auditable after the terminal
closes.
