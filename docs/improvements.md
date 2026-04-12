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

### 1. Document output interpretation

The public README now shows how to run the two jobs. A short follow-on doc that
explains how to read the decision summary, ranking table, and backtest metrics
would help new users move from "the command ran" to "I know what to do with the
result."

### 2. Keep sample-data smoke tests broad

The bundled offline path now covers `AAPL.csv` and `MSFT.csv`. The next useful
follow-on would be keeping docs, install smoke tests, and browser checks aligned
with those fixtures whenever the sample data changes so the deterministic path
stays trustworthy.

### 3. Keep source provenance visible in saved artefacts

The CLI and browser UI now tell the operator which source actually supplied the
prices. A useful follow-on would be persisting that provenance into backtest
artefacts and any future explanatory docs so offline, cached, and live runs
stay auditable after the terminal closes.
