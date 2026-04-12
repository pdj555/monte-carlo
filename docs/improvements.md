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

### 2. Surface actual source provenance

`--source auto` now behaves correctly, but the CLI and browser UI still describe
the requested source mode rather than the source that ultimately supplied the
prices. A small follow-on improvement would be surfacing whether a run used live
downloads or local CSV fallback so operators can tell what data they are acting
on.

### 3. Keep sample-data smoke tests broad

The bundled offline path now covers `AAPL.csv` and `MSFT.csv`. The next useful
follow-on would be keeping docs, install smoke tests, and browser checks aligned
with those fixtures whenever the sample data changes so the deterministic path
stays trustworthy.
