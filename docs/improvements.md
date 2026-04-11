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

### 1. Keep shrinking legacy wrapper surface area

The installed entrypoint now runs through `public_cli.py`, with shared workflow
living in `simulate_cli.py` and shared helpers in `cli_shared.py`. The next
useful step would be trimming dead wrapper-era helpers from `cli.py` so the
deprecated path stays obviously small.

### 2. Expand offline fixture coverage

Most tests use `sample_data/AAPL.csv`, which keeps the suite fast and
deterministic. Adding a second fixture with a different return profile would
broaden regression coverage for ranking, guardrail, and backtest paths without
introducing live network dependency.

### 3. Document output interpretation

The public README now shows how to run the two jobs. A short follow-on doc that
explains how to read the decision summary, ranking table, and backtest metrics
would help new users move from "the command ran" to "I know what to do with the
result."
