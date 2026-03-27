# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Monte Carlo decision engine for equities. Two surfaces — vectorized simulation (historical bootstrap and GBM) and walk-forward backtesting — feed a ranking/allocation layer that produces an action plan (stance, top idea, focus list, avoid list, cash buffer). A lean Flask UI wraps the same code paths for the browser and Vercel deployments. Python 3.9+.

## Development Commands

### Running Simulations

**Public CLI:**
```bash
monte-carlo simulate AAPL MSFT --days 252 --scenarios 5000 --model gbm --output ./results --seed 42
```

```bash
monte-carlo backtest AAPL MSFT --lookback 60 --hold 20 --rebalance 20 --model gbm --scenarios 1000 --seed 42
```

Key CLI options:
- `--model historical` (default) or `--model gbm` for geometric Brownian motion
- `--source offline` to use local CSV data only
- `--source auto` to try live data first, then local CSV files
- `--data-path DIR` to specify a custom CSV directory or file
- `--show` to display plots on screen
- `--seed N` for reproducible simulations
- `--details` for full tables and secondary metrics

### Browser UI

```bash
python3 -m pip install -e .[ui]   # adds Flask
monte-carlo-ui                    # serves http://127.0.0.1:8000
```

The UI starts on the bundled AAPL demo, so first render is deterministic and offline-safe.

### Testing

```bash
pytest                            # full suite (pytest.ini sets testpaths=tests, addopts=-ra)
pytest tests/test_simulation.py   # single file
pytest tests/test_cli.py::test_x  # single test
pytest -v
```

Tests must be deterministic: pass explicit `seed` values and prefer the offline CSV path over live network calls. `tests/conftest.py` inserts the repo root on `sys.path`, so top-level modules import without a package prefix.

### Environment Setup

```bash
python3 -m pip install -e .       # CLI only
python3 -m pip install -e .[ui]   # adds the browser UI
export MPLBACKEND=Agg             # headless plotting (no GUI pop-ups)
```

## Architecture

### Module Structure

The codebase is organized into focused modules with clear separation of concerns:

**Core Simulation Engine (`simulation.py`):**
- `simulate_prices()` - Historical bootstrap model using empirical drift and volatility
- `simulate_gbm()` - Geometric Brownian motion with explicit mu/sigma parameters
- Both functions return DataFrames with shape `(days, scenarios)` and support reproducible seeding
- Internal helper `_as_float()` handles both scalar floats and pandas Series for current_price

**Data Fetching (`data.py`):**
- `fetch_prices()` retrieves closing prices via yfinance with automatic retry logic
- Falls back to local CSV files in `sample_data/` when network requests fail
- Use `prefer_local=True` to skip network entirely (controlled by `--source offline` in the public CLI)
- Raises `PriceDataError` on failures
- CSV files must have `Date` and `Close` columns

**Analytics (`analysis.py`):**
- `summarize_final_prices()` computes statistics on the final row of simulation results
- Returns mean, median, std, min, max, quantiles (default: 5%, 25%, 75%, 95%)
- When `current_price` is provided, adds expected_return, prob_above_current, value_at_risk_95
- Optional knobs surface expected_shortfall_95_pct, max_drawdown_q95, prob_hit_target, prob_breach_max_loss, kelly_fraction, and benchmark-relative metrics — these feed the decision layer's scoring and guardrails

**Decision Layer (`decision.py`):**
- `rank_tickers()` scores summaries into BUY / WATCH / AVOID using expected return, prob above current, VaR/CVaR, drawdown, and (when present) excess-return and Kelly signals
- `apply_risk_guards()` flips rows to AVOID and records `guardrail_reasons` when expected return, hit-probability, VaR, drawdown, or breach-probability thresholds fail
- `recommend_allocations()` converts rankings into weights via a risk-scaled, capped, bisection-balanced allocator
- `enforce_portfolio_risk_budget()` scales weights linearly to keep portfolio 95% VaR within a hard budget (path-aware when supplied, blend-based otherwise)
- `build_action_plan()` produces the stance / headline / primary pick / focus / avoid / cash-weight payload the CLI and UI render
- `build_execution_plan()` translates weights into shares, costs, and cash drift given current prices and capital

**Visualization (`viz.py`):**
- `plot_distribution()` creates histogram + KDE of final prices
- `plot_paths()` plots simulated price trajectories over time
- Both functions return `matplotlib.Figure` objects for flexible display/saving
- Support MultiIndex columns for multi-ticker DataFrames

**Programmatic SDK (`sdk.py`):**
- `MonteCarloSDK` - Single-import, typed API for agent and programmatic use
- `sdk.analyze()` - Full simulation + analysis for a single ticker, returns `TickerResult`
- `sdk.analyze_many()` - Concurrent multi-ticker analysis with `ThreadPoolExecutor`
- `sdk.portfolio()` - End-to-end portfolio construction: simulate, rank, allocate, plan
- `sdk.screen()` - Categorize tickers into BUY/WATCH/AVOID with structured `ScreenResult`
- `sdk.compare()` - Head-to-head ticker comparison with compact metrics
- All results are dataclasses with `.to_dict()` and `.to_json()` for serialisation

**MCP Server (`mcp_server.py`):**
- Model Context Protocol server for direct AI agent integration
- Exposes tools: `analyze_ticker`, `analyze_portfolio`, `screen_tickers`, `compare_tickers`
- JSON-RPC 2.0 over stdin/stdout, zero external dependencies
- Start with `python mcp_server.py` or configure in Claude Code MCP settings

**Agentic Workflows (`agent_workflow.py`):**
- `opportunity_scan()` - Scan a universe of tickers for the best opportunities
- `risk_check()` - Deep risk assessment with multi-dimensional risk scoring
- `what_if()` - Scenario analysis comparing historical vs GBM models
- `rebalance_signal()` - Determine whether a portfolio needs rebalancing
- Each workflow returns a structured, JSON-serialisable dataclass

**Command-Line Interfaces:**
- `monte-carlo` - Public entrypoint with `simulate` and `backtest` subcommands
- `public_cli.py` - Public CLI parser and command runners
  - `build_public_parser()` defines the public argparse structure
  - `run_public_simulate(args)` executes the public simulation surface
  - `main()` is the installed `monte-carlo` entrypoint
- `simulate_cli.py` - Shared simulation workflow used by both CLI surfaces
- `legacy_cli.py` - Full deprecated simulation parser and runner used by the old wrapper
- `cli_shared.py` - Shared version, validation, and detailed-rendering helpers
- `cli.py` - Thin deprecated compatibility facade for `python cli.py ...`
  - `legacy_main()` preserves the old wrapper flow
- `backtest.py` - Walk-forward engine plus deprecated wrapper entrypoint
- `MonteCarlo.py` - Deprecated single-ticker compatibility wrapper

**Browser UI (`app.py`):**
- Flask app exposing `Simulate` and `Backtest` over the same `simulate_cli.py` / `backtest.py` code paths
- `app.main()` is the installed `monte-carlo-ui` entrypoint
- `Demo sample`, `Try live data`, and `Local CSV` modes map onto the `--source` semantics from the CLI
- Module-level `app` object is what Vercel imports — keep it importable without side effects

**AI Summaries (`ai.py`):**
- `generate_ai_summary()` hits the OpenAI Responses API (default `gpt-5.2`) when the legacy CLI is run with `--ai-summary`
- Requires `OPENAI_API_KEY`; honors `OPENAI_BASE_URL`, `OPENAI_MODEL`, and `--ai-model`
- Raises `OpenAIConfigurationError` / `OpenAIRequestError` — callers degrade gracefully rather than failing the run

**Vercel Deployment (`vercel.json`, `api/index.py`, `public/`):**
- All routes rewrite to `api/index.py` except `/styles.css` and `/robots.txt`, which are served from `public/` with long s-maxage caching
- `vercel.json` `includeFiles` enumerates every top-level Python module that must ship in the function bundle — **if you add a new top-level `.py` that the UI/CLI imports, add it both here and to `pyproject.toml` `py-modules`**
- Function `maxDuration=60s`, `memory=1024MB`. Long simulations belong on small scenario counts in this surface
- See `docs/deploy.md` for serverless operational limits

### Data Flow

1. **Fetch** historical prices via `data.fetch_prices()` → returns pd.Series indexed by date
2. **Transform** to returns: `prices.pct_change().dropna()`
3. **Simulate** using either:
   - `simulate_prices(returns, days, scenarios, current_price)` - historical model
   - `simulate_gbm(current_price, mu, sigma, days, scenarios)` - GBM model
4. **Analyze** with `summarize_final_prices(sims, current_price)`
5. **Decide** by piping summaries through `rank_tickers` → `apply_risk_guards` → `recommend_allocations` → `build_action_plan` (and optionally `build_execution_plan`)
6. **Visualize** with `plot_distribution()` and `plot_paths()`

Both CLI surfaces and the Flask UI share steps 3–5; the only thing they differ on is presentation (`cli_shared.py` for terminal output, `app.py` for HTML).

### MultiIndex Convention

The CLI creates simulation DataFrames with MultiIndex columns:
- Level 0: ticker symbol (e.g., "AAPL", "MSFT")
- Level 1: scenario number (1 to scenarios)

Extract single ticker with: `df.xs("AAPL", axis=1, level=0)`

### Agent Integration

**MCP Server** (recommended for AI agents):
```json
{
    "mcpServers": {
        "monte-carlo": {
            "command": "python",
            "args": ["mcp_server.py"],
            "cwd": "/path/to/monte-carlo"
        }
    }
}
```

**Python SDK** (recommended for programmatic use):
```python
from sdk import MonteCarloSDK
sdk = MonteCarloSDK(offline_only=True)
result = sdk.portfolio(["AAPL", "MSFT", "GOOGL"], days=252, scenarios=1000, seed=42)
print(result.to_json(indent=2))
```

**Agentic Workflows** (recommended for complex agent pipelines):
```python
from agent_workflow import opportunity_scan, risk_check
report = opportunity_scan(["AAPL", "MSFT", "GOOGL", "AMZN", "NVDA"])
risk = risk_check("NVDA", scenarios=2000)
```

### Testing Strategy

Tests use pytest fixtures defined in `tests/conftest.py`:
- Tests cover simulation edge cases (empty data, invalid parameters)
- CLI tests verify argument parsing and output structure
- SDK, MCP server, and agent workflow tests use mocked price data
- Mock data used to avoid network dependencies during testing
- `test_repo_config.py` and `test_installation.py` guard packaging — they'll fail if a new top-level module is missing from `pyproject.toml` or `vercel.json` `includeFiles`

### Packaging Gotcha

The project is a flat layout, not a `src/` package. When you add a new top-level `.py` module that other modules import, update **all three** of:
1. `pyproject.toml` `[tool.setuptools] py-modules` — so `pip install -e .` ships it
2. `vercel.json` `functions."api/*.py".includeFiles` — so Vercel deployments include it
3. `pyproject.toml` `[project.scripts]` if it exposes a new console entrypoint

Forgetting any of these tends to surface as "works locally, breaks on install or deploy".
