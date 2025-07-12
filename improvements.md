# GitHub Issue Queue

The following issues should be created in GitHub. Each one is ready for assignment and contains clear tasks and acceptance tests.

---

## Issue 1: Create `data.py` for robust price fetching
**Description**
Build a dedicated data module that wraps `yfinance` and provides clean error handling.

**Tasks**
- Implement `fetch_prices(ticker: str, start: str | None = None, end: str | None = None) -> pd.Series` to download daily close prices.
- Validate the ticker string and retry up to three times on transient network errors.
- Raise a custom `PriceDataError` when the ticker is invalid or the API fails.
- Document the function with a short example in the module docstring.

**Acceptance Criteria**
- Calling `fetch_prices("AAPL")` returns a non-empty pandas Series.
- An invalid ticker results in a `PriceDataError` with a helpful message.
- Network errors are retried and do not cause uncaught exceptions.

---

## Issue 2: Add `simulation.py` with vectorized simulations
**Description**
Move all Monte Carlo logic into a new module so it can be tested in isolation.

**Tasks**
- Create `simulate_prices(returns: pd.Series, days: int, scenarios: int, dt: float = 1.0) -> pd.DataFrame` using NumPy vectorization.
- Accept a random seed parameter to make runs reproducible.
- Document the algorithm (drift, volatility, and cumulative returns) in comments.
- Ensure the function can operate on Series indexed by date or integer.

**Acceptance Criteria**
- Output DataFrame has shape `(days, scenarios)` when called with valid inputs.
- Using the same seed produces identical results across runs.
- Setting `dt=1.0` yields the same behaviour as the current script.

---

## Issue 3: Add `viz.py` for plotting helpers
**Description**
Provide reusable plotting functions so charts remain consistent across notebooks and the CLI.

**Tasks**
- Implement `plot_distribution(df: pd.DataFrame, ticker: str) -> matplotlib.figure.Figure` for histogram plots.
- Implement `plot_paths(df: pd.DataFrame, ticker: str) -> matplotlib.figure.Figure` for simulated price paths.
- Include optional arguments for title and color palette.
- Write docstrings demonstrating basic usage.

**Acceptance Criteria**
- Both functions return `Figure` objects that can be saved or displayed.
- Functions gracefully handle DataFrames containing multiple tickers.
- Example code in the docstrings runs without modification.

---

## Issue 4: Refactor `MonteCarlo.py` to use new modules
**Description**
Update the main script so all core logic resides in the new modules and the script only orchestrates execution.

**Tasks**
- Remove hard-coded constants and read parameters using the CLI interface.
- Replace existing code with calls to `data.py`, `simulation.py`, and `viz.py`.
- Keep backward compatibility by supporting defaults identical to the current behaviour.
- Update inline comments to explain each step at a high level.

**Acceptance Criteria**
- Running `python MonteCarlo.py --ticker AAPL --days 252 --scenarios 1000` produces the same output as before.
- All imports resolve correctly and there are no unused functions left in the script.

---

## Issue 5: Build a CLI in `cli.py`
**Description**
Expose a command-line interface that allows running simulations without editing code.

**Tasks**
- Use `argparse` to accept `--ticker`, `--days`, `--scenarios`, and `--dt` arguments.
- Allow comma-separated tickers so multiple simulations can run in one command.
- Add `--output` to specify a directory where plots are saved.
- Display help text describing each option and provide examples in the module docstring.

**Acceptance Criteria**
- Executing `python cli.py --ticker AAPL --days 252 --scenarios 1000` runs successfully and shows plots.
- When `--output ./results` is provided, figures are saved to that folder.
- Invalid arguments result in a clear error message from argparse.

---

## Issue 6: Introduce pytest-based testing
**Description**
Add automated tests so future changes do not break existing functionality.

**Tasks**
- Create a `tests/` directory with unit tests for `data.py`, `simulation.py`, and `viz.py`.
- Mock network calls for `fetch_prices` to avoid hitting the real API during tests.
- Achieve at least 80% code coverage using `pytest --cov`.
- Add a `pytest.ini` with basic configuration such as markers and test paths.

**Acceptance Criteria**
- Running `pytest` locally passes and reports coverage of 80% or higher.
- Continuous integration fails when tests do not pass.

---

## Issue 7: Add Flake8 linting and GitHub Actions
**Description**
Automate code quality checks so style violations are caught early.

**Tasks**
- Add a `.flake8` configuration file defining line length and other rules.
- Configure a GitHub Actions workflow to run `flake8` and `pytest` on every push and pull request.
- Include a status badge in the README displaying workflow results.

**Acceptance Criteria**
- Pull requests show both lint and test results in the GitHub interface.
- Any Flake8 errors cause the CI workflow to fail.

---

## Issue 8: Pin dependencies
**Description**
Lock package versions so the environment is reproducible across machines.

**Tasks**
- Create `requirements.txt` listing numpy, pandas, matplotlib, seaborn, yfinance, pytest, and flake8 with explicit versions.
- Document how to update dependencies when needed and regenerate the file.

**Acceptance Criteria**
- `pip install -r requirements.txt` installs all packages without warnings.
- Versions remain consistent when the file is used on another machine.

---

## Issue 9: Expand the README
**Description**
Update the project documentation to reflect the new modules and CLI workflow.

**Tasks**
- Add a section describing the command-line interface with example commands.
- Provide instructions for running the tutorial notebook and interpreting plots.
- Clarify prerequisite Python version and typical runtime for 1000 scenarios.

**Acceptance Criteria**
- README gives clear setup steps and sample commands that match the CLI.
- Users can understand how to read the generated distribution and path charts.

---

## Issue 10: Provide a tutorial notebook
**Description**
Give newcomers a walk-through of the full simulation process in Jupyter.

**Tasks**
- Create `docs/tutorial.ipynb` that loads data, runs simulations via `simulation.py`, and generates plots via `viz.py`.
- Include explanatory Markdown cells covering parameter choices and analysis.
- Ensure the notebook works with the CLI-installed package versions.

**Acceptance Criteria**
- Executing all cells in order completes without errors and saves example plots.

---

## Issue 11: Add geometric Brownian motion model
**Description**
Support an alternative simulation method selectable from the CLI.

**Tasks**
- Implement `simulate_gbm` in `simulation.py` following the GBM formula.
- Add a CLI flag `--model gbm` that chooses between the default model and GBM.
- Provide unit tests validating that GBM output has the expected shape.

**Acceptance Criteria**
- Users can run the script with `--model gbm` and receive simulated prices generated by the new function.

---

## Issue 12: Explore performance improvements with Numba
**Description**
Investigate optional speedups for large-scale simulations.

**Tasks**
- Experiment with applying Numba JIT compilation to functions in `simulation.py`.
- Document any measurable speedups and the exact configuration used.
- Ensure the code still runs when Numba is not installed.

**Acceptance Criteria**
- Benchmarks demonstrating performance gains appear in the README or docs.
- Numba is an optional dependency and simulations work without it.

---
