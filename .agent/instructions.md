# Agent Instructions for Monte Carlo Simulation Codebase

## Project Overview

This is a Monte Carlo stock price simulation tool that provides reusable building blocks for running financial simulations. The codebase retrieves historical data from Yahoo! Finance (via `yfinance`) and feeds it into vectorized simulation routines with plotting helpers.

## Codebase Structure

### Core Modules

- **`simulation.py`** - Monte Carlo simulation helpers (`simulate_prices`, `simulate_gbm`)
- **`data.py`** - Price data fetching and management (`fetch_prices`)
- **`analysis.py`** - Statistical analysis and summarization (`summarize_final_prices`)
- **`viz.py`** - Visualization utilities (`plot_distribution`, `plot_paths`)
- **`cli.py`** - Command-line interface for running simulations
- **`MonteCarlo.py`** - Legacy single-ticker workflow script

### Supporting Files

- **`tests/`** - Automated test suite (pytest)
- **`docs/constitution.md`** - Codebase principles and standards (MUST READ)
- **`sample_data/`** - Sample CSV data for offline testing
- **`requirements.txt`** - Python dependencies

## Constitution Adherence

**CRITICAL**: You MUST follow the principles in `docs/constitution.md`:

1. **Demand Excellence** - Well-structured, maintainable code
2. **Embrace Simplicity** - Avoid over-engineering
3. **Future Forward** - Use modern, reliable frameworks
4. **Consistency** - Follow PEP 8, 4-space indentation
5. **Transparency** - Clear documentation and comments
6. **Quality Assurance** - Test thoroughly before changes
7. **Open Collaboration** - Focused, relevant contributions

## Setup and Environment

### Initial Setup

```bash
uv pip install -r requirements.txt
export MPLBACKEND=Agg  # For headless matplotlib operations
```

### Key Dependencies

- `yfinance` - Historical price data
- `pandas`, `numpy` - Data manipulation and vectorized operations
- `matplotlib`, `seaborn` - Visualizations
- `pytest` - Testing framework

## Execution Guidelines

### Running Simulations

**CLI Usage:**

```bash
python cli.py --tickers AAPL,MSFT --days 252 --scenarios 5000 --model gbm --output ./results
```

**Legacy Script:**

```bash
python MonteCarlo.py --ticker AAPL --days 252 --scenarios 1000
```

### Testing

Always run tests before making changes:

```bash
pytest
```

### Offline Mode

For environments without network access:

```bash
python cli.py --offline-only --offline-path sample_data/
```

## Long-Running Tasks (24-Hour Operation)

During extended operation, you should:

1. **Iterate on Improvements**

   - Identify and fix bugs
   - Optimize simulation performance
   - Enhance code quality and documentation
   - Refactor for better maintainability

2. **Run Comprehensive Simulations**

   - Test with various tickers and parameters
   - Validate results across different models (historical, GBM)
   - Generate and analyze output artifacts

3. **Code Quality Maintenance**

   - Ensure all code follows PEP 8
   - Maintain test coverage
   - Update documentation as needed
   - Keep dependencies minimal

4. **Output Management**
   - Save results to appropriate directories (e.g., `./results/`)
   - Avoid cluttering the repository root
   - Clean up temporary files
   - Organize generated artifacts logically

## Code Style Requirements

- **Indentation**: 4 spaces (no tabs)
- **Line length**: Follow PEP 8 (typically 79-99 characters)
- **Naming**: Use descriptive names, follow Python conventions
- **Type hints**: Use type annotations where appropriate
- **Docstrings**: Include docstrings for all functions and classes
- **Comments**: Explain non-obvious logic

## Testing Requirements

- Run `pytest` before committing any changes
- Ensure all existing tests pass
- Add tests for new functionality
- Test edge cases and error conditions
- Validate data inputs and outputs

## Dependency Management

- Use `uv` for Python package management (per project rules)
- Keep dependencies minimal and purposeful
- Avoid introducing proprietary or closed-source dependencies
- Document any new dependencies in `requirements.txt`

## Network and I/O

- Network access is enabled for `yfinance` data fetching
- Use `MPLBACKEND=Agg` for headless matplotlib operations
- Handle offline mode gracefully with `--offline-only` flag
- Support both CSV file and directory inputs for offline data

## Autonomous Operation Guidelines

1. **Work Systematically**

   - Start with understanding existing code
   - Make incremental, focused improvements
   - Test changes immediately
   - Document rationale for significant changes

2. **Respect Existing Structure**

   - Don't reorganize without clear benefit
   - Maintain backward compatibility
   - Follow established patterns
   - Keep modules focused and cohesive

3. **Quality Over Speed**

   - Thoroughly test all changes
   - Review code before finalizing
   - Ensure consistency with codebase style
   - Validate against constitution principles

4. **Error Handling**
   - Handle `PriceDataError` appropriately
   - Validate inputs (days > 0, scenarios > 0, etc.)
   - Provide clear error messages
   - Log issues for debugging

## Output and Artifacts

- Save plots to specified output directories
- Use descriptive filenames (e.g., `{ticker}_distribution.png`)
- Generate summary statistics and reports
- Keep output organized and accessible

## Time Management

You have 24 hours (86400 seconds) to operate autonomously. Use this time to:

- Make meaningful improvements
- Run comprehensive tests
- Enhance documentation
- Optimize performance
- Fix issues and bugs

Work efficiently but thoroughly. Quality and adherence to the constitution are paramount.
