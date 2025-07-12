# Monte Carlo Stock Price Simulation

This Python script uses Monte Carlo simulations to estimate the future price of a given stock (in this example, AAPL). It utilizes historical stock price data obtained from Yahoo! Finance.

## Table of Contents
- [Prerequisites](#prerequisites)
- [Installation](#installation)
- [Usage](#usage)
- [License](#license)

## Prerequisites

Before running this code, you'll need to ensure you have the necessary Python packages installed. These include:
- yfinance
- seaborn
- numpy
- pandas
- matplotlib

You can install these packages using `pip`:

```bash
pip3 install yfinance seaborn numpy pandas matplotlib
```

If you encounter an error such as ``ModuleNotFoundError: No module named
\'matplotlib\'`` when running the script, make sure these packages are
installed in your current Python environment.

If you are running in an environment without a graphical display (for example
on a CI server), set ``MPLBACKEND=Agg`` when invoking the script so
``matplotlib`` does not attempt to open GUI windows.

When internet access is restricted, ``data.py`` will fall back to CSV files in
``sample_data/``. You can add your own ``<TICKER>.csv`` files with ``Date`` and
``Close`` columns to run simulations offline.

## Installation

1. Clone or download the repository to your local machine.

2. Make sure you have the required packages installed (see [Prerequisites](#prerequisites)).

## Usage

Run the simulation from the command line using `MonteCarlo.py`. By default the
script fetches historical prices for ``AAPL`` and simulates 10,000 possible
future paths over one year (365 trading days).

```bash
python MonteCarlo.py --ticker AAPL --days 252 --scenarios 1000
```

The optional arguments allow different tickers, forecast horizons, numbers of
scenarios and the ``dt`` step size. See ``--help`` for details.

The script produces a histogram of final prices and a plot of several simulated
paths using ``matplotlib`` and ``seaborn``.

## License

This project is licensed under the [MIT License](LICENSE).

