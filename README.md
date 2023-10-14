# Monte Carlo Stock Price Simulation

This Python script uses Monte Carlo simulations to estimate the future price of a given stock (in this example, AAPL). It utilizes historical stock price data obtained from Yahoo! Finance.

## Table of Contents
- [Prerequisites](#prerequisites)
- [Installation](#installation)
- [Usage](#usage)
- [Function Details](#function-details)
- [Example](#example)
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
pip install yfinance seaborn numpy pandas matplotlib
```

## Installation

1. Clone or download the repository to your local machine.

2. Make sure you have the required packages installed (see [Prerequisites](#prerequisites)).

## Usage

To run the simulation, execute the Python script `MonteCarlo.py`. The script will download historical stock price data for the specified asset from Yahoo! Finance and perform the Monte Carlo simulation.

## Function Details

The core function `Monte_Carlo` takes several parameters:

- `period` (int): Number of trading days for the simulation.
- `n_scenarios` (int): Number of simulation scenarios.
- `mu` (float): Mean daily return.
- `sigma` (float): Daily volatility.
- `last_price` (float): Last known price.

It returns a DataFrame containing simulated stock prices.

## Example

```python
# Simulate stock prices
sim = Monte_Carlo_example(period=252, n_scenarios=10000, mu=mean, sigma=vol, last_price=last_price)

# Plot the distribution of the estimated price
plt.figure(figsize=(12,8))
ax1 = sns.histplot(data=sim.iloc[-1, :], bins=30, kde=True, color='skyblue', stat='density')
ax1.set(xlabel='Price', ylabel='Density', title=f'Distribution of {asset} Estimated Price in 252 Days')
plt.show()

# Plot the paths of the simulations
plt.figure(figsize=(12,8))
ax2 = sim.plot(legend=False, title=f'Simulated Paths of {asset}')
ax2.set(xlabel='Days', ylabel='Price')
plt.show()
```

This will generate a distribution plot of the estimated price and a plot of the simulated paths for the specified asset.

## License

This project is licensed under the [MIT License](LICENSE).

