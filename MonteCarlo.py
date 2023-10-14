# Import Packages
import yfinance as yf
import seaborn as sns
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
plt.style.use('ggplot')

# Stock to predict
asset = 'AAPL'

# Function to generate Monte Carlo simulations of stock prices
def Monte_Carlo(period, n_scenarios, asset):
    """
    Generate Monte Carlo simulations of stock prices.
    
    Args:
        period (int): Number of trading days.
        n_scenarios (int): Number of simulation scenarios.
        asset (str): Ticker symbol of the stock.

    Returns:
        DataFrame: Simulated stock prices.
    """
    # Use data from Yahoo! Finance
    df = yf.download(asset)['Close']
    df_pct = df.pct_change()

    mean = df_pct.mean()
    vol = df_pct.std()
    last_price = df.iloc[-1]
    
    # Inputs
    dt = 1
    n_steps = int(period)
    xi = np.random.normal(size=(n_steps, n_scenarios))

    rets = mean * dt + vol * np.sqrt(dt) * xi
    rets = pd.DataFrame(rets)

    prices = last_price * (1 + rets).cumprod()
    return prices

# Define period and number of scenarios
periods = 365
n_scenarios = 10000

# Simulate stock prices
sim = Monte_Carlo(periods, n_scenarios, asset)

# Plot the distribution of the estimated price
plt.figure(figsize=(12,8))
ax1 = sns.histplot(data=sim.iloc[-1, :], bins=30, kde=True, color='skyblue', stat='density')
ax1.set(xlabel='Price', ylabel='Density', title=f'Distribution of {asset} Estimated Price in {periods} Days')
plt.show()

# Plot the paths of the simulations
plt.figure(figsize=(12,8))
ax2 = sim.plot(legend=False, title=f'Simulated Paths of {asset}')
ax2.set(xlabel='Days', ylabel='Price')
plt.show()
