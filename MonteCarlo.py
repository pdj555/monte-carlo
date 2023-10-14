# Import Packages
import yfinance as yf
import seaborn as sns
import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
plt.style.use('ggplot')

# Stock to predict
asset = 'AAPL'

# Use data from Yahoo! Finance
df = yf.download(asset)['Close']
df_pct = df.pct_change()

mean = df_pct.mean()
vol = df_pct.std()
last_price = df.iloc[-1]
periods = 252


def Monte_Carlo(period=252, n_scenarios=1000, mu=mean, sigma=vol, last_price=last_price):
    """
    Generate Monte Carlo simulations of stock prices.
    
    Args:
        period (int): Number of trading days.
        n_scenarios (int): Number of simulation scenarios.
        mu (float): Mean daily return.
        sigma (float): Daily volatility.
        last_price (float): Last known price.

    Returns:
        DataFrame: Simulated stock prices.
    """
    # Inputs
    dt = 1
    n_steps = int(period)
    xi = np.random.normal(size=(n_steps, n_scenarios))

    rets = mu * dt + sigma * np.sqrt(dt) * xi
    rets = pd.DataFrame(rets)

    prices = last_price * (1 + rets).cumprod()
    return prices


# Simulate stock prices
sim = Monte_Carlo(period=periods, n_scenarios=10000, mu=mean, sigma=vol, last_price=last_price)

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
