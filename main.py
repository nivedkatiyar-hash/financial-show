import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def gbm(n_years=100, mu=0.142, sigma=0.99, steps_per_year=365, n_paths=1, s0=100):
    """
    Simulates asset prices using Geometric Brownian Motion.
    """
    # Calculate time step
    dt = 1 / steps_per_year
    n_steps = int(n_years * steps_per_year)
    
    # Generate random standard normal variables for all steps and paths
    # size is (n_steps, n_paths) to simulate multiple scenarios simultaneously
    xi = np.random.normal(size=(n_steps, n_paths))
    
    # Calculate the log returns using the standard GBM formula
    drift = (mu - 0.5 * sigma**2) * dt
    diffusion = sigma * np.sqrt(dt) * xi
    log_returns = drift + diffusion
    
    # Calculate cumulative returns and convert to price paths
    # np.vstack adds the initial price (s0) to the beginning of the array
    cumulative_returns = np.cumsum(log_returns, axis=0)
    prices = s0 * np.exp(cumulative_returns)
    prices = np.vstack([np.full(n_paths, s0), prices])
    
    return pd.DataFrame(prices)

# Generate 5 different potential future price paths
simulated_prices = gbm(n_paths=5)

# Optional: Plot the results to verify
simulated_prices.plot(legend=False, title="GBM Simulated Price Paths")
plt.xlabel("Time Steps (Days)")
plt.ylabel("Asset Price")
plt.show()
