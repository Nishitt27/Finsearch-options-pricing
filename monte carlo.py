#!/usr/bin/env python
# coding: utf-8

# In[1]:


import math
from numpy import *
from time import time

# Seed for reproducibility
random.seed(20000)
t0 = time()

# Parameters
S0 = 100.0  # Initial stock price
K = 105.0  # Strike price
T = 1.0  # Time to maturity in years
r = 0.05  # Risk-free interest rate
sigma = 0.2  # Volatility of the underlying asset
M = 50  # Number of time steps
dt = T / M  # Length of each time step
I = 250000  # Number of simulated price paths

# Simulating I paths with M time steps
S = S0 * exp(cumsum((r - 0.5 * sigma ** 2) * dt
                   + sigma * math.sqrt(dt)
                   * random.standard_normal((M + 1, I)), axis=0))

# Setting initial stock price for all paths
S[0] = S0

# Calculating the Monte Carlo estimator
C0 = math.exp(-r * T) * sum(maximum(S[-1] - K, 0)) / I

# Results output
tnp2 = time() - t0

print('The European Option Value is: ', C0)  # The European Option Value is:  8.165807966259603
print('The Execution Time is: ',tnp2)


# In[2]:


import matplotlib.pyplot as plt

# Plotting the first 10 simulated paths
plt.plot(S[:, :10])  # S[:, :10] selects the first 10 paths for all time steps
plt.grid(True)       # Adds a grid to the plot for better readability
plt.xlabel('Steps')  # Labels the x-axis as 'Steps'
plt.ylabel('Index level')  # Labels the y-axis as 'Index level'
plt.title('Simulated Price Paths')  # Adds a title to the graph
plt.show()  # Displays the plot


# In[3]:


import matplotlib.pyplot as plt

# Set the figure size for the plot
plt.rcParams["figure.figsize"] = (15, 8)

# Plotting the histogram of the simulated index levels at the end of the simulation
plt.hist(S[-1], bins=50)  # S[-1] contains the index levels at the last time step (maturity)
plt.grid(True)            # Adds a grid to the plot for better readability
plt.xlabel('Index level')  # Labels the x-axis as 'Index level'
plt.ylabel('Frequency')    # Labels the y-axis as 'Frequency'
plt.title('Distribution of Simulated Index Levels at Maturity')  # Adds a title to the graph
plt.show()  # Displays the plot


# In[4]:


import numpy as np
import matplotlib.pyplot as plt

# Set the figure size for the plot
plt.rcParams["figure.figsize"] = (15, 8)

# Calculate the option inner values at the end of the simulation (intrinsic value)
option_values = np.maximum(S[-1] - K, 0)

# Plotting the histogram of the end-of-period option values
plt.hist(option_values, bins=50)  # 50 bins to visualize the distribution
plt.grid(True)                    # Adds a grid to the plot for better readability
plt.xlabel('Option Inner Value')  # Labels the x-axis as 'option inner value'
plt.ylabel('Frequency')           # Labels the y-axis as 'frequency'
plt.ylim(0, 50000)                # Limits the y-axis to a maximum frequency of 50,000 for better visualization
plt.title('Distribution of Simulated End-of-Period Option Values')  # Adds a title to the graph
plt.show()  # Displays the plot


# In[ ]:




