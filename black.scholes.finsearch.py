import math
import scipy.stats as stats

def black_scholes(S, K, t, r, sigma, option_type='call'):
    """
    Calculate the value of a call or put option using the Black-Scholes model.

    Parameters:
    S (float): The current price of the underlying asset.
    K (float): The strike price of the option.
    t (float): The time to maturity of the option (in years).
    r (float): The risk-free interest rate.
    sigma (float): The volatility of the underlying asset.
    option_type (str): 'call' or 'put' (default is 'call').

    Returns:
    float: The value of the option.
    """
    d1 = (math.log(S / K) + (r + sigma ** 2 / 2) * t) / (sigma * math.sqrt(t))
    d2 = d1 - sigma * math.sqrt(t)

    if option_type == 'call':
        return S * stats.norm.cdf(d1) - K * math.exp(-r * t) * stats.norm.cdf(d2)
    elif option_type == 'put':
        return K * math.exp(-r * t) * stats.norm.cdf(-d2) - S * stats.norm.cdf(-d1)
    else:
        raise ValueError("Invalid option type. Must be 'call' or 'put'.")

# Example usage:
S = 100  # Current stock price
K = 110  # Strike price
t = 1  # Time to maturity (1 year)
r = 0.05  # Risk-free interest rate
sigma = 0.2  # Volatility
option_type = 'call'

option_value = black_scholes(S, K, t, r, sigma, option_type)
print(f"The value of the {option_type} option is {option_value:.2f}")

#This code uses the scipy.stats module to calculate the cumulative distribution function (CDF) of the normal distribution, which is used in the Black-Scholes formula. Note that this implementation assumes a European-style option, which can only be exercised at maturity.