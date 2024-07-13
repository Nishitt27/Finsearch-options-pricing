import numpy as np

def binomial_option_pricing(S, K, T, r, σ, N):
    Δt = T / N
    u = np.exp(σ * np.sqrt(Δt))
    d = np.exp(-σ * np.sqrt(Δt))
    p = (np.exp(r * Δt) - d) / (u - d)

    # Initialize asset prices at maturity
    asset_prices = np.zeros((N + 1, N + 1))
    option_values = np.zeros((N + 1, N + 1))

    for i in range(N + 1):
        for j in range(i + 1):
            asset_prices[j, i] = S * (u ** (i - j)) * (d ** j)

    # Calculate option values at maturity
    for j in range(N + 1):
        option_values[j, N] = max(0, asset_prices[j, N] - K)

    # Backward induction
    for i in range(N - 1, -1, -1):
        for j in range(i + 1):
            option_values[j, i] = np.exp(-r * Δt) * (p * option_values[j, i + 1] + (1 - p) * option_values[j + 1, i + 1])

    return option_values[0, 0]

# Example usage
S_0 = 100
K = 100
T = 1
r = 0.05
σ = 0.2
N = 3

option_price = binomial_option_pricing(S_0, K, T, r, σ, N)
print("Option Price:", option_price)
