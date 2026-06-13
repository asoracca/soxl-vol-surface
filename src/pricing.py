"""
pricing.py
----------
Black-Scholes put pricing and Greeks.
Replaces the rough premium approximation in backtest.py
"""

import numpy as np
from scipy.stats import norm


def black_scholes_put(S, K, T, r=0.05, sigma=None):
    """
    S     = spot price
    K     = strike price
    T     = time to expiration in years (DTE/252)
    r     = risk-free rate (0.05 = current Fed funds approx)
    sigma = implied volatility (annualized)
    """
    if T <= 0 or sigma is None or sigma <= 0:
        return {'price': 0, 'delta': -1, 'gamma': 0, 'theta': 0, 'vega': 0}

    d1 = (np.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)

    put_price = K * np.exp(-r * T) * norm.cdf(-d2) - S * norm.cdf(-d1)
    delta     = -norm.cdf(-d1)
    gamma     = norm.pdf(d1) / (S * sigma * np.sqrt(T))
    theta     = (-(S * norm.pdf(d1) * sigma) / (2 * np.sqrt(T))
                 + r * K * np.exp(-r * T) * norm.cdf(-d2)) / 365
    vega      = S * norm.pdf(d1) * np.sqrt(T) / 100  # per 1% IV move

    return {
        'price': round(put_price, 4),
        'delta': round(delta, 4),
        'gamma': round(gamma, 6),
        'theta': round(theta, 4),
        'vega':  round(vega, 4),
    }


if __name__ == "__main__":
    # Test: SOXL at $180, $150 strike, 7 DTE, IV=222%
    result = black_scholes_put(S=180.65, K=150, T=7/252, sigma=2.22)
    print("Black-Scholes Put Pricing Test")
    print(f"  Price:  ${result['price']:.2f}")
    print(f"  Delta:  {result['delta']:.4f}")
    print(f"  Gamma:  {result['gamma']:.6f}")
    print(f"  Theta:  ${result['theta']:.2f}/day")
    print(f"  Vega:   ${result['vega']:.2f} per 1% IV")