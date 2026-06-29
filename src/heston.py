"""
heston.py — Heston stochastic-volatility model
-----------------------------------------------
Black-Scholes assumes ONE constant volatility, so it predicts a FLAT implied-vol
line across strikes. Reality shows a smile/skew. Heston fixes this by letting
variance follow its own mean-reverting random process, correlated with price:

    dS = r S dt + sqrt(v) S dW1
    dv = kappa (theta - v) dt + sigma sqrt(v) dW2,   corr(dW1, dW2) = rho

Parameters:
  v0     starting variance        kappa  mean-reversion speed
  theta  long-run variance        sigma  vol-of-vol
  rho    price/vol correlation (negative -> downside skew, the usual equity case)

We price European calls via the Heston characteristic function (Gil-Pelaez /
Carr-Madan), then invert Black-Scholes to recover the implied-vol SMILE.

    python src/heston.py
"""

import os
import numpy as np
from scipy.integrate import quad
from scipy.stats import norm
from scipy.optimize import brentq
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def heston_cf(u, S0, T, r, kappa, theta, sigma, rho, v0):
    """Characteristic function of log(S_T) under Heston."""
    i = 1j
    xi = kappa - sigma * rho * i * u
    d = np.sqrt(xi ** 2 + sigma ** 2 * (i * u + u ** 2))
    g = (xi - d) / (xi + d)
    ed = np.exp(-d * T)
    C = r * i * u * T + (kappa * theta / sigma ** 2) * ((xi - d) * T - 2 * np.log((1 - g * ed) / (1 - g)))
    D = ((xi - d) / sigma ** 2) * ((1 - ed) / (1 - g * ed))
    return np.exp(C + D * v0 + i * u * np.log(S0))


def heston_call(S0, K, T, r, kappa, theta, sigma, rho, v0):
    i = 1j
    phi_mi = heston_cf(-i, S0, T, r, kappa, theta, sigma, rho, v0)

    def integ(u, which):
        if which == 1:
            cf = heston_cf(u - i, S0, T, r, kappa, theta, sigma, rho, v0) / phi_mi
        else:
            cf = heston_cf(u, S0, T, r, kappa, theta, sigma, rho, v0)
        return np.real(np.exp(-i * u * np.log(K)) * cf / (i * u))

    P1 = 0.5 + (1 / np.pi) * quad(lambda u: integ(u, 1), 1e-8, 200, limit=200)[0]
    P2 = 0.5 + (1 / np.pi) * quad(lambda u: integ(u, 2), 1e-8, 200, limit=200)[0]
    return S0 * P1 - K * np.exp(-r * T) * P2


def bs_call(S, K, T, r, sig):
    d1 = (np.log(S / K) + (r + 0.5 * sig * sig) * T) / (sig * np.sqrt(T))
    d2 = d1 - sig * np.sqrt(T)
    return S * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)


def implied_vol(price, S, K, T, r):
    intrinsic = max(S - K * np.exp(-r * T), 0.0)
    if price <= intrinsic + 1e-8:
        return np.nan
    try:
        return brentq(lambda s: bs_call(S, K, T, r, s) - price, 1e-4, 5.0)
    except Exception:
        return np.nan


def main():
    S0, r, T = 100.0, 0.03, 0.5
    kappa, theta, sigma, rho, v0 = 2.0, 0.04, 0.6, -0.7, 0.04
    print("Heston: v0=%.3f kappa=%.1f theta=%.3f vol-of-vol=%.2f rho=%.2f" % (v0, kappa, theta, sigma, rho))
    print("Starting vol ~ %.0f%%, long-run vol ~ %.0f%%\n" % (np.sqrt(v0) * 100, np.sqrt(theta) * 100))

    Ks = np.linspace(70, 130, 25)
    ivs = []
    print(" strike   heston_price   implied_vol")
    for K in Ks:
        p = heston_call(S0, K, T, r, kappa, theta, sigma, rho, v0)
        iv = implied_vol(p, S0, K, T, r)
        ivs.append(iv)
        if int(K) % 10 == 0 or abs(K - S0) < 3:
            print("  %5.0f   %10.4f   %7.1f%%" % (K, p, iv * 100))

    plt.figure(figsize=(9, 5))
    plt.plot(Ks, np.array(ivs) * 100, marker="o")
    plt.axvline(S0, ls="--", color="gray", alpha=0.6, label="spot")
    plt.title("Heston implied-volatility smile (rho = %.1f)" % rho)
    plt.xlabel("strike"); plt.ylabel("Black-Scholes implied vol (%)")
    plt.grid(alpha=0.3); plt.legend(); plt.tight_layout()
    os.makedirs("data", exist_ok=True)
    plt.savefig("data/heston_smile.png", dpi=110)
    print("\nSaved: data/heston_smile.png")
    print("Black-Scholes would be a FLAT line. Heston's downward skew comes from rho<0:")
    print("price drops are linked to vol spikes, so low strikes (puts) price at higher vol.")


if __name__ == "__main__":
    main()
