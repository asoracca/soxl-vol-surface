"""
heston.py  --  Heston stochastic-volatility option pricing & calibration
=========================================================================
Drop this into  soxl-vol-surface/  (e.g. as src/heston.py).

WHY THIS MATTERS (the "Month 2" upgrade from Black-Scholes)
-----------------------------------------------------------
Black-Scholes assumes ONE constant volatility. Reality: implied vol changes
with strike (the "smile"/"smirk") and with maturity (the term structure).
BS literally cannot produce a smile — so traders fudge it by quoting a
different sigma per strike. That's a patch, not a model.

Heston (1993) instead makes variance itself a mean-reverting random process:

    dS_t = mu S_t dt + sqrt(v_t) S_t dW1
    dv_t = kappa (theta - v_t) dt + xi sqrt(v_t) dW2
    corr(dW1, dW2) = rho

Five parameters with real economic meaning:
    v0     : current instantaneous variance      (where vol is right now)
    kappa  : mean-reversion speed                 (how fast vol pulls back)
    theta  : long-run variance                    (where vol reverts to)
    xi     : vol-of-vol                           (controls smile CURVATURE)
    rho    : spot/vol correlation (usually < 0)   (controls smile SLOPE/skew)

Because rho < 0 for equities (vol spikes when price drops), Heston naturally
produces the downside skew you see in SOXL puts. That is exactly *why vol
smiles exist* — and why selling puts when implied skew is rich can be an edge.

Pricing uses the semi-analytic characteristic function with the Albrecher
"Little Heston Trap" formulation (numerically stable for long maturities),
integrated by Gauss-Legendre quadrature.

    pip install numpy scipy

USAGE
-----
    from heston import HestonParams, heston_price, implied_vol, calibrate

    p = HestonParams(v0=0.25**2, kappa=2.0, theta=0.28**2, xi=0.6, rho=-0.6)
    call = heston_price(S=100, K=105, T=0.5, r=0.04, q=0.0, params=p)

    # Calibrate to a market vol surface (strikes x maturities of implied vols):
    fitted = calibrate(S, r, q, market_quotes)   # see __main__ for the shape

Run as a script for a self-test:
    python heston.py
"""

from __future__ import annotations

from dataclasses import dataclass, asdict

import numpy as np
from scipy.optimize import brentq, least_squares
from scipy.stats import norm


# -----------------------------------------------------------------------------
# Parameters
# -----------------------------------------------------------------------------
@dataclass
class HestonParams:
    v0: float      # current variance        (e.g. 0.25**2)
    kappa: float   # mean-reversion speed
    theta: float   # long-run variance
    xi: float      # vol of vol
    rho: float     # spot/vol correlation

    def feller_ok(self) -> bool:
        """Feller condition 2*kappa*theta > xi^2 keeps variance strictly > 0."""
        return 2 * self.kappa * self.theta > self.xi ** 2

    def as_vector(self):
        return np.array([self.v0, self.kappa, self.theta, self.xi, self.rho])

    @staticmethod
    def from_vector(x):
        return HestonParams(*x)


# -----------------------------------------------------------------------------
# Characteristic function (Little Heston Trap, Albrecher et al. 2007)
# -----------------------------------------------------------------------------
def _char_func(phi, S, T, r, q, p: HestonParams):
    x = np.log(S)
    a = p.kappa * p.theta
    d = np.sqrt((p.rho * p.xi * 1j * phi - p.kappa) ** 2
                + (p.xi ** 2) * (1j * phi + phi ** 2))
    g = (p.kappa - p.rho * p.xi * 1j * phi - d) / \
        (p.kappa - p.rho * p.xi * 1j * phi + d)

    exp_dt = np.exp(-d * T)
    C = (r - q) * 1j * phi * T + (a / p.xi ** 2) * (
        (p.kappa - p.rho * p.xi * 1j * phi - d) * T
        - 2 * np.log((1 - g * exp_dt) / (1 - g))
    )
    D = ((p.kappa - p.rho * p.xi * 1j * phi - d) / p.xi ** 2) * \
        ((1 - exp_dt) / (1 - g * exp_dt))
    return np.exp(C + D * p.v0 + 1j * phi * x)


# Gauss-Legendre nodes on [0, U]; 128 points is plenty and fast.
_GL_DEG = 128
_gl_x, _gl_w = np.polynomial.legendre.leggauss(_GL_DEG)


def _prob(j, S, K, T, r, q, p: HestonParams, upper=200.0):
    """P_j via the Gil-Pelaez inversion integral."""
    # shift phi by -i for P1 (the in-the-money probability under stock measure)
    half, mid = upper / 2.0, upper / 2.0
    phis = half * _gl_x + mid
    weights = half * _gl_w

    if j == 1:
        cf = _char_func(phis - 1j, S, T, r, q, p) / \
             _char_func(-1j, S, T, r, q, p)
    else:
        cf = _char_func(phis, S, T, r, q, p)

    integrand = np.real(np.exp(-1j * phis * np.log(K)) * cf / (1j * phis))
    return 0.5 + (1.0 / np.pi) * np.sum(weights * integrand)


def heston_price(S, K, T, r, q, params: HestonParams, option="call") -> float:
    """European option price under Heston."""
    P1 = _prob(1, S, K, T, r, q, params)
    P2 = _prob(2, S, K, T, r, q, params)
    call = S * np.exp(-q * T) * P1 - K * np.exp(-r * T) * P2
    call = max(call, 0.0)
    if option == "call":
        return call
    # put via put-call parity
    return call - S * np.exp(-q * T) + K * np.exp(-r * T)


# -----------------------------------------------------------------------------
# Black-Scholes & implied vol (for the smile comparison)
# -----------------------------------------------------------------------------
def bs_price(S, K, T, r, q, sigma, option="call"):
    if sigma <= 0 or T <= 0:
        intrinsic = max(S - K, 0.0) if option == "call" else max(K - S, 0.0)
        return intrinsic
    d1 = (np.log(S / K) + (r - q + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    if option == "call":
        return S * np.exp(-q * T) * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)
    return K * np.exp(-r * T) * norm.cdf(-d2) - S * np.exp(-q * T) * norm.cdf(-d1)


def implied_vol(price, S, K, T, r, q, option="call"):
    """Back out BS implied vol from a price (used to draw the Heston smile)."""
    intrinsic = (max(S - K, 0.0) if option == "call" else max(K - S, 0.0))
    if price <= intrinsic + 1e-10:
        return np.nan
    try:
        return brentq(lambda s: bs_price(S, K, T, r, q, s, option) - price,
                      1e-4, 5.0, maxiter=200, xtol=1e-8)
    except ValueError:
        return np.nan


def heston_smile(S, T, r, q, params: HestonParams, strikes, option="call"):
    """Return BS-implied vols produced by the Heston model across strikes.
    This is the curve that demonstrates *why a smile exists*."""
    vols = []
    for K in strikes:
        px = heston_price(S, K, T, r, q, params, option)
        vols.append(implied_vol(px, S, K, T, r, q, option))
    return np.array(vols)


# -----------------------------------------------------------------------------
# Calibration to a market vol surface
# -----------------------------------------------------------------------------
def calibrate(S, r, q, market_quotes, x0: HestonParams | None = None,
              weights=None, verbose=True):
    """Least-squares calibration of Heston params to market implied vols.

    market_quotes : list of (K, T, market_iv) tuples.
    Returns a fitted HestonParams. We fit in IMPLIED-VOL space (more stable
    and economically even-handed across strikes than fitting raw prices).
    """
    quotes = np.array(market_quotes, dtype=float)
    Ks, Ts, mkt_ivs = quotes[:, 0], quotes[:, 1], quotes[:, 2]
    if weights is None:
        weights = np.ones(len(quotes))

    if x0 is None:
        x0 = HestonParams(v0=mkt_ivs.mean() ** 2, kappa=2.0,
                          theta=mkt_ivs.mean() ** 2, xi=0.5, rho=-0.5)

    # bounds keep the optimizer in an economically sane, stable region
    lb = np.array([1e-4, 1e-2, 1e-4, 1e-2, -0.999])
    ub = np.array([2.0,  20.0, 2.0,   5.0,  0.0])

    def residuals(xv):
        p = HestonParams.from_vector(xv)
        out = np.empty(len(quotes))
        for i, (K, T, miv) in enumerate(zip(Ks, Ts, mkt_ivs)):
            px = heston_price(S, K, T, r, q, p, "call")
            model_iv = implied_vol(px, S, K, T, r, q, "call")
            out[i] = (0.0 if np.isnan(model_iv) else model_iv - miv) * weights[i]
        return out

    sol = least_squares(residuals, x0.as_vector(), bounds=(lb, ub),
                        xtol=1e-10, ftol=1e-10, max_nfev=2000)
    fitted = HestonParams.from_vector(sol.x)

    if verbose:
        rmse = np.sqrt(np.mean(sol.fun ** 2))
        print("Calibration complete.")
        print(f"  RMSE (vol points): {rmse:.4%}")
        print(f"  Feller condition : {'OK' if fitted.feller_ok() else 'VIOLATED'}")
        for k, v in asdict(fitted).items():
            print(f"  {k:<6}= {v:+.4f}")
    return fitted


# -----------------------------------------------------------------------------
# Self-test
# -----------------------------------------------------------------------------
if __name__ == "__main__":
    S, r, q = 100.0, 0.04, 0.0
    true = HestonParams(v0=0.25**2, kappa=2.5, theta=0.28**2, xi=0.6, rho=-0.65)

    # sanity: ATM Heston price vs BS at ~v0 vol
    K, T = 100.0, 0.5
    h = heston_price(S, K, T, r, q, true, "call")
    b = bs_price(S, K, T, r, q, 0.25, "call")
    print(f"ATM call  Heston={h:.4f}   BS(sigma=0.25)={b:.4f}")

    # demonstrate the smile Heston produces
    strikes = np.array([80, 90, 95, 100, 105, 110, 120.0])
    smile = heston_smile(S, T, r, q, true, strikes, "call")
    print("\nHeston-implied smile (T=0.5):")
    for k, v in zip(strikes, smile):
        print(f"  K={k:6.1f}  IV={v:6.2%}")
    print("  ^ note the higher IV on low strikes (downside skew) from rho<0")

    # round-trip calibration: generate quotes from `true`, recover params
    quotes = []
    for T in (0.25, 0.5, 1.0):
        for K in (85, 95, 100, 105, 115):
            px = heston_price(S, K, T, r, q, true, "call")
            iv = implied_vol(px, S, K, T, r, q, "call")
            if not np.isnan(iv):
                quotes.append((K, T, iv))

    print("\nRecovering parameters from a synthetic surface...")
    fitted = calibrate(S, r, q, quotes,
                       x0=HestonParams(0.2**2, 1.5, 0.2**2, 0.4, -0.3))
    print("\nTrue params:", {k: round(v, 4) for k, v in asdict(true).items()})
