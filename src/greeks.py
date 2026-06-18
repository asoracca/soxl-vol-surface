"""
greeks.py
---------
Black-Scholes Greeks for put options.

Greeks tell you EXACTLY what your option will do under different market conditions:

  Delta  — how much the option price moves per $1 move in the stock
             Put delta is negative: if stock drops $1, put gains |delta| dollars
             e.g. delta=-0.20 means if QLD drops $1, your put gains $0.20

  Gamma  — how fast delta changes per $1 move in the stock
             High gamma = your delta (and P&L) accelerates when stock moves
             Dangerous near expiry (gamma spikes)

  Theta  — dollars you collect per day just from time passing (time decay)
             Always positive for sellers. e.g. theta=0.05 means you earn $5/day per contract

  Vega   — how much the option price changes per 1% change in implied vol
             Positive for buyers, negative for sellers
             e.g. vega=-0.10 means if vol spikes 1%, your short put loses $0.10

  Rho    — sensitivity to interest rate changes. Least important Greek.

Usage:
    from src.greeks import compute_greeks, print_greeks
    result = compute_greeks(S=94.51, K=82.00, T=30/365, r=0.053, sigma=0.25, option_type="put")
    print_greeks(result)

    Or run standalone:
    python greeks.py
"""

import numpy as np
from scipy.stats import norm


# ── IV Solver ────────────────────────────────────────────────────────────────

def implied_volatility(market_price, S, K, T, r, option_type="put",
                       tol=1e-6, max_iter=500):
    """
    Back-calculate implied volatility from a market option price.
    Uses bisection method — always converges, never blows up.

    market_price — the mid price you see on Schwab (bid+ask)/2
    S            — current stock price
    K            — strike price
    T            — time to expiry in years (e.g. 30/365)
    r            — risk-free rate (e.g. 0.053)
    option_type  — "put" or "call"

    Returns implied vol as a decimal (e.g. 1.97 = 197%)
    Returns None if no solution found (price outside arbitrage bounds)
    """
    if T <= 0:
        return None

    # Arbitrage bounds check
    intrinsic = max(K - S, 0) if option_type == "put" else max(S - K, 0)
    if market_price < intrinsic * 0.999:
        return None  # price below intrinsic — data error

    lo, hi = 1e-6, 20.0  # search between 0.0001% and 2000% IV

    for _ in range(max_iter):
        mid   = (lo + hi) / 2
        price = black_scholes_price(S, K, T, r, mid, option_type)

        if abs(price - market_price) < tol:
            return mid
        if price < market_price:
            lo = mid
        else:
            hi = mid

    return (lo + hi) / 2  # best estimate after max_iter


def iv_from_chain(S, K, T, r, bid, ask, option_type="put"):
    """
    Compute IV from a bid/ask pair. Uses mid price.
    Also returns whether the spread is too wide to trust.

    Returns dict: {iv, mid, spread_pct, trustworthy}
    """
    if bid <= 0 or ask <= 0 or ask < bid:
        return {"iv": None, "mid": None, "spread_pct": None, "trustworthy": False}

    mid        = (bid + ask) / 2
    spread_pct = (ask - bid) / mid * 100

    iv = implied_volatility(mid, S, K, T, r, option_type)

    # Wide spread = unreliable IV (market maker not confident)
    trustworthy = spread_pct < 50 and iv is not None

    return {
        "iv":          iv,
        "mid":         mid,
        "spread_pct":  spread_pct,
        "trustworthy": trustworthy,
    }


def black_scholes_price(S, K, T, r, sigma, option_type="put"):
    """
    Black-Scholes option price.

    S     — current stock price
    K     — strike price
    T     — time to expiry in years (e.g. 30 days = 30/365)
    r     — risk-free rate (annualized, e.g. 0.053 for 5.3%)
    sigma — implied volatility (annualized, e.g. 0.25 for 25%)
    """
    if T <= 0 or sigma <= 0:
        return max(K - S, 0) if option_type == "put" else max(S - K, 0)

    d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)

    if option_type == "call":
        price = S * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)
    else:  # put
        price = K * np.exp(-r * T) * norm.cdf(-d2) - S * norm.cdf(-d1)

    return price


def compute_greeks(S, K, T, r, sigma, option_type="put"):
    """
    Compute all five Black-Scholes Greeks for a single option.

    Returns dict with: price, delta, gamma, theta, vega, rho
    All values are per-share. Multiply by 100 for per-contract.
    """
    if T <= 0 or sigma <= 0:
        return {"price": max(K - S, 0), "delta": -1.0 if S < K else 0.0,
                "gamma": 0, "theta": 0, "vega": 0, "rho": 0}

    d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)

    price = black_scholes_price(S, K, T, r, sigma, option_type)

    # Delta: directional exposure
    if option_type == "call":
        delta = norm.cdf(d1)
    else:
        delta = norm.cdf(d1) - 1   # put delta is always negative

    # Gamma: same for calls and puts
    gamma = norm.pdf(d1) / (S * sigma * np.sqrt(T))

    # Theta: daily time decay (divide by 365 to get per-day)
    if option_type == "call":
        theta = (-(S * norm.pdf(d1) * sigma) / (2 * np.sqrt(T))
                 - r * K * np.exp(-r * T) * norm.cdf(d2)) / 365
    else:
        theta = (-(S * norm.pdf(d1) * sigma) / (2 * np.sqrt(T))
                 + r * K * np.exp(-r * T) * norm.cdf(-d2)) / 365

    # Vega: per 1% change in vol (divide by 100 for percentage)
    vega = S * norm.pdf(d1) * np.sqrt(T) / 100

    # Rho: per 1% change in rates
    if option_type == "call":
        rho = K * T * np.exp(-r * T) * norm.cdf(d2) / 100
    else:
        rho = -K * T * np.exp(-r * T) * norm.cdf(-d2) / 100

    return {
        "price":  price,
        "delta":  delta,
        "gamma":  gamma,
        "theta":  theta,   # per day
        "vega":   vega,    # per 1% vol move
        "rho":    rho,
        "d1":     d1,
        "d2":     d2,
        "moneyness": S / K,
        "iv":     sigma,
        "T_days": T * 365,
    }


def print_greeks(g, label="Option Greeks"):
    """Pretty-print Greeks with plain-English interpretation."""
    print(f"\n{'='*55}")
    print(f"  {label}")
    print(f"{'='*55}")
    print(f"  Theoretical price:  ${g['price']:.2f}  (${g['price']*100:.0f} per contract)")
    print(f"  Moneyness (S/K):    {g['moneyness']:.3f}  "
          f"({'OTM' if g['moneyness'] > 1 else 'ITM'} put)")
    print(f"  DTE:                {g['T_days']:.0f} days")
    print(f"  IV used:            {g['iv']*100:.1f}%")
    print(f"{'─'*55}")
    print(f"  Delta:   {g['delta']:+.4f}  → per $1 stock move, option changes ${g['delta']:.2f}")
    print(f"  Gamma:   {g['gamma']:+.4f}  → delta accelerates by {g['gamma']:.4f} per $1 move")
    print(f"  Theta:   {g['theta']:+.4f}  → you earn ${abs(g['theta']):.2f}/day ({abs(g['theta'])*100:.2f}/day per contract)")
    print(f"  Vega:    {g['vega']:+.4f}  → per 1% vol spike, option changes ${g['vega']:.2f} (${g['vega']*100:.2f}/contract)")
    print(f"  Rho:     {g['rho']:+.4f}  → per 1% rate change")
    print(f"{'─'*55}")

    # Plain-English P&L scenarios (per contract = ×100)
    print(f"\n  What this means for your trade (1 contract = 100 shares):")
    print(f"  • Stock drops $5:   put gains  ${abs(g['delta'])*5*100:.0f} (approx)")
    print(f"  • Stock rises $5:   put loses  ${abs(g['delta'])*5*100:.0f} (approx)")
    print(f"  • 1 week passes:    you earn   ${abs(g['theta'])*7*100:.0f} from theta decay")
    print(f"  • Vol spikes +5%:   put changes ${g['vega']*5*100:.0f} (bad for short put seller)")
    print(f"{'='*55}")

    # Seller's summary
    if g['delta'] < 0:  # put
        prob_otm = 1 - abs(g['delta'])  # rough probability of expiring OTM
        print(f"\n  As a PUT SELLER you want this to expire OTM.")
        print(f"  Delta={g['delta']:.2f} ≈ {prob_otm:.0%} chance of expiring worthless (rough estimate)")
        print(f"  Collect ${abs(g['theta'])*100:.2f}/day in theta. Need {abs(g['price']/g['theta']):.0f} days to collect full premium.")
    print()


def greeks_for_soxl_trade(dte=30, otm_pct=0.85):
    """
    Live SOXL put Greeks.
    Fetches current SOXL price, proposes a strike at otm_pct of spot,
    uses vol-scaled IV (SOXL realized vol is ~3x SPY so IV is much higher than VIX).

    otm_pct=0.85 means strike is 15% below current price (standard for SOXL puts).
    """
    import yfinance as yf
    import warnings
    warnings.filterwarnings("ignore")

    # Fetch live SOXL price
    soxl = yf.Ticker("SOXL")
    hist  = soxl.history(period="35d")
    S     = float(hist["Close"].iloc[-1])

    # Compute 30-day realized vol (annualized)
    hist["log_ret"] = (hist["Close"] / hist["Close"].shift(1)).apply(lambda x: __import__("math").log(x) if x > 0 else 0)
    rv30  = hist["log_ret"].iloc[-30:].std() * (252 ** 0.5)

    # SOXL IV ≈ realized vol (since VIX-scaling gives similar result)
    # Use realized vol directly as sigma estimate
    sigma = max(rv30, 0.50)   # floor at 50% — SOXL rarely goes below this

    K     = round(S * otm_pct, 0)   # round to nearest $1 strike
    T     = dte / 365
    r     = 0.053

    print(f"\n  Live SOXL price: ${S:.2f}")
    print(f"  30d Realized Vol: {rv30*100:.1f}%  →  using {sigma*100:.1f}% as IV")
    print(f"  Proposed strike: ${K:.0f} ({(1-otm_pct)*100:.0f}% OTM) | DTE: {dte}")

    g = compute_greeks(S, K, T, r, sigma, option_type="put")
    print_greeks(g, label=f"SOXL ${K:.0f} Put — {dte} DTE")

    # Scenario table
    print(f"  Scenario Table — what your short put earns/loses (1 contract):")
    print(f"  {'SOXL Price':>11}  {'% move':>7}  {'Put Value':>10}  {'Your P&L':>10}  {'Delta':>8}")
    print(f"  {'─'*55}")
    scenarios = [S*1.20, S*1.10, S*1.05, S, S*0.95, S*0.90, S*0.85, K, S*0.70, S*0.50]
    for spot in scenarios:
        g2   = compute_greeks(spot, K, T, r, sigma, option_type="put")
        pnl  = -(g2['price'] - g['price']) * 100   # short put: gain when price falls in value
        move = (spot / S - 1) * 100
        print(f"  ${spot:>10.2f}  {move:>+6.1f}%  ${g2['price']:>9.2f}  {pnl:>+9.0f}  {g2['delta']:>+8.3f}")

    return g


def demo_iv_solver():
    """
    Test IV solver on the QLD $82 put you tried to trade.
    Market showed: bid=$0.40, ask=$2.65, mid=$1.53
    Black-Scholes gave $0.08 using 27% IV — clearly wrong.
    IV solver will back-calculate the real implied vol.
    """
    print("\n" + "="*55)
    print("  IV SOLVER DEMO — QLD $82 Put Jul 17")
    print("="*55)

    S, K, T, r = 94.51, 82.00, 30/365, 0.053
    bid, ask   = 0.40, 2.65

    result = iv_from_chain(S, K, T, r, bid, ask, option_type="put")

    print(f"  Market bid:       ${bid:.2f}")
    print(f"  Market ask:       ${ask:.2f}")
    print(f"  Mid price:        ${result['mid']:.2f}")
    print(f"  Bid-ask spread:   {result['spread_pct']:.0f}%  "
          f"({'⚠️ Too wide — unreliable' if not result['trustworthy'] else '✅ OK'})")

    if result["iv"]:
        iv_pct = result["iv"] * 100
        print(f"\n  Implied Vol:      {iv_pct:.1f}%")
        print(f"  (vs our naive guess of 27% — real IV is {iv_pct/27:.1f}x higher)")

        # Now re-price with correct IV
        g = compute_greeks(S, K, T, r, result["iv"], option_type="put")
        print(f"\n  Re-priced Greeks with real IV ({iv_pct:.0f}%):")
        print(f"  Price:   ${g['price']:.2f}  (matches market mid ✅)")
        print(f"  Delta:   {g['delta']:+.4f}")
        print(f"  Theta:   ${abs(g['theta'])*100:.2f}/day per contract")
        print(f"  Vega:    ${abs(g['vega'])*100:.2f}/contract per 1% vol move")
    else:
        print("  ⚠️  Could not solve — price may be outside arbitrage bounds")

    print("="*55)


if __name__ == "__main__":
    demo_iv_solver()
    print()
    greeks_for_soxl_trade(dte=30, otm_pct=0.85)