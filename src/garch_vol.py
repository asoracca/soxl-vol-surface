"""
garch_vol.py  —  add to soxl-vol-surface/src/
----------------------------------------------
GARCH(1,1) volatility model for SOXL.

Why GARCH instead of rolling std?
  Rolling 30-day std treats every day equally — yesterday's move counts the same
  as a move from 29 days ago. That's wrong.

  GARCH (Generalised AutoRegressive Conditional Heteroskedasticity) says:
    1. Volatility clusters — big moves follow big moves, calm follows calm
    2. Recent moves matter MORE than old ones
    3. Volatility is mean-reverting — it eventually returns to a long-run average

  The GARCH(1,1) model:
    σ²_t = ω + α·ε²_(t-1) + β·σ²_(t-1)

  Where:
    σ²_t  = today's variance forecast
    ω     = long-run variance (the floor volatility reverts to)
    α     = weight on yesterday's shock (how reactive to new info)
    β     = weight on yesterday's variance (how persistent volatility is)
    ε_(t-1) = yesterday's return shock

  α + β < 1 guarantees mean reversion.
  Typical values: α≈0.10, β≈0.85 for stocks.

  In plain English: GARCH says "if the market just had a big move,
  expect more big moves soon — but they'll eventually calm down."

Install:
    pip install arch

Run standalone:
    python src/garch_vol.py

Or import:
    from src.garch_vol import fit_garch, get_garch_forecast, compare_vol_estimates
"""

import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import yfinance as yf
from pathlib import Path

try:
    from arch import arch_model
    ARCH_AVAILABLE = True
except ImportError:
    ARCH_AVAILABLE = False
    print("⚠️  arch library not installed. Run: pip install arch --break-system-packages")
    print("    Falling back to EWMA vol (exponentially weighted moving average)")


# ── EWMA fallback (no arch needed) ──────────────────────────────────────────

def compute_ewma_vol(returns, lam=0.94):
    """
    EWMA (Exponentially Weighted Moving Average) volatility.
    RiskMetrics standard: lambda=0.94 for daily data.

    Simpler than GARCH but still better than rolling std.
    Recent moves get weight lam^0, lam^1, lam^2, ... (exponentially decaying).
    """
    var = returns.var()  # initialise
    variances = []

    for r in returns:
        var = lam * var + (1 - lam) * r**2
        variances.append(var)

    ewma_var = pd.Series(variances, index=returns.index)
    ewma_vol = np.sqrt(ewma_var * 252) * 100   # annualised %
    ewma_vol.name = "ewma_vol"
    return ewma_vol


# ── GARCH ────────────────────────────────────────────────────────────────────

def fit_garch(returns, p=1, q=1):
    """
    Fit GARCH(p,q) model to return series.
    p=1, q=1 is the standard starting point (almost always sufficient).

    Returns the fitted model object.
    """
    if not ARCH_AVAILABLE:
        return None

    # arch library wants returns in % (multiply by 100) for numerical stability
    r_pct = returns * 100

    model  = arch_model(r_pct, vol="Garch", p=p, q=q, dist="Normal")
    result = model.fit(disp="off", options={"maxiter": 300})

    # Print fitted parameters
    params = result.params
    omega = params.get("omega", np.nan)
    alpha = params.get("alpha[1]", np.nan)
    beta  = params.get("beta[1]", np.nan)
    persistence = alpha + beta if not (np.isnan(alpha) or np.isnan(beta)) else np.nan

    print(f"\n── GARCH(1,1) Fitted Parameters ─────────────────────")
    print(f"  ω (long-run floor):    {omega:.6f}")
    print(f"  α (shock weight):      {alpha:.4f}  ← how reactive to new moves")
    print(f"  β (persistence):       {beta:.4f}  ← how long vol stays elevated")
    print(f"  α + β (persistence):   {persistence:.4f}  (< 1 = mean-reverting ✅)")
    if persistence < 1:
        half_life = -np.log(2) / np.log(persistence)
        print(f"  Half-life of vol:      {half_life:.1f} days  "
              f"(how long a vol spike takes to decay 50%)")
    print(f"  Log-likelihood:        {result.loglikelihood:.1f}")

    return result


def get_garch_forecast(result, returns, horizon=1):
    """
    Get GARCH conditional volatility series + next-day forecast.

    Returns:
        cond_vol  — historical conditional vol (annualised %)
        forecast  — next `horizon` days forecast (annualised %)
    """
    if result is None:
        return None, None

    # Conditional volatility (in-sample, annualised %)
    cond_vol = result.conditional_volatility * np.sqrt(252)
    cond_vol.index = returns.index
    cond_vol.name = "garch_vol"

    # Out-of-sample forecast
    fc = result.forecast(horizon=horizon, reindex=False)
    forecast_var = fc.variance.values[-1]   # variance for next `horizon` days
    forecast_vol = np.sqrt(forecast_var * 252) * 1.0  # already in % from arch

    return cond_vol, forecast_vol


# ── Comparison ───────────────────────────────────────────────────────────────

def compare_vol_estimates(ticker="SOXL", period="2y"):
    """
    Compare three volatility estimates side by side:
      1. Rolling 30-day std (naive)
      2. EWMA (better — exponential weighting)
      3. GARCH (best — models clustering and mean reversion)

    This is what gets added to the daily signal output.
    """
    print(f"\nFetching {ticker} data ({period})...")
    data = yf.Ticker(ticker).history(period=period)
    data.index = pd.to_datetime(data.index).tz_localize(None)
    data["log_ret"] = np.log(data["Close"] / data["Close"].shift(1))
    data = data.dropna(subset=["log_ret"])

    returns = data["log_ret"]

    # 1. Rolling 30d std
    rv30 = returns.rolling(30).std() * np.sqrt(252) * 100
    rv30.name = "rv30"

    # 2. EWMA
    ewma = compute_ewma_vol(returns, lam=0.94)

    # 3. GARCH
    garch_result = fit_garch(returns)
    if garch_result is not None:
        garch_vol, garch_forecast = get_garch_forecast(garch_result, returns, horizon=5)
    else:
        garch_vol = ewma.copy()
        garch_vol.name = "garch_vol"
        garch_forecast = [float(ewma.iloc[-1])] * 5

    # Current values
    current_rv30  = float(rv30.iloc[-1])
    current_ewma  = float(ewma.iloc[-1])
    current_garch = float(garch_vol.iloc[-1]) if garch_vol is not None else current_ewma
    forecast_5d   = float(np.mean(garch_forecast)) if garch_forecast is not None else current_garch

    print(f"\n── {ticker} Volatility Comparison ──────────────────────")
    print(f"  Rolling 30d std:     {current_rv30:.1f}%  (naive)")
    print(f"  EWMA (λ=0.94):       {current_ewma:.1f}%  (better)")
    print(f"  GARCH(1,1) today:    {current_garch:.1f}%  (best)")
    print(f"  GARCH 5-day forecast: {forecast_5d:.1f}%  ← use THIS for option pricing")

    if forecast_5d > current_garch * 1.1:
        print(f"\n  ⚠️  Vol rising — GARCH expects more turbulence ahead")
    elif forecast_5d < current_garch * 0.9:
        print(f"\n  ✅ Vol falling — GARCH expects calmer period ahead")
    else:
        print(f"\n  → Vol stable — GARCH sees no major regime shift coming")

    return rv30, ewma, garch_vol, garch_forecast, data


def plot_vol_comparison(rv30, ewma, garch_vol, data, ticker="SOXL"):
    """3-panel: price, three vol estimates, vol spread."""
    fig, axes = plt.subplots(3, 1, figsize=(13, 11), sharex=True)
    fig.suptitle(f"{ticker} Volatility Comparison: Rolling vs EWMA vs GARCH",
                 fontsize=13, fontweight="bold")

    # Panel 1: Price
    axes[0].plot(data.index, data["Close"], color="#378ADD", linewidth=1.5)
    axes[0].set_ylabel("Price ($)")
    axes[0].set_title(f"{ticker} Price")
    axes[0].grid(True, alpha=0.25)

    # Panel 2: Three vol estimates
    axes[1].plot(rv30.index,    rv30.values,    color="grey",    linewidth=1.2,
                 alpha=0.7, label="Rolling 30d (naive)")
    axes[1].plot(ewma.index,    ewma.values,    color="#378ADD", linewidth=1.5,
                 alpha=0.8, label="EWMA λ=0.94")
    if garch_vol is not None:
        axes[1].plot(garch_vol.index, garch_vol.values, color="#A32D2D", linewidth=1.8,
                     label="GARCH(1,1)")
    axes[1].set_ylabel("Annualised Vol (%)")
    axes[1].set_title("Volatility Estimates")
    axes[1].legend(fontsize=9)
    axes[1].grid(True, alpha=0.25)

    # Panel 3: GARCH vs Rolling spread
    if garch_vol is not None:
        aligned = rv30.align(garch_vol, join="inner")
        spread = aligned[0] - aligned[1]
        colors = ["#1D9E75" if v >= 0 else "#A32D2D" for v in spread]
        axes[2].bar(spread.index, spread.values, color=colors, alpha=0.6, width=1)
        axes[2].axhline(0, color="black", linewidth=0.8)
        axes[2].set_ylabel("Rolling30 − GARCH (%)")
        axes[2].set_title("Rolling Overestimates Vol When Positive (GARCH is more accurate)")
        axes[2].grid(True, alpha=0.25)

    axes[2].xaxis.set_major_formatter(mdates.DateFormatter("%b %Y"))
    axes[2].xaxis.set_major_locator(mdates.MonthLocator(interval=2))
    plt.setp(axes[2].xaxis.get_majorticklabels(), rotation=30, ha="right")

    plt.tight_layout()
    Path("data").mkdir(exist_ok=True)
    plt.savefig("data/garch_vol.png", dpi=150, bbox_inches="tight")
    print("\nSaved: data/garch_vol.png")


if __name__ == "__main__":
    # Install arch if needed
    import subprocess, sys
    try:
        import arch
    except ImportError:
        print("Installing arch library...")
        subprocess.check_call([sys.executable, "-m", "pip", "install", "arch",
                               "--break-system-packages", "-q"])
        from arch import arch_model
        
        ARCH_AVAILABLE = True

    rv30, ewma, garch_vol, forecast, data = compare_vol_estimates("SOXL", period="2y")
    plot_vol_comparison(rv30, ewma, garch_vol, data, ticker="SOXL")

    print("\n── Next step ─────────────────────────────────────────")
    print("  Add get_garch_forecast() output to iv_rank.py signal.")
    print("  Replace rv30 in the signal with garch_vol for better IV estimate.")
