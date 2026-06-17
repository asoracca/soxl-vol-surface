"""
iv_rank.py
----------
IV Rank and IV Percentile using REAL implied volatility from VIX.

Previously used realized vol as a proxy — this version uses actual VIX data,
which is the market's true implied volatility estimate.

Also adds VRP (Variance Risk Premium) = VIX - Realized Vol.
Positive VRP = market overpaying for protection = put sellers have edge.

IV Rank  = (current VIX - 52w low VIX) / (52w high VIX - 52w low VIX) * 100
IV Pctile = % of days in past year where VIX was BELOW today
VRP      = VIX - 30d realized vol of SOXL (annualized %)
"""

import warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import yfinance as yf
from pathlib import Path


def fetch_vix(period="2y"):
    """Fetch real VIX implied volatility from Yahoo Finance."""
    raw = yf.download("^VIX", period=period, progress=False)
    # Flatten MultiIndex columns if present
    raw.columns = ["_".join(c).strip("_") if isinstance(c, tuple) else c
                   for c in raw.columns]
    # Find whichever column contains Close
    close_col = next(c for c in raw.columns if "Close" in c)
    vix = raw[[close_col]].copy()
    vix.columns = ["VIX"]
    vix.index = pd.to_datetime(vix.index).tz_localize(None)
    return vix.dropna()


def compute_soxl_realized_vol(period="2y"):
    """Fetch SOXL prices and compute 30-day realized vol (annualized %)."""
    data = yf.Ticker("SOXL").history(period=period)
    data.index = pd.to_datetime(data.index).tz_localize(None)
    data["log_ret"] = np.log(data["Close"] / data["Close"].shift(1))
    data["rv30"] = data["log_ret"].rolling(30).std() * np.sqrt(252) * 100
    return data[["Close", "rv30"]].dropna()


def compute_iv_rank(iv_series: pd.Series, window: int = 252) -> pd.Series:
    """Rolling IV Rank over `window` trading days. Returns 0-100."""
    roll_min = iv_series.rolling(window).min()
    roll_max = iv_series.rolling(window).max()
    return (iv_series - roll_min) / (roll_max - roll_min) * 100


def compute_iv_percentile(iv_series: pd.Series, window: int = 252) -> pd.Series:
    """Rolling IV Percentile over `window` trading days. Returns 0-100."""
    def pctile(x):
        return (x < x.iloc[-1]).sum() / len(x) * 100
    return iv_series.rolling(window).apply(pctile, raw=False)


def get_current_signal(ticker_symbol="SOXL"):
    """
    Returns today's IV Rank, IV Percentile, VRP, and trade signal.
    Uses real VIX data instead of realized vol proxy.
    """
    vix  = fetch_vix(period="2y")
    soxl = compute_soxl_realized_vol(period="2y")

    combined = vix.join(soxl[["rv30"]], how="inner").dropna()

    iv_series  = combined["VIX"]
    iv_rank    = compute_iv_rank(iv_series)
    iv_pctile  = compute_iv_percentile(iv_series)
    vrp        = combined["VIX"] - combined["rv30"]

    current_vix    = float(iv_series.iloc[-1])
    current_rv     = float(combined["rv30"].iloc[-1])
    current_rank   = float(iv_rank.iloc[-1])
    current_pctile = float(iv_pctile.iloc[-1])
    current_vrp    = float(vrp.iloc[-1])

    if current_rank >= 50:
        signal = "SELL PUTS  ✅"
        rationale = (f"IV Rank {current_rank:.1f} — VIX in top "
                     f"{100 - current_rank:.0f}% of annual range. Premium elevated.")
    elif current_rank >= 30:
        signal = "NEUTRAL  ⚠️"
        rationale = f"IV Rank {current_rank:.1f} — middling volatility."
    else:
        signal = "AVOID SELLING  ❌"
        rationale = f"IV Rank {current_rank:.1f} — VIX is low, premium is thin."

    vrp_note = "✅ Sellers have edge" if current_vrp > 0 else "❌ Vol cheap vs realized"

    print("\n" + "="*55)
    print(f"  SOXL IV SIGNAL (VIX-based) — {combined.index[-1].date()}")
    print("="*55)
    print(f"  VIX (real IV):        {current_vix:.1f}")
    print(f"  SOXL Realized Vol:    {current_rv:.1f}%")
    print(f"  VRP (VIX - RV):       {current_vrp:+.1f}  {vrp_note}")
    print(f"  IV Rank (1Y):         {current_rank:.1f} / 100")
    print(f"  IV Percentile (1Y):   {current_pctile:.1f} / 100")
    print(f"  Signal:               {signal}")
    print(f"\n  {rationale}")
    print("="*55)

    return {
        "date":        combined.index[-1].date(),
        "vix":         current_vix,
        "rv30":        current_rv,
        "vrp":         current_vrp,
        "iv_rank":     current_rank,
        "iv_pctile":   current_pctile,
        "signal":      signal,
    }


def plot_iv_rank_history(ticker_symbol="SOXL", save=True):
    """4-panel dashboard: VIX, Realized Vol, IV Rank, VRP."""
    vix  = fetch_vix(period="2y")
    soxl = compute_soxl_realized_vol(period="2y")

    combined  = vix.join(soxl[["rv30"]], how="inner").dropna()
    iv_series = combined["VIX"]
    iv_rank   = compute_iv_rank(iv_series)
    iv_pctile = compute_iv_percentile(iv_series)
    vrp       = combined["VIX"] - combined["rv30"]

    fig, axes = plt.subplots(4, 1, figsize=(13, 14), sharex=True)
    fig.suptitle("SOXL Volatility Dashboard (VIX-Based)",
                 fontsize=14, fontweight="bold", y=0.99)

    # Panel 1: VIX vs Realized Vol
    axes[0].plot(combined.index, combined["VIX"],  color="#A32D2D", linewidth=1.5, label="VIX (implied)")
    axes[0].plot(combined.index, combined["rv30"], color="#378ADD", linewidth=1.5, label="SOXL Realized Vol (30d)")
    axes[0].set_ylabel("Volatility (%)")
    axes[0].set_title("VIX vs SOXL Realized Volatility", fontsize=11)
    axes[0].legend(fontsize=9)
    axes[0].grid(True, alpha=0.25)

    # Panel 2: VRP
    colors = ["#1D9E75" if v >= 0 else "#A32D2D" for v in vrp]
    axes[1].bar(vrp.index, vrp, color=colors, alpha=0.6, width=1)
    axes[1].axhline(0, color="black", linewidth=0.8)
    axes[1].set_ylabel("VRP (VIX - RV)")
    axes[1].set_title("Variance Risk Premium — Positive = Put Sellers Have Edge", fontsize=11)
    axes[1].grid(True, alpha=0.25)

    # Panel 3: IV Rank
    axes[2].plot(combined.index, iv_rank, color="#A32D2D", linewidth=1.5)
    axes[2].axhline(50, color="black",  linestyle="--", alpha=0.4, label="Sell zone (50)")
    axes[2].axhline(30, color="orange", linestyle="--", alpha=0.4, label="Neutral (30)")
    axes[2].fill_between(combined.index, iv_rank, 50,
                         where=(iv_rank >= 50), alpha=0.15, color="#1D9E75", label="Sell puts region")
    axes[2].set_ylabel("IV Rank (0-100)")
    axes[2].set_title("IV Rank — 252-Day Rolling (VIX-Based)", fontsize=11)
    axes[2].set_ylim(0, 100)
    axes[2].legend(fontsize=8, loc="upper left")
    axes[2].grid(True, alpha=0.25)

    # Panel 4: IV Percentile
    axes[3].plot(combined.index, iv_pctile, color="#8A4FD8", linewidth=1.5)
    axes[3].axhline(50, color="black", linestyle="--", alpha=0.4)
    axes[3].set_ylabel("IV Percentile (0-100)")
    axes[3].set_title("IV Percentile — 252-Day Rolling", fontsize=11)
    axes[3].set_ylim(0, 100)
    axes[3].grid(True, alpha=0.25)

    axes[3].xaxis.set_major_formatter(mdates.DateFormatter("%b %Y"))
    axes[3].xaxis.set_major_locator(mdates.MonthLocator(interval=2))
    plt.setp(axes[3].xaxis.get_majorticklabels(), rotation=30, ha="right")

    plt.tight_layout()
    if save:
        Path("data").mkdir(exist_ok=True)
        plt.savefig("data/iv_rank_history.png", dpi=150, bbox_inches="tight")
        print("Saved: data/iv_rank_history.png")


if __name__ == "__main__":
    get_current_signal()
    plot_iv_rank_history()
# Alias for backward compatibility with backtest.py
def compute_historical_iv(ticker_symbol="SOXL", period="2y"):
    return compute_soxl_realized_vol(period=period)

