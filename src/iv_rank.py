"""
iv_rank.py
----------
Computes IV Rank and IV Percentile for SOXL — the two most commonly used
signals in options trading to assess whether implied volatility is
historically "cheap" or "expensive".

IV Rank  = (current IV - 52w low) / (52w high - 52w low)  * 100
IV Pctile = % of days in the past year where IV was BELOW today's IV

Interpretation for put sellers:
  IV Rank > 50  →  IV is in the upper half of its annual range → sell premium
  IV Rank < 30  →  IV is cheap → avoid selling, consider buying instead
"""

import warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import yfinance as yf
from pathlib import Path


# ── Helpers ──────────────────────────────────────────────────────────────────

def compute_historical_iv(ticker_symbol="SOXL", period="2y"):
    """
    Approximate historical IV using 30-day rolling realized volatility
    annualized. Real IV requires option history (expensive data), so we
    use realized vol as a practical proxy for backtesting.
    """
    ticker = yf.Ticker(ticker_symbol)
    data = ticker.history(period=period)
    data['log_ret'] = np.log(data['Close'] / data['Close'].shift(1))

    # 30-day rolling realized vol, annualized
    data['rv30'] = data['log_ret'].rolling(30).std() * np.sqrt(252)
    data.dropna(inplace=True)
    return data


def compute_iv_rank(iv_series: pd.Series, window: int = 252) -> pd.Series:
    """
    Rolling IV Rank over `window` trading days.
    Returns a 0–100 score.
    """
    roll_min = iv_series.rolling(window).min()
    roll_max = iv_series.rolling(window).max()
    iv_rank = (iv_series - roll_min) / (roll_max - roll_min) * 100
    return iv_rank


def compute_iv_percentile(iv_series: pd.Series, window: int = 252) -> pd.Series:
    """
    Rolling IV Percentile over `window` trading days.
    Returns a 0–100 score.
    """
    def pctile(x):
        return (x < x.iloc[-1]).sum() / len(x) * 100

    return iv_series.rolling(window).apply(pctile, raw=False)


# ── Current Signal ────────────────────────────────────────────────────────────

def get_current_signal(ticker_symbol="SOXL"):
    """
    Returns today's IV Rank, IV Percentile, and a plain-English signal.
    """
    data = compute_historical_iv(ticker_symbol, period="2y")
    iv_series = data['rv30']

    iv_rank    = compute_iv_rank(iv_series)
    iv_pctile  = compute_iv_percentile(iv_series)

    current_iv     = iv_series.iloc[-1]
    current_rank   = iv_rank.iloc[-1]
    current_pctile = iv_pctile.iloc[-1]

    # Signal logic
    if current_rank >= 50:
        signal = "SELL PUTS  ✅"
        rationale = (f"IV Rank {current_rank:.1f} — volatility is in the TOP "
                     f"{100 - current_rank:.0f}% of its annual range. "
                     f"Premium is elevated; good time to collect it.")
    elif current_rank >= 30:
        signal = "NEUTRAL  ⚠️"
        rationale = (f"IV Rank {current_rank:.1f} — volatility is middling. "
                     f"Selling puts is acceptable but not optimal.")
    else:
        signal = "AVOID SELLING  ❌"
        rationale = (f"IV Rank {current_rank:.1f} — volatility is LOW. "
                     f"Premium is thin; risk/reward favors buyers not sellers.")

    print("\n" + "="*50)
    print(f"  SOXL IV SIGNAL — {data.index[-1].date()}")
    print("="*50)
    print(f"  Realized Vol (30d):  {current_iv:.1%}")
    print(f"  IV Rank (1Y):        {current_rank:.1f} / 100")
    print(f"  IV Percentile (1Y):  {current_pctile:.1f} / 100")
    print(f"  Signal:              {signal}")
    print(f"\n  {rationale}")
    print("="*50)

    return {
        "date":        data.index[-1].date(),
        "rv30":        current_iv,
        "iv_rank":     current_rank,
        "iv_pctile":   current_pctile,
        "signal":      signal,
    }


# ── Plots ─────────────────────────────────────────────────────────────────────

def plot_iv_rank_history(ticker_symbol="SOXL", save=True):
    data = compute_historical_iv(ticker_symbol, period="2y")
    iv_series = data['rv30']

    iv_rank   = compute_iv_rank(iv_series)
    iv_pctile = compute_iv_percentile(iv_series)

    fig, axes = plt.subplots(3, 1, figsize=(13, 11), sharex=True)
    fig.suptitle(f'{ticker_symbol} Implied Volatility Dashboard',
                 fontsize=14, fontweight='bold', y=0.98)

    # ── Panel 1: Realized Vol ────────────────────────────────────────────────
    axes[0].plot(data.index, iv_series * 100, color='#378ADD', linewidth=1.5)
    axes[0].fill_between(data.index, iv_series * 100, alpha=0.15, color='#378ADD')
    axes[0].set_ylabel('Realized Vol (%)')
    axes[0].set_title('30-Day Realized Volatility (Annualized)', fontsize=11)
    axes[0].grid(True, alpha=0.25)

    # ── Panel 2: IV Rank ─────────────────────────────────────────────────────
    axes[1].plot(data.index, iv_rank, color='#A32D2D', linewidth=1.5)
    axes[1].axhline(50, color='black',  linestyle='--', alpha=0.4, label='Sell zone (50)')
    axes[1].axhline(30, color='orange', linestyle='--', alpha=0.4, label='Neutral (30)')
    axes[1].fill_between(data.index, iv_rank, 50,
                         where=(iv_rank >= 50), alpha=0.15, color='#1D9E75',
                         label='Sell puts region')
    axes[1].fill_between(data.index, iv_rank, 30,
                         where=(iv_rank <= 30), alpha=0.15, color='#A32D2D',
                         label='Avoid selling region')
    axes[1].set_ylabel('IV Rank (0–100)')
    axes[1].set_title('IV Rank — 252-Day Rolling', fontsize=11)
    axes[1].set_ylim(0, 100)
    axes[1].legend(fontsize=8, loc='upper left')
    axes[1].grid(True, alpha=0.25)

    # ── Panel 3: IV Percentile ───────────────────────────────────────────────
    axes[2].plot(data.index, iv_pctile, color='#8A4FD8', linewidth=1.5)
    axes[2].axhline(50, color='black', linestyle='--', alpha=0.4)
    axes[2].set_ylabel('IV Percentile (0–100)')
    axes[2].set_title('IV Percentile — 252-Day Rolling', fontsize=11)
    axes[2].set_ylim(0, 100)
    axes[2].grid(True, alpha=0.25)

    axes[2].xaxis.set_major_formatter(mdates.DateFormatter('%b %Y'))
    axes[2].xaxis.set_major_locator(mdates.MonthLocator(interval=2))
    plt.setp(axes[2].xaxis.get_majorticklabels(), rotation=30, ha='right')

    plt.tight_layout()
    if save:
        Path("data").mkdir(exist_ok=True)
        plt.savefig("data/iv_rank_history.png", dpi=150, bbox_inches='tight')
        print("Saved: data/iv_rank_history.png")
    #plt.show()


if __name__ == "__main__":
    get_current_signal()
    plot_iv_rank_history()