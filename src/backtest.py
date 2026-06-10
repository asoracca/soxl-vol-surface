"""
backtest.py
-----------
Simple backtest of a "sell puts when IV Rank > 50" strategy on SOXL.

Strategy rules:
  - Entry:  IV Rank crosses above 50 → sell a 30-DTE, 15%-OTM cash-secured put
  - Exit:   Hold to expiration (30 days later)
  - Profit: Full premium if SOXL closes above strike at expiration
  - Loss:   (Strike - Close) - Premium if SOXL closes below strike
  - Premium approximated as: ATM_IV * Strike * sqrt(DTE/252) * 0.4
    (simplified Black-Scholes estimate for OTM puts)

This is a simplified backtest — real fills, commissions, and margin are
not modeled. Treat as directional signal validation, not exact P&L.
"""

import warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import yfinance as yf
from pathlib import Path

from src.iv_rank import compute_historical_iv, compute_iv_rank


# ── Core Backtest ─────────────────────────────────────────────────────────────

def run_backtest(
    ticker_symbol: str = "SOXL",
    otm_pct: float = 0.15,        # how far OTM the put strike is (15%)
    dte_entry: int = 30,          # DTE at entry
    iv_rank_threshold: float = 50 # min IV rank to sell
) -> pd.DataFrame:
    """
    Returns a DataFrame of all trades with entry/exit dates and P&L.
    """
    data = compute_historical_iv(ticker_symbol, period="3y")
    iv_series = data['rv30']
    iv_rank = compute_iv_rank(iv_series, window=252)

    data['iv_rank'] = iv_rank
    data['signal'] = (data['iv_rank'] >= iv_rank_threshold).astype(int)
    # Only enter on first day of a new signal (avoid re-entering every day)
    data['new_signal'] = (data['signal'] == 1) & (data['signal'].shift(1) == 0)

    trades = []
    close_prices = data['Close']

    for entry_date in data[data['new_signal']].index:
        # Get entry price
        entry_price = close_prices.loc[entry_date]
        strike = entry_price * (1 - otm_pct)

        # Approximate premium (simplified BS estimate for OTM put)
        iv_at_entry = iv_series.loc[entry_date]
        premium = iv_at_entry * strike * np.sqrt(dte_entry / 252) * 0.40

        # Find exit date ~30 calendar days later
        future_dates = close_prices.index[close_prices.index > entry_date]
        if len(future_dates) < dte_entry:
            continue

        # Closest trading day to 30 calendar days out
        target_exit = entry_date + pd.Timedelta(days=dte_entry)
        exit_date = future_dates[future_dates >= target_exit][0] \
            if any(future_dates >= target_exit) else future_dates[-1]

        exit_price = close_prices.loc[exit_date]

        # P&L
        if exit_price >= strike:
            pnl = premium           # put expires worthless, keep full premium
            outcome = "WIN"
        else:
            pnl = premium - (strike - exit_price)  # assigned
            outcome = "LOSS"

        trades.append({
            "entry_date":   entry_date.date(),
            "exit_date":    exit_date.date(),
            "entry_price":  round(entry_price, 2),
            "strike":       round(strike, 2),
            "premium":      round(premium, 2),
            "exit_price":   round(exit_price, 2),
            "pnl":          round(pnl, 2),
            "iv_rank":      round(data.loc[entry_date, 'iv_rank'], 1),
            "outcome":      outcome,
        })

    return pd.DataFrame(trades)


# ── Analytics ─────────────────────────────────────────────────────────────────

def print_backtest_stats(trades: pd.DataFrame):
    if trades.empty:
        print("No trades generated.")
        return

    wins  = trades[trades['outcome'] == 'WIN']
    loss  = trades[trades['outcome'] == 'LOSS']
    total_pnl = trades['pnl'].sum()

    print("\n" + "="*50)
    print("  BACKTEST RESULTS — Sell Puts @ IV Rank > 50")
    print("="*50)
    print(f"  Total trades:        {len(trades)}")
    print(f"  Win rate:            {len(wins)/len(trades):.1%}")
    print(f"  Avg P&L per trade:   ${trades['pnl'].mean():.2f}")
    print(f"  Total P&L:           ${total_pnl:.2f}")
    print(f"  Best trade:          ${trades['pnl'].max():.2f}")
    print(f"  Worst trade:         ${trades['pnl'].min():.2f}")
    print(f"  Avg premium:         ${trades['premium'].mean():.2f}")
    print(f"  Avg IV Rank at entry:{trades['iv_rank'].mean():.1f}")
    print("="*50)

    print("\nAll trades:")
    print(trades.to_string(index=False))


# ── Plots ─────────────────────────────────────────────────────────────────────

def plot_backtest(trades: pd.DataFrame, save=True):
    if trades.empty:
        print("No trades to plot.")
        return

    trades = trades.copy()
    trades['entry_date'] = pd.to_datetime(trades['entry_date'])
    trades['cumulative_pnl'] = trades['pnl'].cumsum()

    fig, axes = plt.subplots(2, 2, figsize=(14, 9))
    fig.suptitle('SOXL Put Selling Backtest (IV Rank > 50 Entry)',
                 fontsize=14, fontweight='bold')

    # ── Cumulative P&L ───────────────────────────────────────────────────────
    axes[0, 0].plot(trades['entry_date'], trades['cumulative_pnl'],
                    'o-', color='#1D9E75', linewidth=2, markersize=5)
    axes[0, 0].fill_between(trades['entry_date'], trades['cumulative_pnl'],
                             alpha=0.15, color='#1D9E75')
    axes[0, 0].axhline(0, color='black', linestyle='--', alpha=0.3)
    axes[0, 0].set_title('Cumulative P&L ($)')
    axes[0, 0].set_ylabel('P&L ($)')
    axes[0, 0].grid(True, alpha=0.25)

    # ── Per-Trade P&L ────────────────────────────────────────────────────────
    colors = ['#1D9E75' if o == 'WIN' else '#A32D2D' for o in trades['outcome']]
    axes[0, 1].bar(range(len(trades)), trades['pnl'], color=colors, alpha=0.8)
    axes[0, 1].axhline(0, color='black', linestyle='--', alpha=0.3)
    axes[0, 1].set_title('Per-Trade P&L ($)')
    axes[0, 1].set_xlabel('Trade #')
    axes[0, 1].set_ylabel('P&L ($)')
    axes[0, 1].grid(True, alpha=0.25)

    # ── Win/Loss Pie ─────────────────────────────────────────────────────────
    win_count  = (trades['outcome'] == 'WIN').sum()
    loss_count = (trades['outcome'] == 'LOSS').sum()
    axes[1, 0].pie(
        [win_count, loss_count],
        labels=[f'WIN ({win_count})', f'LOSS ({loss_count})'],
        colors=['#1D9E75', '#A32D2D'],
        autopct='%1.0f%%', startangle=90,
        textprops={'fontsize': 11}
    )
    axes[1, 0].set_title('Win / Loss Rate')

    # ── IV Rank at Entry Histogram ───────────────────────────────────────────
    axes[1, 1].hist(trades['iv_rank'], bins=15, color='#378ADD', alpha=0.75,
                    edgecolor='white')
    axes[1, 1].axvline(trades['iv_rank'].mean(), color='red',
                       linestyle='--', label=f"Mean: {trades['iv_rank'].mean():.0f}")
    axes[1, 1].set_title('IV Rank at Entry')
    axes[1, 1].set_xlabel('IV Rank')
    axes[1, 1].set_ylabel('Frequency')
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.25)

    plt.tight_layout()
    if save:
        Path("data").mkdir(exist_ok=True)
        plt.savefig("data/backtest_results.png", dpi=150, bbox_inches='tight')
        print("Saved: data/backtest_results.png")
    #plt.show()


if __name__ == "__main__":
    print("Running backtest...")
    trades = run_backtest()
    print_backtest_stats(trades)
    plot_backtest(trades)