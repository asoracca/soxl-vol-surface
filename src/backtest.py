"""
backtest.py - SOXL put selling backtest with BS pricing and full metrics
"""
import warnings
warnings.filterwarnings('ignore')
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pathlib import Path
from src.iv_rank import compute_historical_iv, compute_iv_rank
from src.pricing import black_scholes_put

def run_backtest(ticker_symbol="SOXL", otm_pct=0.15, dte_entry=7, iv_rank_threshold=30):
    data = compute_historical_iv(ticker_symbol, period="2y")
    iv_series = data['rv30']
    iv_rank = compute_iv_rank(iv_series, window=252)
    data['iv_rank'] = iv_rank
    data['signal'] = (data['iv_rank'] >= iv_rank_threshold).astype(int)
    data['new_signal'] = (data['signal'] == 1) & (data['signal'].shift(1) == 0)
    trades = []
    close_prices = data['Close']
    for entry_date in data[data['new_signal']].index:
        entry_price = float(close_prices.loc[entry_date])
        strike = entry_price * (1 - otm_pct)
        T = dte_entry / 252
        iv_at_entry = float(iv_series.loc[entry_date])
        bs = black_scholes_put(S=entry_price, K=strike, T=T, sigma=iv_at_entry)
        mid_premium = bs['price']
        premium = max(0, mid_premium * 0.90 - 0.65)
        future_dates = close_prices.index[close_prices.index > entry_date]
        if len(future_dates) < dte_entry:
            continue
        target_exit = entry_date + pd.Timedelta(days=dte_entry)
        exit_date = (future_dates[future_dates >= target_exit][0]
                     if any(future_dates >= target_exit) else future_dates[-1])
        exit_price = float(close_prices.loc[exit_date])
        if exit_price >= strike:
            pnl = premium
            outcome = "WIN"
        else:
            pnl = premium - (strike - exit_price)
            outcome = "LOSS"
        trades.append({
            "entry_date": entry_date.date(),
            "exit_date": exit_date.date(),
            "entry_price": round(entry_price, 2),
            "strike": round(strike, 2),
            "premium": round(premium, 2),
            "exit_price": round(exit_price, 2),
            "pnl": round(pnl, 2),
            "iv_rank": round(float(data.loc[entry_date, 'iv_rank']), 1),
            "outcome": outcome,
        })
    return pd.DataFrame(trades)

def print_backtest_stats(trades):
    if trades.empty:
        print("No trades generated.")
        return
    wins = trades[trades['outcome'] == 'WIN']
    loss = trades[trades['outcome'] == 'LOSS']
    win_rate = len(wins) / len(trades)
    avg_win = wins['pnl'].mean() if len(wins) > 0 else 0
    avg_loss = loss['pnl'].mean() if len(loss) > 0 else 0
    profit_factor = (wins['pnl'].sum() / abs(loss['pnl'].sum())
                     if loss['pnl'].sum() != 0 else float('inf'))
    trades = trades.copy()
    trades['return_pct'] = trades['pnl'] / (trades['strike'] * 100)
    sharpe = (trades['return_pct'].mean() / trades['return_pct'].std() *
              np.sqrt(252 / 7)) if trades['return_pct'].std() > 0 else 0
    cumulative = trades['pnl'].cumsum()
    max_drawdown = (cumulative - cumulative.cummax()).min()
    ev = (win_rate * avg_win) + ((1 - win_rate) * avg_loss)
    print("\n" + "="*55)
    print("  BACKTEST RESULTS — Sell Puts @ IV Rank > 30, 7DTE")
    print("="*55)
    print(f"  Total trades:         {len(trades)}")
    print(f"  Win rate:             {win_rate:.1%}")
    print(f"  Avg win:              ${avg_win:.2f}")
    print(f"  Avg loss:             ${avg_loss:.2f}")
    print(f"  Expected value/trade: ${ev:.2f}")
    print(f"  Profit factor:        {profit_factor:.2f}")
    print(f"  Sharpe ratio:         {sharpe:.2f}")
    print(f"  Max drawdown:         ${max_drawdown:.2f}")
    print(f"  Total P&L:            ${trades['pnl'].sum():.2f}")
    print("="*55)
    if sharpe >= 2.0:
        print("  Sharpe > 2.0 — excellent risk-adjusted returns")
    elif sharpe >= 1.0:
        print("  Sharpe > 1.0 — respectable, institutional quality")
    elif sharpe >= 0.5:
        print("  Sharpe 0.5-1.0 — positive edge but needs work")
    else:
        print("  Sharpe < 0.5 — weak edge, review parameters")
    print(f"\nAll trades:")
    print(trades[['entry_date','exit_date','strike','premium','exit_price','pnl','iv_rank','outcome']].to_string(index=False))

def plot_backtest(trades, save=True):
    if trades.empty:
        return
    trades = trades.copy()
    trades['entry_date'] = pd.to_datetime(trades['entry_date'])
    trades['cumulative_pnl'] = trades['pnl'].cumsum()
    fig, axes = plt.subplots(2, 2, figsize=(14, 9))
    fig.suptitle('SOXL Put Selling Backtest (IV Rank > 30, 7DTE)', fontsize=14, fontweight='bold')
    axes[0,0].plot(trades['entry_date'], trades['cumulative_pnl'], 'o-', color='#1D9E75', linewidth=2, markersize=4)
    axes[0,0].fill_between(trades['entry_date'], trades['cumulative_pnl'], alpha=0.15, color='#1D9E75')
    axes[0,0].axhline(0, color='black', linestyle='--', alpha=0.3)
    axes[0,0].set_title('Cumulative P&L ($)')
    axes[0,0].grid(True, alpha=0.25)
    colors = ['#1D9E75' if o == 'WIN' else '#A32D2D' for o in trades['outcome']]
    axes[0,1].bar(range(len(trades)), trades['pnl'], color=colors, alpha=0.8)
    axes[0,1].axhline(0, color='black', linestyle='--', alpha=0.3)
    axes[0,1].set_title('Per-Trade P&L ($)')
    axes[0,1].grid(True, alpha=0.25)
    win_count = (trades['outcome'] == 'WIN').sum()
    loss_count = (trades['outcome'] == 'LOSS').sum()
    axes[1,0].pie([win_count, loss_count], labels=[f'WIN ({win_count})', f'LOSS ({loss_count})'],
                  colors=['#1D9E75', '#A32D2D'], autopct='%1.0f%%', startangle=90)
    axes[1,0].set_title('Win / Loss Rate')
    axes[1,1].hist(trades['iv_rank'], bins=15, color='#378ADD', alpha=0.75, edgecolor='white')
    axes[1,1].axvline(trades['iv_rank'].mean(), color='red', linestyle='--', label=f"Mean: {trades['iv_rank'].mean():.0f}")
    axes[1,1].set_title('IV Rank at Entry')
    axes[1,1].legend()
    axes[1,1].grid(True, alpha=0.25)
    plt.tight_layout()
    if save:
        Path("data").mkdir(exist_ok=True)
        plt.savefig("data/backtest_results.png", dpi=150, bbox_inches='tight')
        print("Saved: data/backtest_results.png")

if __name__ == "__main__":
    trades = run_backtest()
    print_backtest_stats(trades)
    plot_backtest(trades)
