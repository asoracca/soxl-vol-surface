"""
walk_forward.py
---------------
Runs the walk-forward backtest and prints results.
Unlike the regular backtest, this only counts trades from TEST periods —
data the strategy never saw during training.
"""
import pandas as pd
import numpy as np
from src.backtest import walk_forward_backtest, stress_test, monte_carlo_significance


def print_walk_forward_stats(trades):
    if trades.empty:
        print("No trades generated in walk-forward test.")
        return

    wins = trades[trades['outcome'] == 'WIN']
    loss = trades[trades['outcome'] == 'LOSS']
    win_rate = len(wins) / len(trades)
    total_pnl = trades['pnl'].sum()
    avg_win = wins['pnl'].mean() if len(wins) > 0 else 0
    avg_loss = loss['pnl'].mean() if len(loss) > 0 else 0
    pnl_vals = trades['pnl'].values
    sharpe = (pnl_vals.mean() / pnl_vals.std() * np.sqrt(252/7)) if pnl_vals.std() > 0 else 0
    cumulative = trades['pnl'].cumsum()
    max_dd = (cumulative - cumulative.cummax()).min()

    print("\n" + "="*60)
    print("  WALK-FORWARD BACKTEST RESULTS")
    print("  (Only out-of-sample test periods counted)")
    print("="*60)
    print(f"  Total trades:     {len(trades)}")
    print(f"  Win rate:         {win_rate:.1%}")
    print(f"  Avg win:          ${avg_win:.2f}")
    print(f"  Avg loss:         ${avg_loss:.2f}")
    print(f"  Sharpe ratio:     {sharpe:.2f}")
    print(f"  Max drawdown:     ${max_dd:.2f}")
    print(f"  Total P&L:        ${total_pnl:.2f}")
    print("="*60)

    print(f"\n  Results by test period:")
    for period, group in trades.groupby('period'):
        w = (group['outcome'] == 'WIN').sum()
        print(f"  {period}  |  trades: {len(group)}  win: {w/len(group):.0%}  P&L: ${group['pnl'].sum():.2f}")

    print(f"\n  Interpretation:")
    if win_rate >= 0.6 and sharpe >= 1.0:
        print("  ✅ Strategy holds up out-of-sample — edge looks real")
    elif win_rate >= 0.5:
        print("  ⚠️  Marginal — edge exists but is weak out-of-sample")
    else:
        print("  ❌ Strategy fails out-of-sample — likely overfit in-sample")


if __name__ == "__main__":
    print("Running walk-forward backtest (train 12mo, test 3mo rolling)...")
    trades = walk_forward_backtest(train_months=12, test_months=3)
    print_walk_forward_stats(trades)
    stress_test(trades)
    monte_carlo_significance(trades)
