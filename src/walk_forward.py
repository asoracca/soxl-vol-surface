"""
walk_forward.py
---------------
Walk-forward backtest, Kelly sizing, stress test, Monte Carlo.
"""
import numpy as np
import pandas as pd
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


def kelly_sizing(trades):
    """
    Kelly Criterion: how much of your account to risk per trade.
    Formula: f = W - (1-W)/R
    W = win rate, R = avg win / avg loss ratio
    Half-Kelly is standard practice (safer).
    """
    if trades.empty:
        return

    wins = trades[trades['outcome'] == 'WIN']
    loss = trades[trades['outcome'] == 'LOSS']

    if len(loss) == 0 or len(wins) == 0:
        print("\n── Kelly Criterion ─────────────────────────────────")
        print("  Need both wins and losses to compute Kelly.")
        return

    W = len(wins) / len(trades)
    avg_win  = wins['pnl'].mean()
    avg_loss = abs(loss['pnl'].mean())
    R = avg_win / avg_loss

    kelly      = W - (1 - W) / R
    half_kelly = kelly / 2

    print("\n── Kelly Criterion Position Sizing ────────────────")
    print(f"  Win rate (W):        {W:.1%}")
    print(f"  Avg win:             ${avg_win:.2f}")
    print(f"  Avg loss:            ${avg_loss:.2f}")
    print(f"  Win/loss ratio (R):  {R:.2f}")
    print(f"  Full Kelly:          {kelly:.1%} of account per trade")
    print(f"  Half Kelly (safer):  {half_kelly:.1%} of account per trade")

    account = 19941  # your real portfolio value
    print(f"\n  On your ${account:,} portfolio:")
    print(f"  Full Kelly bet:      ${account * kelly:,.0f} per trade")
    print(f"  Half Kelly bet:      ${account * half_kelly:,.0f} per trade")

    if kelly <= 0:
        print(f"\n  ❌ Kelly = {kelly:.1%} — negative edge, don't trade this")
    elif kelly > 0.25:
        print(f"\n  ⚠️  Full Kelly is high — always use Half Kelly in practice")
    else:
        print(f"\n  ✅ Reasonable sizing — Half Kelly recommended")


if __name__ == "__main__":
    print("Running walk-forward backtest (train 12mo, test 3mo rolling)...")
    trades = walk_forward_backtest(train_months=12, test_months=3)
    print_walk_forward_stats(trades)
    stress_test(trades)
    monte_carlo_significance(trades)
    kelly_sizing(trades)