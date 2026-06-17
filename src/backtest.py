"""
backtest.py - SOXL put selling backtest with BS pricing and full metrics
"""
import warnings
warnings.filterwarnings('ignore')
import numpy as np
import pandas as pd
import yfinance as yf
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
def walk_forward_backtest(ticker_symbol="SOXL", 
                          train_months=12, 
                          test_months=3):
    """
    Walk-forward validation:
    Train on 12 months, test on next 3, roll forward.
    Only trades generated in TEST period count.
    This simulates what you would have actually known at each point.
    """
    from src.iv_rank import compute_soxl_scaled_iv
    combined = compute_soxl_scaled_iv(period="5y")
    data = yf.Ticker(ticker_symbol).history(period="5y")
    data.index = pd.to_datetime(data.index).tz_localize(None)
    data["soxl_iv"] = combined["soxl_iv"].reindex(data.index).ffill()
    data["rv30"] = combined["soxl_rv"].reindex(data.index).ffill()
    data.dropna(subset=["soxl_iv"], inplace=True)
    iv_series = data["soxl_iv"]
    iv_rank = compute_iv_rank(iv_series, window=252)
    data['iv_rank'] = iv_rank
    
    all_test_trades = []
    start_idx = train_months * 21  # ~21 trading days per month
    
    while start_idx + (test_months * 21) < len(data):
        # Test window — only trades here count
        test_start = data.index[start_idx]
        test_end   = data.index[min(start_idx + test_months * 21, 
                                    len(data) - 1)]
        
        # Only look at test period
        test_data = data[(data.index >= test_start) & 
                         (data.index <= test_end)]
        
        # Run trades on test period only
        test_data = test_data.copy()
        test_data['signal'] = (test_data['iv_rank'] >= 30).astype(int)
        test_data['new_signal'] = ((test_data['signal'] == 1) & 
                                    (test_data['signal'].shift(1) == 0))
        
        for entry_date in test_data[test_data['new_signal']].index:
            entry_price = float(data['Close'].loc[entry_date])
            strike = entry_price * 0.85
            T = 7 / 252
            iv_at_entry = float(iv_series.loc[entry_date])
            
            from src.pricing import black_scholes_put
            bs = black_scholes_put(S=entry_price, K=strike, T=T, 
                                   sigma=iv_at_entry)
            premium = max(0, bs['price'] * 0.90 - 0.65)
            
            future = data['Close'].index[data['Close'].index > entry_date]
            if len(future) < 7:
                continue
                
            target = entry_date + pd.Timedelta(days=7)
            exit_date = (future[future >= target][0] 
                        if any(future >= target) else future[-1])
            exit_price = float(data['Close'].loc[exit_date])
            
            pnl = premium if exit_price >= strike else premium - (strike - exit_price)
            
            all_test_trades.append({
                'entry_date': entry_date.date(),
                'exit_date':  exit_date.date(),
                'period':     f"{test_start.date()} to {test_end.date()}",
                'strike':     round(strike, 2),
                'premium':    round(premium, 2),
                'exit_price': round(exit_price, 2),
                'pnl':        round(pnl, 2),
                'iv_rank':    round(float(iv_rank.loc[entry_date]), 1),
                'outcome':    'WIN' if exit_price >= strike else 'LOSS'
            })
        
        start_idx += test_months * 21  # roll forward
    
    return pd.DataFrame(all_test_trades)
def stress_test(trades_df):
    """
    How did the strategy perform during known market crashes?
    If it fails in every crash, it's not robust.
    """
    crashes = {
        'COVID crash (Mar 2020)':   ('2020-02-15', '2020-04-01'),
        'Rate hike selloff (2022)': ('2022-01-01', '2022-12-31'),
        'SOXL crash (Mar 2026)':    ('2026-02-15', '2026-04-15'),
    }
    
    print("\n── Stress Test by Market Regime ──────────────────")
    trades_df = trades_df.copy()
    trades_df['entry_date'] = pd.to_datetime(trades_df['entry_date'])
    
    for event, (start, end) in crashes.items():
        mask = ((trades_df['entry_date'] >= start) & 
                (trades_df['entry_date'] <= end))
        period_trades = trades_df[mask]
        
        if len(period_trades) == 0:
            print(f"  {event}: No trades")
            continue
            
        wins = (period_trades['outcome'] == 'WIN').sum()
        total = len(period_trades)
        pnl = period_trades['pnl'].sum()
        
        print(f"  {event}:")
        print(f"    Trades: {total}, Win rate: {wins/total:.0%}, P&L: ${pnl:.2f}")
def monte_carlo_significance(trades_df, n_simulations=10000):
    """
    Randomly shuffle win/loss outcomes and compute Sharpe distribution.
    If your real Sharpe beats 95% of random shuffles, 
    the edge is likely real (p < 0.05).
    """
    if trades_df.empty:
        return
        
    real_pnl = trades_df['pnl'].values
    real_sharpe = (real_pnl.mean() / real_pnl.std() * 
                   np.sqrt(252/7)) if real_pnl.std() > 0 else 0
    
    random_sharpes = []
    for _ in range(n_simulations):
        shuffled = np.random.permutation(real_pnl)
        s = (shuffled.mean() / shuffled.std() * 
             np.sqrt(252/7)) if shuffled.std() > 0 else 0
        random_sharpes.append(s)
    
    random_sharpes = np.array(random_sharpes)
    p_value = (random_sharpes >= real_sharpe).mean()
    
    print(f"\n── Monte Carlo Significance Test ──────────────────")
    print(f"  Real Sharpe:        {real_sharpe:.2f}")
    print(f"  Random mean Sharpe: {random_sharpes.mean():.2f}")
    print(f"  P-value:            {p_value:.3f}")
    
    if p_value < 0.05:
        print(f"  Result: STATISTICALLY SIGNIFICANT (p < 0.05)")
        print(f"  Edge is likely real, not random luck")
    elif p_value < 0.10:
        print(f"  Result: MARGINAL (p < 0.10) — suggestive but not conclusive")
    else:
        print(f"  Result: NOT SIGNIFICANT (p = {p_value:.2f})")
        print(f"  Cannot distinguish from random chance with current data")