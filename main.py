"""
main.py
-------
Runs the full SOXL options analysis pipeline:
  1. Price history chart
  2. IV Rank + Percentile dashboard
  3. Live options chain → vol surface + smile
  4. Skew analysis
  5. Backtest of put selling strategy
  6. Today's signal
"""

import warnings
warnings.filterwarnings('ignore')

from pathlib import Path

from src.fetch_data   import fetch_price_history, fetch_options_data
from src.vol_surface  import (plot_price_history, plot_vol_surface_3d,
                               plot_vol_surface_interactive, plot_vol_smile)
from src.iv_rank      import plot_iv_rank_history, get_current_signal
from src.backtest     import run_backtest, print_backtest_stats, plot_backtest
from src.signal       import run_signal


def main():
    Path("data").mkdir(exist_ok=True)

    print("\n── 1. Price History ────────────────────────────────")
    data = fetch_price_history("SOXL", period="1y")
    print(data[['Close', 'Volume']].tail(5).to_string())
    plot_price_history(data)

    print("\n── 2. IV Rank Dashboard ────────────────────────────")
    get_current_signal("SOXL")
    plot_iv_rank_history("SOXL")

    print("\n── 3. Live Options Chain ───────────────────────────")
    df, spot_price = fetch_options_data("SOXL")

    print("\n── 4. Vol Surface ──────────────────────────────────")
    plot_vol_surface_3d(df, spot_price)
    plot_vol_surface_interactive(df, spot_price)
    plot_vol_smile(df, spot_price)

    print("\n── 5. Backtest ─────────────────────────────────────")
    trades = run_backtest("SOXL")
    print_backtest_stats(trades)
    plot_backtest(trades)

    print("\n── 6. Today's Signal ───────────────────────────────")
    run_signal()


if __name__ == "__main__":
    main()