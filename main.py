"""
main.py
-------
Runs the full SOXL options analysis pipeline:
  1. Price history chart
  2. IV Rank + Percentile dashboard
  3. Live options chain → vol surface + smile
  4. Skew analysis
  5. Backtest of put-selling strategy (in-sample baseline + walk-forward OOS)
  6. Today's signal
  7. VIX term-structure regime (lookback-free IV-Rank proxy)
"""

import warnings
warnings.filterwarnings('ignore')

from pathlib import Path

from src.fetch_data   import fetch_price_history, fetch_options_data
from src.vol_surface  import (plot_price_history, plot_vol_surface_3d,
                               plot_vol_surface_interactive, plot_vol_smile)
from src.iv_rank      import plot_iv_rank_history, get_current_signal
from src.backtest     import (run_backtest, print_backtest_stats, plot_backtest,
                              walk_forward_backtest, monte_carlo_significance,
                              stress_test)
from src.trade_signal import run_signal
from vix_term_structure import report_term_structure


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
    IV_RANK_THRESHOLD = 50  # matches the live discipline rule (see README)

    # 5a. In-sample (naive) — baseline ONLY. NOT tradeable:
    #     same data used to pick and test the rule -> look-ahead / overfit.
    trades = run_backtest("SOXL", iv_rank_threshold=IV_RANK_THRESHOLD)
    in_sample_sharpe = print_backtest_stats(
        trades, label="in-sample, NOT tradeable", threshold=IV_RANK_THRESHOLD)
    plot_backtest(trades)

    # 5b. Walk-forward out-of-sample — the result that actually counts.
    wf_trades = walk_forward_backtest("SOXL", iv_rank_threshold=IV_RANK_THRESHOLD)
    close_5y = fetch_price_history("SOXL", period="5y")["Close"]
    mc = monte_carlo_significance(wf_trades, close_5y)
    stress_test(wf_trades)

    # 5c. Headline summary
    print("\n── HEADLINE RESULT (out-of-sample) ──────────────────")
    print(f"  In-sample Sharpe:      {in_sample_sharpe:8.1f}   (in-sample, NOT tradeable)")
    print(f"  Walk-forward trades:   {len(wf_trades):8d}   (out-of-sample)")
    if mc:
        print(f"  Walk-forward Sharpe:   {mc['real_sharpe']:8.2f}")
        print(f"  Monte Carlo p-value:   {mc['p_value']:8.3f}")
        verdict = ("edge plausibly real" if mc['p_value'] < 0.05
                   else "NOT distinguishable from luck")
        print(f"  Verdict:               {verdict}")
    print("  Note: the in-sample Sharpe is an artifact of a tiny, in-sample,")
    print("        near-deterministic premium series — it is not a real edge.")

    print("\n── 6. Today's Signal ───────────────────────────────")
    run_signal()

    print("\n── 7. VIX Term Structure (regime / IV proxy) ───────")
    # Reuse the project's proven fetcher so the index pulls work on your network.
    report_term_structure(fetch_fn=fetch_price_history)


if __name__ == "__main__":
    main()