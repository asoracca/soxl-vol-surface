"""
signal.py
---------
Prints today's actionable SOXL options trading signal.
Run this daily before the market open to get a go/no-go on selling puts.

Output includes:
  - Current IV Rank and IV Percentile
  - Put skew from live options chain
  - Suggested strike and expiration if signal is active
  - Risk warning for SOXL's 3x leverage
"""

import warnings
warnings.filterwarnings('ignore')

import pandas as pd
import numpy as np
from datetime import date
from src.fetch_data import fetch_options_data, fetch_price_history
from src.iv_rank import get_current_signal


# ── Skew helper ───────────────────────────────────────────────────────────────

def get_skew_snapshot(df: pd.DataFrame) -> dict:
    """
    From live options data, return ATM IV and 15%-OTM put IV
    for the nearest expiration with >= 20 DTE.
    """
    near_term = df[df['DTE'] >= 20].copy()
    if near_term.empty:
        return {}

    exp = near_term.sort_values('DTE')['expiration'].iloc[0]
    exp_data = near_term[near_term['expiration'] == exp]

    atm_row = exp_data.iloc[(exp_data['moneyness'] - 1.0).abs().argsort()[:1]]
    otm_data = exp_data[
        (exp_data['moneyness'] >= 0.80) & (exp_data['moneyness'] <= 0.90)
    ]

    if atm_row.empty or otm_data.empty:
        return {}

    atm_iv  = float(atm_row['impliedVolatility'].values[0])
    otm_iv  = float(otm_data['impliedVolatility'].mean())
    dte     = int(exp_data['DTE'].iloc[0])

    return {
        "expiration": exp,
        "DTE":        dte,
        "atm_iv":     atm_iv,
        "otm_put_iv": otm_iv,
        "skew":       otm_iv - atm_iv,
    }


def suggest_strike(df: pd.DataFrame, spot_price: float) -> dict:
    """
    Suggest the best put strike for the nearest 25-45 DTE expiration.
    Targets 80-85% moneyness (15-20% OTM).
    """
    window = df[(df['DTE'] >= 25) & (df['DTE'] <= 45)]
    if window.empty:
        window = df[(df['DTE'] >= 15) & (df['DTE'] <= 60)]
    if window.empty:
        return {}

    exp = window.sort_values('DTE')['expiration'].iloc[0]
    exp_data = window[window['expiration'] == exp]

    # Target 82% moneyness (18% OTM)
    target_moneyness = 0.82
    best = exp_data.iloc[(exp_data['moneyness'] - target_moneyness).abs().argsort()[:1]]

    if best.empty:
        return {}

    row = best.iloc[0]
    return {
        "expiration":  exp,
        "DTE":         int(row['DTE']),
        "strike":      row['strike'],
        "moneyness":   round(row['moneyness'], 3),
        "iv":          round(row['impliedVolatility'], 3),
        "bid":         row.get('bid', None),
        "ask":         row.get('ask', None),
    }


# ── Main signal ───────────────────────────────────────────────────────────────

def run_signal():
    print(f"\n{'='*55}")
    print(f"  SOXL OPTIONS SIGNAL — {date.today()}")
    print(f"{'='*55}")

    # 1. IV Rank signal
    iv_data = get_current_signal("SOXL")

    # 2. Live options data
    print("\nFetching live options chain...")
    try:
        df, spot_price = fetch_options_data("SOXL")
        skew = get_skew_snapshot(df)
        suggestion = suggest_strike(df, spot_price)

        print(f"\n── Put Skew Snapshot ──────────────────────────────")
        if skew:
            print(f"  Expiration:    {skew['expiration']}  ({skew['DTE']} DTE)")
            print(f"  ATM IV:        {skew['atm_iv']:.1%}")
            print(f"  OTM Put IV:    {skew['otm_put_iv']:.1%}")
            print(f"  Skew:          {skew['skew']:.1%}  "
                  f"{'(elevated ✅)' if skew['skew'] > 0.08 else '(normal)'}")

        print(f"\n── Suggested Trade ─────────────────────────────────")
        if suggestion and iv_data['iv_rank'] >= 50:
            bid  = suggestion.get('bid')
            ask  = suggestion.get('ask')
            mid  = round((bid + ask) / 2, 2) if bid and ask else "N/A"
            print(f"  Action:        SELL PUT")
            print(f"  Strike:        ${suggestion['strike']:.2f}  "
                  f"({suggestion['moneyness']:.1%} moneyness)")
            print(f"  Expiration:    {suggestion['expiration']}  "
                  f"({suggestion['DTE']} DTE)")
            print(f"  Mid premium:   ${mid}")
            print(f"  IV at strike:  {suggestion['iv']:.1%}")
        elif iv_data['iv_rank'] < 50:
            print(f"  Action:        NO TRADE — IV Rank too low "
                  f"({iv_data['iv_rank']:.1f})")
        else:
            print("  No suitable strike found in the chain.")

    except Exception as e:
        print(f"  Could not fetch options data: {e}")

    print(f"\n── Risk Warning ────────────────────────────────────")
    print("  SOXL is a 3x leveraged ETF. Losses can be severe.")
    print("  Size positions to risk no more than 1-2% of capital.")
    print("  Always verify fills and Greeks with your broker.")
    print(f"{'='*55}\n")


if __name__ == "__main__":
    run_signal()