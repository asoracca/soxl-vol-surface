"""
fetch_data.py
-------------
Pulls SOXL price history and options chain from Yahoo Finance.
"""

import warnings
warnings.filterwarnings('ignore')

import pandas as pd
import yfinance as yf
from pathlib import Path


def fetch_price_history(ticker_symbol="SOXL", period="1y"):
    ticker = yf.Ticker(ticker_symbol)
    data = ticker.history(period=period, interval="1d")
    data.index = pd.to_datetime(data.index)
    return data


def fetch_options_data(ticker_symbol="SOXL"):
    ticker = yf.Ticker(ticker_symbol)
    spot_price = ticker.history(period="1d")['Close'].iloc[-1]
    expirations = ticker.options

    print(f"Spot price:          ${spot_price:.2f}")
    print(f"Expirations found:   {len(expirations)}")

    all_puts = []
    for exp in expirations:
        try:
            chain = ticker.option_chain(exp)
            puts = chain.puts.copy()
            puts['expiration'] = exp
            puts['spot_price'] = spot_price

            exp_date = pd.to_datetime(exp)
            today = pd.Timestamp.today()
            puts['DTE'] = (exp_date - today).days
            puts['moneyness'] = puts['strike'] / spot_price
            all_puts.append(puts)
        except Exception as e:
            print(f"  Skipping {exp}: {e}")

    df = pd.concat(all_puts, ignore_index=True)

    # Filter to liquid, meaningful options
    df = df[df['impliedVolatility'] > 0.01]
    df = df[df['volume'] > 0]
    df = df[df['DTE'] > 0]
    df = df[df['DTE'] <= 365]
    df = df[(df['moneyness'] >= 0.4) & (df['moneyness'] <= 1.6)]

    print(f"Clean options rows:  {len(df)}")
    return df, spot_price


if __name__ == "__main__":
    Path("data").mkdir(exist_ok=True)
    data = fetch_price_history()
    print(data.tail())
    df, spot = fetch_options_data()
    print(df[['strike', 'DTE', 'moneyness', 'impliedVolatility']].head(10))