"""
vix_term_structure.py  —  soxl-vol-surface

WHY THIS EXISTS
---------------
IV Rank needs a 252-day lookback of implied vol you don't have for free.
The VIX *term structure* gives a regime read RIGHT NOW, no lookback:

    VIX9D (9-day)  <  VIX (30-day)  <  VIX3M (3-month)   => CONTANGO  (calm; near-term IV cheap)
    VIX9D          >  VIX           >  VIX3M             => BACKWARDATION (stress; near-term IV rich)

It also appends every reading to data/vix_term_structure.csv, so over a few
months you build your OWN history and can rank today's reading against it.

USAGE
-----
    python vix_term_structure.py                       # standalone
    from vix_term_structure import report_term_structure
    report_term_structure(fetch_fn=fetch_price_history)  # reuse your working fetcher
"""

from __future__ import annotations
import os
from datetime import datetime, timezone

import numpy as np
import pandas as pd
import yfinance as yf

# Yahoo tickers, front of the curve -> back
TICKERS = {
    "VIX9D": "^VIX9D",   # 9-day   (very near term)
    "VIX":   "^VIX",     # 30-day  (the headline VIX)
    "VIX3M": "^VIX3M",   # 3-month
    "VIX6M": "^VIX6M",   # 6-month (optional; shown if available)
}

DATA_DIR = "data"
HIST_CSV = os.path.join(DATA_DIR, "vix_term_structure.csv")


# ----------------------------------------------------------------------
# Data  (robust: project fetcher -> Ticker.history -> download)
# ----------------------------------------------------------------------
def _close_from_df(df) -> float | None:
    """Pull the last non-null Close out of any yfinance-style frame."""
    if df is None or len(df) == 0:
        return None
    col = df["Close"] if "Close" in df else df.get("close")
    if col is None:
        return None
    s = col.dropna()
    return float(s.iloc[-1]) if len(s) else None


def _latest_close(ticker: str, fetch_fn=None) -> float | None:
    """Most recent close for a Yahoo index, trying several methods in order."""
    # 1. Your project's own fetcher (already proven to work on your network)
    if fetch_fn is not None:
        try:
            v = _close_from_df(fetch_fn(ticker, period="5d"))
            if v is not None:
                return v
        except Exception:
            pass
    # 2. yfinance Ticker.history — far more reliable than download() for indices
    try:
        v = _close_from_df(yf.Ticker(ticker).history(period="1mo"))
        if v is not None:
            return v
    except Exception:
        pass
    # 3. last resort: download()
    try:
        v = _close_from_df(yf.download(ticker, period="5d", interval="1d",
                                       progress=False, auto_adjust=False))
        if v is not None:
            return v
    except Exception:
        pass
    print(f"    [warn] could not fetch {ticker}")
    return None


def fetch_curve(fetch_fn=None) -> dict:
    """Return {name: level or None} for each VIX maturity."""
    return {name: _latest_close(tkr, fetch_fn) for name, tkr in TICKERS.items()}


# ----------------------------------------------------------------------
# Signal logic
# ----------------------------------------------------------------------
def _score_from_ratio(ratio: float) -> float:
    """
    Map the VIX / VIX3M ratio to a 0-100 'term-structure stress score'.
      ratio < 1 -> contango (calm, near-term vol cheap)  -> low score
      ratio > 1 -> backwardation (stress, vol rich)       -> high score
    Linear from 0.85 (=0) to 1.15 (=100), clamped to [0, 100].
    """
    lo, hi = 0.85, 1.15
    return float(np.clip((ratio - lo) / (hi - lo) * 100.0, 0, 100))


def _regime(main_slope: float) -> str:
    if main_slope < 0.95:
        return "CONTANGO — calm, near-term IV cheap"
    if main_slope < 1.00:
        return "FLATTENING — vol firming up"
    return "BACKWARDATION — stress, near-term IV elevated"


def history_percentile(score: float) -> float | None:
    """Percentile of today's score within YOUR logged history (None until 20 rows)."""
    if not os.path.exists(HIST_CSV):
        return None
    h = pd.read_csv(HIST_CSV)
    if "ts_score" not in h or len(h) < 20:
        return None
    return float((h["ts_score"] < score).mean() * 100.0)


def get_term_structure_signal(fetch_fn=None) -> dict:
    """Compute the live term-structure signal. Pass fetch_fn to reuse your fetcher."""
    curve = fetch_curve(fetch_fn)
    vix9d, vix, vix3m = curve.get("VIX9D"), curve.get("VIX"), curve.get("VIX3M")

    if vix is None or vix3m is None:
        return {"ok": False, "reason": "VIX or VIX3M unavailable", "curve": curve}

    main_slope = vix / vix3m                        # 30d vs 3m  (primary)
    front_slope = (vix9d / vix) if vix9d else None  # 9d vs 30d (front kink)
    score = _score_from_ratio(main_slope)
    regime = _regime(main_slope)
    pctile = history_percentile(score)

    if score < 35:
        read = "Near-term premium is THIN — consistent with AVOID SELLING."
    elif score < 65:
        read = "Premium is firming, but no clear edge — neutral."
    else:
        read = ("Near-term IV is RICH but regime is STRESSED — premium is high AND "
                "crash risk is high; respect position-size rules.")

    return {
        "ok": True, "curve": curve,
        "front_slope_9d_30d": front_slope,
        "main_slope_30d_3m": main_slope,
        "ts_score": score, "ts_percentile_self": pctile,
        "regime": regime, "read": read,
    }


# ----------------------------------------------------------------------
# Logging
# ----------------------------------------------------------------------
def log_row(sig: dict) -> None:
    if not sig.get("ok"):
        return
    os.makedirs(DATA_DIR, exist_ok=True)
    today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    row = {
        "date": today,
        "VIX9D": sig["curve"].get("VIX9D"), "VIX": sig["curve"].get("VIX"),
        "VIX3M": sig["curve"].get("VIX3M"), "VIX6M": sig["curve"].get("VIX6M"),
        "front_slope": sig["front_slope_9d_30d"], "main_slope": sig["main_slope_30d_3m"],
        "ts_score": sig["ts_score"], "regime": sig["regime"],
    }
    if os.path.exists(HIST_CSV):
        h = pd.read_csv(HIST_CSV)
        h = h[h["date"] != today]                    # replace today's row if re-run
        h = pd.concat([h, pd.DataFrame([row])], ignore_index=True)
    else:
        h = pd.DataFrame([row])
    h.to_csv(HIST_CSV, index=False)


# ----------------------------------------------------------------------
# Dashboard (call this from main.py)
# ----------------------------------------------------------------------
def report_term_structure(fetch_fn=None) -> dict:
    sig = get_term_structure_signal(fetch_fn)
    print("=" * 55)
    print(f"  VIX TERM STRUCTURE — {datetime.now():%Y-%m-%d}")
    print("=" * 55)
    if not sig["ok"]:
        print(f"  Unavailable: {sig['reason']}")
        print(f"  Curve fetched: {sig['curve']}")
        return sig

    c = sig["curve"]
    fmt = lambda x: f"{x:5.2f}" if x is not None else "  n/a"
    print(f"  VIX9D : {fmt(c['VIX9D'])}   (9-day)")
    print(f"  VIX   : {fmt(c['VIX'])}   (30-day)")
    print(f"  VIX3M : {fmt(c['VIX3M'])}   (3-month)")
    if c.get("VIX6M"):
        print(f"  VIX6M : {fmt(c['VIX6M'])}   (6-month)")
    print("-" * 55)
    fs = sig["front_slope_9d_30d"]
    print(f"  Front slope  VIX9D/VIX : {fs:5.3f}" if fs else "  Front slope  VIX9D/VIX :   n/a")
    print(f"  Main  slope  VIX/VIX3M : {sig['main_slope_30d_3m']:5.3f}")
    print(f"  TS Score (0-100)       : {sig['ts_score']:5.1f}   (higher = more near-term stress)")
    p = sig["ts_percentile_self"]
    print(f"  Self-history percentile: {p:5.1f}" if p is not None
          else "  Self-history percentile:   n/a  (need 20+ logged days)")
    print(f"  Regime  : {sig['regime']}")
    print(f"  Read    : {sig['read']}")
    print("=" * 55)

    log_row(sig)
    print(f"  Logged -> {HIST_CSV}")
    return sig


def main() -> None:
    # Standalone: try to reuse the project fetcher if present, else pure yfinance.
    fetch_fn = None
    try:
        from src.fetch_data import fetch_price_history
        fetch_fn = fetch_price_history
    except Exception:
        pass
    report_term_structure(fetch_fn)


if __name__ == "__main__":
    main()