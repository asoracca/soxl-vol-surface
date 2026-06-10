# SOXL Options Analysis & IV Signal

A quantitative options analysis toolkit for **SOXL** (Direxion Dailys Semiconductor Bull 3X ETF), built to explore implied volatility structure and generate systematic put-selling signals.

---

## What this does

| Module | What it produces |
|---|---|
| `fetch_data.py` | Live price history + full options chain from Yahoo Finance |
| `vol_surface.py` | 3D implied volatility surface (static + interactive) and vol smile charts |
| `iv_rank.py` | Rolling IV Rank and IV Percentile — the two core signals for premium sellers |
| `backtest.py` | Historical backtest of "sell 15%-OTM puts when IV Rank > 50" |
| `signal.py` | Today's go/no-go signal with suggested strike and expiration |

---

## Motivation

SOXL is one of the most volatile liquid ETFs in the US market, which makes it interesting for options premium sellers — but only when volatility is *expensive* relative to its own history.

This project builds the infrastructure to answer: **is now a good time to sell puts on SOXL?**

The core insight: implied volatility (IV) mean-reverts. When IV Rank is high (>50), the market is overpaying for downside protection. Systematic put sellers can collect that fear premium and profit when volatility normalizes.

---

## Outputs

- `data/price_history.png` — 1-year SOXL price chart
- `data/iv_rank_history.png` — Rolling IV Rank + IV Percentile dashboard
- `data/vol_surface_3d.png` — Static 3D vol surface
- `data/vol_surface_interactive.html` — Rotate/zoom vol surface in browser
- `data/vol_smile.png` — Volatility smile per expiration
- `data/backtest_results.png` — Backtest P&L, win rate, trade distribution

---

## Quickstart

```bash
# Clone
git clone https://github.com/YOUR_USERNAME/soxl-vol-surface.git
cd soxl-vol-surface

# Install dependencies
pip install -r requirements.txt

# Run full pipeline
python main.py

# Or run just today's signal
python src/signal.py
```

---

## Signal logic

```
IV Rank ≥ 50  →  SELL PUTS   ✅  (vol in top half of 1Y range)
IV Rank 30–50 →  NEUTRAL     ⚠️  (acceptable but not optimal)
IV Rank < 30  →  AVOID       ❌  (vol cheap, premium thin)
```

**IV Rank** = (Current IV − 52w Low) / (52w High − 52w Low) × 100

---

## Backtest methodology

- **Entry**: IV Rank crosses above 50
- **Strike**: 15% out-of-the-money put
- **Expiration**: ~30 DTE
- **Exit**: Hold to expiration
- **Premium estimate**: Simplified Black-Scholes approximation using realized vol
- **Note**: This is a signal validation backtest, not a production P&L model. Real fills, commissions, and margin are not included.

---

## Risk warning

SOXL is a **3x leveraged ETF**. Daily rebalancing causes volatility decay, and the ETF can move 20–40% in a single session. Put sellers face significant assignment risk. Size positions conservatively — no more than 1–2% of capital per trade.

---

## Tech stack

- `yfinance` — market data
- `pandas` / `numpy` — data processing
- `scipy` — surface interpolation
- `matplotlib` — static charts
- `plotly` — interactive 3D surface

---

## Project structure

```
soxl-vol-surface/
├── main.py                  ← run everything
├── requirements.txt
├── README.md
├── .gitignore
├── src/
│   ├── fetch_data.py        ← price + options data
│   ├── vol_surface.py       ← 3D surface + smile plots
│   ├── iv_rank.py           ← IV Rank / IV Percentile
│   ├── backtest.py          ← historical strategy backtest
│   └── signal.py            ← today's trade recommendation
└── data/                    ← generated outputs (gitignored)
```

---

*Built as a quant portfolio project. Not financial advice.*