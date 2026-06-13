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
## Known Limitations & Future Work

- **IV proxy:** Uses 30-day realized volatility as a proxy for implied volatility. 
  Real IV history requires paid data (CBOE, OptionMetrics). This understates 
  true premium for deep OTM puts, causing near-zero premiums in the backtest 
  for older low-price periods.

- **Backtest sample size:** Only 5-14 trades generated depending on lookback. 
  Statistically insufficient — 50+ trades needed for meaningful conclusions.

- **No Greeks hedging:** Strategy assumes hold-to-expiration with no delta hedging.
  Real implementation would manage delta exposure dynamically.

## Planned upgrades
- Integrate CBOE VIX term structure as IV proxy
- Add Kelly Criterion position sizing
- Add variance risk premium analysis
- Extend to multi-ticker (QLD, FNGO, NVDA)

---

## Statistical Disclaimer & Limitations

This project is a quantitative research tool built for learning and 
exploration. The following limitations are acknowledged explicitly:

### Backtesting caveats
- **Past performance does not guarantee future results.** All backtest 
  results are hypothetical and subject to look-ahead bias, overfitting, 
  and regime change risk.
- **In-sample testing:** The strategy parameters were developed and tested 
  on the same dataset. Walk-forward validation and out-of-sample testing 
  are planned but not yet implemented.
- **Small sample size:** The backtest generates 5–14 trades depending on 
  the lookback period. This is statistically insufficient to draw 
  definitive conclusions — a minimum of 50+ trades is required for 
  meaningful inference.
- **IV proxy limitation:** True implied volatility history requires paid 
  data (CBOE, OptionMetrics). This project uses 30-day realized volatility 
  as a proxy, which systematically understates true IV for deep OTM puts 
  and distorts premium estimates for older low-price periods.
- **No transaction cost modeling beyond estimates:** Real fills, margin 
  requirements, early assignment risk, and liquidity constraints are not 
  fully modeled.
- **Survivorship bias:** SOXL has survived as an ETF. Strategies tested 
  on surviving instruments overstate expected returns.

### Strategy risk factors
- SOXL is a 3x leveraged ETF. A 33% drop in semiconductors causes 
  approximately 100% loss in SOXL. Put sellers face assignment risk 
  that can exceed the premium collected by orders of magnitude.
- The variance risk premium (the documented edge underlying this 
  strategy) is known to compress or disappear during sustained 
  bear markets and liquidity crises — precisely when losses are largest.
- IV Rank is a backward-looking signal. High IV Rank indicates 
  volatility has been elevated historically, not that it will remain 
  so or revert predictably.

### What would make this more robust
- Walk-forward validation with train/test splits
- Monte Carlo significance testing (p-value on Sharpe ratio)
- Stress testing against known crash periods (2020, 2022, 2026)
- Real implied volatility data from CBOE or OptionMetrics
- Out-of-sample paper trading with live trade journal
- Kelly Criterion position sizing with drawdown limits

### Intended use
This tool is intended as a **signal generation and research aid**, 
not a fully autonomous trading system. All signals should be verified 
manually against live options chain data before execution. Position 
sizing should never exceed 1–2% of total capital per trade.

*Built as a quant portfolio project. Not financial advice.*