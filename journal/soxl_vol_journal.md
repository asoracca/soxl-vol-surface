
## 2026-07-01 — SOXL Signal Run

**Market**: SOXL $217.55 (violent 5-day range: $215.60–$266.71)
**IV Rank**: 17.7/100 | IV Percentile: 38.9/100 | VRP: -204.3 (vol cheap vs realized)
**Signal**: AVOID SELLING — VIX (16.6) far below SOXL realized vol (220.8%); premium thin.
**VIX term structure**: contango (9D 13.1 < 30D 16.6 < 3M 19.2 < 6M 21.6), TS score 5.3/100 — calm.

**Backtest sanity check**: in-sample Sharpe 765 (n=9, obviously fake — tiny near-deterministic
premium series). Walk-forward: 37 trades, Sharpe 6.33. Monte Carlo test vs random-entry null
(2000 sims): p=0.991 — NOT distinguishable from random timing. System correctly flagged its
own in-sample number as noise.

**Takeaway**: signal correctly abstains in low-IV-rank regime. Edge still unproven at n=37 —
need 50+ walk-forward trades before claiming anything beyond methodology. Keep logging weekly.
