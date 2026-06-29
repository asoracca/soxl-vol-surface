# Walk-forward validation vs in-sample — 2026-06-29

First run wiring `walk_forward_backtest` + `monte_carlo_significance` +
`stress_test` into the pipeline, replacing the in-sample-only backtest.

| Metric              | In-sample (naive) | Walk-forward (OOS) |
|---------------------|-------------------|--------------------|
| Threshold           | IV Rank >= 50     | IV Rank >= 50      |
| Trades              | 9                 | 37                 |
| Win rate            | 100%              | (mixed)            |
| Sharpe              | 765.1             | 6.33               |
| Monte Carlo p-value | n/a               | 0.991              |

Takeaway: the in-sample Sharpe of 765 is meaningless — 9 trades, 100% win rate,
same data used to fit and test, on a near-deterministic premium series (15%-OTM
7-DTE puts almost never breach, and the realized-vol premium model overstates
credit). Out-of-sample, the walk-forward produced 37 trades with a Monte Carlo
p-value of 0.991 under a random-entry-timing null — i.e. the apparent edge is
NOT distinguishable from random timing. Stress test: survived 2022 (5 trades,
80% win) and the Mar-2026 SOXL crash (3 trades, 100% win), but with too few
trades per regime to conclude anything.

Caveat on the MC null: random-entry Sharpes come out very high (~478 mean)
because premium is held at the realized mean and the win structure is nearly
constant, so variance is tiny. This inflates the null distribution but does not
change the conclusion (real Sharpe still sits far below it). A real-IV premium
model and 50+ OOS trades are needed before any edge claim.

Also fixed this run: monte_carlo_significance previously permuted PnL, which
leaves Sharpe unchanged (p always 1.0). Replaced with a random-entry null.
