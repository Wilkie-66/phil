# quant — protected validation engine for 4h crypto strategies (Kraken)

Step 2 of the plan: an engine an AI can propose strategies to but cannot
flatter. Strategies are cheap; this is the part that says no.

```
quant/
  config.json        PROTECTED  costs, walk-forward windows, gate thresholds, risk caps
  engine/            PROTECTED  data, backtest, metrics, leak test, DSR, walk-forward, gates, sizing
  tests/             PROTECTED  30 tests pinning the engine's honesty (CI runs them)
  strategies/        AGENT      one file per hypothesis (see strategies/tsmom.py)
  journal/trials.jsonl  APPEND-ONLY  every parameter set ever evaluated (CI enforces)
  journal/evaluations/  local reports (gitignored)
  data/                 local Kraken cache (gitignored)
```

## Quick start (on your machine — Kraken is not reachable from the cloud sandbox)

```bash
pip install -r quant/requirements.txt

# 1. history: Kraken's REST API only serves the last 720 bars (120 days of 4h).
#    Download Kraken's "Downloadable historical OHLCVT data" archive
#    (support.kraken.com), unzip, and import the 4h file for each pair:
python -m quant import-csv ~/Downloads/Kraken_OHLCVT/XBTUSD_240.csv --symbol XBTUSD
python -m quant import-csv ~/Downloads/Kraken_OHLCVT/ETHUSD_240.csv --symbol ETHUSD

# 2. top up to the latest closed bar (safe to run any time)
python -m quant fetch --symbol XBTUSD

# 3. run a strategy through the three gates
python -m quant evaluate quant/strategies/tsmom.py --symbol XBTUSD
python -m quant evaluate quant/strategies/tsmom.py --synthetic   # no data needed

python -m quant trials      # honest count of everything tried so far
python -m quant validate    # config ceilings + registry integrity
python -m pytest quant/tests -q
```

USD pairs (XBTUSD, ETHUSD) are much deeper than the AUD books; research on
USD, decide later whether to execute on AUD pairs and re-check costs.

## What the gates do

| Gate | Check | Fixes vs. the guide |
|---|---|---|
| 1 Leak | Signal at bar *t* must be identical when computed on history truncated at *t* (40 random cut points × every param set) | Automated: catches centred MAs, bfill, full-sample z-scores, `shift(-1)` — no LLM review needed |
| 2 Deflated Sharpe | Stitched out-of-sample Sharpe vs. the expected max of *N* noise strategies, *N* = every trial in the registry | Uses per-bar Sharpe and the cross-trial spread; the guide's version mixes units and can pass a lucky Sharpe of 3 (see `test_guide_formula_passes_a_lucky_noise_strategy`) |
| 3 Walk-forward | ≥6 folds, ≥60% positive, worst fold ≥ −10%, ≥30 OOS trades, Sharpe beats buy-and-hold | Engine (not the strategy) picks params on each train window; each fold pays entry and exit costs |

Execution model: signal known at bar close, filled at the next bar's open
(+ `approval_delay_bars` to model you approving late), 0.40% Kraken taker fee
+ 0.05% slippage per side, long-only spot, 2190 bars/year (not 365).

## Writing a strategy

Copy `strategies/tsmom.py`. A strategy must state a `mechanism`, a
`counterparty` and an `invalidation` rule, keep `PARAM_GRID` ≤ 50 entries, and
implement `signal(df, **params)` returning a target position in [0, 1] per bar
using data up to that bar's close only. Every grid entry is logged as a trial
the moment it is evaluated — including the ones you delete afterwards.

## Expect rejections

On a random walk the example momentum strategy loses ~80% out of sample,
almost all of it to fees: ~150 round trips × 0.9% each. At $1k on Kraken's
lowest fee tier, a 4h strategy needs either few trades or a large per-trade
edge. That is the first thing the agent should learn.
