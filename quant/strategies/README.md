# Strategies

Each file is one hypothesis. Every `PARAM_GRID` entry is logged as a trial the
first time it is evaluated on real data, and every trial raises the bar
(deflated Sharpe) for all strategies - keep grids small and honest.

| File | Idea | Grid | Expected trades |
|---|---|---|---|
| `tsmom.py` | fast time-series momentum (baseline; expected to die on fees) | 4 | many |
| `trend_slow.py` | 30/60-day average with an entry/exit band | 4 | few per year |
| `trend_voltarget.py` | trend filter, exposure scaled down when volatility spikes (25% steps) | 4 | moderate |
| `breakout_donchian.py` | buy 20/40/60-day high breakouts, exit on half-length lows | 3 | few per year |
| `capitulation_rebound.py` | buy extreme down bars on volume spikes (liquidation cascades), hold 1-3 days | 4 | rare |

Rules for new files: self-contained (no imports from sibling strategy files -
the code hash only covers the file itself), long-only target in [0, 1], data
up to the bar close only, and few position changes: a round trip costs ~0.9%.
