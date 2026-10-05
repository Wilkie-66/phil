"""Slow trend with a hysteresis band, long-only.

Same idea as tsmom.py but built to survive Kraken's fees: a 30-60 day
moving average with an entry/exit band, so the position changes a handful of
times a year instead of every few days.
"""
import numpy as np
import pandas as pd

HYPOTHESIS = {
    "mechanism": "Time-series momentum is one of the few crypto effects documented out of "
                 "sample (Liu & Tsyvinski 2021): attention and flows build over weeks, "
                 "leverage is added into rising prices and unwound into falling ones, so "
                 "multi-week direction persists. A slow average with a band captures the "
                 "persistence while keeping turnover low enough to clear 0.9% round-trip costs.",
    "counterparty": "Early profit-takers and mean-reversion sellers who fade multi-week moves, "
                    "and holders who sell into strength because they anchor to old prices.",
    "invalidation": "Fewer than 40% of paper round trips profitable over 15+ trades, or "
                    "paper drawdown worse than the backtest's worst fold.",
}

# 180 bars = 30 days, 360 = 60 days; band = how far through the average price
# must go before the position flips
PARAM_GRID = [{"n": n, "band": b} for n in (180, 360) for b in (0.02, 0.05)]


def signal(df, n, band):
    close = df["close"].to_numpy()
    sma = df["close"].rolling(n).mean().to_numpy()
    out = np.full(len(close), np.nan)
    pos = 0.0
    for i in range(len(close)):
        if np.isnan(sma[i]):
            continue
        if pos == 0.0 and close[i] > sma[i] * (1 + band):
            pos = 1.0
        elif pos == 1.0 and close[i] < sma[i] * (1 - band):
            pos = 0.0
        out[i] = pos
    return pd.Series(out, index=df.index)
