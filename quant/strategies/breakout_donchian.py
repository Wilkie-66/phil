"""Range breakout (Donchian channel), long-only: buy a close above the prior
N-bar high, exit on a close below the prior M-bar low.
"""
import numpy as np
import pandas as pd

HYPOTHESIS = {
    "mechanism": "Traders anchor to visible ranges: stop-losses of shorts and breakout buy "
                 "orders cluster just above prior highs, and option/perp hedgers chase once "
                 "they are taken out. Clearing a multi-week high releases that resting flow "
                 "and tends to start a volatility expansion in the breakout's direction.",
    "counterparty": "Short sellers and range traders who sold the top of the range and must "
                    "cover, plus holders who sold at the old high expecting it to hold.",
    "invalidation": "More than 65% of paper breakouts exiting at a loss over 15+ trades "
                    "(false breakouts dominate), or OOS edge confined to one regime.",
}

# (entry, exit) channel lengths in 4h bars: 20/10, 40/20, 60/30 days
PARAM_GRID = [{"entry": e, "exit": x} for e, x in ((120, 60), (240, 120), (360, 180))]


def signal(df, entry, exit):
    # prior channel only: shift(1) so the current bar is compared with history
    hi = df["high"].rolling(entry).max().shift(1).to_numpy()
    lo = df["low"].rolling(exit).min().shift(1).to_numpy()
    close = df["close"].to_numpy()
    out = np.full(len(close), np.nan)
    pos = 0.0
    for i in range(len(close)):
        if np.isnan(hi[i]) or np.isnan(lo[i]):
            continue
        if pos == 0.0 and close[i] > hi[i]:
            pos = 1.0
        elif pos == 1.0 and close[i] < lo[i]:
            pos = 0.0
        out[i] = pos
    return pd.Series(out, index=df.index)
