"""Buy forced selling: enter after an extreme down bar on a volume spike,
exit after a fixed hold. Long-only, rare trades.
"""
import numpy as np
import pandas as pd

HYPOTHESIS = {
    "mechanism": "Crypto is dominated by leveraged perpetual futures. Sharp drops trigger "
                 "liquidation cascades: positions are closed by the exchange at any price, "
                 "pushing spot below where unforced buyers and sellers would clear. Once the "
                 "forced flow is exhausted price partially reverts; whoever supplies liquidity "
                 "into the cascade earns that rebound.",
    "counterparty": "Liquidated leveraged longs and panicking spot holders - price-insensitive "
                    "sellers who must sell now, not sellers who know something.",
    "invalidation": "Average 4h return over the hold period after entry not positive across "
                    "20+ paper trades, or most entries followed by further >5% declines "
                    "(i.e. drops are information, not forced flow).",
}

# z: how extreme the bar's drop must be (in units of recent 4h volatility);
# hold: bars to stay long (6 = 1 day, 18 = 3 days); volume spike fixed at 2x median
PARAM_GRID = [{"z": z, "hold": h, "vol_mult": 2.0, "window": 180}
              for z in (3.0, 4.0) for h in (6, 18)]


def signal(df, z, hold, vol_mult, window):
    r = df["close"].pct_change()
    sd = r.rolling(window).std()
    vol_med = df["volume"].rolling(window).median()
    trigger = ((r < -z * sd) & (df["volume"] > vol_mult * vol_med)).to_numpy()
    ready = sd.notna().to_numpy() & vol_med.notna().to_numpy()
    out = np.full(len(r), np.nan)
    left = 0
    for i in range(len(r)):
        if not ready[i]:
            continue
        if trigger[i]:
            left = hold  # a fresh capitulation bar restarts the hold
        out[i] = 1.0 if left > 0 else 0.0
        if left > 0:
            left -= 1
    return pd.Series(out, index=df.index)
