"""Trend filter with volatility-scaled exposure, long-only.

Long only above the trend average, and size the position so the sleeve's
volatility stays near a target: full size in calm uptrends, a fraction in
violent ones. Exposure moves in 25% steps and only when the target moves a
full step, so fees stay bounded.
"""
import numpy as np
import pandas as pd

HYPOTHESIS = {
    "mechanism": "Volatility clusters and high-volatility regimes in crypto carry worse "
                 "risk-adjusted returns than calm ones (volatility-managed portfolios, "
                 "Moreira & Muir 2017, replicated on crypto). Cutting exposure when realised "
                 "vol spikes avoids the worst of crash regimes while keeping the trend's drift.",
    "counterparty": "Partly a risk-timing effect rather than a pure counterparty trade: the "
                    "other side is constant-exposure holders and late leveraged longs who "
                    "carry full risk through volatility spikes and are forced out at the lows.",
    "invalidation": "Paper Sharpe not above the plain trend filter's over the same period, "
                    "or average exposure in high-vol bars not lower than in calm ones.",
}

PARAM_GRID = [{"n": n, "target_vol": tv, "vol_window": 180}
              for n in (180, 360) for tv in (0.4, 0.6)]

BARS_PER_YEAR = 2190
STEP = 0.25


def signal(df, n, target_vol, vol_window):
    close = df["close"]
    trend_up = (close > close.rolling(n).mean()).to_numpy()
    ann_vol = (close.pct_change().rolling(vol_window).std() * np.sqrt(BARS_PER_YEAR)).to_numpy()
    sma_ready = close.rolling(n).mean().notna().to_numpy()
    out = np.full(len(close), np.nan)
    level = 0.0
    for i in range(len(close)):
        if not sma_ready[i] or np.isnan(ann_vol[i]) or ann_vol[i] <= 0:
            continue
        raw = min(1.0, target_vol / ann_vol[i]) if trend_up[i] else 0.0
        # move only when the target is a full step away; exits to 0 are immediate
        if raw == 0.0 or abs(raw - level) >= STEP:
            level = np.floor(raw / STEP) * STEP
        out[i] = level
    return pd.Series(out, index=df.index)
