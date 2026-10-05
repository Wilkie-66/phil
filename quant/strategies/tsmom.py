"""Time-series momentum on 4h bars, long-only.

Example strategy and template for the research agent. It is NOT expected to
pass the gates - it exists so the pipeline has something honest to reject.
"""

HYPOTHESIS = {
    "mechanism": "Crypto price discovery is slow: news and flows diffuse over days through "
                 "a retail-heavy market that under-reacts, then herds. Trailing return over "
                 "the lookback predicts the next bars' drift.",
    "counterparty": "Early sellers and mean-reversion traders who fade moves, and retail who "
                    "anchor to recent prices; they supply liquidity into the trend.",
    "invalidation": "Live hit-rate on entries below 35% over 30+ trades, or OOS Sharpe "
                    "not above buy-and-hold over 6 months of paper trading.",
}

PARAM_GRID = [{"lookback": lb, "vol_window": 60} for lb in (30, 60, 120, 240)]


def signal(df, lookback, vol_window):
    ret = df["close"].pct_change(lookback)
    vol = df["close"].pct_change().rolling(vol_window).std()
    # long when the trailing move is more than half a sigma-scaled move; flat otherwise
    thresh = 0.5 * vol * (lookback ** 0.5)
    return (ret > thresh).astype(float).where(ret.notna() & vol.notna())
