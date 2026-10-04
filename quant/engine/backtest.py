"""Bar-by-bar backtest with honest fills. PROTECTED (operator-owned).

Timing (the guide's shift(1), made explicit):
  - signal[t] is a target position computed from data up to the CLOSE of bar t
  - it is filled at the OPEN of bar t+1+delay (delay = human approval lag)
  - the position held over the gap close[t]->open[t+1] is the OLD one

Returns are simple (not log) so shorts and costs compound correctly, and
costs are charged per side on every change in position.
"""
import pandas as pd


def positions_from_signal(signal, delay_bars=0, long_only=True):
    """Position held during each bar (from its open to its close)."""
    sig = signal.astype(float).clip(0.0 if long_only else -1.0, 1.0)
    return sig.shift(1 + delay_bars).fillna(0.0)


def backtest(df, signal, cfg):
    """df: OHLC frame (see data.py); signal: Series on df.index in [-1, 1]."""
    ex = cfg["execution"]
    if not signal.index.equals(df.index):
        raise ValueError("signal index must equal the data index")
    if signal.isna().all():
        raise ValueError("signal is entirely NaN")
    pos = positions_from_signal(signal.fillna(0.0), ex["approval_delay_bars"], ex["long_only"])
    prev_pos = pos.shift(1).fillna(0.0)

    gap = (df["open"] / df["close"].shift(1) - 1.0).fillna(0.0)
    intra = df["close"] / df["open"] - 1.0
    turnover = (pos - prev_pos).abs()
    cost_rate = (ex["fee_bps_per_side"] + ex["slippage_bps_per_side"]) / 1e4
    costs = turnover * cost_rate

    # equity moves with the old position over the gap, pays costs to rebalance
    # at the open, then moves with the new position to the close
    gross = (1 + prev_pos * gap) * (1 + pos * intra) - 1.0
    net = (1 + prev_pos * gap) * (1 + pos * intra - costs) - 1.0
    equity = cfg["capital_usd"] * (1 + net).cumprod()
    return pd.DataFrame({"position": pos, "turnover": turnover, "gross": gross,
                         "costs": costs, "net": net, "equity": equity})


def buy_and_hold(df, cfg):
    sig = pd.Series(1.0, index=df.index)
    return backtest(df, sig, cfg)


def count_trades(bt):
    return int((bt["turnover"] > 1e-9).sum())


def window_backtest(df, signal, cfg, start, stop):
    """Backtest only bars [start, stop) as if the account were flat before
    `start` and flattened after `stop` - each walk-forward fold pays its own
    entry and exit costs. Signals outside the window are ignored."""
    d = cfg["execution"]["approval_delay_bars"]
    first_sig = max(start - 1 - d, 0)
    masked = pd.Series(0.0, index=signal.index)
    masked.iloc[first_sig:stop - 1 - d] = signal.iloc[first_sig:stop - 1 - d].fillna(0.0)
    lo = max(first_sig - 1, 0)
    bt = backtest(df.iloc[lo:stop], masked.iloc[lo:stop], cfg).iloc[start - lo:].copy()
    ex = cfg["execution"]
    exit_cost = abs(bt["position"].iloc[-1]) * (ex["fee_bps_per_side"] + ex["slippage_bps_per_side"]) / 1e4
    bt.iloc[-1, bt.columns.get_loc("net")] -= exit_cost
    bt.iloc[-1, bt.columns.get_loc("costs")] += exit_cost
    bt["equity"] = cfg["capital_usd"] * (1 + bt["net"]).cumprod()
    return bt
