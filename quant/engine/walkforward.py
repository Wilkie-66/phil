"""Walk-forward: fit on the past, trade the future, roll. PROTECTED (operator-owned)."""
import pandas as pd

from .backtest import count_trades, window_backtest
from .metrics import metrics, sharpe_per_bar


def folds(n, wf):
    tr, te, emb = wf["train_bars"], wf["test_bars"], wf["embargo_bars"]
    out, i = [], 0
    while i + tr + emb + te <= n:
        out.append(((i, i + tr), (i + tr + emb, i + tr + emb + te)))
        i += te
    return out


def walk_forward(df, signals, cfg):
    """signals: {params_json: Series} precomputed on the full history (valid
    only after the leak test passed). Returns per-fold results and the
    stitched out-of-sample net returns of the strategy and of buy-and-hold."""
    hold = pd.Series(1.0, index=df.index)
    rows, oos, oos_hold = [], [], []
    for (a, b), (c, d) in folds(len(df), cfg["walk_forward"]):
        scores = {k: sharpe_per_bar(window_backtest(df, s, cfg, a, b)["net"])
                  for k, s in signals.items()}
        best = max(scores, key=scores.get)
        bt = window_backtest(df, signals[best], cfg, c, d)
        bh = window_backtest(df, hold, cfg, c, d)
        m = metrics(bt["net"], cfg, trades=count_trades(bt))
        rows.append({"test_start": str(df.index[c]), "test_end": str(df.index[d - 1]),
                     "chosen_params": best, "train_sharpe_bar": round(scores[best], 5),
                     **{k: m[k] for k in ("sharpe_ann", "total_return", "max_drawdown", "trades")},
                     "buy_hold_return": round(float((1 + bh["net"]).prod() - 1), 4)})
        oos.append(bt["net"])
        oos_hold.append(bh["net"])
    if not rows:
        return {"folds": pd.DataFrame(), "oos_net": pd.Series(dtype=float),
                "oos_hold_net": pd.Series(dtype=float), "oos_trades": 0}
    folds_df = pd.DataFrame(rows)
    return {"folds": folds_df, "oos_net": pd.concat(oos), "oos_hold_net": pd.concat(oos_hold),
            "oos_trades": int(folds_df["trades"].sum())}


def regime_report(df, oos_net, cfg, ma_bars=200, slope_bars=20):
    """OOS performance split by a causal trend regime label."""
    sma = df["close"].rolling(ma_bars).mean()
    slope = sma - sma.shift(slope_bars)
    label = pd.Series("chop", index=df.index)
    label[(df["close"] > sma) & (slope > 0)] = "bull"
    label[(df["close"] < sma) & (slope < 0)] = "bear"
    lab = label.reindex(oos_net.index)
    out = {}
    for name in ("bull", "bear", "chop"):
        r = oos_net[lab == name]
        if len(r) > 30:
            m = metrics(r, cfg)
            out[name] = {"bars": m["n_bars"], "sharpe_ann": m["sharpe_ann"],
                         "total_return": m["total_return"]}
        else:
            out[name] = {"bars": int(len(r)), "note": "too few bars"}
    return out
