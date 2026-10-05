"""Performance metrics on per-bar simple returns. PROTECTED (operator-owned).

Sharpe is computed per bar first and annualised with the config's
bars_per_year (2190 for 24/7 4h crypto) - never a hardcoded 365.
"""
import numpy as np
import pandas as pd
from scipy import stats as sps


def sharpe_per_bar(r):
    r = pd.Series(r).dropna()
    sd = r.std(ddof=1)
    return float(r.mean() / sd) if len(r) > 1 and sd > 0 else 0.0


def metrics(net, cfg, trades=None):
    r = pd.Series(net).dropna()
    bpy = cfg["bars_per_year"]
    if len(r) < 2:
        return {"error": "insufficient_data", "n_bars": int(len(r))}
    sr = sharpe_per_bar(r)
    equity = (1 + r).cumprod()
    dd = equity / equity.cummax() - 1
    underwater = dd < 0
    runs = underwater.groupby((underwater != underwater.shift()).cumsum()).sum()
    longest_bars = int(runs.max()) if underwater.any() else 0
    total = float(equity.iloc[-1] - 1)
    years = len(r) / bpy
    cagr = float(equity.iloc[-1] ** (1 / years) - 1) if equity.iloc[-1] > 0 else -1.0
    max_dd = float(dd.min())
    out = {
        "n_bars": int(len(r)),
        "sharpe_bar": round(sr, 5),
        "sharpe_ann": round(sr * np.sqrt(bpy), 3),
        "total_return": round(total, 4),
        "cagr": round(cagr, 4),
        "max_drawdown": round(max_dd, 4),
        "longest_dd_days": round(longest_bars * cfg["timeframe_minutes"] / 1440, 1),
        "calmar": round(cagr / abs(max_dd), 3) if max_dd < 0 else None,
        "skew": round(float(sps.skew(r)), 4),
        "kurtosis": round(float(sps.kurtosis(r, fisher=False)), 4),
    }
    if trades is not None:
        out["trades"] = int(trades)
    return out
