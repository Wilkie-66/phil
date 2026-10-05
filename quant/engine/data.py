"""Kraken OHLC data: fetch, import, clean, cache. PROTECTED (operator-owned).

Conventions (every series in this engine follows them):
  - index: tz-aware UTC DatetimeIndex of bar OPEN times, regular grid
  - columns: open, high, low, close, volume (floats)
  - a bar is only present once it has CLOSED (the in-progress bar is dropped)

Kraken's REST OHLC endpoint returns at most the 720 most recent bars
(120 days of 4h bars) - too short for walk-forward. For history, download
Kraken's official OHLCVT archive (support.kraken.com, "Downloadable historical
OHLCVT data"), unzip, and `import-csv` the <PAIR>_240.csv file; then `fetch`
keeps it topped up from the API.
"""
import time
from pathlib import Path

import numpy as np
import pandas as pd
import requests

from .config import DATA_DIR

KRAKEN_OHLC_URL = "https://api.kraken.com/0/public/OHLC"
COLUMNS = ["open", "high", "low", "close", "volume"]


def cache_path(symbol, timeframe_minutes, data_dir=DATA_DIR):
    return Path(data_dir) / "kraken" / f"{symbol}_{timeframe_minutes}.csv"


def fetch_kraken(symbol, timeframe_minutes, since=None, now=None):
    """Fetch up to 720 recent CLOSED bars from Kraken's public API."""
    params = {"pair": symbol, "interval": timeframe_minutes}
    if since is not None:
        params["since"] = int(since)
    resp = requests.get(KRAKEN_OHLC_URL, params=params, timeout=30,
                        headers={"User-Agent": "quant-research"})
    resp.raise_for_status()
    body = resp.json()
    if body.get("error"):
        raise RuntimeError(f"Kraken error: {body['error']}")
    result = body["result"]
    keys = [k for k in result if k != "last"]
    if len(keys) != 1:
        raise RuntimeError(f"unexpected Kraken result keys: {list(result)}")
    rows = result[keys[0]]
    # [time, open, high, low, close, vwap, volume, count]
    df = pd.DataFrame(rows, columns=["time", "open", "high", "low", "close",
                                     "vwap", "volume", "count"])
    df = _to_frame(df["time"], df[COLUMNS])
    return drop_unclosed(df, timeframe_minutes, now=now)


def read_kraken_archive_csv(path):
    """Read a file from Kraken's OHLCVT archive: no header,
    columns = timestamp, open, high, low, close, volume, trades."""
    raw = pd.read_csv(path, header=None,
                      names=["time", "open", "high", "low", "close", "volume", "trades"])
    return _to_frame(raw["time"], raw[COLUMNS])


def _to_frame(times, values):
    df = values.astype(float).copy()
    df.index = pd.to_datetime(times.astype("int64"), unit="s", utc=True)
    df.index.name = "time"
    return df


def drop_unclosed(df, timeframe_minutes, now=None):
    """Drop any bar whose close time is in the future (Kraken returns the live bar)."""
    now = pd.Timestamp.now(tz="UTC") if now is None else pd.Timestamp(now)
    closes = df.index + pd.Timedelta(minutes=timeframe_minutes)
    return df[closes <= now]


def clean(df, timeframe_minutes):
    """Sort, de-duplicate and regularise to the bar grid.

    Missing bars (no trades) are filled flat at the previous close with zero
    volume - the honest reading of "nothing traded". Returns (df, report).
    """
    report = {"rows_in": int(len(df))}
    df = df.sort_index()
    dupes = int(df.index.duplicated(keep="last").sum())
    df = df[~df.index.duplicated(keep="last")]
    freq = pd.Timedelta(minutes=timeframe_minutes)
    if len(df) == 0:
        raise ValueError("no data")
    off_grid = int(((df.index - df.index[0]) % freq != pd.Timedelta(0)).sum())
    if off_grid:
        raise ValueError(f"{off_grid} bars are not on the {timeframe_minutes}m grid")
    grid = pd.date_range(df.index[0], df.index[-1], freq=freq, tz="UTC", name="time")
    missing = grid.difference(df.index)
    df = df.reindex(grid)
    prev_close = df["close"].ffill()
    for col in ("open", "high", "low", "close"):
        df[col] = df[col].fillna(prev_close)
    df["volume"] = df["volume"].fillna(0.0)
    bad = (df["high"] < df[["open", "close"]].max(axis=1) - 1e-9) | \
          (df["low"] > df[["open", "close"]].min(axis=1) + 1e-9) | (df["close"] <= 0)
    if bad.any():
        raise ValueError(f"{int(bad.sum())} bars fail OHLC sanity (e.g. {df.index[bad][0]})")
    report.update({"rows_out": int(len(df)), "duplicates_dropped": dupes,
                   "missing_bars_filled": int(len(missing)),
                   "start": str(df.index[0]), "end": str(df.index[-1])})
    return df, report


def load(symbol, timeframe_minutes, data_dir=DATA_DIR):
    path = cache_path(symbol, timeframe_minutes, data_dir)
    if not path.exists():
        raise FileNotFoundError(
            f"no cached data at {path}; run `python -m quant import-csv` and/or `fetch` first")
    df = pd.read_csv(path, index_col="time", parse_dates=["time"])
    df.index = pd.DatetimeIndex(df.index, tz="UTC") if df.index.tz is None else df.index
    return clean(df[COLUMNS], timeframe_minutes)


def save_merged(new, symbol, timeframe_minutes, data_dir=DATA_DIR):
    """Merge new bars into the cache (new rows win on overlap) and save."""
    path = cache_path(symbol, timeframe_minutes, data_dir)
    if path.exists():
        old = pd.read_csv(path, index_col="time", parse_dates=["time"])
        old.index = pd.DatetimeIndex(old.index, tz="UTC") if old.index.tz is None else old.index
        new = pd.concat([old[COLUMNS], new[COLUMNS]])
    df, report = clean(new, timeframe_minutes)
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index_label="time")
    report["path"] = str(path)
    return report


def update_from_api(symbol, timeframe_minutes, data_dir=DATA_DIR):
    path = cache_path(symbol, timeframe_minutes, data_dir)
    since = None
    if path.exists():
        last = pd.read_csv(path, usecols=["time"], parse_dates=["time"])["time"].iloc[-1]
        since = pd.Timestamp(last).timestamp() - 1
    new = fetch_kraken(symbol, timeframe_minutes, since=since)
    time.sleep(1.0)  # be polite to the public endpoint
    report = save_merged(new, symbol, timeframe_minutes, data_dir)
    if since is None or new.index[0].timestamp() > since + timeframe_minutes * 60:
        report["warning"] = ("API returned no overlap with the cache - there may be a gap "
                             "that was filled flat. Import the archive CSV to cover it.")
    return report


def synthetic(n_bars=8000, seed=0, drift=0.0, vol=0.012, timeframe_minutes=240):
    """Random-walk OHLC for tests and demos. Contains no edge by construction."""
    rng = np.random.default_rng(seed)
    r = rng.standard_t(df=4, size=n_bars) * vol / np.sqrt(2) + drift
    close = 30000 * np.exp(np.cumsum(r))
    open_ = np.concatenate([[30000], close[:-1]]) * np.exp(rng.normal(0, vol / 20, n_bars))
    hi = np.maximum(open_, close) * (1 + np.abs(rng.normal(0, vol / 3, n_bars)))
    lo = np.minimum(open_, close) * (1 - np.abs(rng.normal(0, vol / 3, n_bars)))
    idx = pd.date_range("2019-01-01", periods=n_bars,
                        freq=pd.Timedelta(minutes=timeframe_minutes), tz="UTC", name="time")
    return pd.DataFrame({"open": open_, "high": hi, "low": lo, "close": close,
                         "volume": rng.uniform(10, 100, n_bars)}, index=idx)
