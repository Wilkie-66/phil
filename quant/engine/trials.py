"""Append-only trial registry. PROTECTED (operator-owned).

Every parameter combination the engine evaluates is logged here BY THE
ENGINE, so the trial count fed to the deflated Sharpe cannot be
under-reported. CI rejects any agent commit that rewrites existing lines.
A trial is identified by (strategy code hash, params, symbol): re-running
the same thing on more data is not a new trial; editing the code is.
"""
import hashlib
import json
from datetime import datetime, timezone

from .config import TRIALS_PATH


def code_hash(path):
    with open(path, "rb") as f:
        return hashlib.sha256(f.read()).hexdigest()[:16]


def trial_key(chash, params, symbol):
    return f"{chash}|{json.dumps(params, sort_keys=True)}|{symbol}"


def read_all(path=TRIALS_PATH):
    if not path.exists():
        return []
    with open(path) as f:
        return [json.loads(line) for line in f if line.strip()]


def record(rows, path=TRIALS_PATH):
    """Append rows whose key is new. Returns the number appended."""
    seen = {r["key"] for r in read_all(path)}
    now = datetime.now(timezone.utc).isoformat(timespec="seconds")
    fresh = []
    for r in rows:
        if r["key"] in seen:
            continue
        seen.add(r["key"])
        fresh.append({"ts": now, **r})
    if fresh:
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "a") as f:
            for r in fresh:
                f.write(json.dumps(r, sort_keys=True) + "\n")
    return len(fresh)


def family(timeframe_minutes, path=TRIALS_PATH):
    """All trials on this timeframe - the honest denominator, across every
    strategy and symbol ever tried, not just the current file."""
    return [r for r in read_all(path) if r.get("timeframe_minutes") == timeframe_minutes]
