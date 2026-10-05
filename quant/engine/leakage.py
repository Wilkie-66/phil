"""Automated look-ahead detector. PROTECTED (operator-owned).

An LLM critic reading code misses leaks; this test cannot. A causal signal's
value at bar t must be identical whether it is computed on the full history
or on the history truncated at t. Centred moving averages, bfill, full-sample
z-scores, shift(-1), resampling without closed='left' - all fail it.
"""
import numpy as np


def truncation_test(df, signal_fn, params, n_checks=40, seed=0, warmup=None):
    full = signal_fn(df, **params)
    if not full.index.equals(df.index):
        return {"passed": False, "reason": "signal index differs from data index"}
    n = len(df)
    lo = warmup if warmup is not None else min(n // 10, 500)
    rng = np.random.default_rng(seed)
    cuts = sorted(set(rng.integers(lo, n - 1, size=n_checks).tolist()) | {n - 2})
    for t in cuts:
        part = signal_fn(df.iloc[: t + 1].copy(), **params)
        a, b = full.iloc[t], part.iloc[-1]
        both_nan = np.isnan(a) and np.isnan(b)
        if not both_nan and not np.isclose(a, b, rtol=1e-9, atol=1e-12, equal_nan=True):
            return {"passed": False, "params": params, "bar": str(df.index[t]),
                    "full_history_value": float(a), "truncated_value": float(b),
                    "reason": "signal at bar t changes when later bars are removed: it uses future data"}
    return {"passed": True, "checks": len(cuts), "params": params}
