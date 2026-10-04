import json

import numpy as np
import pandas as pd
import pytest

from quant.engine import paper, trials
from quant.engine.backtest import window_backtest

TF = pd.Timedelta(hours=4)

STRAT = '''
HYPOTHESIS = {"mechanism": "m", "counterparty": "c", "invalidation": "i"}
PARAM_GRID = [{"n": 20}]
def signal(df, n):
    return (df["close"] > df["close"].rolling(n).mean()).astype(float)
'''


def planted(n=3000, seed=11, step=0.004):
    rng = np.random.default_rng(seed)
    regime = np.repeat(rng.choice([-1, 1], size=n // 300 + 1), 300)[:n]
    r = regime * step + rng.normal(0, 0.008, n)
    close = 30000 * np.exp(np.cumsum(r))
    open_ = np.concatenate([[30000], close[:-1]]) * np.exp(rng.normal(0, 0.001, n))
    idx = pd.date_range("2024-01-01", periods=n, freq="4h", tz="UTC", name="time")
    return pd.DataFrame({"open": open_, "high": np.maximum(open_, close) * 1.002,
                         "low": np.minimum(open_, close) * 0.998, "close": close, "volume": 1.0}, index=idx)


@pytest.fixture
def setup(tmp_path, cfg):
    spath = tmp_path / "strat.py"
    spath.write_text(STRAT)
    jdir = tmp_path / "journal"
    report = {"symbol": "XBTUSD", "data": {"end": "x"},
              "oos_metrics": {"sharpe_ann": 2.0, "max_drawdown": -0.15, "trades": 100},
              "gates": {"deflated_sharpe": {"deflated_sharpe": 0.99}}}
    paper.record_pass(report, spath, jdir)
    return spath, jdir, cfg


def simulate(df, cfg, jdir, first, last, lag=pd.Timedelta(minutes=5)):
    """Run a paper cycle after each bar closes, seeing only closed bars."""
    for k in range(first, last + 1):
        visible = df.iloc[:k]
        now = visible.index[-1] + TF + lag
        paper.run(cfg, lambda s, v=visible: v, journal_dir=jdir, now=now)


def ledger(jdir):
    return paper._read(jdir / "paper-ledger.jsonl")


def test_forward_only_and_honest_fills(setup):
    _, jdir, cfg = setup
    df = planted()
    simulate(df, cfg, jdir, 1000, 1400)
    rows = ledger(jdir)
    start = rows[0]
    assert start["type"] == "start" and start["start_after"] == df.index[999].isoformat()
    decisions = [r for r in rows if r["type"] == "decision"]
    assert pd.Timestamp(decisions[0]["bar"]) == df.index[1000]  # nothing before activation
    fills = [r for r in rows if r["type"] == "fill"]
    assert fills, "planted trend should produce trades"
    slip = cfg["execution"]["slippage_bps_per_side"] / 1e4
    for f in fills:
        o = df.loc[pd.Timestamp(f["bar"]), "open"]
        assert f["open"] == pytest.approx(o)
        assert f["price"] == pytest.approx(o * (1 + slip) if f["units_delta"] > 0 else o * (1 - slip))
    # every fill is at the bar after the decision that ordered it
    ordered = {r["fill_bar"] for r in decisions if r["order"]}
    assert {f["bar"] for f in fills} <= ordered
    assert all(pd.Timestamp(r["fill_bar"]) == pd.Timestamp(r["bar"]) + TF for r in decisions if r["order"])


def test_paper_matches_backtest(setup):
    """Same strategy, same bars: paper equity should track the protected backtest."""
    _, jdir, cfg = setup
    df = planted()
    simulate(df, cfg, jdir, 1000, 1600)
    sleeve = paper.status(cfg, jdir)["sleeves"][0]
    ns = {}
    exec(STRAT, ns)
    sig = ns["signal"](df, 20)
    bt = window_backtest(df.iloc[:1600], sig.iloc[:1600], cfg, 1001, 1600)
    bt_ret = float((1 + bt["net"]).prod() - 1)
    assert sleeve["return"] == pytest.approx(bt_ret, abs=0.02)
    assert sleeve["return"] > 0.05  # the planted edge is real


def test_late_bars_are_missed_not_traded(setup):
    _, jdir, cfg = setup
    df = planted()
    simulate(df, cfg, jdir, 1000, 1000)
    # machine was off for 200 bars; next run happens a day after the last close
    paper.run(cfg, lambda s: df.iloc[:1200], journal_dir=jdir,
              now=df.index[1199] + TF + pd.Timedelta(days=1))
    late = [r for r in ledger(jdir) if r["type"] == "decision" and not r["live"]]
    assert len(late) == 200
    assert not any(r["order"] for r in late)
    assert any(r["missed"] for r in late)


def test_edited_strategy_is_flattened_then_retired(setup):
    spath, jdir, cfg = setup
    df = planted()
    k = 1000
    simulate(df, cfg, jdir, k, k)
    while paper.status(cfg, jdir)["sleeves"][0]["units"] == 0:  # wait until long
        k += 1
        simulate(df, cfg, jdir, k, k)
    spath.write_text(STRAT + "\n# tweak\n")
    n_before = len(ledger(jdir))
    simulate(df, cfg, jdir, k + 1, k + 6)
    new = ledger(jdir)[n_before:]
    first = next(r for r in new if r["type"] == "decision")
    assert first["stop_reason"].startswith("code changed") and first["target"] == 0.0
    s = paper.status(cfg, jdir)["sleeves"][0]
    assert s["stale"] and s["units"] == 0.0 and s["position_frac"] == 0.0
    # flat and stale -> retired: it stops writing decisions well before the 6 bars run out
    assert sum(r["type"] == "decision" for r in new) <= 3


def test_gap_crash_halts_sleeve_and_account(setup):
    _, jdir, cfg = setup
    base = planted().iloc[:1000]
    # strong uptrend (strategy goes long), then the market gaps down 70% in one bar
    idx = pd.date_range(base.index[-1] + TF, periods=60, freq="4h", tz="UTC", name="time")
    px = base["close"].iloc[-1] * np.concatenate([np.linspace(1.0, 1.3, 40), np.full(20, 0.39)])
    tail = pd.DataFrame({"open": px, "high": px * 1.001, "low": px * 0.999, "close": px, "volume": 1.0},
                        index=idx)
    full = pd.concat([base, tail])
    simulate(full, cfg, jdir, 1000, len(full))
    halts = [r for r in ledger(jdir) if r["type"] == "halt"]
    assert any(r.get("sleeve") for r in halts), "70% gap in a long sleeve must trip its halt"
    assert any(r.get("scope") == "account" for r in halts), "~17% account loss must trip the 15% halt"
    st = paper.status(cfg, jdir)
    assert st["account"]["halted"] and st["sleeves"][0]["halted"] and st["sleeves"][0]["units"] == 0.0
    # halts stick until the operator lifts them
    simulate(full, cfg, jdir, len(full), len(full))
    assert paper.status(cfg, jdir)["account"]["halted"]
    paper.resume("account", journal_dir=jdir, cfg=cfg)
    assert not paper.status(cfg, jdir)["account"]["halted"]


def test_sleeve_limit(tmp_path, cfg):
    jdir = tmp_path / "j"
    cfg["paper"]["max_sleeves"] = 1
    for i in range(2):
        sp = tmp_path / f"s{i}.py"
        sp.write_text(STRAT + f"\n# {i}\n")
        paper.record_pass({"symbol": "XBTUSD", "data": {"end": "x"},
                           "oos_metrics": {"sharpe_ann": 1, "max_drawdown": -0.1, "trades": 50},
                           "gates": {"deflated_sharpe": {"deflated_sharpe": 0.97}}}, sp, jdir)
    df = planted()
    out = paper.run(cfg, lambda s: df.iloc[:1000], journal_dir=jdir, now=df.index[999] + TF)
    assert len(out["started"]) == 1 and out["skipped"][0]["reason"] == "max_sleeves reached"


def test_pass_record_has_code_hash(setup):
    spath, jdir, _ = setup
    p = paper.latest_passes(jdir)[0]
    assert p["code_hash"] == trials.code_hash(spath) and p["stale"] is None
    json.dumps(p)
