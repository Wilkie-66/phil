import json
import types
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from quant.engine import data, gates, stats, strategy, trials, validate
from quant.engine.backtest import backtest, window_backtest
from quant.engine.leakage import truncation_test
from quant.engine.metrics import sharpe_per_bar
from quant.engine.sizing import kill_switch, position_size
from quant.engine.walkforward import folds

ROOT = Path(__file__).resolve().parent.parent


# ---- backtest timing and costs -------------------------------------------

def test_signal_fills_next_bar_open(df, cfg):
    sig = pd.Series(0.0, index=df.index)
    sig.iloc[100] = 1.0  # decided at close of bar 100
    bt = backtest(df, sig, cfg)
    assert bt["position"].iloc[100] == 0.0
    assert bt["position"].iloc[101] == 1.0
    cfg["execution"]["approval_delay_bars"] = 2
    bt = backtest(df, sig, cfg)
    assert bt["position"].iloc[102] == 0.0 and bt["position"].iloc[103] == 1.0


def test_bar_return_uses_open_to_close_for_new_position(df, cfg):
    cfg["execution"]["fee_bps_per_side"] = cfg["execution"]["slippage_bps_per_side"] = 0.0
    sig = pd.Series(0.0, index=df.index)
    sig.iloc[100] = 1.0
    bt = backtest(df, sig, cfg)
    expect = df["close"].iloc[101] / df["open"].iloc[101] - 1
    assert bt["net"].iloc[101] == pytest.approx(expect)
    assert bt["net"].iloc[100] == 0.0  # the bar the signal was computed on earns nothing


def test_round_trip_cost_charged_twice(df, cfg):
    sig = pd.Series(0.0, index=df.index)
    sig.iloc[100:110] = 1.0
    bt = backtest(df, sig, cfg)
    per_side = (cfg["execution"]["fee_bps_per_side"] + cfg["execution"]["slippage_bps_per_side"]) / 1e4
    assert bt["costs"].sum() == pytest.approx(2 * per_side)


def test_long_only_clips_shorts(df, cfg):
    bt = backtest(df, pd.Series(-1.0, index=df.index), cfg)
    assert (bt["position"] == 0).all()


def test_window_backtest_starts_flat_and_charges_exit(df, cfg):
    sig = pd.Series(1.0, index=df.index)
    bt = window_backtest(df, sig, cfg, 500, 600)
    per_side = (cfg["execution"]["fee_bps_per_side"] + cfg["execution"]["slippage_bps_per_side"]) / 1e4
    assert len(bt) == 100 and bt.index[0] == df.index[500]
    assert bt["costs"].sum() == pytest.approx(2 * per_side)


# ---- leakage detector -----------------------------------------------------

def causal(df, n):
    return (df["close"] > df["close"].rolling(n).mean()).astype(float)


def centred(df, n):  # the classic repainting indicator
    return (df["close"] > df["close"].rolling(n, center=True).mean()).astype(float)


def full_sample_zscore(df, n):
    r = df["close"].pct_change(n)
    return ((r - r.mean()) / r.std() > 0).astype(float)


def peeks_next_bar(df, n):
    return (df["close"].shift(-1) > df["close"]).astype(float)


def test_leak_detector_passes_causal(df):
    assert truncation_test(df, causal, {"n": 50})["passed"]


@pytest.mark.parametrize("fn", [centred, full_sample_zscore, peeks_next_bar])
def test_leak_detector_catches_lookahead(df, fn):
    assert not truncation_test(df, fn, {"n": 50})["passed"]


# ---- deflated Sharpe ------------------------------------------------------

def test_psr_is_half_at_benchmark():
    assert stats.probabilistic_sharpe(0.02, 0.02, 1000, 0, 3) == pytest.approx(0.5)


def test_dsr_falls_as_trials_rise():
    kw = dict(sr=0.06, n_obs=2000, skew=0, kurtosis=3, trial_sharpes=[0.0, 0.03, -0.02, 0.01])
    vals = [stats.deflated_sharpe(n_trials=n, **kw)["deflated_sharpe"] for n in (1, 10, 100, 1000)]
    assert vals == sorted(vals, reverse=True) and vals[0] > vals[-1]


def test_best_of_many_noise_strategies_is_rejected():
    """The guide's failure case: test 200 random strategies, keep the best."""
    rng = np.random.default_rng(42)
    n_obs, n_trials = 2190, 200
    srs = [sharpe_per_bar(rng.normal(0, 0.01, n_obs)) for _ in range(n_trials)]
    best = max(srs)
    assert best * np.sqrt(2190) > 1.5  # looks great annualised...
    naive = stats.probabilistic_sharpe(best, 0, n_obs, 0, 3)
    dsr = stats.deflated_sharpe(best, n_obs, 0, 3, n_trials, srs)["deflated_sharpe"]
    assert naive > 0.95 and dsr < 0.95  # ...and only deflation sees through it


def test_guide_formula_passes_a_lucky_noise_strategy():
    """Documents the guide's unit bug: annualised Sharpe in a per-bar formula,
    no cross-trial spread. The same noise winner PASSES under it."""
    from scipy.stats import norm
    sharpe_ann, n_trials, n_obs = 3.0, 80, 2190
    e = stats.EULER
    expected_max = (1 - e) * norm.ppf(1 - 1 / n_trials) + e * norm.ppf(1 - 1 / (n_trials * np.e))
    denom = np.sqrt(1 + (3 - 1) / 4 * sharpe_ann ** 2)
    guide_dsr = norm.cdf((sharpe_ann - expected_max) * np.sqrt(n_obs - 1) / denom)
    assert guide_dsr > 0.95
    srs_bar = np.random.default_rng(0).normal(0, 1 / np.sqrt(n_obs), n_trials)
    ours = stats.deflated_sharpe(sharpe_ann / np.sqrt(2190), n_obs // 4, 0, 3, n_trials, srs_bar)
    assert ours["deflated_sharpe"] < guide_dsr


# ---- walk-forward ---------------------------------------------------------

def test_folds_are_causal_and_non_overlapping(cfg):
    fs = folds(5000, cfg["walk_forward"])
    assert len(fs) > 3
    for (a, b), (c, d) in fs:
        assert b + cfg["walk_forward"]["embargo_bars"] == c and d - c == cfg["walk_forward"]["test_bars"]
    tests = [range(c, d) for _, (c, d) in fs]
    assert all(t1.stop <= t2.start for t1, t2 in zip(tests, tests[1:]))


# ---- trial registry -------------------------------------------------------

def test_trials_dedupe_and_append(tmp_path):
    p = tmp_path / "t.jsonl"
    row = {"key": "a", "code_hash": "h", "params": {}, "symbol": "X",
           "timeframe_minutes": 240, "sharpe_bar_full_sample": 0.0}
    assert trials.record([row, row], p) == 1
    assert trials.record([row, {**row, "key": "b"}], p) == 1
    assert [r["key"] for r in trials.read_all(p)] == ["a", "b"]


# ---- data -----------------------------------------------------------------

def test_clean_fills_gaps_flat_and_drops_duplicates():
    df = data.synthetic(n_bars=20)
    broken = pd.concat([df.drop(df.index[5]), df.iloc[[3]]])
    out, rep = data.clean(broken, 240)
    assert rep["missing_bars_filled"] == 1 and rep["duplicates_dropped"] == 1
    assert out["close"].iloc[5] == out["close"].iloc[4] == out["open"].iloc[5]
    assert out["volume"].iloc[5] == 0


def test_drop_unclosed_removes_live_bar():
    df = data.synthetic(n_bars=10)
    now = df.index[-1] + pd.Timedelta(hours=1)  # last bar still open
    assert len(data.drop_unclosed(df, 240, now=now)) == 9


def test_kraken_archive_csv(tmp_path):
    p = tmp_path / "XBTUSD_240.csv"
    p.write_text("1609459200,29000,29400,28800,29300,12.5,300\n"
                 "1609473600,29300,29500,29100,29200,8.1,210\n")
    df = data.read_kraken_archive_csv(p)
    assert str(df.index[0]) == "2021-01-01 00:00:00+00:00" and df["close"].iloc[1] == 29200


# ---- strategy contract ------------------------------------------------------

def _write(tmp_path, body):
    p = tmp_path / "s.py"
    p.write_text(body)
    return p


def test_strategy_without_mechanism_is_rejected(tmp_path):
    p = _write(tmp_path, "HYPOTHESIS={'mechanism':'','counterparty':'x','invalidation':'y'}\n"
                         "PARAM_GRID=[{}]\ndef signal(df): return df['close']*0\n")
    with pytest.raises(strategy.StrategyError, match="mechanism"):
        strategy.load(p)


def test_parameter_sweep_is_rejected(tmp_path):
    p = _write(tmp_path, "HYPOTHESIS={'mechanism':'m','counterparty':'c','invalidation':'i'}\n"
                         "PARAM_GRID=[{'n':i} for i in range(51)]\ndef signal(df,n): return df['close']*0\n")
    with pytest.raises(strategy.StrategyError, match="sweep"):
        strategy.load(p)


# ---- sizing / kill switch ---------------------------------------------------

def test_position_size_respects_risk_and_cap(cfg):
    s = position_size(1000, 60000, 58800, cfg)  # 2% stop
    assert s["loss_if_stopped"] <= 10.0 + 1e-6
    tight = position_size(1000, 60000, 59990, cfg)
    assert tight["capped"] and tight["notional"] <= 250.0 + 1e-6


def test_kill_switch(cfg):
    assert kill_switch([1000, 1100, 930], cfg)["action"] == "HALT"
    assert kill_switch([1000, 1100, 1000], cfg)["action"] == "CONTINUE"


# ---- config tripwires -------------------------------------------------------

def test_shipped_config_is_valid(cfg):
    assert validate.check(cfg) == []


@pytest.mark.parametrize("path,value", [
    (("execution", "fee_bps_per_side"), 0.0),
    (("gates", "dsr_min"), 0.5),
    (("risk", "risk_per_trade_pct"), 0.1),
    (("live_trading_enabled",), True),
])
def test_loosened_config_trips(cfg, path, value):
    d = cfg
    for k in path[:-1]:
        d = d[k]
    d[path[-1]] = value
    assert validate.check(cfg)


# ---- end to end -------------------------------------------------------------

def _strat(signal_fn, grid):
    return types.SimpleNamespace(HYPOTHESIS={"mechanism": "m", "counterparty": "c", "invalidation": "i"},
                                 PARAM_GRID=grid, signal=signal_fn)


def test_example_strategy_rejected_on_noise(cfg, tmp_path):
    path = ROOT / "strategies" / "tsmom.py"
    rep = gates.evaluate(data.synthetic(n_bars=6000, seed=3), strategy.load(path), path,
                         "SYNTH", cfg, trials_path=tmp_path / "t.jsonl", write_report=False)
    assert rep["verdict"] == "REJECT" and rep["gates"]["leak"]["passed"]


def test_leaky_strategy_stops_at_gate_one(cfg, tmp_path, df):
    path = _write(tmp_path, "x = 1\n")
    rep = gates.evaluate(df, _strat(peeks_next_bar, [{"n": 1}]), path, "SYNTH", cfg,
                         trials_path=tmp_path / "t.jsonl", write_report=False)
    assert rep["verdict"] == "REJECT" and not rep["gates"]["leak"]["passed"]
    assert not (tmp_path / "t.jsonl").exists()  # never got far enough to log trials


def test_real_edge_can_pass(cfg, tmp_path):
    """Gates must not be impossible: plant persistent drift regimes and a
    strategy that follows them; it should clear all three."""
    rng = np.random.default_rng(11)
    n = 9000
    regime = np.repeat(rng.choice([-1, 1], size=n // 300 + 1), 300)[:n]
    r = regime * 0.004 + rng.normal(0, 0.008, n)
    close = 30000 * np.exp(np.cumsum(r))
    idx = pd.date_range("2019-01-01", periods=n, freq="4h", tz="UTC", name="time")
    open_ = np.concatenate([[30000], close[:-1]])
    df = pd.DataFrame({"open": open_, "high": np.maximum(open_, close) * 1.001,
                       "low": np.minimum(open_, close) * 0.999, "close": close, "volume": 1.0}, index=idx)
    path = _write(tmp_path, "x = 2\n")
    rep = gates.evaluate(df, _strat(causal, [{"n": 20}, {"n": 40}]), path, "PLANTED", cfg,
                         trials_path=tmp_path / "t.jsonl", write_report=False)
    assert rep["verdict"] == "PASS", json.dumps(rep["gates"], default=str)
