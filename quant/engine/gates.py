"""The three gates. PROTECTED (operator-owned).

  1. LEAK   - automated truncation test passes for every grid entry
  2. DSR    - out-of-sample Sharpe survives deflation for ALL trials ever logged
  3. WALK   - walk-forward: enough folds, most positive, worst fold bounded,
              enough trades, and better risk-adjusted than just holding

A strategy is a candidate for paper trading only if all three pass.
"""
import json
from datetime import datetime, timezone

from . import trials
from .backtest import backtest
from .config import JOURNAL_DIR
from .leakage import truncation_test
from .metrics import metrics, sharpe_per_bar
from .stats import deflated_sharpe
from .walkforward import regime_report, walk_forward


def evaluate(df, strat, strat_path, symbol, cfg, trials_path=None, write_report=True):
    g = cfg["gates"]
    tpath = trials_path or trials.TRIALS_PATH
    grid = strat.PARAM_GRID
    report = {"strategy": strat_path.name, "symbol": symbol,
              "hypothesis": strat.HYPOTHESIS, "data": {"start": str(df.index[0]),
              "end": str(df.index[-1]), "bars": int(len(df))},
              "execution": cfg["execution"], "gates": {}}

    # Gate 1: leakage
    leak = [truncation_test(df, strat.signal, p, n_checks=g["leak_checks"]) for p in grid]
    failed = [x for x in leak if not x["passed"]]
    report["gates"]["leak"] = {"passed": not failed, "detail": failed[0] if failed else
                               f"{len(grid)} param sets x {leak[0]['checks']} truncation checks"}
    if failed:
        return _finish(report, False, write_report)

    # every grid entry is a trial; log it before looking at OOS results
    signals, rows = {}, []
    chash = trials.code_hash(strat_path)
    for p in grid:
        key = json.dumps(p, sort_keys=True)
        s = strat.signal(df, **p).reindex(df.index)
        signals[key] = s
        sr = sharpe_per_bar(backtest(df, s, cfg)["net"])
        rows.append({"key": trials.trial_key(chash, p, symbol), "strategy": strat_path.name,
                     "code_hash": chash, "params": p, "symbol": symbol,
                     "timeframe_minutes": cfg["timeframe_minutes"], "n_bars": int(len(df)),
                     "sharpe_bar_full_sample": round(sr, 6)})
    report["trials_added"] = trials.record(rows, tpath)

    # Gate 3 data: walk-forward
    wf = walk_forward(df, signals, cfg)
    n_folds = len(wf["folds"])
    if n_folds == 0:
        report["gates"]["walk_forward"] = {"passed": False, "reason": "not enough data for one fold"}
        return _finish(report, False, write_report)
    oos, hold = wf["oos_net"], wf["oos_hold_net"]
    m_oos = metrics(oos, cfg, trades=wf["oos_trades"])
    m_hold = metrics(hold, cfg)
    f = wf["folds"]
    pos_frac = float((f["total_return"] > 0).mean())
    worst = float(f["total_return"].min())
    checks = {
        "folds": (n_folds >= g["min_folds"], f"{n_folds} (min {g['min_folds']})"),
        "positive_folds": (pos_frac >= g["min_positive_fold_frac"],
                           f"{int((f['total_return'] > 0).sum())}/{n_folds}"),
        "worst_fold_return": (worst >= g["worst_fold_return_min"],
                              f"{worst:.2%} (min {g['worst_fold_return_min']:.0%})"),
        "oos_trades": (wf["oos_trades"] >= g["min_oos_trades"],
                       f"{wf['oos_trades']} (min {g['min_oos_trades']})"),
    }
    if g["must_beat_buy_and_hold_sharpe"]:
        checks["beats_buy_and_hold"] = (m_oos["sharpe_ann"] > m_hold["sharpe_ann"],
                                        f"{m_oos['sharpe_ann']} vs hold {m_hold['sharpe_ann']}")
    report["gates"]["walk_forward"] = {"passed": all(v[0] for v in checks.values()),
                                       "checks": {k: {"ok": v[0], "value": v[1]} for k, v in checks.items()}}

    # Gate 2: deflated Sharpe on stitched OOS, against every trial on this timeframe
    fam = trials.family(cfg["timeframe_minutes"], tpath)
    dsr = deflated_sharpe(sr=m_oos["sharpe_bar"], n_obs=m_oos["n_bars"], skew=m_oos["skew"],
                          kurtosis=m_oos["kurtosis"], n_trials=len(fam),
                          trial_sharpes=[r["sharpe_bar_full_sample"] for r in fam])
    report["gates"]["deflated_sharpe"] = {"passed": dsr["deflated_sharpe"] >= g["dsr_min"],
                                          "min": g["dsr_min"], **dsr}

    report["oos_metrics"] = m_oos
    report["buy_and_hold_oos_metrics"] = m_hold
    report["regimes"] = regime_report(df, oos, cfg)
    report["folds"] = f.to_dict(orient="records")
    ok = all(report["gates"][k]["passed"] for k in ("leak", "walk_forward", "deflated_sharpe"))
    return _finish(report, ok, write_report)


def _finish(report, ok, write_report):
    report["verdict"] = "PASS" if ok else "REJECT"
    report["evaluated_at"] = datetime.now(timezone.utc).isoformat(timespec="seconds")
    if write_report:
        out = JOURNAL_DIR / "evaluations"
        out.mkdir(parents=True, exist_ok=True)
        stamp = report["evaluated_at"].replace(":", "").replace("-", "")[:15]
        path = out / f"{stamp}-{report['strategy'].removesuffix('.py')}-{report['symbol']}.json"
        path.write_text(json.dumps(report, indent=2, default=str))
        report["report_path"] = str(path)
    return report
