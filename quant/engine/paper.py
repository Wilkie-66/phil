"""Stage 3: forward-only paper trading of strategies that passed all gates.
PROTECTED (operator-owned).

Rules this module enforces:
  - only strategies with a PASS record in journal/passes.jsonl are traded, and
    only while the file's code hash still matches the hash that passed
    (edit the strategy -> it is flattened until re-evaluated)
  - forward only: a sleeve starts AFTER the last bar that existed when it was
    activated; nothing is ever traded on bars the backtest already saw
  - decisions use data up to the bar close only (the signal is recomputed on
    the truncated history); fills happen at a LATER bar's open with fees and
    slippage, exactly as in the backtest
  - an order is placed only if the decision was made live (within
    paper.live_grace_minutes of the bar close). Bars processed late because
    the machine was off are recorded as `missed`, never traded in hindsight
  - each strategy-symbol pair is a sleeve with its own cash
    (capital_usd * risk.max_position_pct); sleeves halt on their own drawdown
    limit, the whole account halts at risk.max_drawdown_halt_pct, and only
    `paper resume` (an operator act) lifts a halt

journal/paper-ledger.jsonl is written only here and is append-only (CI).
"""
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from . import strategy as strategy_mod
from . import trials
from .backtest import window_backtest
from .config import JOURNAL_DIR, QUANT_DIR
from .metrics import sharpe_per_bar

REPO_ROOT = QUANT_DIR.parent


def _now():
    return datetime.now(timezone.utc)


def _iso(ts):
    return pd.Timestamp(ts).isoformat()


def _read(path):
    path = Path(path)
    if not path.exists():
        return []
    with open(path) as f:
        return [json.loads(line) for line in f if line.strip()]


def _append(path, rows):
    if not rows:
        return
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a") as f:
        for r in rows:
            f.write(json.dumps(r, sort_keys=True, default=str) + "\n")


def _paths(journal_dir):
    j = Path(journal_dir)
    return j / "passes.jsonl", j / "paper-ledger.jsonl"


def _resolve(p):
    p = Path(p)
    return p if p.is_absolute() else REPO_ROOT / p


# ---- PASS records -----------------------------------------------------------

def record_pass(report, strat_path, journal_dir=JOURNAL_DIR):
    """Called by `evaluate` on a PASS against real data."""
    strat_path = Path(strat_path).resolve()
    try:
        rel = str(strat_path.relative_to(REPO_ROOT))
    except ValueError:
        rel = str(strat_path)
    m = report["oos_metrics"]
    row = {"ts": _iso(_now()), "strategy": strat_path.name, "path": rel,
           "code_hash": trials.code_hash(strat_path), "symbol": report["symbol"],
           "data_end": report["data"]["end"], "oos_sharpe_ann": m["sharpe_ann"],
           "oos_max_drawdown": m["max_drawdown"], "oos_trades": m["trades"],
           "deflated_sharpe": report["gates"]["deflated_sharpe"]["deflated_sharpe"]}
    _append(_paths(journal_dir)[0], [row])
    return row


def pass_key(p):
    return f"{p['strategy']}|{p['symbol']}"


def sleeve_id(p):
    """One sleeve per (strategy, symbol, code version): re-passing edited code
    starts a fresh sleeve and flattens the old one."""
    return f"{pass_key(p)}|{p['code_hash'][:8]}"


def latest_passes(journal_dir=JOURNAL_DIR):
    """Most recent PASS per sleeve, newest first, with a staleness flag."""
    latest = {}
    for p in _read(_paths(journal_dir)[0]):
        latest[pass_key(p)] = p
    out = []
    for p in sorted(latest.values(), key=lambda r: r["ts"], reverse=True):
        path = _resolve(p["path"])
        stale = None
        if not path.exists():
            stale = "strategy file missing"
        elif trials.code_hash(path) != p["code_hash"]:
            stale = "code changed since PASS - re-run evaluate"
        out.append({**p, "key": pass_key(p), "sleeve": sleeve_id(p), "stale": stale})
    return out


# ---- ledger replay ----------------------------------------------------------

class Sleeve:
    def __init__(self, start):
        self.id = start["sleeve"]
        self.start = start
        self.cash = start["cash"]
        self.units = 0.0
        self.last_bar = pd.Timestamp(start["start_after"])
        self.frac = 0.0
        self.params = None
        self.refit_bar = None
        self.pending = None
        self.peak = start["cash"]
        self.halted = False
        self.equity_marks = []  # (bar, equity at close)
        self.fills = 0
        self.missed = 0

    def apply(self, r):
        t = r["type"]
        if t == "decision":
            self.last_bar = pd.Timestamp(r["bar"])
            self.params = r["params"]
            self.refit_bar = pd.Timestamp(r["refit_bar"]) if r.get("refit_bar") else None
            self.peak = max(self.peak, r["equity"])
            self.equity_marks.append((self.last_bar, r["equity"]))
            self.missed += int(r.get("missed", False))
            if r["order"]:
                self.frac = r["target"]
                self.pending = {"fill_bar": r["fill_bar"], "target": r["target"]}
        elif t == "fill":
            self.cash, self.units = r["cash_after"], r["units_after"]
            self.pending = None
            self.fills += int(r["units_delta"] != 0)
        elif t == "cancel":
            self.pending = None
        elif t == "halt":
            self.halted = True
        elif t == "resume":
            self.halted = False
            self.peak = r.get("equity", self.peak)

    def equity(self, price):
        return self.cash + self.units * price


def replay(ledger):
    """-> (sleeves, account_halted, last account resume row or None)"""
    sleeves, account_halted, account_resume = {}, False, None
    for r in ledger:
        if r["type"] == "start":
            sleeves[r["sleeve"]] = Sleeve(r)
        elif r.get("scope") == "account":
            if r["type"] == "halt":
                account_halted = True
            elif r["type"] == "resume":
                account_halted = False
                account_resume = r
        elif r.get("sleeve") in sleeves:
            sleeves[r["sleeve"]].apply(r)
    return sleeves, account_halted, account_resume


# ---- one sleeve, one run ----------------------------------------------------

def _select_params(df_t, strat, cfg):
    """Best grid entry by Sharpe over the most recent training window -
    the same choice walk-forward made on every fold."""
    n = len(df_t)
    a = max(0, n - cfg["walk_forward"]["train_bars"])
    scores = {}
    for p in strat.PARAM_GRID:
        sig = strat.signal(df_t, **p).reindex(df_t.index)
        scores[json.dumps(p, sort_keys=True)] = sharpe_per_bar(window_backtest(df_t, sig, cfg, a, n)["net"])
    best = max(scores, key=scores.get)
    return json.loads(best)


def _fill(sl, bar, open_px, cfg):
    ex, pc = cfg["execution"], cfg["paper"]
    fee, slip = ex["fee_bps_per_side"] / 1e4, ex["slippage_bps_per_side"] / 1e4
    eq = sl.equity(open_px)
    want = sl.pending["target"] * eq / open_px
    buying = want > sl.units
    px = open_px * (1 + slip) if buying else open_px * (1 - slip)
    if buying:  # leave room for the fee so cash never goes negative
        want = sl.pending["target"] * eq / (px * (1 + fee))
    delta = want - sl.units
    if abs(delta) * px < pc["min_order_usd"] and sl.pending["target"] > 0:
        delta = 0.0  # below Kraken's minimum order size: no trade
    cost = abs(delta) * px * fee
    cash = sl.cash - delta * px - cost
    units = sl.units + delta
    if sl.pending["target"] == 0:
        units = 0.0 if abs(units) < 1e-12 else units
    return {"type": "fill", "sleeve": sl.id, "bar": _iso(bar), "open": open_px, "price": px,
            "units_delta": delta, "fee_usd": cost, "units_after": units, "cash_after": cash,
            "target": sl.pending["target"]}


def step_sleeve(sl, info, df, strat, cfg, now, account_halted):
    """Process every closed bar after the sleeve's last processed bar."""
    tf = pd.Timedelta(minutes=cfg["timeframe_minutes"])
    pc, d = cfg["paper"], cfg["execution"]["approval_delay_bars"]
    grace = pd.Timedelta(minutes=pc["live_grace_minutes"])
    dd_limit = min(pc["sleeve_dd_halt_mult"] * abs(sl.start["oos_max_drawdown"]),
                   pc["sleeve_dd_halt_max"])
    rows = []
    for t in np.flatnonzero(df.index > sl.last_bar):
        bar = df.index[t]
        o, c = float(df["open"].iloc[t]), float(df["close"].iloc[t])
        if sl.pending and pd.Timestamp(sl.pending["fill_bar"]) == bar:
            r = _fill(sl, bar, o, cfg)
            sl.apply(r)
            rows.append(r)
        elif sl.pending and pd.Timestamp(sl.pending["fill_bar"]) < bar:
            r = {"type": "cancel", "sleeve": sl.id, "bar": _iso(bar), "reason": "fill bar missing from data"}
            sl.apply(r)
            rows.append(r)

        eq = sl.equity(c)
        if not sl.halted and eq / max(sl.peak, eq) - 1 <= -dd_limit:
            r = {"type": "halt", "sleeve": sl.id, "bar": _iso(bar), "equity": eq,
                 "reason": f"sleeve drawdown beyond {dd_limit:.0%}"}
            sl.apply(r)
            rows.append(r)

        stop_reason = info["stale"] or ("sleeve halted" if sl.halted else None) or \
            ("account halted" if account_halted else None)
        df_t = df.iloc[: t + 1]
        refit_every = cfg["walk_forward"]["test_bars"]
        refit = None
        if stop_reason:
            target = 0.0
        else:
            if sl.params is None or sl.refit_bar is None or (bar - sl.refit_bar) >= refit_every * tf:
                sl.params, sl.refit_bar, refit = _select_params(df_t, strat, cfg), bar, True
            s = strat.signal(df_t, **sl.params).iloc[-1]
            target = 0.0 if pd.isna(s) else float(np.clip(s, 0.0, 1.0))

        live = (pd.Timestamp(now) - (bar + tf)) <= grace
        wants = target != sl.frac
        order = bool(wants and live and sl.pending is None)
        r = {"type": "decision", "sleeve": sl.id, "bar": _iso(bar), "close": c, "equity": eq,
             "params": sl.params, "refit_bar": _iso(sl.refit_bar) if sl.refit_bar is not None else None,
             "refit": bool(refit), "target": target, "position_frac": sl.frac, "order": order,
             "fill_bar": _iso(bar + (1 + d) * tf) if order else None, "live": bool(live),
             "missed": bool(wants and not live), "stop_reason": stop_reason,
             "decided_at": _iso(now)}
        sl.apply(r)
        rows.append(r)
    return rows


# ---- account level ----------------------------------------------------------

def account_curve(sleeves, cfg):
    """Account equity at each bar close = capital + sum of sleeve P&L."""
    series = {}
    for sl in sleeves.values():
        if sl.equity_marks:
            s = pd.Series(dict(sl.equity_marks)) - sl.start["cash"]
            series[sl.id] = s[~s.index.duplicated(keep="last")]
    if not series:
        return pd.Series(dtype=float)
    pnl = pd.DataFrame(series).sort_index().ffill().fillna(0.0).sum(axis=1)
    return cfg["capital_usd"] + pnl


def run(cfg, load_df, journal_dir=JOURNAL_DIR, now=None):
    """One paper cycle. load_df(symbol) -> cleaned OHLC frame of CLOSED bars."""
    now = pd.Timestamp(now or _now())
    passes_path, ledger_path = _paths(journal_dir)
    ledger = _read(ledger_path)
    sleeves, account_halted, account_resume = replay(ledger)
    passes = {p["key"]: p for p in latest_passes(journal_dir)}
    summary = {"started": [], "skipped": [], "processed": {}}
    new_rows = []

    def stale_reason(sl):
        p = passes.get(f"{sl.start['strategy']}|{sl.start['symbol']}")
        if p is None:
            return "no PASS record"
        if p["code_hash"] != sl.start["code_hash"]:
            return "superseded by a newer PASS - flattening"
        return p["stale"]

    # activate new PASSes, up to the sleeve limit (counting live sleeves only)
    live_count = sum(1 for sl in sleeves.values() if not stale_reason(sl))
    for p in passes.values():
        k = p["sleeve"]
        if k in sleeves:
            continue
        if p["stale"]:
            summary["skipped"].append({"sleeve": k, "reason": p["stale"]})
            continue
        if live_count >= cfg["paper"]["max_sleeves"]:
            summary["skipped"].append({"sleeve": k, "reason": "max_sleeves reached"})
            continue
        df = load_df(p["symbol"])
        r = {"type": "start", "sleeve": k, "ts": _iso(now), "strategy": p["strategy"],
             "path": p["path"], "code_hash": p["code_hash"], "symbol": p["symbol"],
             "start_after": _iso(df.index[-1]), "cash": cfg["capital_usd"] * cfg["risk"]["max_position_pct"],
             "oos_sharpe_ann": p["oos_sharpe_ann"], "oos_max_drawdown": p["oos_max_drawdown"]}
        new_rows.append(r)
        sleeves[k] = Sleeve(r)
        live_count += 1
        summary["started"].append(k)

    for k, sl in sleeves.items():
        stale = stale_reason(sl)
        if stale and sl.frac == 0 and sl.pending is None and sl.units == 0:
            continue  # retired: flat and never trading again
        strat = None if stale else strategy_mod.load(_resolve(sl.start["path"]))
        df = load_df(sl.start["symbol"])
        rows = step_sleeve(sl, {"stale": stale}, df, strat, cfg, now, account_halted)
        new_rows += rows
        summary["processed"][k] = {"bars": sum(r["type"] == "decision" for r in rows),
                                   "orders": sum(bool(r.get("order")) for r in rows),
                                   "fills": sum(r["type"] == "fill" for r in rows),
                                   **({"stale": stale} if stale else {})}

    curve = account_curve(sleeves, cfg)
    if len(curve) and not account_halted:
        if account_resume is not None:
            after = curve[curve.index > pd.Timestamp(account_resume["as_of"])]
            peak = max([account_resume["equity"], *after.tolist()])
        else:
            peak = float(curve.max())
        dd = curve.iloc[-1] / peak - 1
        if dd <= -cfg["risk"]["max_drawdown_halt_pct"]:
            new_rows.append({"type": "halt", "scope": "account", "ts": _iso(now),
                             "equity": float(curve.iloc[-1]),
                             "reason": f"account drawdown {dd:.1%} beyond "
                                       f"{cfg['risk']['max_drawdown_halt_pct']:.0%}"})
            summary["account_halted"] = True
    _append(ledger_path, new_rows)
    summary["rows_written"] = len(new_rows)
    return summary


def resume(scope, journal_dir=JOURNAL_DIR, cfg=None, now=None):
    """Operator act: lift a halt. scope = 'account' or a sleeve id."""
    _, ledger_path = _paths(journal_dir)
    sleeves, _, _ = replay(_read(ledger_path))
    row = {"type": "resume", "ts": _iso(now or _now())}
    if scope == "account":
        row["scope"] = "account"
        if cfg is not None:
            curve = account_curve(sleeves, cfg)
            if len(curve):  # the drawdown peak restarts from here
                row["equity"], row["as_of"] = float(curve.iloc[-1]), _iso(curve.index[-1])
            else:
                row["equity"], row["as_of"] = cfg["capital_usd"], _iso(now or _now())
    else:
        if scope not in sleeves:
            raise ValueError(f"unknown sleeve {scope!r}; known: {sorted(sleeves)}")
        sl = sleeves[scope]
        row["sleeve"] = scope
        if sl.equity_marks:
            row["equity"] = sl.equity_marks[-1][1]
    _append(ledger_path, [row])
    return row


def _status_stale(sl, passes):
    p = passes.get(f"{sl.start['strategy']}|{sl.start['symbol']}")
    if p is None:
        return "no PASS record"
    if p["code_hash"] != sl.start["code_hash"]:
        return "superseded by a newer PASS"
    return p["stale"]


def status(cfg, journal_dir=JOURNAL_DIR):
    _, ledger_path = _paths(journal_dir)
    ledger = _read(ledger_path)
    sleeves, account_halted, _ = replay(ledger)
    passes = {p["key"]: p for p in latest_passes(journal_dir)}
    out = {"account": {}, "sleeves": []}
    curve = account_curve(sleeves, cfg)
    if len(curve):
        out["account"] = {"equity": round(float(curve.iloc[-1]), 2),
                          "return": round(float(curve.iloc[-1] / cfg["capital_usd"] - 1), 4),
                          "drawdown": round(float(curve.iloc[-1] / curve.cummax().iloc[-1] - 1), 4),
                          "as_of": _iso(curve.index[-1])}
    out["account"]["halted"] = account_halted
    hmin = cfg["paper"]["health_min_bars"]
    for k, sl in sleeves.items():
        marks = pd.Series(dict(sl.equity_marks)) if sl.equity_marks else pd.Series(dtype=float)
        eq = float(marks.iloc[-1]) if len(marks) else sl.start["cash"]
        row = {"sleeve": k, "equity": round(eq, 2),
               "return": round(eq / sl.start["cash"] - 1, 4), "position_frac": sl.frac,
               "units": sl.units, "params": sl.params, "fills": sl.fills, "missed_bars": sl.missed,
               "bars": len(marks), "halted": sl.halted, "pending": sl.pending,
               "stale": _status_stale(sl, passes),
               "backtest_oos_sharpe_ann": sl.start["oos_sharpe_ann"]}
        if len(marks) >= hmin:
            r = marks.pct_change().dropna()
            live_sr = sharpe_per_bar(r) * np.sqrt(cfg["bars_per_year"])
            row["paper_sharpe_ann"] = round(live_sr, 3)
            if live_sr < 0.5 * sl.start["oos_sharpe_ann"]:
                row["warning"] = "paper Sharpe below half the backtest's - edge may be decaying"
        else:
            row["paper_sharpe_ann"] = f"n/a until {hmin} bars"
        out["sleeves"].append(row)
    return out
