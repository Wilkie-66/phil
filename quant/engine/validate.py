"""Integrity tripwires for quant/. PROTECTED (operator-owned). Run by CI.

Hard ceilings live here, so loosening a gate or cap takes two operator
edits (config.json AND this file) - never a strategy commit.
"""
import json
import py_compile
import sys

from .config import CONFIG_PATH, QUANT_DIR, TRIALS_PATH, load_config


def check(cfg):
    errs = []

    def need(cond, msg):
        if not cond:
            errs.append(msg)

    ex, wf, g, r = cfg["execution"], cfg["walk_forward"], cfg["gates"], cfg["risk"]
    need(cfg["live_trading_enabled"] is False, "live_trading_enabled must be false (no live executor exists yet)")
    need(cfg["bars_per_year"] == 365 * 24 * 60 // cfg["timeframe_minutes"],
         "bars_per_year must match the timeframe for a 24/7 market")
    need(ex["fee_bps_per_side"] >= 25, "fee_bps_per_side below Kraken's lowest-tier maker fee (25 bps)")
    need(ex["slippage_bps_per_side"] >= 2, "slippage_bps_per_side must be >= 2")
    need(0 <= ex["approval_delay_bars"] <= 6, "approval_delay_bars must be in [0, 6]")
    need(ex["long_only"] is True, "long_only must stay true (spot account, no margin)")
    need(wf["train_bars"] >= 500 and wf["test_bars"] >= 120 and wf["embargo_bars"] >= 0,
         "walk-forward windows too short")
    need(g["dsr_min"] >= 0.90, "dsr_min below 0.90")
    need(g["min_folds"] >= 4, "min_folds below 4")
    need(g["min_positive_fold_frac"] >= 0.5, "min_positive_fold_frac below 0.5")
    need(g["worst_fold_return_min"] >= -0.25, "worst_fold_return_min looser than -25%")
    need(g["min_oos_trades"] >= 20, "min_oos_trades below 20")
    need(g["leak_checks"] >= 20, "leak_checks below 20")
    need(0 < r["risk_per_trade_pct"] <= 0.02, "risk_per_trade_pct must be in (0, 2%]")
    need(0 < r["max_position_pct"] <= 0.5, "max_position_pct must be in (0, 50%]")
    need(1 <= r["max_open_positions"] <= 5, "max_open_positions must be in [1, 5]")
    need(0 < r["max_drawdown_halt_pct"] <= 0.25, "max_drawdown_halt_pct must be in (0, 25%]")
    return errs


def check_trials(path=TRIALS_PATH):
    errs = []
    if not path.exists():
        return errs
    keys = set()
    for i, line in enumerate(path.read_text().splitlines(), 1):
        if not line.strip():
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError:
            errs.append(f"trials.jsonl line {i} is not JSON")
            continue
        for k in ("key", "code_hash", "params", "symbol", "timeframe_minutes", "sharpe_bar_full_sample"):
            if k not in row:
                errs.append(f"trials.jsonl line {i} missing {k}")
        if row.get("key") in keys:
            errs.append(f"trials.jsonl line {i} duplicates key {row.get('key')}")
        keys.add(row.get("key"))
    return errs


def main():
    errs = []
    try:
        errs += check(load_config(CONFIG_PATH))
    except (KeyError, json.JSONDecodeError) as e:
        errs.append(f"config.json unreadable: {e!r}")
    errs += check_trials()
    for f in sorted(QUANT_DIR.rglob("*.py")):
        try:
            py_compile.compile(str(f), doraise=True)
        except py_compile.PyCompileError as e:
            errs.append(f"{f}: {e.msg}")
    for e in errs:
        print(f"FAIL: {e}")
    print("quant validate: OK" if not errs else f"quant validate: {len(errs)} problem(s)")
    return 1 if errs else 0


if __name__ == "__main__":
    sys.exit(main())
