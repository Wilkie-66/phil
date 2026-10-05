"""CLI: python -m quant <command>   (run from the repo root)

  import-csv PATH --symbol XBTUSD     load a Kraken OHLCVT archive file into the cache
  fetch --symbol XBTUSD               top the cache up from Kraken's public API
  evaluate STRATEGY.py --symbol XBTUSD [--synthetic]
  trials                              how many variations have been tried so far
  paper run                           stage 3: process new closed bars for every passed strategy
  paper status                        sleeves, positions, P&L vs backtest, halts
  paper resume --scope account|SLEEVE operator act: lift a halt
  validate                            integrity tripwires (CI runs this)
"""
import argparse
import json
import sys
from pathlib import Path

from .engine import data, gates, paper, strategy, trials
from .engine.config import JOURNAL_DIR, load_config
from .engine.validate import main as validate_main


def cmd_import(a, cfg):
    df = data.read_kraken_archive_csv(a.path)
    print(json.dumps(data.save_merged(df, a.symbol, cfg["timeframe_minutes"]), indent=2))


def cmd_fetch(a, cfg):
    print(json.dumps(data.update_from_api(a.symbol, cfg["timeframe_minutes"]), indent=2))


def cmd_evaluate(a, cfg):
    if a.synthetic:
        df, info = data.synthetic(seed=a.seed), {"source": "synthetic random walk (no edge)"}
        symbol = f"SYNTH{a.seed}"
    else:
        df, info = data.load(a.symbol, cfg["timeframe_minutes"])
        symbol = a.symbol
    path = Path(a.strategy).resolve()
    strat = strategy.load(path)
    # synthetic runs must never inflate (or dilute) the real trial count
    tpath = JOURNAL_DIR / "trials-synthetic.jsonl" if a.synthetic else None
    rep = gates.evaluate(df, strat, path, symbol, cfg, trials_path=tpath)
    print(f"\n{rep['strategy']} on {symbol}  ({rep['data']['start'][:10]} -> {rep['data']['end'][:10]}, "
          f"{rep['data']['bars']} bars)")
    for name, gate in rep["gates"].items():
        print(f"  [{'PASS' if gate['passed'] else 'FAIL'}] {name}")
        if name == "walk_forward" and "checks" in gate:
            for k, v in gate["checks"].items():
                print(f"         {'ok ' if v['ok'] else 'NO '} {k}: {v['value']}")
        elif name == "deflated_sharpe":
            print(f"         DSR {gate['deflated_sharpe']} (min {gate['min']}) over "
                  f"{gate['n_trials']} logged trials")
        elif not gate["passed"]:
            print(f"         {gate.get('detail') or gate.get('reason')}")
    if "oos_metrics" in rep:
        m, h = rep["oos_metrics"], rep["buy_and_hold_oos_metrics"]
        print(f"  OOS: Sharpe {m['sharpe_ann']}  return {m['total_return']:.1%}  maxDD "
              f"{m['max_drawdown']:.1%}  trades {m['trades']}   | hold: Sharpe {h['sharpe_ann']} "
              f"return {h['total_return']:.1%} maxDD {h['max_drawdown']:.1%}")
    print(f"  VERDICT: {rep['verdict']}   (report: {rep.get('report_path')})\n")
    if rep["verdict"] == "PASS" and not a.synthetic:
        paper.record_pass(rep, path)
        print("  Recorded in journal/passes.jsonl - `python -m quant paper run` will start paper trading it.\n")
    return 0 if rep["verdict"] == "PASS" else 2


def cmd_trials(a, cfg):
    fam = trials.family(cfg["timeframe_minutes"])
    by = {}
    for r in fam:
        by.setdefault(r["strategy"], 0)
        by[r["strategy"]] += 1
    print(f"{len(fam)} trials on {cfg['timeframe_minutes']}m bars")
    for k, v in sorted(by.items(), key=lambda kv: -kv[1]):
        print(f"  {v:4d}  {k}")


def cmd_paper(a, cfg):
    if a.action == "run":
        def load_df(symbol):
            return data.load(symbol, cfg["timeframe_minutes"])[0]
        out = paper.run(cfg, load_df)
    elif a.action == "resume":
        if not a.scope:
            raise SystemExit("paper resume needs --scope account|<sleeve id>")
        out = paper.resume(a.scope, cfg=cfg)
    else:
        out = paper.status(cfg)
    print(json.dumps(out, indent=2, default=str))


def main(argv=None):
    p = argparse.ArgumentParser(prog="python -m quant")
    sub = p.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("import-csv"); s.add_argument("path"); s.add_argument("--symbol", required=True)
    s = sub.add_parser("fetch"); s.add_argument("--symbol", required=True)
    s = sub.add_parser("evaluate"); s.add_argument("strategy"); s.add_argument("--symbol", default="XBTUSD")
    s.add_argument("--synthetic", action="store_true"); s.add_argument("--seed", type=int, default=0)
    sub.add_parser("trials")
    s = sub.add_parser("paper"); s.add_argument("action", choices=["run", "status", "resume"])
    s.add_argument("--scope")
    sub.add_parser("validate")
    a = p.parse_args(argv)
    if a.cmd == "validate":
        return validate_main()
    cfg = load_config()
    return {"import-csv": cmd_import, "fetch": cmd_fetch, "evaluate": cmd_evaluate,
            "trials": cmd_trials, "paper": cmd_paper}[a.cmd](a, cfg) or 0


if __name__ == "__main__":
    sys.exit(main())
