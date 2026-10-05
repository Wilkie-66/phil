"""Strategy loader and interface contract. PROTECTED (operator-owned).

A strategy is one file in quant/strategies/ defining:

  HYPOTHESIS = {
      "mechanism":    why this should make money (economic reason, not a pattern),
      "counterparty": who is on the other side and why they keep losing,
      "invalidation": what live behaviour would prove the idea wrong,
  }
  PARAM_GRID = [ {...}, {...} ]          # every entry is a logged trial
  def signal(df, **params) -> pd.Series   # target position per bar in [-1, 1],
                                          # using data up to that bar's close only

The engine does the fitting (picks params on each training window), so a
strategy cannot peek at test data, and every grid entry counts as a trial.
"""
import importlib.util
from pathlib import Path

MAX_GRID = 50
REQUIRED_HYPOTHESIS = ("mechanism", "counterparty", "invalidation")


class StrategyError(ValueError):
    pass


def load(path):
    path = Path(path)
    spec = importlib.util.spec_from_file_location(f"strategy_{path.stem}", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    hyp = getattr(mod, "HYPOTHESIS", None)
    if not isinstance(hyp, dict):
        raise StrategyError("HYPOTHESIS dict missing")
    for k in REQUIRED_HYPOTHESIS:
        if not str(hyp.get(k, "")).strip():
            raise StrategyError(f"HYPOTHESIS['{k}'] is empty - no mechanism, no test")
    grid = getattr(mod, "PARAM_GRID", None)
    if not isinstance(grid, list) or not grid or not all(isinstance(g, dict) for g in grid):
        raise StrategyError("PARAM_GRID must be a non-empty list of dicts")
    if len(grid) > MAX_GRID:
        raise StrategyError(f"PARAM_GRID has {len(grid)} entries (max {MAX_GRID}): "
                            "that is a parameter sweep, not a hypothesis")
    if not callable(getattr(mod, "signal", None)):
        raise StrategyError("signal(df, **params) missing")
    return mod
