"""Position sizing and the kill switch. PROTECTED (operator-owned)."""


def position_size(capital, entry, stop, cfg, open_positions=0):
    """Size so that hitting the stop (plus round-trip costs) loses
    risk_per_trade_pct of capital, capped at max_position_pct of capital."""
    risk, ex = cfg["risk"], cfg["execution"]
    if open_positions >= risk["max_open_positions"]:
        return {"units": 0.0, "notional": 0.0, "reason": "max_open_positions reached"}
    if entry <= 0 or stop <= 0 or stop == entry:
        raise ValueError("entry and stop must be positive and different")
    cost_rate = 2 * (ex["fee_bps_per_side"] + ex["slippage_bps_per_side"]) / 1e4
    loss_per_unit = abs(entry - stop) + entry * cost_rate
    units = capital * risk["risk_per_trade_pct"] / loss_per_unit
    cap = capital * risk["max_position_pct"]
    capped = units * entry > cap
    if capped:
        units = cap / entry
    notional = units * entry
    return {"units": round(units, 8), "notional": round(notional, 2),
            "pct_of_capital": round(notional / capital * 100, 1),
            "loss_if_stopped": round(units * loss_per_unit, 2), "capped": capped}


def kill_switch(equity, cfg):
    """HALT once peak-to-current drawdown exceeds the pre-committed limit."""
    values = [float(x) for x in equity]
    peak, cur = max(values), values[-1]
    dd = cur / peak - 1
    limit = -cfg["risk"]["max_drawdown_halt_pct"]
    return {"drawdown": round(dd, 4), "limit": limit, "action": "HALT" if dd <= limit else "CONTINUE"}
