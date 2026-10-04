#!/usr/bin/env bash
# Stage 3 paper cycle - run on YOUR machine (Kraken is unreachable from the cloud sandbox).
# PROTECTED (operator-owned). Safe to run any time: only new closed bars are processed.
#
# Suggested crontab (hourly at :07 - works in any timezone and across daylight saving;
# Kraken's 4h bars close at 00/04/08/12/16/20 UTC, so 2 of every 4 runs do nothing):
#   7 * * * * cd /path/to/phil && ./quant/paper.sh >> quant/journal/paper-cron.log 2>&1
set -euo pipefail
cd "$(dirname "$0")/.."
PY=${PYTHON:-python3}

echo "=== $(date -u +%Y-%m-%dT%H:%M:%SZ) paper cycle"
symbols=$($PY - <<'PY'
import json, pathlib
p = pathlib.Path("quant/journal/passes.jsonl")
rows = [json.loads(l) for l in p.read_text().splitlines() if l.strip()] if p.exists() else []
print(" ".join(sorted({r["symbol"] for r in rows})))
PY
)
if [[ -z "$symbols" ]]; then
  echo "no strategy has passed the gates yet - nothing to paper trade"
  exit 0
fi
for s in $symbols; do
  $PY -m quant fetch --symbol "$s" > /dev/null || echo "WARN: fetch failed for $s (paper run will use cached bars; late bars are recorded as missed)"
done
$PY -m quant paper run
$PY -m quant paper status
