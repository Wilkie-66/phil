"""Load the protected config. PROTECTED (operator-owned)."""
import json
from pathlib import Path

QUANT_DIR = Path(__file__).resolve().parent.parent
CONFIG_PATH = QUANT_DIR / "config.json"
DATA_DIR = QUANT_DIR / "data"
JOURNAL_DIR = QUANT_DIR / "journal"
TRIALS_PATH = JOURNAL_DIR / "trials.jsonl"


def load_config(path=CONFIG_PATH):
    with open(path) as f:
        return json.load(f)
