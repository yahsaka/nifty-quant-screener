import json
import os

PAPER_TRADES_FILE = os.getenv("PAPER_TRADES_PATH", "data/paper_trades.json")

def load_paper_trades() -> list[dict]:
    """Load the paper trades ledger, returning an empty list if it doesn't exist."""
    if not os.path.exists(PAPER_TRADES_FILE):
        return []
    
    try:
        with open(PAPER_TRADES_FILE, "r") as f:
            return json.load(f)
    except json.JSONDecodeError:
        return []

def save_paper_trades(trades: list[dict]) -> None:
    """Safely save the paper trades ledger to disk."""
    os.makedirs(os.path.dirname(PAPER_TRADES_FILE), exist_ok=True)
    
    with open(PAPER_TRADES_FILE, "w", encoding="utf-8") as f:
        json.dump(trades, f, indent=2)

def add_paper_trade(new_trade: dict) -> None:
    """Append a new trade to the ledger."""
    trades = load_paper_trades()
    trades.append(new_trade)
    save_paper_trades(trades)
