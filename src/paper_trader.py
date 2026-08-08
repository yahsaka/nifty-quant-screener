import os
import logging
import pandas as pd
from datetime import datetime
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

from paper_store import load_paper_trades, save_paper_trades

# Setup Logging
logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
LOGGER = logging.getLogger(__name__)

# Load Environment Variables
CACHE_DIR = os.getenv("CACHE_DIR", "data/ohlcv_cache")
EXIT_SLIPPAGE_BPS = int(os.getenv("BACKTEST_EXIT_SLIPPAGE_BPS", "5"))
ROUND_TRIP_COST_BPS = int(os.getenv("BACKTEST_ROUND_TRIP_COST_BPS", "20"))

def apply_friction(gross_exit_price: float) -> float:
    """Deduct slippage and round-trip costs to get the realistic net exit price."""
    friction_pct = (EXIT_SLIPPAGE_BPS + ROUND_TRIP_COST_BPS) / 10_000
    return gross_exit_price * (1 - friction_pct)

def process_open_trades():
    trades = load_paper_trades()
    open_trades = [t for t in trades if t.get("status") == "OPEN"]
    
    if not open_trades:
        LOGGER.info("No open paper trades to process.")
        return

    updates_made = False

    for trade in open_trades:
        ticker = trade["ticker"]
        cache_path = os.path.join(CACHE_DIR, f"{ticker}.parquet")
        
        if not os.path.exists(cache_path):
            LOGGER.warning(f"No cache found for {ticker}, skipping evaluation.")
            continue
            
        # Load the stock's price history
        df = pd.read_parquet(cache_path)
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)
            
        # Filter for dates AFTER the entry date
        entry_date = pd.to_datetime(trade["entry_date"])
        future_data = df[df.index > entry_date]
        
        if future_data.empty:
            continue # No new trading days have occurred yet

        stop_price = trade["stop_price"]
        exit_target_date = pd.to_datetime(trade["exit_target_date"])
        
        # Iterate through the unfolding days chronologically
        for date, row in future_data.iterrows():
            day_open = float(row["Open"])
            day_low = float(row["Low"])
            day_close = float(row["Close"])
            
            exit_raw = None
            exit_reason = None
            
            # 1. Check ATR Stop-Loss
            if day_low <= stop_price:
                if day_open <= stop_price:
                    exit_raw = day_open
                    exit_reason = "Stop Loss (Gap)"
                else:
                    exit_raw = stop_price
                    exit_reason = "Stop Loss"
            
            # 2. Check Time Exit (if stop wasn't hit)
            elif date >= exit_target_date:
                exit_raw = day_close
                exit_reason = "Time Exit"
                
            # If an exit condition was met, record it and stop evaluating this trade
            if exit_raw is not None:
                net_exit_price = apply_friction(exit_raw)
                
                trade["status"] = "CLOSED"
                trade["exit_date"] = date.strftime("%Y-%m-%d")
                trade["exit_price"] = round(net_exit_price, 2)
                trade["exit_reason"] = exit_reason
                trade["realized_pnl_pct"] = round(((net_exit_price / trade["entry_price"]) - 1) * 100, 2)
                
                LOGGER.info(f"Closed {ticker} | Reason: {exit_reason} | P&L: {trade['realized_pnl_pct']}%")
                updates_made = True
                break # Exit the chronological loop for this specific stock

    if updates_made:
        save_paper_trades(trades)
        LOGGER.info("Paper trades ledger updated successfully.")
    else:
        LOGGER.info("No trades met exit conditions today.")

if __name__ == "__main__":
    LOGGER.info("Starting Daily Paper Trade Evaluation...")
    process_open_trades()
    LOGGER.info("Evaluation Complete.")
