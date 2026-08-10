import os
import json
import logging
import pandas as pd
import pandas_ta as ta
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
SCREENER_JSON_PATH = os.getenv("SCREENER_JSON_PATH", "data/latest_screener_results.json")
HOLD_DAYS = int(os.getenv("BACKTEST_HOLD_DAYS", "20"))
ATR_MULTIPLIER = float(os.getenv("BACKTEST_ATR_MULTIPLIER", "2.0"))
ENTRY_SLIPPAGE_BPS = int(os.getenv("BACKTEST_ENTRY_SLIPPAGE_BPS", "5"))
EXIT_SLIPPAGE_BPS = int(os.getenv("BACKTEST_EXIT_SLIPPAGE_BPS", "5"))
ROUND_TRIP_COST_BPS = int(os.getenv("BACKTEST_ROUND_TRIP_COST_BPS", "20"))

def apply_friction(gross_exit_price: float) -> float:
    """Deduct slippage and round-trip costs to get the realistic net exit price."""
    friction_pct = (EXIT_SLIPPAGE_BPS + ROUND_TRIP_COST_BPS) / 10_000
    return gross_exit_price * (1 - friction_pct)

def ingest_new_signals():
    """Reads latest screener results and auto-adds Trade-Ready setups to the ledger as PENDING."""
    if not os.path.exists(SCREENER_JSON_PATH):
        LOGGER.warning(f"No screener results found at {SCREENER_JSON_PATH}")
        return

    with open(SCREENER_JSON_PATH, "r") as f:
        screener_data = json.load(f)

    trades = load_paper_trades()
    active_tickers = {t["ticker"] for t in trades if t["status"] in ["OPEN", "PENDING"]}
    updates_made = False

    for signal in screener_data.get("signals", []):
        if signal.get("status") == "Trade-Ready":
            ticker = signal["ticker"]
            signal_date = signal["last_date"]
            
            # Avoid duplicate concurrent trades for the same ticker
            if ticker in active_tickers:
                continue

            # Load the cache to calculate the ATR for the exact signal date
            cache_path = os.path.join(CACHE_DIR, f"{ticker}.parquet")
            if not os.path.exists(cache_path):
                continue
                
            df = pd.read_parquet(cache_path)
            if isinstance(df.columns, pd.MultiIndex):
                df.columns = df.columns.get_level_values(0)
            
            df["ATR_14"] = ta.atr(df["High"], df["Low"], df["Close"], length=14)
            
            # Extract the ATR value on the signal date
            try:
                atr_value = df.loc[signal_date, "ATR_14"]
                if pd.isna(atr_value):
                    continue
            except KeyError:
                continue

            new_trade = {
                "trade_id": f"{ticker}_{signal_date}",  # <-- Add this line
                "ticker": ticker,
                "status": "PENDING",
                "signal_date": signal_date,
                "atr_at_signal": round(float(atr_value), 2),
                "target_hold_days": HOLD_DAYS
            }
            trades.append(new_trade)
            active_tickers.add(ticker)
            updates_made = True
            LOGGER.info(f"Ingested new PENDING trade for {ticker} from {signal_date} signal.")

    if updates_made:
        save_paper_trades(trades)

def process_active_trades():
    """Simulates market entry for PENDING trades and checks exits for OPEN trades."""
    trades = load_paper_trades()
    active_trades = [t for t in trades if t.get("status") in ["PENDING", "OPEN"]]
    
    if not active_trades:
        LOGGER.info("No active paper trades to process.")
        return

    updates_made = False

    for trade in active_trades:
        ticker = trade["ticker"]
        cache_path = os.path.join(CACHE_DIR, f"{ticker}.parquet")
        
        if not os.path.exists(cache_path):
            continue
            
        df = pd.read_parquet(cache_path)
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)
            
        # Filter for dates strictly AFTER the initial signal date
        signal_date = pd.to_datetime(trade["signal_date"])
        future_data = df[df.index > signal_date]
        
        if future_data.empty:
            continue 
        
        # Iterate through the unfolding days chronologically
        for date, row in future_data.iterrows():
            day_open = float(row["Open"])
            day_low = float(row["Low"])
            day_close = float(row["Close"])
            date_str = date.strftime("%Y-%m-%d")

            # --- HANDLE PENDING TO OPEN TRANSITION ---
            if trade["status"] == "PENDING":
                trade["entry_date"] = date_str
                trade["entry_price"] = round(day_open * (1 + (ENTRY_SLIPPAGE_BPS / 10_000)), 2)
                trade["stop_price"] = round(trade["entry_price"] - (ATR_MULTIPLIER * trade["atr_at_signal"]), 2)
                trade["sessions_held"] = 1
                trade["status"] = "OPEN"
                LOGGER.info(f"Opened {ticker} at {trade['entry_price']} | Stop: {trade['stop_price']}")
                updates_made = True
                continue # Move to the next day in future_data to check for exits

            # --- HANDLE OPEN TRADE EXITS ---
            if trade["status"] == "OPEN":
                trade["sessions_held"] = trade.get("sessions_held", 1) + 1
                stop_price = trade["stop_price"]
                
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
                
                # 2. Check Time Exit
                elif trade["sessions_held"] >= trade["target_hold_days"]:
                    exit_raw = day_close
                    exit_reason = "Time Exit"
                    
                # Execute Exit
                if exit_raw is not None:
                    net_exit_price = apply_friction(exit_raw)
                    trade["status"] = "CLOSED"
                    trade["exit_date"] = date_str
                    trade["exit_price"] = round(net_exit_price, 2)
                    trade["exit_reason"] = exit_reason
                    trade["realized_pnl_pct"] = round(((net_exit_price / trade["entry_price"]) - 1) * 100, 2)
                    
                    LOGGER.info(f"Closed {ticker} | Reason: {exit_reason} | P&L: {trade['realized_pnl_pct']}%")
                    updates_made = True
                    break 

    if updates_made:
        save_paper_trades(trades)
        LOGGER.info("Paper trades ledger updated successfully.")
    else:
        LOGGER.info("No trades met entry/exit conditions today.")

if __name__ == "__main__":
    LOGGER.info("Starting Daily Paper Trade Evaluation...")
    ingest_new_signals()
    process_active_trades()
    LOGGER.info("Evaluation Complete.")
