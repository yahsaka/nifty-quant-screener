import os
import pandas as pd
import numpy as np
import yfinance as yf
import pandas_ta as ta
from indicators import calculate_indicators
import warnings

warnings.filterwarnings('ignore')

CACHE_DIR = "../data/ohlcv_cache"
HOLD_DAYS = 20
ATR_MULTIPLIER = 2.0 

def get_market_regime() -> set:
    """Fetches Nifty 50 and returns a set of dates where the market is in an uptrend."""
    print("Fetching Nifty 50 benchmark for trend filtering...")
    nifty = yf.download("^NSEI", period="5y", progress=False)
    
    if isinstance(nifty.columns, pd.MultiIndex):
        nifty.columns = nifty.columns.get_level_values(0)
        
    nifty['EMA_50'] = nifty['Close'].ewm(span=50, adjust=False).mean()
    bullish_dates = nifty[nifty['Close'] > nifty['EMA_50']].index.date
    return set(bullish_dates)

def extract_historical_signals(ticker: str, df: pd.DataFrame, market_bullish_dates: set) -> list:
    """Scans for triggers and applies ATR stop-loss with next-day OPEN entries."""
    if df.empty or len(df) < 200:
        return []

    df = calculate_indicators(df)
    df['ATR_14'] = ta.atr(df['High'], df['Low'], df['Close'], length=14)
    df = df.dropna(subset=['EMA_200', 'RSI_14', 'VOL_SMA_20', 'ATR_14']).copy()
    
    # Define Conditions
    ema_breakout = (df['Close'] > df['EMA_200']) & (df['Close'].shift(1) <= df['EMA_200'].shift(1))
    volume_spike = df['Volume'] > (2.0 * df['VOL_SMA_20'])
    rsi_cross = (df['RSI_14'] > 60) & (df['RSI_14'].shift(1) <= 60)
    golden_setup = rsi_cross & volume_spike
    
    valid_dates = df.index.date
    is_market_bullish = np.array([d in market_bullish_dates for d in valid_dates])

    strategies = {
        "EMA_200_BREAKOUT (ATR)": ema_breakout & is_market_bullish,
        "GOLDEN_SETUP (ATR)": golden_setup & is_market_bullish,
        "RSI_CROSS_60 (ATR)": rsi_cross & is_market_bullish,
        "VOLUME_SPIKE_2X (ATR)": volume_spike & is_market_bullish
    }

    signals = []
    
    for strategy_name, condition_mask in strategies.items():
        triggered_indices = np.where(condition_mask)[0]
        
        for idx in triggered_indices:
            # FIX: Ensure we have enough data to enter TOMORROW (idx+1) and hold for HOLD_DAYS
            if idx + 1 + HOLD_DAYS >= len(df):
                continue 
                
            # FIX: Enter at the Open of the day AFTER the signal triggered
            entry_price = float(df['Open'].iloc[idx + 1])
            current_atr = float(df['ATR_14'].iloc[idx])
            
            stop_price = entry_price - (ATR_MULTIPLIER * current_atr)
            
            # Intraday lows evaluated starting from the entry day
            future_lows = df['Low'].iloc[idx + 1 : idx + 1 + HOLD_DAYS]
            hit_sl = future_lows <= stop_price
            
            if hit_sl.any():
                trade_return = (stop_price / entry_price) - 1.0
                sl_hit = True
            else:
                exit_price = float(df['Close'].iloc[idx + HOLD_DAYS])
                trade_return = (exit_price / entry_price) - 1.0
                sl_hit = False
                
            signals.append({
                "Ticker": ticker,
                "Date": df.index[idx],
                "Year": str(df.index[idx].year), # Extracted for grouping
                "Strategy": strategy_name,
                "Return": trade_return,
                "Hit_SL": sl_hit
            })

    return signals

def run_backtest():
    print("="*70)
    print("Initializing Fixed-Execution ATR Backtest...")
    print("Parameters: Nifty 50 Filter | Next-Day Open Entry | 2x ATR Stop Loss")
    print("="*70)
    
    if not os.path.exists(CACHE_DIR):
        print("No cache directory found.")
        return

    market_bullish_dates = get_market_regime()
    cache_files = [f for f in os.listdir(CACHE_DIR) if f.endswith('.parquet')]
    all_signals = []

    total_files = len(cache_files)
    for i, file in enumerate(cache_files, 1):
        if i % 50 == 0:
            print(f"Processed {i}/{total_files} stocks...")
            
        ticker = file.replace('.parquet', '')
        filepath = os.path.join(CACHE_DIR, file)
        
        try:
            df = pd.read_parquet(filepath)
            signals = extract_historical_signals(ticker, df, market_bullish_dates)
            all_signals.extend(signals)
        except Exception:
            pass 

    results_df = pd.DataFrame(all_signals)
    
    if results_df.empty:
        print("No signals found in historical data.")
        return

    print("\n" + "="*80)
    print("YEARLY BACKTEST RESULTS")
    print("="*80)
    
    # FIX: Grouped by Strategy AND Year
    summary = []
    for (strategy, year), group in results_df.groupby(["Strategy", "Year"]):
        summary.append({
            "Strategy": strategy,
            "Year": year,
            "Trades": len(group),
            "Win Rate": f"{(group['Return'] > 0).mean() * 100:.1f}%",
            "Avg Return": f"{group['Return'].mean() * 100:.2f}%",
            "Stopped Out": f"{group['Hit_SL'].mean() * 100:.1f}%"
        })

    summary_df = pd.DataFrame(summary).set_index(["Strategy", "Year"])
    print(summary_df.to_markdown())
    print("="*80)

if __name__ == "__main__":
    run_backtest()