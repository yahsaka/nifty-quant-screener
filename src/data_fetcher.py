import os
import pandas as pd
import yfinance as yf
from tenacity import retry, stop_after_attempt, wait_exponential
from datetime import datetime, timedelta

CACHE_DIR = "../data/ohlcv_cache"

# Retry up to 3 times, waiting exponentially between 2 and 10 seconds if it fails
@retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=2, max=10))
def fetch_ticker_data(ticker_symbol: str, start_date: str, end_date: str) -> pd.DataFrame:
    """Fetches OHLCV data from yfinance with retry/backoff logic."""
    print(f"Fetching {ticker_symbol} from {start_date} to {end_date}...")
    
    # yfinance requires the .NS suffix for Indian NSE stocks
    ns_ticker = f"{ticker_symbol}.NS"
    df = yf.download(ns_ticker, start=start_date, end=end_date, progress=False)
    
    if df.empty:
        raise ValueError(f"No data returned for {ticker_symbol} from yfinance")
        
    return df

def update_ticker_cache(ticker_symbol: str):
    """Updates the local Parquet cache incrementally for a given ticker."""
    os.makedirs(CACHE_DIR, exist_ok=True)
    cache_file = os.path.join(CACHE_DIR, f"{ticker_symbol}.parquet")
    
    end_date = datetime.today().strftime('%Y-%m-%d')
    
    if os.path.exists(cache_file):
        # 1. Cache exists: Load it to find the last fetched date
        existing_df = pd.read_parquet(cache_file)
        
        # Ensure the index is a datetime object for comparison
        if not pd.api.types.is_datetime64_any_dtype(existing_df.index):
             existing_df.index = pd.to_datetime(existing_df.index)
             
        last_date = existing_df.index.max()
        start_date = (last_date + timedelta(days=1)).strftime('%Y-%m-%d')
        
        if start_date >= end_date:
            print(f"[{ticker_symbol}] Cache is already up to date.")
            return
            
        try:
            # Fetch only the missing days
            new_df = fetch_ticker_data(ticker_symbol, start_date, end_date)
            
            # Combine, deduplicate, and save
            updated_df = pd.concat([existing_df, new_df])
            updated_df = updated_df[~updated_df.index.duplicated(keep='last')]
            updated_df.to_parquet(cache_file)
            print(f"[{ticker_symbol}] Appended new data and saved to cache.")
            
        except Exception as e:
            print(f"[{ticker_symbol}] Failed to update incrementally: {e}")
            
    else:
        # 2. No cache exists: Fetch the last 2 years of history
        start_date = (datetime.today() - timedelta(days=730)).strftime('%Y-%m-%d')
        try:
            df = fetch_ticker_data(ticker_symbol, start_date, end_date)
            df.to_parquet(cache_file)
            print(f"[{ticker_symbol}] Created new 2-year history cache file.")
        except Exception as e:
            print(f"[{ticker_symbol}] Failed to create initial cache: {e}")

if __name__ == "__main__":
    print("Fetching official Nifty 500 list from NSE...")
    try:
        # Fetch directly from NSE's official repository
        url = "https://archives.nseindia.com/content/indices/ind_nifty500list.csv"
        nifty_df = pd.read_csv(url)
        
        # Save it to our data folder
        ticker_file = "../data/nifty500_tickers.csv"
        nifty_df.to_csv(ticker_file, index=False)
        
        # Extract just the symbols
        tickers = nifty_df['Symbol'].tolist()
        print(f"Successfully loaded {len(tickers)} Nifty 500 tickers.")
        
    except Exception as e:
        print(f"Failed to fetch Nifty 500 list: {e}")
        print("Falling back to local CSV if it exists...")
        try:
            tickers = pd.read_csv("../data/nifty500_tickers.csv")['Symbol'].tolist()
        except:
            print("No local CSV found. Exiting.")
            exit()
    
    print("\nStarting full Nifty 500 data fetch (This will take a few minutes)...")
    
    # Track progress
    total = len(tickers)
    for i, ticker in enumerate(tickers, 1):
        print(f"[{i}/{total}] ", end="")
        update_ticker_cache(ticker)
        
    print("\nData fetch complete! You can now run screener.py")