"""Build and safely refresh the OHLCV cache used by the daily screener."""

import logging
import os
import tempfile
from datetime import datetime, timedelta
from io import StringIO

import pandas as pd
import requests
import yfinance as yf
from tenacity import retry, stop_after_attempt, wait_exponential

CACHE_DIR = "data/ohlcv_cache"
LOGGER = logging.getLogger(__name__)


def _write_parquet_atomically(dataframe: pd.DataFrame, destination: str) -> None:
    """Avoid a partially written cache file if a runner is interrupted."""
    with tempfile.NamedTemporaryFile(suffix=".parquet", dir=os.path.dirname(destination), delete=False) as temporary:
        temporary_name = temporary.name
    try:
        dataframe.to_parquet(temporary_name)
        os.replace(temporary_name, destination)
    finally:
        if os.path.exists(temporary_name):
            os.unlink(temporary_name)


@retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=2, max=10))
def fetch_ticker_data(ticker_symbol: str, start_date: str, end_date: str) -> pd.DataFrame:
    """Fetch complete OHLCV data; ``end_date`` is exclusive in yfinance."""
    LOGGER.info("Fetching %s from %s to %s", ticker_symbol, start_date, end_date)
    dataframe = yf.download(f"{ticker_symbol}.NS", start=start_date, end=end_date, progress=False)
    if isinstance(dataframe.columns, pd.MultiIndex):
        dataframe.columns = dataframe.columns.get_level_values(0)
    required = {"Open", "High", "Low", "Close", "Volume"}
    if dataframe.empty or not required.issubset(dataframe.columns):
        raise ValueError(f"Incomplete OHLCV data for {ticker_symbol}")
    return dataframe.sort_index()


def update_ticker_cache(ticker_symbol: str) -> bool:
    """Refresh a ticker cache and return False when it cannot be trusted."""
    os.makedirs(CACHE_DIR, exist_ok=True)
    cache_file = os.path.join(CACHE_DIR, f"{ticker_symbol}.parquet")
    # yfinance excludes end_date; tomorrow includes today's completed session.
    end_date = (datetime.today().date() + timedelta(days=1)).isoformat()
    try:
        if os.path.exists(cache_file):
            existing = pd.read_parquet(cache_file)
            if existing.empty:
                raise ValueError("existing cache is empty")
            if not pd.api.types.is_datetime64_any_dtype(existing.index):
                existing.index = pd.to_datetime(existing.index)
            start_date = (existing.index.max() + timedelta(days=1)).date().isoformat()
            if start_date >= end_date:
                LOGGER.info("[%s] Cache is already up to date.", ticker_symbol)
                return True
            updated = pd.concat([existing, fetch_ticker_data(ticker_symbol, start_date, end_date)])
            updated = updated[~updated.index.duplicated(keep="last")].sort_index()
            _write_parquet_atomically(updated, cache_file)
            return True

        start_date = (datetime.today().date() - timedelta(days=730)).isoformat()
        _write_parquet_atomically(fetch_ticker_data(ticker_symbol, start_date, end_date), cache_file)
        return True
    except Exception as exc:
        LOGGER.error("[%s] Cache update failed: %s", ticker_symbol, exc)
        return False


def load_tickers() -> list[str]:
    """Prefer the current NSE list, with a local list as a transparent fallback."""
    ticker_file = "data/nifty500_tickers.csv"
    try:
        response = requests.get(
            "https://archives.nseindia.com/content/indices/ind_nifty500list.csv",
            headers={"User-Agent": "Mozilla/5.0"}, timeout=30,
        )
        response.raise_for_status()
        nifty = pd.read_csv(StringIO(response.text))
        
        if "Symbol" not in nifty.columns:
            raise ValueError("NSE list did not contain a Symbol column")
            
        # Validate row count to protect against truncated or empty responses
        if len(nifty) < 400:
            raise ValueError(f"Truncated NSE response: received only {len(nifty)} rows (expected >= 400)")
            
        nifty.to_csv(ticker_file, index=False)
        return nifty["Symbol"].dropna().astype(str).tolist()
    except Exception as exc:
        LOGGER.warning("Could not refresh NSE ticker list: %s", exc)
        try:
            return pd.read_csv(ticker_file)["Symbol"].dropna().astype(str).tolist()
        except (FileNotFoundError, KeyError, pd.errors.EmptyDataError) as fallback_exc:
            raise RuntimeError(f"No usable Nifty 500 ticker list: {fallback_exc}") from fallback_exc


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    os.makedirs("data", exist_ok=True)
    tickers = load_tickers()
    LOGGER.info("Refreshing %s Nifty 500 tickers.", len(tickers))
    failed = [ticker for ticker in tickers if not update_ticker_cache(ticker)]
    if failed:
        LOGGER.error("Cache refresh failed for %s ticker(s): %s", len(failed), ", ".join(failed))
        raise SystemExit(1)
    LOGGER.info("Data fetch complete; all %s ticker caches are available.", len(tickers))
