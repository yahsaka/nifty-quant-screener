"""Backtest the same six-point setup and execution model shown by the screener."""

import logging
import os
import warnings
from collections import Counter
from typing import Optional

import numpy as np
import pandas as pd
import pandas_ta as ta
import yfinance as yf

from indicators import calculate_indicators
from screener import BEARISH, BULLISH, UNKNOWN, evaluate_setup

warnings.filterwarnings("ignore")
logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
LOGGER = logging.getLogger(__name__)

CACHE_DIR = "data/ohlcv_cache"
HOLD_DAYS = 20
ATR_MULTIPLIER = 2.0
# Conservative, configurable assumptions for a cash-equity swing trade.
ENTRY_SLIPPAGE_BPS = 5
EXIT_SLIPPAGE_BPS = 5
ROUND_TRIP_COST_BPS = 20


def _finite(value: object) -> Optional[float]:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if np.isfinite(number) else None


def get_market_regimes(period: str = "10y") -> dict:
    """Map each Nifty session to Bullish/Bearish under the shared 200/3 rule."""
    try:
        nifty = yf.download("^NSEI", period=period, progress=False)
        if isinstance(nifty.columns, pd.MultiIndex):
            nifty.columns = nifty.columns.get_level_values(0)
        if nifty.empty or "Close" not in nifty or len(nifty) < 203:
            raise ValueError("insufficient Nifty history for 200 EMA")
        nifty = nifty.copy()
        nifty["EMA_200"] = nifty["Close"].ewm(span=200, adjust=False).mean()
        nifty["below_ema"] = nifty["Close"] < nifty["EMA_200"]
        nifty["bearish"] = nifty["below_ema"].rolling(3).sum().eq(3)
        return {
            timestamp.date(): BEARISH if bool(is_bearish) else BULLISH
            for timestamp, is_bearish in nifty["bearish"].items()
            if pd.notna(is_bearish)
        }
    except Exception as exc:
        LOGGER.error("Cannot run a trustworthy backtest without Nifty regime data: %s", exc)
        return {}


def _execute_trade(df: pd.DataFrame, signal_idx: int, atr: float) -> Optional[dict]:
    """Simulate next-open entry, gap-aware ATR stop, and an inclusive holding window."""
    entry_idx = signal_idx + 1
    exit_idx = entry_idx + HOLD_DAYS - 1
    if exit_idx >= len(df):
        return None

    raw_entry = _finite(df["Open"].iloc[entry_idx])
    if raw_entry is None or raw_entry <= 0 or atr <= 0:
        return None
    entry_price = raw_entry * (1 + ENTRY_SLIPPAGE_BPS / 10_000)
    stop_price = entry_price - ATR_MULTIPLIER * atr
    if stop_price <= 0:
        return None

    exit_raw = None
    exit_reason = "Time exit"
    actual_exit_idx = exit_idx
    for day_idx in range(entry_idx, exit_idx + 1):
        day_open = _finite(df["Open"].iloc[day_idx])
        day_low = _finite(df["Low"].iloc[day_idx])
        if day_open is None or day_low is None:
            return None
        if day_low <= stop_price:
            # A gap through the stop can only be filled at the weaker opening price.
            exit_raw = day_open if day_open <= stop_price else stop_price
            exit_reason = "ATR stop (gap)" if day_open <= stop_price else "ATR stop"
            actual_exit_idx = day_idx
            break
    if exit_raw is None:
        exit_raw = _finite(df["Close"].iloc[exit_idx])
        if exit_raw is None:
            return None

    exit_price = exit_raw * (1 - (EXIT_SLIPPAGE_BPS + ROUND_TRIP_COST_BPS) / 10_000)
    return {
        "Entry Price": entry_price,
        "Stop Price": stop_price,
        "Exit Price": exit_price,
        "Return": exit_price / entry_price - 1.0,
        "Exit Reason": exit_reason,
        "Exit Date": df.index[actual_exit_idx],
        "Hit_SL": exit_reason.startswith("ATR stop"),
    }


def extract_historical_signals(ticker: str, df: pd.DataFrame, market_regimes: dict) -> list[dict]:
    """Backtest score >=3 exactly as the live screener scores it on each date."""
    if df.empty or len(df) < 221:
        return []
    indicators = calculate_indicators(df.copy())
    indicators["ATR_14"] = ta.atr(indicators["High"], indicators["Low"], indicators["Close"], length=14)
    signals: list[dict] = []

    # The final HOLD_DAYS rows cannot be fully evaluated after tomorrow's entry.
    for idx in range(1, len(indicators) - HOLD_DAYS):
        date = indicators.index[idx].date()
        setup = evaluate_setup(indicators, idx, market_regimes.get(date, UNKNOWN), ticker)
        if not setup or setup["score"] < 3:
            continue
        atr = _finite(indicators["ATR_14"].iloc[idx])
        if atr is None:
            continue
        trade = _execute_trade(indicators, idx, atr)
        if not trade:
            continue
        signals.append(
            {
                "Ticker": ticker,
                "Date": indicators.index[idx],
                "Year": str(indicators.index[idx].year),
                "Score": setup["score"],
                "Status": setup["status"],
                "Triggers": setup["triggers_str"],
                **trade,
            }
        )
    return signals


def _summary(results: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (score, status), group in results.groupby(["Score", "Status"], sort=True):
        wins = group.loc[group["Return"] > 0, "Return"]
        losses = group.loc[group["Return"] < 0, "Return"]
        profit_factor = wins.sum() / abs(losses.sum()) if not losses.empty else np.inf
        rows.append(
            {
                "Score": score,
                "Status": status,
                "Trades": len(group),
                "Win Rate": f"{(group['Return'] > 0).mean() * 100:.1f}%",
                "Avg Return": f"{group['Return'].mean() * 100:.2f}%",
                "Median Return": f"{group['Return'].median() * 100:.2f}%",
                "Stop Rate": f"{group['Hit_SL'].mean() * 100:.1f}%",
                "Profit Factor": f"{profit_factor:.2f}" if np.isfinite(profit_factor) else "∞",
            }
        )
    return pd.DataFrame(rows)


def run_backtest() -> None:
    LOGGER.info(
        "Backtest: score >=3 | next-day open | %s-session hold | %.1fx ATR stop | entry/exit slippage %s/%s bps | costs %s bps",
        HOLD_DAYS, ATR_MULTIPLIER, ENTRY_SLIPPAGE_BPS, EXIT_SLIPPAGE_BPS, ROUND_TRIP_COST_BPS,
    )
    if not os.path.exists(CACHE_DIR):
        LOGGER.error("No cache directory found.")
        return
    market_regimes = get_market_regimes()
    if not market_regimes:
        return

    files = [name for name in os.listdir(CACHE_DIR) if name.endswith(".parquet")]
    all_signals: list[dict] = []
    failures: Counter[str] = Counter()
    for number, filename in enumerate(files, 1):
        if number % 50 == 0:
            LOGGER.info("Processed %s/%s stocks.", number, len(files))
        ticker = filename.removesuffix(".parquet")
        try:
            all_signals.extend(extract_historical_signals(ticker, pd.read_parquet(os.path.join(CACHE_DIR, filename)), market_regimes))
        except Exception as exc:
            failures[ticker] += 1
            LOGGER.warning("Could not backtest %s: %s", ticker, exc)

    LOGGER.info("Completed %s files; %s failures.", len(files) - len(failures), len(failures))
    if failures:
        LOGGER.info("Failed tickers: %s", ", ".join(failures))
    results = pd.DataFrame(all_signals)
    if results.empty:
        LOGGER.warning("No qualifying score >=3 signals found.")
        return
    print("\nSIX-POINT SCREENER BACKTEST RESULTS")
    print(_summary(results).to_markdown(index=False))


if __name__ == "__main__":
    run_backtest()
