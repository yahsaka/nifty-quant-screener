"""Live Nifty 500 screener and canonical six-point setup evaluator."""

import json
import logging
import os
import tempfile
import warnings
from datetime import datetime, timedelta, timezone
from typing import Optional

import numpy as np
import pandas as pd
import yfinance as yf

from indicators import calculate_indicators

warnings.filterwarnings("ignore")
logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
LOGGER = logging.getLogger(__name__)

CACHE_DIR = "data/ohlcv_cache"
OUTPUT_FILE = "data/latest_screener_results.json"
BULLISH = "Bullish"
BEARISH = "Bearish"
UNKNOWN = "Unknown"


def _finite(value: object) -> Optional[float]:
    """Return a finite scalar, or None for missing/non-numeric values."""
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if np.isfinite(number) else None


def get_market_regime(period: str = "2y") -> str:
    """Return Bullish, Bearish, or Unknown using 200 EMA/three-session rule.

    Bearish means the latest three Nifty closes are all below the 200-day EMA.
    A failed or insufficient download is deliberately Unknown, never Bullish.
    """
    try:
        nifty = yf.download("^NSEI", period=period, progress=False)
        if isinstance(nifty.columns, pd.MultiIndex):
            nifty.columns = nifty.columns.get_level_values(0)
        if nifty.empty or "Close" not in nifty or len(nifty) < 203:
            LOGGER.warning("Nifty data is unavailable or too short for the 200 EMA rule.")
            return UNKNOWN

        ema_200 = nifty["Close"].ewm(span=200, adjust=False).mean()
        below_ema = (nifty["Close"] < ema_200).dropna()
        if len(below_ema) < 3:
            return UNKNOWN
        return BEARISH if bool(below_ema.iloc[-3:].all()) else BULLISH
    except Exception as exc:
        LOGGER.warning("Failed to determine Nifty market regime: %s", exc)
        return UNKNOWN


def evaluate_setup(
    df: pd.DataFrame, idx: int, market_regime: str, ticker: str = ""
) -> Optional[dict]:
    """Evaluate the canonical six-point screener rules for one historical row.

    The same function is used by the live screener and the backtest.  It returns
    every valid, eligible score (including scores below 3); callers decide which
    score thresholds to display or simulate.
    """
    if idx < 1 or idx >= len(df) or market_regime != BULLISH:
        return None

    latest, previous = df.iloc[idx], df.iloc[idx - 1]
    close = _finite(latest.get("Close"))
    previous_close = _finite(previous.get("Close"))
    if close is None or previous_close is None:
        return None

    ema_50 = _finite(latest.get("EMA_50"))
    ema_200 = _finite(latest.get("EMA_200"))
    previous_ema_200 = _finite(previous.get("EMA_200"))
    rsi_14 = _finite(latest.get("RSI_14"))
    volume = _finite(latest.get("Volume"))
    volume_sma_20 = _finite(latest.get("VOL_SMA_20"))

    # Do not score a setup that lacks a valid long-term trend baseline.
    if ema_50 is None or ema_200 is None:
        return None
    pct_above_50 = (close - ema_50) / ema_50 if ema_50 != 0 else None
    if pct_above_50 is None or pct_above_50 > 0.15:
        return None

    triggers: list[str] = []
    if close > ema_200:
        triggers.append("ABOVE_EMA_200")
    if (
        previous_ema_200 is not None
        and close > ema_200
        and previous_close <= previous_ema_200
    ):
        triggers.append("EMA_200_BREAKOUT")
    if close > ema_50:
        triggers.append("ABOVE_EMA_50")
    if volume is not None and volume_sma_20 is not None and volume_sma_20 > 0:
        if volume > 2.0 * volume_sma_20:
            triggers.append("VOLUME_SPIKE_2X")
    if rsi_14 is not None and 60 < rsi_14 <= 70:
        triggers.append("RSI_60_TO_70")

    macd_columns = [c for c in df.columns if str(c).startswith("MACD_")]
    signal_columns = [c for c in df.columns if str(c).startswith("MACDs_")]
    if macd_columns and signal_columns:
        macd = _finite(latest[macd_columns[0]])
        signal = _finite(latest[signal_columns[0]])
        previous_macd = _finite(previous[macd_columns[0]])
        previous_signal = _finite(previous[signal_columns[0]])
        if None not in (macd, signal, previous_macd, previous_signal):
            if macd > signal and previous_macd <= previous_signal:
                triggers.append("MACD_BULLISH_CROSS")

    score = len(triggers)
    status = "Trade-Ready" if score >= 5 else "Watchlist" if score >= 3 else "No setup"
    last_date = latest.name.date() if hasattr(latest.name, "date") else latest.name
    return {
        "ticker": ticker,
        "last_date": str(last_date),
        "close": round(close, 2),
        "rsi_14": round(rsi_14, 2) if rsi_14 is not None else None,
        "pct_above_50": round(pct_above_50 * 100, 2),
        "volume_ratio": round(volume / volume_sma_20, 2)
        if volume is not None and volume_sma_20 is not None and volume_sma_20 > 0
        else None,
        "score": score,
        "status": status,
        "triggers": triggers,
        "triggers_str": ", ".join(triggers),
    }


def evaluate_signals(ticker: str, df: pd.DataFrame, market_regime: str) -> Optional[dict]:
    """Evaluate the latest row and return only displayable (score >= 3) setups."""
    if df.empty or len(df) < 201:
        return None
    indicators = calculate_indicators(df.copy())
    setup = evaluate_setup(indicators, len(indicators) - 1, market_regime, ticker)
    return setup if setup and setup["score"] >= 3 else None


def run_screener() -> None:
    LOGGER.info("Checking broader market regime (200 EMA + three-session rule)...")
    market_regime = get_market_regime()
    LOGGER.info("Nifty 50 regime: %s", market_regime)
    if market_regime != BULLISH:
        LOGGER.warning("Long setup classification is suspended while regime is %s.", market_regime)

    if not os.path.exists(CACHE_DIR):
        LOGGER.error("No cache directory found. Run data_fetcher.py first.")
        return

    cache_files = [name for name in os.listdir(CACHE_DIR) if name.endswith(".parquet")]
    results: list[dict] = []
    failures: list[dict] = []
    successful_scans = 0
    for filename in cache_files:
        ticker = filename.removesuffix(".parquet")
        try:
            dataframe = pd.read_parquet(os.path.join(CACHE_DIR, filename))
            signal = evaluate_signals(ticker, dataframe, market_regime)
            successful_scans += 1
            if signal:
                results.append(signal)
                LOGGER.info("[%s] %s | Score %s/6", signal["status"], ticker, signal["score"])
        except Exception as exc:
            failures.append({"ticker": ticker, "error": str(exc)})
            LOGGER.warning("Could not screen %s: %s", ticker, exc)

    results.sort(key=lambda item: (item["score"], item.get("volume_ratio") or 0), reverse=True)
    ist = timezone(timedelta(hours=5, minutes=30))
    output_data = {
        "updated_at": datetime.now(ist).isoformat(),
        "market_regime": market_regime,
        "universe_size": len(cache_files),
        "total_scanned": successful_scans,
        "failed_count": len(failures),
        "failures": failures,
        "total_signals": len(results),
        "signals": results,
    }
    os.makedirs(os.path.dirname(OUTPUT_FILE), exist_ok=True)
    with tempfile.NamedTemporaryFile("w", dir=os.path.dirname(OUTPUT_FILE), delete=False, encoding="utf-8") as temporary:
        json.dump(output_data, temporary, indent=2)
        temporary_name = temporary.name
    os.replace(temporary_name, OUTPUT_FILE)
    LOGGER.info("Screener complete: %s setups; %s failed files.", len(results), len(failures))


if __name__ == "__main__":
    run_screener()
