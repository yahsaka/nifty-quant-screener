# tests/test_core_logic.py

import pytest
import pandas as pd
import numpy as np

# Now these imports will resolve correctly thanks to conftest.py
from screener import evaluate_setup, BULLISH, BEARISH
from backtest import _execute_trade

# ==========================================
# TEST: evaluate_setup()
# ==========================================

def test_evaluate_setup_perfect_score():
    """
    Feeds a 2-row synthetic DataFrame to trigger all 6 conditions simultaneously:
    ABOVE_EMA_200, EMA_200_BREAKOUT, ABOVE_EMA_50, VOLUME_SPIKE_2X, 
    RSI_60_TO_70, and MACD_BULLISH_CROSS.
    """
    df = pd.DataFrame({
        "Close": [99.0, 105.0],           # Crosses 100 to trigger Breakout
        "EMA_50": [95.0, 95.0],           # Close > 95
        "EMA_200": [100.0, 100.0],        # Prev < 100, Curr > 100
        "RSI_14": [50.0, 65.0],           # Lands in 60-70 range
        "Volume": [1000.0, 5000.0],       # 5000 > (2 * 1000)
        "VOL_SMA_20": [1000.0, 1000.0],
        "MACD_12_26_9": [-0.5, 1.0],      # Crosses above signal
        "MACDs_12_26_9": [0.0, 0.5]
    })
    
    idx = 1 # Evaluate the latest row
    
    result = evaluate_setup(df, idx, market_regime=BULLISH, ticker="TEST")
    
    assert result is not None
    assert result["score"] == 6
    assert result["status"] == "Trade-Ready"
    assert "EMA_200_BREAKOUT" in result["triggers"]
    assert "MACD_BULLISH_CROSS" in result["triggers"]

def test_evaluate_setup_fails_overextended():
    """
    Ensures a setup is discarded (returns None) if the price is > 15% above the 50 EMA,
    even if all other indicators are bullish.
    """
    df = pd.DataFrame({
        "Close": [99.0, 120.0],           # 120 is 26% above EMA_50 (95)
        "EMA_50": [95.0, 95.0],
        "EMA_200": [100.0, 100.0],
        "RSI_14": [50.0, 65.0],
        "Volume": [1000.0, 5000.0],
        "VOL_SMA_20": [1000.0, 1000.0],
        "MACD_12_26_9": [-0.5, 1.0],
        "MACDs_12_26_9": [0.0, 0.5]
    })
    
    result = evaluate_setup(df, idx=1, market_regime=BULLISH, ticker="TEST")
    assert result is None, "Should reject stock if pct_above_50 > MAX_PCT_ABOVE_50EMA"

def test_evaluate_setup_fails_bearish_regime():
    """Returns None immediately if the market regime is not BULLISH."""
    df = pd.DataFrame({
        "Close": [99.0, 105.0], "EMA_50": [95.0, 95.0], "EMA_200": [100.0, 100.0],
        "RSI_14": [65.0, 65.0], "Volume": [5000, 5000], "VOL_SMA_20": [1000, 1000],
        "MACD_12_26_9": [1.0, 1.0], "MACDs_12_26_9": [0.5, 0.5]
    })
    result = evaluate_setup(df, idx=1, market_regime=BEARISH)
    assert result is None

# ==========================================
# TEST: _execute_trade()
# ==========================================

@pytest.fixture
def df_trade_base():
    """Creates a 25-session OHLC dataframe to test the 20-day hold logic."""
    return pd.DataFrame({
        "Open": [100.0] * 25,
        "High": [105.0] * 25,
        "Low": [98.0] * 25,
        "Close": [102.0] * 25
    })

def test_execute_trade_time_exit(df_trade_base):
    """Test full 20-day hold with no stop loss hit."""
    # signal_idx = 0. Entry is on idx 1. 
    # Entry Price = 100.0 * (1 + 0.0005 slippage) = 100.05
    # ATR = 5.0. Stop = 100.05 - (2.0 * 5.0) = 90.05
    
    trade = _execute_trade(df_trade_base, signal_idx=0, atr=5.0)
    
    assert trade is not None
    assert trade["Exit Reason"] == "Time exit"
    assert np.isclose(trade["Entry Price"], 100.05)
    
    # Exit on idx 20 (entry + 20 - 1). Close is 102.0
    # Exit Price = 102.0 * (1 - 0.0025 friction) = 101.745
    assert np.isclose(trade["Exit Price"], 101.745)
    assert trade["Hit_SL"] is False

def test_execute_trade_intraday_atr_stop(df_trade_base):
    """Test intraday stop hit where Open is safe, but Low breaches stop."""
    df = df_trade_base.copy()
    
    # Entry is 100.05, Stop is 90.05.
    # On day 5, price drops to 89 during the session, but opens at 100.
    df.loc[5, "Open"] = 100.0
    df.loc[5, "Low"] = 89.0 
    
    trade = _execute_trade(df, signal_idx=0, atr=5.0)
    
    assert trade["Exit Reason"] == "ATR stop"
    assert trade["Hit_SL"] is True
    # Exits at exact stop price (90.05) minus friction
    expected_exit = 90.05 * (1 - 0.0025)
    assert np.isclose(trade["Exit Price"], expected_exit)

def test_execute_trade_gap_down_stop(df_trade_base):
    """Test catastrophic gap down where stock opens below the stop loss."""
    df = df_trade_base.copy()
    
    # Entry is 100.05, Stop is 90.05.
    # On day 5, stock violently gaps down and opens at 85.0
    df.loc[5, "Open"] = 85.0
    df.loc[5, "Low"] = 84.0 
    
    trade = _execute_trade(df, signal_idx=0, atr=5.0)
    
    assert trade["Exit Reason"] == "ATR stop (gap)"
    assert trade["Hit_SL"] is True
    # Exits at the weaker opening price (85.0) minus friction, NOT the stop price
    expected_exit = 85.0 * (1 - 0.0025)
    assert np.isclose(trade["Exit Price"], expected_exit)
