# tests/test_core_logic.py

import pytest
import pandas as pd
import numpy as np

# Adjust these imports based on your actual file structure
from src.screener import evaluate_setup
from src.paper_trader import _execute_trade 

# ==========================================
# TEST: evaluate_setup()
# ==========================================

def test_evaluate_setup_triggers_on_valid_conditions():
    """
    Feeds synthetic OHLCV data with known EMA/RSI/MACD values to ensure
    the trigger fires ONLY when conditions are met.
    """
    # Synthetic data simulating a fresh Bullish MACD crossover
    data = {
        "Open": [100.0, 101.0],
        "High": [102.0, 105.0],
        "Low": [99.0, 100.5],
        "Close": [101.0, 104.0],
        "Volume": [1000, 2000],
        "EMA_20": [98.0, 98.5],       # Close > EMA_20
        "EMA_50": [95.0, 95.2],       # EMA_20 > EMA_50
        "RSI_14": [48.0, 55.0],       # RSI crossed above 50
        "MACD_12_26_9": [-0.1, 0.5],  # MACD line crossed above signal
        "MACDs_12_26_9": [0.0, 0.1],  # Signal line
        "MACDh_12_26_9": [-0.1, 0.4]  # Histogram turned positive
    }
    df = pd.DataFrame(data)
    
    # Assert the setup fires
    assert evaluate_setup(df) is True, "Setup should trigger on valid bullish momentum"

def test_evaluate_setup_fails_on_weak_rsi():
    """Ensures the trigger fails if MACD is good but RSI is below the threshold."""
    data = {
        "Close": [101.0, 104.0],
        "EMA_20": [98.0, 98.5],
        "EMA_50": [95.0, 95.2],
        "RSI_14": [45.0, 49.0],       # <--- RSI below 50
        "MACD_12_26_9": [-0.1, 0.5],
        "MACDs_12_26_9": [0.0, 0.1],
        "MACDh_12_26_9": [-0.1, 0.4]
    }
    df = pd.DataFrame(data)
    assert evaluate_setup(df) is False, "Setup should NOT trigger if RSI is weak"


# ==========================================
# TEST: _execute_trade()
# ==========================================

# Assuming _execute_trade signature looks something like:
# _execute_trade(entry_price, stop_loss_price, slippage_pct, df_future_data)

@pytest.fixture
def slippage_pct():
    return 0.001 # 0.1% slippage

def test_execute_trade_gap_down_below_stop_loss(slippage_pct):
    """
    Costliest bug: If a stock gaps down below your Stop Loss at the open, 
    you CANNOT exit at your Stop Loss. You exit at the Open - slippage.
    """
    entry_price = 100.0
    stop_loss = 95.0
    
    # Next day opens massively lower than the SL
    future_data = pd.DataFrame({
        "Open": [90.0], 
        "High": [92.0],
        "Low": [88.0],
        "Close": [91.0]
    })
    
    # Expected logic: 
    # Open (90.0) < SL (95.0) -> Immediate market order at Open.
    # Exit Price = 90.0 - (90.0 * 0.001) = 89.91
    expected_exit = 90.0 * (1 - slippage_pct)
    
    exit_price, exit_reason = _execute_trade(entry_price, stop_loss, slippage_pct, future_data)
    
    assert exit_reason == "SL_GAP_DOWN"
    assert np.isclose(exit_price, expected_exit), f"Expected {expected_exit}, got {exit_price}"

def test_execute_trade_intraday_stop_loss_hit(slippage_pct):
    """
    Tests standard intraday SL hit. The price opens above SL, but drops below it during the day.
    Exit should occur exactly at SL price - slippage.
    """
    entry_price = 100.0
    stop_loss = 95.0
    
    future_data = pd.DataFrame({
        "Open": [98.0], 
        "High": [99.0],
        "Low": [94.0],  # <--- Drops below SL intraday
        "Close": [96.0]
    })
    
    # Expected logic:
    # Low (94.0) <= SL (95.0) -> Stop limit triggered.
    # Exit Price = 95.0 - (95.0 * 0.001) = 94.905
    expected_exit = stop_loss * (1 - slippage_pct)
    
    exit_price, exit_reason = _execute_trade(entry_price, stop_loss, slippage_pct, future_data)
    
    assert exit_reason == "SL_HIT"
    assert np.isclose(exit_price, expected_exit), f"Expected {expected_exit}, got {exit_price}"

def test_execute_trade_time_based_exit(slippage_pct):
    """
    Tests that if the SL is never breached, the trade exits at the Close price
    of the final period (Time Exit), accounting for slippage.
    """
    entry_price = 100.0
    stop_loss = 95.0
    
    future_data = pd.DataFrame({
        "Open": [101.0], 
        "High": [106.0],
        "Low": [99.0],   # SL never hit
        "Close": [105.0] # Exits here
    })
    
    # Expected logic:
    # SL intact. Exit at End of Day/Period.
    # Exit Price = 105.0 - (105.0 * 0.001) = 104.895
    expected_exit = 105.0 * (1 - slippage_pct)
    
    exit_price, exit_reason = _execute_trade(entry_price, stop_loss, slippage_pct, future_data)
    
    assert exit_reason == "TIME_EXIT"
    assert np.isclose(exit_price, expected_exit), f"Expected {expected_exit}, got {exit_price}"
