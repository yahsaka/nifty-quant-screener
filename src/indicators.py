import pandas as pd
import pandas_ta as ta

def calculate_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """
    Calculates key technical indicators on a stock's OHLCV DataFrame:
    - 50-day & 200-day Exponential Moving Averages (EMA)
    - 14-period Relative Strength Index (RSI)
    - Moving Average Convergence Divergence (MACD)
    - 20-day Volume Simple Moving Average (SMA)
    """
    if df.empty or len(df) < 200:
        return df

    # Ensure columns are 1D (flattens yfinance multi-index columns if present)
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)

    # Calculate EMAs
    df['EMA_50'] = ta.ema(df['Close'], length=50)
    df['EMA_200'] = ta.ema(df['Close'], length=200)

    # Calculate RSI
    df['RSI_14'] = ta.rsi(df['Close'], length=14)

    # Calculate Volume SMA (20 days)
    df['VOL_SMA_20'] = ta.sma(df['Volume'], length=20)

    # Calculate MACD (returns MACD_12_26_9, MACDh_12_26_9, MACDs_12_26_9)
    macd = ta.macd(df['Close'])
    if macd is not None and not macd.empty:
        df = pd.concat([df, macd], axis=1)

    return df