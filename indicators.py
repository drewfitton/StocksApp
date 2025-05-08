import numpy as np
import pandas as pd


def calc_Bollinger(df, symbol, window=20, num_std=2):
    df = df.copy()
    df['adj_close'] = pd.to_numeric(df['adj_close'], errors='coerce')

    # Calculate SMA (Simple Moving Average)
    df['SMA'] = df[symbol].rolling(window=window).mean()

    # Calculate standard deviation
    df['std_dev'] = df[symbol].rolling(window=window).std()

    # Calculate Bollinger Bands
    df['upper_bollinger'] = df['SMA'] + (num_std * df['std_dev'])
    df['lower_bollinger'] = df['SMA'] - (num_std * df['std_dev'])

    # Drop rows with NaN values due to rolling window calculations
    # df = df.dropna()

    # Create a new DataFrame to return with just the Bollinger columns
    bollinger_df = df[['lower_bollinger', 'upper_bollinger']]
    
    return bollinger_df

def calc_RSI(df, symbol, window=14):
    df = df.copy()
    df['adj_close'] = pd.to_numeric(df['adj_close'], errors='coerce')

    # Calculate price differences
    delta = df[symbol].diff()

    # Separate gains and losses
    gain = (delta.where(delta > 0, 0)).rolling(window=window).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=window).mean()

    # Calculate RS (Relative Strength)
    rs = gain / loss

    # Calculate RSI (Relative Strength Index)
    df['RSI'] = 100 - (100 / (1 + rs))
    
    # Drop rows with NaN values due to rolling window calculations
    # df = df.dropna()

    rsi_df = df[['RSI']]

    return rsi_df

def calc_MACD(df, column='adj_close', short_per=12, long_per=26, signal_per=9):
    df = df.copy()

    # Calculate EMAs
    ema_short = df[column].ewm(span=short_per, adjust=False).mean()
    ema_long = df[column].ewm(span=long_per, adjust=False).mean()

    # MACD line
    df["MACD"] = ema_short - ema_long

    # Signal line
    df["MACD_Signal"] = df["MACD"].ewm(span=signal_per, adjust=False).mean()

    # Drop NaN values caused by initial periods
    # df = df.dropna(subset=["MACD", "MACD_Signal"])

    return df[["MACD", "MACD_Signal"]]

def calc_sma(df, symbol, window=20):
    df = df.copy()
    df['SMA'] = df[symbol].rolling(window=window).mean()
    df = df.dropna()
    sma_df = df['SMA']
    return sma_df

def calc_cci(close_df, high_df, low_df, symbol, window=20):
    df = close_df.copy()

    df['TP'] = (close_df[symbol] + high_df[symbol] + low_df[symbol]) / 3
    df['SMA'] = df['TP'].rolling(window=window).mean()

    df['MAD'] = df['TP'].rolling(window=window).apply(lambda x: np.mean(np.abs(x - np.mean(x))), raw=True)
    df['CCI'] = (df['TP'] - df['SMA']) / (0.015 * df['MAD'])
    df = df.dropna()
    cci_df = df['CCI']
    return cci_df




def calc_momentum(df, symbol, window=20):
    df = df.copy()
    df['momentum'] = (df[symbol] / df[symbol].shift(window)) - 1
    df = df.dropna()
    mom_df = df['momentum']
    return mom_df