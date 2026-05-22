"""
app/data/fetcher.py — Market Data Retrieval
============================================
PURPOSE
-------
This is the single point of entry for all raw market data in the project.
Whenever any part of the codebase needs price history for a stock, it calls
fetch_ohlcv() from here — nothing else reaches out to the internet directly.

WHY A THIN WRAPPER?
-------------------
yfinance's API is not guaranteed to be stable; column names, MultiIndex
behaviour, and timezone handling have changed across versions.  By isolating
all yfinance calls here we only ever have one place to update if the library
changes, and the rest of the codebase never has to know about it.

WHAT IS OHLCV?
--------------
OHLCV stands for the five columns that describe a price bar (one row per
trading day by default):
    Open   — price at market open
    High   — highest price reached during the bar
    Low    — lowest price reached during the bar
    Close  — price at market close  ← most indicators use this column
    Volume — number of shares traded

HOW IT FITS IN
--------------
    fetch_ohlcv()  →  indicators.py (calc_sma / calc_rsi / calc_macd)
                   →  strategy.generate_signals()
                   →  (future) engine.run()

LEARNING NOTE
-------------
`auto_adjust=True` tells yfinance to apply split and dividend adjustments
automatically so that historical prices are comparable across time.  Without
it, a 2-for-1 stock split would make the price look like it halved overnight.
"""

import pandas as pd
import yfinance as yf


def fetch_ohlcv(ticker: str, period: str = "1y", interval: str = "1d") -> pd.DataFrame:
    """Download OHLCV data for a ticker via yfinance.

    Parameters
    ----------
    ticker : str
        The stock symbol (e.g. 'AAPL').
    period : str
        Lookback window accepted by yfinance (e.g. '1y', '6mo', '3mo').
    interval : str
        Bar interval accepted by yfinance (e.g. '1d', '1h', '15m').

    Returns
    -------
    pd.DataFrame
        DataFrame with columns [Open, High, Low, Close, Volume] and a
        DatetimeIndex (timezone-naive UTC). Raises ValueError if no data
        is returned.
    """
    raw = yf.download(ticker, period=period, interval=interval, auto_adjust=True, progress=False)

    if raw.empty:
        raise ValueError(f"No data returned for ticker '{ticker}' (period={period}, interval={interval})")

    df = raw[["Open", "High", "Low", "Close", "Volume"]].copy()

    # Flatten MultiIndex columns that yfinance may produce when downloading a single ticker
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)

    df.index = pd.to_datetime(df.index).tz_localize(None)
    df.index.name = "Date"

    return df
