"""
app/data/indicators.py — Technical Indicator Library
=====================================================
PURPOSE
-------
This file is a collection of pure functions that turn raw OHLCV price data
into derived numeric signals (indicators).  Every function takes a DataFrame
in and returns a single Series out — no side effects, no state.

WHY KEEP INDICATORS SEPARATE FROM STRATEGIES?
---------------------------------------------
An indicator is just maths on price data.  A strategy is a decision rule that
uses one or more indicators.  Keeping them separate means:
  - The same calc_sma() can be reused in five different strategies without
    copy-pasting code.
  - You can unit-test indicators independently from strategy logic.
  - Swapping out the underlying library (e.g. replacing pandas-ta with TA-Lib)
    only requires changes here, not inside every strategy file.

INDICATORS IN THIS FILE
-----------------------
  calc_sma   — Simple Moving Average: smooths price noise by averaging the
               last N closing prices.  Slower to react, good for trend direction.

  calc_rsi   — Relative Strength Index: oscillates between 0 and 100.
               < 30 = potentially oversold (market may bounce up).
               > 70 = potentially overbought (market may pull back).

  calc_macd  — Moving Average Convergence/Divergence: the difference between
               a fast (12-period) and slow (26-period) EMA.  Used to spot
               momentum shifts and trend changes.

LEARNING NOTE
-------------
All functions delegate to pandas-ta (https://github.com/twopirllc/pandas-ta),
a library that implements 130+ indicators on top of pandas.  We call it here
rather than implementing the maths ourselves so the project stays focused on
strategy logic, not indicator arithmetic.
"""

import pandas as pd
import pandas_ta as ta


def calc_sma(data: pd.DataFrame, period: int) -> pd.Series:
    """Simple moving average of the Close price."""
    return ta.sma(data["Close"], length=period)


def calc_rsi(data: pd.DataFrame, period: int = 14) -> pd.Series:
    """Relative Strength Index of the Close price."""
    return ta.rsi(data["Close"], length=period)


def calc_macd(data: pd.DataFrame) -> pd.Series:
    """MACD line (12/26/9). Returns the MACD line only (not signal or histogram)."""
    result = ta.macd(data["Close"])
    return result.iloc[:, 0]
