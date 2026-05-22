"""
tests/test_sma_crossover.py — Unit Tests for SMACrossover
==========================================================
PURPOSE
-------
This file verifies that the SMACrossover strategy behaves correctly in
isolation, without needing real market data or network access.

WHY TEST WITH SYNTHETIC DATA?
------------------------------
Real stock data changes every day, is noisy, and can contain edge cases
(trading halts, missing bars, stock splits) that make it hard to write
deterministic tests.  Instead we construct simple, predictable price series
where we know exactly what the strategy should output.  If the test says "a
sharp price jump at bar 20 should produce a buy signal", we can be confident
the logic is correct without relying on any external data source.

HELPER FUNCTION — make_ohlcv()
-------------------------------
Builds a minimal OHLCV DataFrame from a list of closing prices.  Open, High,
Low are all set to the same value as Close (a flat bar), which is enough for
any indicator that only uses the Close column.  The DatetimeIndex starts at
2020-01-01 with a daily frequency.

THE THREE TEST CASES
--------------------
Good strategy tests cover three different concerns:

  test_signals_are_valid_integers
      → CONTRACT TEST: checks that the output only contains allowed values.
        If the strategy ever returns 2, 0.5, or NaN the system would break
        silently, so we catch that here.

  test_signals_length_matches_input
      → SHAPE TEST: checks that the output Series lines up with the input
        DataFrame row-for-row.  A length mismatch would cause index-alignment
        bugs downstream in the engine.

  test_buy_signal_on_known_crossover
      → BEHAVIOUR TEST: the most important one.  We engineer a price series
        where we *know* a crossover must happen (flat prices then a big jump)
        and assert that the strategy actually emits a +1 buy signal.  This
        confirms the core logic is correct, not just the output format.

LEARNING NOTE
-------------
pytest automatically discovers functions whose names start with `test_` in
files whose names start with `test_`.  Run all tests from the project root with:
    pytest -v
"""

import numpy as np
import pandas as pd
import pytest

from app.strategies.sma_crossover import SMACrossover


def make_ohlcv(closes: list[float]) -> pd.DataFrame:
    index = pd.date_range("2020-01-01", periods=len(closes), freq="D")
    close = pd.Series(closes, index=index, dtype=float)
    return pd.DataFrame(
        {"Open": close, "High": close, "Low": close, "Close": close, "Volume": 1000},
        index=index,
    )


def test_signals_are_valid_integers():
    """All signals must be in {-1, 0, 1}."""
    prices = [float(i) for i in range(1, 61)]
    data = make_ohlcv(prices)
    strategy = SMACrossover(fast_period=5, slow_period=10)
    signals = strategy.generate_signals(data)
    assert set(signals.unique()).issubset({-1, 0, 1}), f"Unexpected values: {set(signals.unique())}"


def test_signals_length_matches_input():
    """Returned Series must have the same length as the input DataFrame."""
    prices = [float(i) for i in range(1, 61)]
    data = make_ohlcv(prices)
    strategy = SMACrossover(fast_period=5, slow_period=10)
    signals = strategy.generate_signals(data)
    assert len(signals) == len(data)


def test_buy_signal_on_known_crossover():
    """Fast SMA crossing above slow SMA should produce a +1 signal.

    Prices rise from 10 to 10+N, so the fast SMA (5-period) catches up to and
    crosses above the slow SMA (10-period) early in the series.  After enough
    warmup bars we expect at least one +1 in the signals.
    """
    # Flat prices for warmup, then a sharp jump to force a crossover
    flat = [10.0] * 20
    jump = [30.0] * 20
    prices = flat + jump
    data = make_ohlcv(prices)
    strategy = SMACrossover(fast_period=5, slow_period=10)
    signals = strategy.generate_signals(data)
    assert 1 in signals.values, "Expected a buy signal (+1) on the known crossover"
