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
