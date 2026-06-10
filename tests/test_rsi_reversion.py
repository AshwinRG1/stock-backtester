import pandas as pd

from app.strategies.rsi_reversion import RSIReversion


def make_ohlcv(closes: list[float]) -> pd.DataFrame:
    index = pd.date_range("2020-01-01", periods=len(closes), freq="D")
    close = pd.Series(closes, index=index, dtype=float)
    return pd.DataFrame(
        {"Open": close, "High": close, "Low": close, "Close": close, "Volume": 1000},
        index=index,
    )


def test_signals_are_valid_integers():
    """All signals must be in {-1, 0, 1} with no NaN values."""
    prices = [100.0 - i * 0.5 for i in range(60)]
    data = make_ohlcv(prices)
    strategy = RSIReversion()
    signals = strategy.generate_signals(data)
    assert set(signals.unique()).issubset({-1, 0, 1}), f"Unexpected values: {set(signals.unique())}"


def test_signals_length_matches_input():
    """Returned Series must have the same length as the input DataFrame."""
    prices = [100.0 - i * 0.5 for i in range(60)]
    data = make_ohlcv(prices)
    strategy = RSIReversion()
    signals = strategy.generate_signals(data)
    assert len(signals) == len(data)


def test_buy_signal_on_known_oversold():
    """Sharp sustained decline should drive RSI below 30 and produce a +1 signal.

    15 flat bars cover the RSI warmup period, then 20 bars of steep decline
    push RSI toward 0 — well below the default oversold threshold of 30.
    """
    flat = [100.0] * 15
    declining = [100.0 - i * 5 for i in range(20)]
    data = make_ohlcv(flat + declining)
    strategy = RSIReversion(period=14, oversold=30, overbought=70)
    signals = strategy.generate_signals(data)
    assert 1 in signals.values, "Expected a buy signal (+1) during the sustained decline"
