import pandas as pd

from app.strategies.bollinger_breakout import BollingerBreakout


def make_ohlcv(closes: list[float]) -> pd.DataFrame:
    index = pd.date_range("2020-01-01", periods=len(closes), freq="D")
    close = pd.Series(closes, index=index, dtype=float)
    return pd.DataFrame(
        {"Open": close, "High": close, "Low": close, "Close": close, "Volume": 1000},
        index=index,
    )


def test_signals_are_valid_integers():
    """All signals must be in {-1, 0, 1} with no NaN values."""
    prices = [10.0 + i * 0.1 for i in range(60)]
    data = make_ohlcv(prices)
    strategy = BollingerBreakout()
    signals = strategy.generate_signals(data)
    assert set(signals.unique()).issubset({-1, 0, 1}), f"Unexpected values: {set(signals.unique())}"


def test_signals_length_matches_input():
    """Returned Series must have the same length as the input DataFrame."""
    prices = [10.0 + i * 0.1 for i in range(60)]
    data = make_ohlcv(prices)
    strategy = BollingerBreakout()
    signals = strategy.generate_signals(data)
    assert len(signals) == len(data)


def test_buy_signal_on_price_spike_above_upper_band():
    """A price far above the established range must break the upper Bollinger Band.

    40 bars of gradual drift establish a mean and standard deviation; the
    subsequent spike to 200 places Close many standard deviations above the
    upper band, guaranteeing a +1 signal.
    """
    base = [10.0 + i * 0.1 for i in range(40)]
    spike = [200.0] * 10
    data = make_ohlcv(base + spike)
    strategy = BollingerBreakout()
    signals = strategy.generate_signals(data)
    assert 1 in signals.values, "Expected a buy signal (+1) when price spikes above the upper band"
