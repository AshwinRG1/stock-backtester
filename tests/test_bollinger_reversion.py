import pandas as pd

from app.strategies.bollinger_reversion import BollingerReversion


def make_ohlcv(closes: list[float]) -> pd.DataFrame:
    index = pd.date_range("2020-01-01", periods=len(closes), freq="D")
    close = pd.Series(closes, index=index, dtype=float)
    return pd.DataFrame(
        {"Open": close, "High": close, "Low": close, "Close": close, "Volume": 1000},
        index=index,
    )


def test_signals_are_valid_integers():
    """All signals must be in {-1, 0, 1} with no NaN values."""
    prices = [100.0 + i * 0.1 for i in range(60)]
    data = make_ohlcv(prices)
    strategy = BollingerReversion()
    signals = strategy.generate_signals(data)
    assert set(signals.unique()).issubset({-1, 0, 1}), f"Unexpected values: {set(signals.unique())}"


def test_signals_length_matches_input():
    """Returned Series must have the same length as the input DataFrame."""
    prices = [100.0 + i * 0.1 for i in range(60)]
    data = make_ohlcv(prices)
    strategy = BollingerReversion()
    signals = strategy.generate_signals(data)
    assert len(signals) == len(data)


def test_buy_signal_on_price_drop_below_lower_band():
    """A price far below the established range must touch the lower Bollinger Band.

    40 bars of gradual drift establish a mean and standard deviation; the
    subsequent crash to 1.0 places Close many standard deviations below the
    lower band, guaranteeing a +1 signal.
    """
    base = [100.0 + i * 0.1 for i in range(40)]
    crash = [1.0] * 10
    data = make_ohlcv(base + crash)
    strategy = BollingerReversion()
    signals = strategy.generate_signals(data)
    assert 1 in signals.values, "Expected a buy signal (+1) when price crashes below the lower band"
