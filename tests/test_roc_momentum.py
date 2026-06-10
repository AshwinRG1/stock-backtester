import pandas as pd

from app.strategies.roc_momentum import ROCMomentum


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
    strategy = ROCMomentum()
    signals = strategy.generate_signals(data)
    assert set(signals.unique()).issubset({-1, 0, 1}), f"Unexpected values: {set(signals.unique())}"


def test_signals_length_matches_input():
    """Returned Series must have the same length as the input DataFrame."""
    prices = [100.0 - i * 0.5 for i in range(60)]
    data = make_ohlcv(prices)
    strategy = ROCMomentum()
    signals = strategy.generate_signals(data)
    assert len(signals) == len(data)


def test_buy_signal_on_strong_upward_momentum():
    """A price doubling over 10 bars produces 100% ROC, well above the 2% threshold.

    15 flat bars give ROC a baseline (close[t-10] = 100); the jump to 200
    then yields ROC = (200 - 100) / 100 * 100 = 100%, triggering a +1 signal.
    """
    flat = [100.0] * 15
    jump = [200.0] * 15
    data = make_ohlcv(flat + jump)
    strategy = ROCMomentum(period=10, threshold=2.0)
    signals = strategy.generate_signals(data)
    assert 1 in signals.values, "Expected a buy signal (+1) on strong upward momentum"
