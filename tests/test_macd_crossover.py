import pandas as pd

from app.strategies.macd_crossover import MACDCrossover


def make_ohlcv(closes: list[float]) -> pd.DataFrame:
    index = pd.date_range("2020-01-01", periods=len(closes), freq="D")
    close = pd.Series(closes, index=index, dtype=float)
    return pd.DataFrame(
        {"Open": close, "High": close, "Low": close, "Close": close, "Volume": 1000},
        index=index,
    )


def test_signals_are_valid_integers():
    """All signals must be in {-1, 0, 1} with no NaN values."""
    prices = [float(i) for i in range(1, 81)]
    data = make_ohlcv(prices)
    strategy = MACDCrossover()
    signals = strategy.generate_signals(data)
    assert set(signals.unique()).issubset({-1, 0, 1}), f"Unexpected values: {set(signals.unique())}"


def test_signals_length_matches_input():
    """Returned Series must have the same length as the input DataFrame."""
    prices = [float(i) for i in range(1, 81)]
    data = make_ohlcv(prices)
    strategy = MACDCrossover()
    signals = strategy.generate_signals(data)
    assert len(signals) == len(data)


def test_buy_signal_on_known_crossover():
    """Flat prices followed by a sharp jump forces the MACD line above the signal line.

    During the flat period both lines converge to 0. The price jump causes the
    MACD line (more reactive) to spike above the lagging signal line.
    """
    flat = [10.0] * 40
    jump = [30.0] * 30
    data = make_ohlcv(flat + jump)
    strategy = MACDCrossover()
    signals = strategy.generate_signals(data)
    assert 1 in signals.values, "Expected a buy signal (+1) after the price jump"
