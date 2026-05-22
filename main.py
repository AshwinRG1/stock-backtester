import app.engine as engine
import app.data.fetcher as fetcher
import app.metrics as metrics
from app.strategies.sma_crossover import SMACrossover


TICKER = "AAPL"  # Example ticker, can be changed to any stock symbol
PERIOD = "10y"     # Lookback window for historical data (e.g. '1y', '6mo', '3mo')
INTERVAL = "1d"   # Bar interval (e.g. '1d',
INITIAL_CAPITAL = 10_000.0  # Starting capital for backtesting


data = fetcher.fetch_ohlcv(TICKER, period=PERIOD, interval=INTERVAL)
strategy = SMACrossover(fast_period=20, slow_period=50)
result = engine.run_engine(data, strategy, initial_capital=INITIAL_CAPITAL, ticker=TICKER)
metrics.summary(result)
