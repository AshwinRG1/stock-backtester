from app.strategies.bollinger_breakout import BollingerBreakout
from app.strategies.bollinger_reversion import BollingerReversion
from app.strategies.ema_crossover import EMACrossover
from app.strategies.macd_crossover import MACDCrossover
from app.strategies.obv_trend import OBVTrend
from app.strategies.roc_momentum import ROCMomentum
from app.strategies.rsi_reversion import RSIReversion
from app.strategies.sma_crossover import SMACrossover
from app.strategies.zscore_reversion import ZScoreReversion

STRATEGY_REGISTRY = {
    "SMACrossover": SMACrossover,
    "EMACrossover": EMACrossover,
    "MACDCrossover": MACDCrossover,
    "BollingerBreakout": BollingerBreakout,
    "BollingerReversion": BollingerReversion,
    "RSIReversion": RSIReversion,
    "ZScoreReversion": ZScoreReversion,
    "ROCMomentum": ROCMomentum,
    "OBVTrend": OBVTrend,
}
