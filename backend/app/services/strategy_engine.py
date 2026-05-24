"""Strategy engine — base classes and signal generation framework."""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum

import numpy as np
import pandas as pd


class SignalType(str, Enum):
    BUY = "BUY"
    SELL = "SELL"
    HOLD = "HOLD"


@dataclass
class Signal:
    code: str
    date: str
    signal_type: SignalType
    quantity: int = 0
    price: float | None = None
    weight: float = 0.0  # target portfolio weight
    reason: str = ""


@dataclass
class StrategyConfig:
    name: str = "base"
    initial_capital: float = 100_000
    max_position_pct: float = 0.2  # max single stock position
    max_positions: int = 10  # max concurrent positions
    stop_loss_pct: float = -0.08  # -8% stop loss
    take_profit_pct: float = 0.25  # +25% take profit
    holding_days_min: int = 1  # T+1 minimum (A-share)
    holding_days_max: int = 60


class BaseStrategy(ABC):
    """Abstract strategy — subclass and implement generate_signals()."""

    def __init__(self, config: StrategyConfig | None = None):
        self.config = config or StrategyConfig()

    @abstractmethod
    def generate_signals(
        self, data: dict[str, pd.DataFrame], current_date: str
    ) -> list[Signal]:
        """Generate trading signals for a given date.

        Args:
            data: {code: DataFrame with columns [trade_date, open, high, low, close, volume, ...]}
            current_date: YYYY-MM-DD

        Returns:
            List of Signal objects.
        """
        ...

    def compute_position_size(self, price: float, capital: float) -> int:
        """Compute number of shares to buy given price and available capital."""
        max_value = capital * self.config.max_position_pct
        return max(int(max_value / price // 100) * 100, 0)  # round to 100-share lots

    # ── Common indicators (static helpers) ─────────────────────────

    @staticmethod
    def sma(series: pd.Series, window: int) -> pd.Series:
        return series.rolling(window).mean()

    @staticmethod
    def ema(series: pd.Series, window: int) -> pd.Series:
        return series.ewm(span=window, adjust=False).mean()

    @staticmethod
    def macd(close: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9):
        """Returns (macd_line, signal_line, histogram)."""
        ema_fast = close.ewm(span=fast, adjust=False).mean()
        ema_slow = close.ewm(span=slow, adjust=False).mean()
        macd_line = ema_fast - ema_slow
        signal_line = macd_line.ewm(span=signal, adjust=False).mean()
        histogram = macd_line - signal_line
        return macd_line, signal_line, histogram

    @staticmethod
    def rsi(close: pd.Series, window: int = 14) -> pd.Series:
        delta = close.diff()
        gain = delta.clip(lower=0)
        loss = (-delta).clip(lower=0)
        avg_gain = gain.ewm(alpha=1 / window, adjust=False).mean()
        avg_loss = loss.ewm(alpha=1 / window, adjust=False).mean()
        rs = avg_gain / avg_loss.replace(0, np.nan)
        return 100.0 - (100.0 / (1.0 + rs))

    @staticmethod
    def bollinger_bands(close: pd.Series, window: int = 20, n_std: float = 2.0):
        """Returns (middle, upper, lower)."""
        middle = close.rolling(window).mean()
        std = close.rolling(window).std()
        upper = middle + n_std * std
        lower = middle - n_std * std
        return middle, upper, lower

    @staticmethod
    def atr(high: pd.Series, low: pd.Series, close: pd.Series, window: int = 14):
        """Average True Range."""
        prev_close = close.shift(1)
        tr1 = high - low
        tr2 = (high - prev_close).abs()
        tr3 = (low - prev_close).abs()
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        return tr.ewm(span=window, adjust=False).mean()

    @staticmethod
    def momentum(close: pd.Series, window: int = 20) -> pd.Series:
        return close.pct_change(window)

    @staticmethod
    def volume_ratio(volume: pd.Series, window: int = 5) -> pd.Series:
        return volume / volume.rolling(window).mean()


# ── Signal Pipeline ────────────────────────────────────────────────

class SignalPipeline:
    """Chain multiple strategies, aggregate signals, resolve conflicts."""

    def __init__(self, strategies: list[BaseStrategy]):
        self.strategies = strategies

    def run(
        self, data: dict[str, pd.DataFrame], current_date: str
    ) -> list[Signal]:
        """Run all strategies, deduplicate and prioritize signals."""
        all_signals: list[Signal] = []
        for strat in self.strategies:
            signals = strat.generate_signals(data, current_date)
            all_signals.extend(signals)
        return self._resolve(all_signals)

    @staticmethod
    def _resolve(signals: list[Signal]) -> list[Signal]:
        """Deduplicate: per stock, keep strongest signal (BUY > SELL)."""
        resolved: dict[str, Signal] = {}
        for s in signals:
            if s.code not in resolved:
                resolved[s.code] = s
            elif s.signal_type == SignalType.BUY:
                resolved[s.code] = s  # BUY overrides SELL
        return list(resolved.values())
