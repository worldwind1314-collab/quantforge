"""Momentum breakout strategy — buy when price breaks above recent high."""

import pandas as pd

from ..strategy_engine import BaseStrategy, Signal, SignalType, StrategyConfig


class MomentumBreakoutStrategy(BaseStrategy):
    """Buy when close > highest high of lookback period (breakout).
    Sell when close < lowest low of lookback period (breakdown).
    """

    def __init__(self, lookback: int = 20, config: StrategyConfig | None = None):
        super().__init__(config or StrategyConfig(name=f"Breakout_{lookback}"))
        self.lookback = lookback

    def generate_signals(self, data: dict[str, pd.DataFrame], current_date: str) -> list[Signal]:
        signals = []
        for code, df in data.items():
            if current_date not in df.index:
                continue
            if len(df.loc[:current_date]) < self.lookback + 1:
                continue

            close_series = df.loc[:current_date, "close"]
            if isinstance(close_series, pd.DataFrame):
                close_series = close_series.iloc[:, 0]
            close_series = pd.to_numeric(close_series, errors="coerce")

            high_series = df.loc[:current_date, "high"]
            if isinstance(high_series, pd.DataFrame):
                high_series = high_series.iloc[:, 0]
            high_series = pd.to_numeric(high_series, errors="coerce")

            # Highest high of previous N days (excluding today)
            highest_high = high_series.iloc[-(self.lookback + 1):-1].max()
            current_close = close_series.iloc[-1]

            if pd.isna(highest_high) or pd.isna(current_close):
                continue

            if current_close > highest_high:
                signals.append(
                    Signal(
                        code=code,
                        date=current_date,
                        signal_type=SignalType.BUY,
                        weight=1.0,
                        reason=f"Breakout: close {current_close:.2f} > {self.lookback}d high {highest_high:.2f}",
                    )
                )
            elif current_close < highest_high * 0.95:
                signals.append(
                    Signal(
                        code=code,
                        date=current_date,
                        signal_type=SignalType.SELL,
                        weight=0.0,
                        reason=f"Breakdown: close below 95% of {self.lookback}d high",
                    )
                )
        return signals
