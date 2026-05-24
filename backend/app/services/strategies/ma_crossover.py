"""Moving average crossover strategy — classic trend-following."""

import pandas as pd

from ..strategy_engine import BaseStrategy, Signal, SignalType, StrategyConfig


class MACrossoverStrategy(BaseStrategy):
    """Golden cross / dead cross strategy.

    Buy when fast MA crosses above slow MA (golden cross).
    Sell when fast MA crosses below slow MA (dead cross).
    """

    def __init__(self, fast: int = 10, slow: int = 30, config: StrategyConfig | None = None):
        super().__init__(config or StrategyConfig(name=f"MA_{fast}_{slow}"))
        self.fast = fast
        self.slow = slow

    def generate_signals(self, data: dict[str, pd.DataFrame], current_date: str) -> list[Signal]:
        signals = []
        for code, df in data.items():
            if current_date not in df.index:
                continue
            if len(df.loc[:current_date]) < self.slow + 1:
                continue

            close_series = df.loc[:current_date, "close"]
            if isinstance(close_series, pd.DataFrame):
                close_series = close_series.iloc[:, 0]
            close_series = pd.to_numeric(close_series, errors="coerce")

            fast_ma = self.sma(close_series, self.fast)
            slow_ma = self.sma(close_series, self.slow)

            if pd.isna(fast_ma.iloc[-1]) or pd.isna(slow_ma.iloc[-1]):
                continue
            if pd.isna(fast_ma.iloc[-2]) or pd.isna(slow_ma.iloc[-2]):
                continue

            # Golden cross: fast crosses above slow
            if fast_ma.iloc[-2] <= slow_ma.iloc[-2] and fast_ma.iloc[-1] > slow_ma.iloc[-1]:
                signals.append(
                    Signal(
                        code=code,
                        date=current_date,
                        signal_type=SignalType.BUY,
                        weight=1.0,
                        reason=f"Golden cross: MA{self.fast}>{self.slow}",
                    )
                )
            # Dead cross: fast crosses below slow
            elif fast_ma.iloc[-2] >= slow_ma.iloc[-2] and fast_ma.iloc[-1] < slow_ma.iloc[-1]:
                signals.append(
                    Signal(
                        code=code,
                        date=current_date,
                        signal_type=SignalType.SELL,
                        weight=0.0,
                        reason=f"Dead cross: MA{self.fast}<{self.slow}",
                    )
                )
        return signals
