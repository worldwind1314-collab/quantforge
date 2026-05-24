"""Mean reversion strategy — buy oversold, sell overbought using Bollinger Bands."""

import pandas as pd

from ..strategy_engine import BaseStrategy, Signal, SignalType, StrategyConfig


class MeanReversionStrategy(BaseStrategy):
    """Buy when price touches lower Bollinger Band (oversold).
    Sell when price crosses above middle band.

    Also uses RSI filter to confirm oversold/overbought.
    """

    def __init__(
        self,
        bb_window: int = 20,
        bb_std: float = 2.0,
        rsi_window: int = 14,
        rsi_oversold: float = 30,
        rsi_overbought: float = 70,
        config: StrategyConfig | None = None,
    ):
        super().__init__(config or StrategyConfig(name="MeanReversion"))
        self.bb_window = bb_window
        self.bb_std = bb_std
        self.rsi_window = rsi_window
        self.rsi_oversold = rsi_oversold
        self.rsi_overbought = rsi_overbought

    def generate_signals(self, data: dict[str, pd.DataFrame], current_date: str) -> list[Signal]:
        signals = []
        for code, df in data.items():
            if current_date not in df.index:
                continue
            min_len = max(self.bb_window, self.rsi_window) + 1
            if len(df.loc[:current_date]) < min_len:
                continue

            close_series = df.loc[:current_date, "close"]
            if isinstance(close_series, pd.DataFrame):
                close_series = close_series.iloc[:, 0]
            close_series = pd.to_numeric(close_series, errors="coerce")

            _, upper, lower = self.bollinger_bands(close_series, self.bb_window, self.bb_std)
            rsi = self.rsi(close_series, self.rsi_window)

            current_close = close_series.iloc[-1]
            current_lower = lower.iloc[-1]
            current_upper = upper.iloc[-1]
            current_rsi = rsi.iloc[-1]

            if any(pd.isna(x) for x in [current_close, current_lower, current_upper, current_rsi]):
                continue

            # Buy: close near or below lower band + RSI oversold
            if current_close <= current_lower * 1.01 and current_rsi <= self.rsi_oversold:
                signals.append(
                    Signal(
                        code=code,
                        date=current_date,
                        signal_type=SignalType.BUY,
                        weight=1.0,
                        reason=f"Oversold: close {current_close:.2f} <= lower {current_lower:.2f}, RSI {current_rsi:.1f}",
                    )
                )
            # Sell: close near or above upper band + RSI overbought
            elif current_close >= current_upper * 0.99 and current_rsi >= self.rsi_overbought:
                signals.append(
                    Signal(
                        code=code,
                        date=current_date,
                        signal_type=SignalType.SELL,
                        weight=0.0,
                        reason=f"Overbought: close {current_close:.2f} >= upper {current_upper:.2f}, RSI {current_rsi:.1f}",
                    )
                )
        return signals
