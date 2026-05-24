"""ML-based strategy — uses XGBoost predictions for stock selection."""

import pandas as pd

from ..strategy_engine import BaseStrategy, Signal, SignalType, StrategyConfig


class MLStrategy(BaseStrategy):
    """Trade based on ML model predictions.

    Buys top-ranked stocks, sells when they fall below a rank threshold.
    """

    def __init__(
        self,
        predictions: list[dict] | None = None,
        top_n: int = 10,
        sell_rank_threshold: int = 50,
        config: StrategyConfig | None = None,
    ):
        super().__init__(config or StrategyConfig(name="ML_MultiFactor"))
        self._predictions = predictions or []
        self.top_n = top_n
        self.sell_rank_threshold = sell_rank_threshold

        # Build prediction lookup
        self._pred_map: dict[str, dict] = {}
        for p in self._predictions:
            self._pred_map[p["code"]] = p

    def generate_signals(self, data: dict[str, pd.DataFrame], current_date: str) -> list[Signal]:
        """Generate BUY for top-N ranked stocks, SELL for stocks that dropped below threshold."""
        signals = []

        # Sort by predicted return
        ranked = sorted(self._predictions, key=lambda x: x.get("predicted_return", -999), reverse=True)

        buy_codes = {r["code"] for r in ranked[: self.top_n]}

        for r in ranked:
            code = r["code"]
            if code not in data:
                continue
            if r["prediction_rank"] is None:
                continue

            if code in buy_codes:
                signals.append(
                    Signal(
                        code=code,
                        date=current_date,
                        signal_type=SignalType.BUY,
                        weight=r.get("confidence", 0.5),
                        reason=f"ML rank #{r['prediction_rank']}, pred_return={r['predicted_return']:.2f}%",
                    )
                )
            elif r["prediction_rank"] > self.sell_rank_threshold:
                signals.append(
                    Signal(
                        code=code,
                        date=current_date,
                        signal_type=SignalType.SELL,
                        weight=0.0,
                        reason=f"ML rank #{r['prediction_rank']} > threshold {self.sell_rank_threshold}",
                    )
                )

        return signals
