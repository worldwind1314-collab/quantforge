"""Vector-based backtesting engine with A-share market rules.

Cost model:
  - Stamp duty (印花税): 0.05% on sell only (halved Aug 2023)
  - Commission (佣金): 0.025%, min 5 CNY
  - Slippage: 0.1% on both buy and sell

Rules:
  - T+1: cannot sell on same day as buy
  - No short selling (only long positions)
  - Round lot: multiples of 100 shares
"""

import json
import logging
from dataclasses import dataclass, field
from datetime import timedelta

import numpy as np
import pandas as pd
from sqlalchemy.orm import Session

from ..core.database import SessionLocal
from ..models.market import DailyQuote
from ..models.stock import Stock
from ..models.trading import BacktestResult as BacktestResultModel
from .strategy_engine import BaseStrategy, Signal, SignalType

logger = logging.getLogger(__name__)

# ── A-share trading costs ──────────────────────────────────────────

STAMP_DUTY_SELL = 0.0005  # 0.05% on sell
COMMISSION_RATE = 0.00025  # 0.025%
MIN_COMMISSION = 5.0  # 5 CNY minimum per trade
SLIPPAGE = 0.001  # 0.1%


def trade_cost(price: float, quantity: int, direction: str) -> float:
    """Compute total trading cost for a single trade."""
    value = price * quantity
    commission = max(value * COMMISSION_RATE, MIN_COMMISSION)
    stamp = value * STAMP_DUTY_SELL if direction == "SELL" else 0
    slippage = value * SLIPPAGE
    return commission + stamp + slippage


# ── Result types ───────────────────────────────────────────────────

@dataclass
class Trade:
    code: str
    entry_date: str
    exit_date: str
    entry_price: float
    exit_price: float
    quantity: int
    pnl: float
    pnl_pct: float
    reason: str = ""


@dataclass
class BacktestReport:
    strategy_name: str
    start_date: str
    end_date: str
    initial_capital: float
    final_value: float
    total_return: float
    annual_return: float
    sharpe_ratio: float
    max_drawdown: float
    win_rate: float
    total_trades: int
    profit_factor: float
    daily_values: list[dict] = field(default_factory=list)
    trades: list[dict] = field(default_factory=list)
    benchmark_return: float = 0.0


# ── Engine ─────────────────────────────────────────────────────────

class BacktestEngine:
    """Run a strategy against historical data."""

    def __init__(self, db: Session | None = None):
        self._db = db

    def _get_db(self) -> Session:
        return self._db or SessionLocal()

    def load_data(
        self,
        codes: list[str],
        start_date: str,
        end_date: str,
    ) -> dict[str, pd.DataFrame]:
        """Load daily quotes from DB, return {code: DataFrame} keyed by code."""
        db = self._get_db()
        rows = (
            db.query(DailyQuote)
            .filter(
                DailyQuote.code.in_(codes),
                DailyQuote.trade_date >= start_date,
                DailyQuote.trade_date <= end_date,
            )
            .order_by(DailyQuote.code, DailyQuote.trade_date)
            .all()
        )

        data: dict[str, list[dict]] = {}
        for r in rows:
            if r.code not in data:
                data[r.code] = []
            data[r.code].append(
                {
                    "trade_date": r.trade_date,
                    "open": r.open,
                    "high": r.high,
                    "low": r.low,
                    "close": r.close,
                    "volume": r.volume,
                    "amount": r.amount,
                    "pct_change": r.pct_change,
                    "turnover": r.turnover,
                }
            )

        result = {}
        for code, records in data.items():
            df = pd.DataFrame(records)
            df["trade_date"] = df["trade_date"].astype(str)
            df = df.sort_values("trade_date").set_index("trade_date")
            result[code] = df

        return result

    def run(
        self,
        strategy: BaseStrategy,
        codes: list[str],
        start_date: str,
        end_date: str,
        initial_capital: float = 100_000,
    ) -> BacktestReport:
        """Execute backtest and return report."""
        # Load data
        data = self.load_data(codes, start_date, end_date)

        # Get all trading dates from union of all stock dates
        all_dates: set[str] = set()
        for df in data.values():
            all_dates.update(df.index.tolist())
        trading_dates = sorted(all_dates)

        if len(trading_dates) < 20:
            raise ValueError(f"Not enough trading data: {len(trading_dates)} days")

        # State
        cash = initial_capital
        positions: dict[str, dict] = {}  # code → {quantity, avg_cost, buy_date}
        daily_values: list[dict] = []
        trades: list[Trade] = []

        for i, current_date in enumerate(trading_dates):
            # Update position market values
            portfolio_value = cash
            for code, pos in positions.items():
                close = self._get_price(data, code, current_date, "close")
                if close:
                    pos["current_price"] = close
                    pos["market_value"] = pos["quantity"] * close
                    portfolio_value += pos["market_value"]

            daily_values.append(
                {
                    "date": current_date,
                    "cash": round(cash, 2),
                    "equity": round(portfolio_value - cash, 2),
                    "total": round(portfolio_value, 2),
                }
            )

            # Check stop loss / take profit
            for code in list(positions.keys()):
                pos = positions[code]
                close = pos.get("current_price", 0)
                if close <= 0:
                    continue
                pnl_pct = (close - pos["avg_cost"]) / pos["avg_cost"]

                # Cannot sell if bought on same day (T+1)
                if current_date <= pos["buy_date"]:
                    continue

                if pnl_pct <= strategy.config.stop_loss_pct:
                    self._execute_sell(code, pos, close, current_date, cash, positions, trades, "stop_loss")
                elif pnl_pct >= strategy.config.take_profit_pct:
                    self._execute_sell(code, pos, close, current_date, cash, positions, trades, "take_profit")

            # Generate signals
            try:
                signals = strategy.generate_signals(data, current_date)
            except Exception as e:
                logger.debug(f"Signal error on {current_date}: {e}")
                continue

            # Execute signals
            for sig in signals:
                if sig.signal_type != SignalType.BUY:
                    continue
                if sig.code in positions:
                    continue  # already holding
                if len(positions) >= strategy.config.max_positions:
                    continue

                price = self._get_price(data, sig.code, current_date, "close")
                if not price or price <= 0:
                    continue

                qty = strategy.compute_position_size(price, cash)
                if qty <= 0:
                    continue

                cost = price * qty + trade_cost(price, qty, "BUY")
                if cost > cash:
                    continue

                cash -= cost
                positions[sig.code] = {
                    "quantity": qty,
                    "avg_cost": price,
                    "buy_date": current_date,
                    "current_price": price,
                    "market_value": price * qty,
                }

        # Close all remaining positions at last date
        last_date = trading_dates[-1]
        for code in list(positions.keys()):
            pos = positions[code]
            close = self._get_price(data, code, last_date, "close") or pos["avg_cost"]
            self._execute_sell(code, pos, close, last_date, cash, positions, trades, "close")

        # Compute final portfolio value
        final_value = cash
        for pos in positions.values():
            final_value += pos.get("market_value", 0)

        # Compute metrics
        return self._compute_report(
            strategy=strategy,
            start_date=start_date,
            end_date=end_date,
            initial_capital=initial_capital,
            final_value=final_value,
            daily_values=daily_values,
            trades=trades,
        )

    def _execute_sell(
        self,
        code: str,
        pos: dict,
        price: float,
        date: str,
        cash: float,
        positions: dict,
        trades: list[Trade],
        reason: str,
    ) -> float:
        """Execute a sell, update cash, remove position, record trade."""
        qty = pos["quantity"]
        proceeds = price * qty - trade_cost(price, qty, "SELL")
        pnl = proceeds - (pos["avg_cost"] * qty)
        pnl_pct = pnl / (pos["avg_cost"] * qty) if pos["avg_cost"] > 0 else 0

        trades.append(
            Trade(
                code=code,
                entry_date=pos["buy_date"],
                exit_date=date,
                entry_price=pos["avg_cost"],
                exit_price=price,
                quantity=qty,
                pnl=round(pnl, 2),
                pnl_pct=round(pnl_pct * 100, 2),
                reason=reason,
            )
        )

        cash += proceeds
        del positions[code]
        return cash

    @staticmethod
    def _get_price(
        data: dict[str, pd.DataFrame], code: str, date: str, field: str
    ) -> float | None:
        """Get a price field for a stock on a given date, with fallback."""
        df = data.get(code)
        if df is None or date not in df.index:
            return None
        val = df.loc[date, field]
        if isinstance(val, pd.Series):
            val = val.iloc[0]
        if pd.isna(val) or val <= 0:
            return None
        return float(val)

    @staticmethod
    def _compute_report(
        strategy: BaseStrategy,
        start_date: str,
        end_date: str,
        initial_capital: float,
        final_value: float,
        daily_values: list[dict],
        trades: list[Trade],
    ) -> BacktestReport:
        """Compute performance metrics."""
        total_return = (final_value - initial_capital) / initial_capital * 100
        n_days = len(daily_values)

        # Annual return
        years = n_days / 252 if n_days > 0 else 1
        annual_return = ((1 + total_return / 100) ** (1 / years) - 1) * 100 if years > 0 else 0

        # Daily returns for Sharpe
        totals = [d["total"] for d in daily_values]
        daily_rets = np.diff(totals) / np.array(totals[:-1]) if len(totals) > 1 else [0]
        daily_rets = daily_rets[np.isfinite(daily_rets)]
        mean_ret = np.mean(daily_rets) if len(daily_rets) > 0 else 0
        std_ret = np.std(daily_rets) if len(daily_rets) > 0 else 1
        sharpe = (mean_ret / std_ret * np.sqrt(252)) if std_ret > 0 else 0

        # Max drawdown
        peak = totals[0] if totals else initial_capital
        max_dd = 0.0
        for v in totals:
            if v > peak:
                peak = v
            dd = (v - peak) / peak * 100 if peak > 0 else 0
            if dd < max_dd:
                max_dd = dd

        # Win rate & profit factor
        wins = [t for t in trades if t.pnl > 0]
        losses = [t for t in trades if t.pnl <= 0]
        win_rate = len(wins) / len(trades) * 100 if trades else 0
        gross_profit = sum(t.pnl for t in wins) if wins else 0
        gross_loss = abs(sum(t.pnl for t in losses)) if losses else 0
        profit_factor = gross_profit / gross_loss if gross_loss > 0 else 0

        return BacktestReport(
            strategy_name=strategy.config.name,
            start_date=start_date,
            end_date=end_date,
            initial_capital=initial_capital,
            final_value=round(final_value, 2),
            total_return=round(total_return, 2),
            annual_return=round(annual_return, 2),
            sharpe_ratio=round(sharpe, 2),
            max_drawdown=round(max_dd, 2),
            win_rate=round(win_rate, 2),
            total_trades=len(trades),
            profit_factor=round(profit_factor, 2),
            daily_values=[
                {"date": d["date"], "total": d["total"]} for d in daily_values
            ],
            trades=[t.__dict__ for t in trades],
        )


def _native(val):
    """Convert numpy types to Python native types for DB insertion."""
    import numpy as np
    if isinstance(val, (np.floating, np.integer)):
        return val.item()
    if isinstance(val, np.bool_):
        return bool(val)
    return val


def save_backtest_result(db: Session, report: BacktestReport) -> int:
    """Persist backtest report to DB, return ID."""
    result = BacktestResultModel(
        strategy_name=report.strategy_name,
        start_date=report.start_date,
        end_date=report.end_date,
        initial_capital=_native(report.initial_capital),
        final_value=_native(report.final_value),
        total_return=_native(report.total_return),
        annual_return=_native(report.annual_return),
        sharpe_ratio=_native(report.sharpe_ratio),
        max_drawdown=_native(report.max_drawdown),
        win_rate=_native(report.win_rate),
        total_trades=_native(report.total_trades),
        profit_factor=_native(report.profit_factor),
        daily_values_json=json.dumps(report.daily_values),
        trade_log_json=json.dumps(report.trades),
    )
    db.add(result)
    db.commit()
    db.refresh(result)
    return result.id
