"""Trading models — paper accounts, orders, positions, trades."""

from datetime import datetime

from sqlalchemy import DateTime, Float, Integer, String, func
from sqlalchemy.orm import Mapped, mapped_column

from ..core.database import Base


class PaperAccount(Base):
    """Virtual trading account for paper trading."""

    __tablename__ = "paper_accounts"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    name: Mapped[str] = mapped_column(String(50), default="default")
    initial_capital: Mapped[float] = mapped_column(Float, default=100_000)
    cash: Mapped[float] = mapped_column(Float, default=100_000)
    total_value: Mapped[float] = mapped_column(Float, default=100_000)
    created_at: Mapped[datetime] = mapped_column(DateTime, server_default=func.now())
    updated_at: Mapped[datetime] = mapped_column(DateTime, server_default=func.now(), onupdate=func.now())


class PaperPosition(Base):
    """Current holdings in paper account."""

    __tablename__ = "paper_positions"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    account_id: Mapped[int] = mapped_column(Integer, nullable=False)
    code: Mapped[str] = mapped_column(String(12), nullable=False)
    quantity: Mapped[int] = mapped_column(Integer, default=0)
    avg_cost: Mapped[float] = mapped_column(Float, default=0.0)
    current_price: Mapped[float] = mapped_column(Float, default=0.0)
    market_value: Mapped[float] = mapped_column(Float, default=0.0)
    unrealized_pnl: Mapped[float] = mapped_column(Float, default=0.0)
    updated_at: Mapped[datetime] = mapped_column(DateTime, server_default=func.now(), onupdate=func.now())


class PaperOrder(Base):
    """Order records — both filled and cancelled."""

    __tablename__ = "paper_orders"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    account_id: Mapped[int] = mapped_column(Integer, nullable=False)
    code: Mapped[str] = mapped_column(String(12), nullable=False)
    direction: Mapped[str] = mapped_column(String(4), nullable=False)  # BUY/SELL
    quantity: Mapped[int] = mapped_column(Integer, nullable=False)
    price: Mapped[float] = mapped_column(Float, nullable=False)
    status: Mapped[str] = mapped_column(String(10), default="filled")  # filled/cancelled
    filled_at: Mapped[datetime] = mapped_column(DateTime, server_default=func.now())
    strategy: Mapped[str | None] = mapped_column(String(50))


class BacktestResult(Base):
    """Store backtest run results."""

    __tablename__ = "backtest_results"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    strategy_name: Mapped[str] = mapped_column(String(50))
    start_date: Mapped[str] = mapped_column(String(10))
    end_date: Mapped[str] = mapped_column(String(10))
    initial_capital: Mapped[float] = mapped_column(Float)
    final_value: Mapped[float] = mapped_column(Float)
    total_return: Mapped[float] = mapped_column(Float)  # %
    annual_return: Mapped[float] = mapped_column(Float)  # %
    sharpe_ratio: Mapped[float | None] = mapped_column(Float, nullable=True)
    max_drawdown: Mapped[float] = mapped_column(Float)  # %
    win_rate: Mapped[float] = mapped_column(Float)  # %
    total_trades: Mapped[int] = mapped_column(Integer)
    profit_factor: Mapped[float | None] = mapped_column(Float, nullable=True)
    daily_values_json: Mapped[str | None] = mapped_column(String)  # JSON array of daily P&L
    trade_log_json: Mapped[str | None] = mapped_column(String)  # JSON array of trade records
    created_at: Mapped[datetime] = mapped_column(DateTime, server_default=func.now())
