"""Daily market quote (K-line) table."""

from sqlalchemy import BigInteger, Date, Float, String
from sqlalchemy.orm import Mapped, mapped_column

from ..core.database import Base


class DailyQuote(Base):
    __tablename__ = "daily_quotes"

    id: Mapped[int] = mapped_column(BigInteger, primary_key=True, autoincrement=True)
    code: Mapped[str] = mapped_column(String(12), nullable=False, index=True, comment="股票代码")
    trade_date: Mapped[str] = mapped_column(String(10), nullable=False, comment="交易日期 YYYY-MM-DD")
    open: Mapped[float | None] = mapped_column(Float, comment="开盘价")
    high: Mapped[float | None] = mapped_column(Float, comment="最高价")
    low: Mapped[float | None] = mapped_column(Float, comment="最低价")
    close: Mapped[float | None] = mapped_column(Float, comment="收盘价")
    volume: Mapped[float | None] = mapped_column(Float, comment="成交量(股)")
    amount: Mapped[float | None] = mapped_column(Float, comment="成交额(元)")
    amplitude: Mapped[float | None] = mapped_column(Float, comment="振幅(%)")
    pct_change: Mapped[float | None] = mapped_column(Float, comment="涨跌幅(%)")
    change: Mapped[float | None] = mapped_column(Float, comment="涨跌额")
    turnover: Mapped[float | None] = mapped_column(Float, comment="换手率(%)")
