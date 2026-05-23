"""Stock basic information table."""

from sqlalchemy import Date, String
from sqlalchemy.orm import Mapped, mapped_column

from ..core.database import Base


class Stock(Base):
    __tablename__ = "stocks"

    code: Mapped[str] = mapped_column(String(12), primary_key=True, comment="股票代码")
    name: Mapped[str] = mapped_column(String(20), nullable=False, comment="股票名称")
    market: Mapped[str] = mapped_column(String(4), nullable=False, comment="市场 SH/SZ/BJ")
    industry: Mapped[str | None] = mapped_column(String(50), comment="所属行业")
    area: Mapped[str | None] = mapped_column(String(20), comment="地区")
    list_date: Mapped[str | None] = mapped_column(String(10), comment="上市日期")
    is_active: Mapped[bool] = mapped_column(default=True, comment="是否正常交易")
