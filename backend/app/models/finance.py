"""Financial data models — indicators, factors, ML predictions."""

from datetime import datetime

from sqlalchemy import DateTime, Float, Integer, String, func
from sqlalchemy.orm import Mapped, mapped_column

from ..core.database import Base


class FinancialIndicator(Base):
    """Latest financial analysis indicators per stock (from AKShare)."""

    __tablename__ = "financial_indicators"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    code: Mapped[str] = mapped_column(String(12), nullable=False, index=True)
    report_date: Mapped[str] = mapped_column(String(10), nullable=False, comment="报告期 YYYY-MM-DD")

    # Core profitability
    roe: Mapped[float | None] = mapped_column(Float, comment="净资产收益率(%)")
    eps: Mapped[float | None] = mapped_column(Float, comment="每股收益(元)")
    gross_margin: Mapped[float | None] = mapped_column(Float, comment="销售毛利率(%)")
    net_margin: Mapped[float | None] = mapped_column(Float, comment="销售净利率(%)")

    # Growth
    revenue_growth: Mapped[float | None] = mapped_column(Float, comment="主营业务收入增长率(%)")
    profit_growth: Mapped[float | None] = mapped_column(Float, comment="净利润增长率(%)")
    asset_growth: Mapped[float | None] = mapped_column(Float, comment="总资产增长率(%)")

    # Valuation
    pe: Mapped[float | None] = mapped_column(Float, comment="市盈率")
    pb: Mapped[float | None] = mapped_column(Float, comment="市净率")
    ps: Mapped[float | None] = mapped_column(Float, comment="市销率")

    # Asset quality
    debt_ratio: Mapped[float | None] = mapped_column(Float, comment="资产负债率(%)")
    current_ratio: Mapped[float | None] = mapped_column(Float, comment="流动比率")
    asset_turnover: Mapped[float | None] = mapped_column(Float, comment="总资产周转率(次)")

    # Cash flow
    cf_per_share: Mapped[float | None] = mapped_column(Float, comment="每股经营性现金流(元)")

    updated_at: Mapped[datetime] = mapped_column(DateTime, server_default=func.now(), onupdate=func.now())


class FactorScore(Base):
    """Daily multi-factor scores for each stock."""

    __tablename__ = "factor_scores"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    code: Mapped[str] = mapped_column(String(12), nullable=False, index=True)
    trade_date: Mapped[str] = mapped_column(String(10), nullable=False)

    # Factor sub-scores (z-score normalized)
    value_score: Mapped[float | None] = mapped_column(Float, comment="价值因子得分")
    quality_score: Mapped[float | None] = mapped_column(Float, comment="质量因子得分")
    momentum_score: Mapped[float | None] = mapped_column(Float, comment="动量因子得分")
    volatility_score: Mapped[float | None] = mapped_column(Float, comment="波动率因子得分")

    # Composite
    composite_score: Mapped[float | None] = mapped_column(Float, comment="综合因子得分")


class MLPrediction(Base):
    """Daily ML model predictions — expected return ranking."""

    __tablename__ = "ml_predictions"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    code: Mapped[str] = mapped_column(String(12), nullable=False, index=True)
    trade_date: Mapped[str] = mapped_column(String(10), nullable=False)
    predicted_return: Mapped[float | None] = mapped_column(Float, comment="预测未来N日收益率(%)")
    prediction_rank: Mapped[int | None] = mapped_column(Integer, comment="全市场排名")
    confidence: Mapped[float | None] = mapped_column(Float, comment="置信度(0-1)")
    created_at: Mapped[datetime] = mapped_column(DateTime, server_default=func.now())
