"""Financial data models — indicators, factors, ML predictions."""

from datetime import datetime

from sqlalchemy import DateTime, Float, Integer, String, Text, UniqueConstraint, func
from sqlalchemy.orm import Mapped, mapped_column

from ..core.database import Base


class FinancialIndicator(Base):
    """Latest financial analysis indicators per stock (from AKShare)."""

    __tablename__ = "financial_indicators"
    __table_args__ = (UniqueConstraint("code", "report_date", name="uq_fi_code_date"),)

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

    # Valuation (PE/PB/PS computed from per-share data * price during factor calc)
    pe: Mapped[float | None] = mapped_column(Float, comment="市盈率")
    pb: Mapped[float | None] = mapped_column(Float, comment="市净率")
    ps: Mapped[float | None] = mapped_column(Float, comment="市销率")
    bv_per_share: Mapped[float | None] = mapped_column(Float, comment="每股净资产(元)")
    revenue_per_share: Mapped[float | None] = mapped_column(Float, comment="每股营业收入(元)")

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
    __table_args__ = (UniqueConstraint("code", "trade_date", name="uq_fs_code_date"),)

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


class FundFlow(Base):
    """Daily fund flow data per stock (主力资金流向 + 北向资金)."""

    __tablename__ = "fund_flows"
    __table_args__ = (UniqueConstraint("code", "trade_date", name="uq_ff_code_date"),)

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    code: Mapped[str] = mapped_column(String(12), nullable=False, index=True)
    trade_date: Mapped[str] = mapped_column(String(10), nullable=False)

    main_net_inflow: Mapped[float | None] = mapped_column(Float, comment="主力净流入(万元)")
    super_large_net: Mapped[float | None] = mapped_column(Float, comment="超大单净流入(万元)")
    large_net: Mapped[float | None] = mapped_column(Float, comment="大单净流入(万元)")
    medium_net: Mapped[float | None] = mapped_column(Float, comment="中单净流入(万元)")
    small_net: Mapped[float | None] = mapped_column(Float, comment="小单净流入(万元)")
    north_bound_net: Mapped[float | None] = mapped_column(Float, comment="北向资金净流入(万元)")

    updated_at: Mapped[datetime] = mapped_column(DateTime, server_default=func.now(), onupdate=func.now())


class MLPrediction(Base):
    """Daily ML model predictions — expected return ranking."""

    __tablename__ = "ml_predictions"
    __table_args__ = (UniqueConstraint("code", "trade_date", name="uq_mlp_code_date"),)

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    code: Mapped[str] = mapped_column(String(12), nullable=False, index=True)
    trade_date: Mapped[str] = mapped_column(String(10), nullable=False)
    predicted_return: Mapped[float | None] = mapped_column(Float, comment="预测未来N日收益率(%)")
    prediction_rank: Mapped[int | None] = mapped_column(Integer, comment="全市场排名")
    confidence: Mapped[float | None] = mapped_column(Float, comment="置信度(0-1)")
    created_at: Mapped[datetime] = mapped_column(DateTime, server_default=func.now())


class MarginTrading(Base):
    """Daily margin trading & short selling data (融资融券)."""

    __tablename__ = "margin_trading"
    __table_args__ = (UniqueConstraint("code", "trade_date", name="uq_mt_code_date"),)

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    code: Mapped[str] = mapped_column(String(12), nullable=False, index=True)
    trade_date: Mapped[str] = mapped_column(String(10), nullable=False)

    margin_balance: Mapped[float | None] = mapped_column(Float, comment="融资余额(元)")
    margin_buy: Mapped[float | None] = mapped_column(Float, comment="融资买入额(元)")
    margin_repay: Mapped[float | None] = mapped_column(Float, comment="融资偿还额(元)")
    short_balance: Mapped[float | None] = mapped_column(Float, comment="融券余量(股)")
    short_sell: Mapped[float | None] = mapped_column(Float, comment="融券卖出量(股)")
    short_repay: Mapped[float | None] = mapped_column(Float, comment="融券偿还量(股)")
    margin_ratio: Mapped[float | None] = mapped_column(Float, comment="融资买入占比(%)")

    updated_at: Mapped[datetime] = mapped_column(DateTime, server_default=func.now(), onupdate=func.now())


class ShareholderCount(Base):
    """Periodic shareholder count (股东户数变化 — 筹码集中度)."""

    __tablename__ = "shareholder_counts"
    __table_args__ = (UniqueConstraint("code", "end_date", name="uq_sh_code_date"),)

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    code: Mapped[str] = mapped_column(String(12), nullable=False, index=True)
    end_date: Mapped[str] = mapped_column(String(10), nullable=False, comment="统计截止日期")

    holder_count: Mapped[int | None] = mapped_column(Integer, comment="股东户数")
    avg_holding: Mapped[float | None] = mapped_column(Float, comment="户均持股(股)")
    holder_change_pct: Mapped[float | None] = mapped_column(Float, comment="股东户数环比变化(%)")
    top10_hold_pct: Mapped[float | None] = mapped_column(Float, comment="前十大股东持股比例(%)")

    updated_at: Mapped[datetime] = mapped_column(DateTime, server_default=func.now(), onupdate=func.now())


class DragonTiger(Base):
    """Dragon-Tiger list (龙虎榜 — 每日龙虎榜上榜股票)."""

    __tablename__ = "dragon_tiger"
    __table_args__ = (UniqueConstraint("code", "trade_date", name="uq_dt_code_date"),)

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    code: Mapped[str] = mapped_column(String(12), nullable=False, index=True)
    trade_date: Mapped[str] = mapped_column(String(10), nullable=False)

    reason: Mapped[str | None] = mapped_column(String(50), comment="上榜原因")
    buy_amount: Mapped[float | None] = mapped_column(Float, comment="龙虎榜买入总额(元)")
    sell_amount: Mapped[float | None] = mapped_column(Float, comment="龙虎榜卖出总额(元)")
    net_amount: Mapped[float | None] = mapped_column(Float, comment="净买入额(元)")
    institution_buy: Mapped[float | None] = mapped_column(Float, comment="机构席位买入(元)")
    institution_sell: Mapped[float | None] = mapped_column(Float, comment="机构席位卖出(元)")

    created_at: Mapped[datetime] = mapped_column(DateTime, server_default=func.now())


class LockupRelease(Base):
    """Upcoming lockup release schedule (限售解禁)."""

    __tablename__ = "lockup_releases"
    __table_args__ = (UniqueConstraint("code", "release_date", name="uq_lr_code_date"),)

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    code: Mapped[str] = mapped_column(String(12), nullable=False, index=True)
    release_date: Mapped[str] = mapped_column(String(10), nullable=False)

    release_shares: Mapped[float | None] = mapped_column(Float, comment="解禁数量(股)")
    release_ratio: Mapped[float | None] = mapped_column(Float, comment="解禁占总股本比例(%)")
    release_market_value: Mapped[float | None] = mapped_column(Float, comment="解禁市值(元)")

    created_at: Mapped[datetime] = mapped_column(DateTime, server_default=func.now())
