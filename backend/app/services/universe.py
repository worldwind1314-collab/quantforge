"""Dynamic stock universe selection — filter, rank, and select tradable stocks."""

import logging
from datetime import date, timedelta

from sqlalchemy import func
from sqlalchemy.orm import Session

from ..core.database import SessionLocal
from ..models.market import DailyQuote
from ..models.stock import Stock

logger = logging.getLogger(__name__)

# CSI 300 approximate constituent list (top 300 by market cap, frequently updated)
# This is a fallback when live API fails; actual constituents fetched via AKShare
_CSI300_FALLBACK = None  # loaded lazily


def get_active_stocks(
    db: Session,
    min_list_days: int = 60,
    exclude_st: bool = True,
    markets: list[str] | None = None,
) -> list[str]:
    """Return all actively trading stock codes with basic filters.

    Args:
        min_list_days: Exclude stocks listed fewer than N days ago (avoid IPO volatility)
        exclude_st: Exclude ST (special treatment) stocks
        markets: Limit to specific markets, e.g. ['SH', 'SZ']
    """
    q = db.query(Stock.code).filter(Stock.is_active == True)

    if markets:
        q = q.filter(Stock.market.in_(markets))

    if min_list_days > 0:
        cutoff = (date.today() - timedelta(days=min_list_days)).isoformat()
        q = q.filter(Stock.list_date <= cutoff)

    # ST stocks have names containing *ST or ST
    if exclude_st:
        q = q.filter(
            ~Stock.name.ilike("%ST%"),
            ~Stock.name.ilike("%退%"),
        )

    return [r[0] for r in q.order_by(Stock.code).all()]


def get_liquid_stocks(
    db: Session,
    top_n: int = 300,
    lookback_days: int = 20,
) -> list[str]:
    """Return top N most liquid stocks by average daily turnover (成交额).

    Uses recent trading data to rank by liquidity — reliable for backtesting.
    """
    start = (date.today() - timedelta(days=lookback_days)).isoformat()

    rows = (
        db.query(
            DailyQuote.code,
            func.avg(DailyQuote.amount).label("avg_amount"),
        )
        .filter(
            DailyQuote.trade_date >= start,
            DailyQuote.amount > 0,
        )
        .group_by(DailyQuote.code)
        .order_by(func.avg(DailyQuote.amount).desc())
        .limit(top_n)
        .all()
    )

    return [r[0] for r in rows]


def get_universe(
    db: Session,
    name: str = "liquid_300",
    top_n: int = 300,
) -> list[str]:
    """Get a named stock universe. Supported names:

    - 'all': All active non-ST non-new-listing stocks
    - 'liquid_300': Top 300 by liquidity (default, good for backtesting)
    - 'liquid_500': Top 500 by liquidity
    - 'liquid_100': Top 100 by liquidity
    - 'csi300': CSI 300 constituents (fetched live or fallback)
    """
    if name == "all":
        return get_active_stocks(db)

    if name == "csi300":
        from .data_pipeline import DataPipeline

        codes = DataPipeline.sync_index_constituents("000300")
        if codes:
            # Filter to only stocks we have data for
            active = set(get_active_stocks(db))
            return [c for c in codes if c in active]
        # Fallback: top 300 liquid (close approximation)
        logger.warning("CSI300 constituents unavailable, falling back to liquid_300")
        return get_liquid_stocks(db, top_n=300)

    if name.startswith("liquid_"):
        n = int(name.split("_")[1])
        return get_liquid_stocks(db, top_n=n)

    # Default
    return get_liquid_stocks(db, top_n=top_n)


def get_multi_sector_stocks(db: Session, per_sector: int = 5) -> list[str]:
    """Select a diverse set of stocks across major industries.

    Returns up to `per_sector` stocks from each major industry,
    prioritizing by liquidity.
    """
    active_codes = get_active_stocks(db)
    if not active_codes:
        return []

    sectors = db.query(Stock.industry, func.count(Stock.code))\
        .filter(Stock.code.in_(active_codes), Stock.industry.isnot(None))\
        .group_by(Stock.industry)\
        .order_by(func.count(Stock.code).desc())\
        .limit(12).all()

    result = []
    seen = set()
    for industry, _ in sectors:
        start = (date.today() - timedelta(days=20)).isoformat()
        rows = (
            db.query(DailyQuote.code)
            .join(Stock, Stock.code == DailyQuote.code)
            .filter(
                Stock.industry == industry,
                DailyQuote.trade_date >= start,
                DailyQuote.amount > 0,
            )
            .group_by(DailyQuote.code)
            .order_by(func.avg(DailyQuote.amount).desc())
            .limit(per_sector)
            .all()
        )
        for r in rows:
            if r[0] not in seen:
                result.append(r[0])
                seen.add(r[0])

    return result
