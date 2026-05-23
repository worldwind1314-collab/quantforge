"""Market data query endpoints."""

from fastapi import APIRouter, Depends, Query
from sqlalchemy.orm import Session

from ..core.database import get_db
from ..models.market import DailyQuote
from ..models.stock import Stock

router = APIRouter(prefix="/market", tags=["market-data"])


@router.get("/stocks")
def list_stocks(
    market: str | None = Query(None, description="筛选市场 SH/SZ/BJ"),
    active_only: bool = Query(True, description="仅显示正常交易"),
    db: Session = Depends(get_db),
):
    """获取股票列表。"""
    q = db.query(Stock)
    if market:
        q = q.filter(Stock.market == market.upper())
    if active_only:
        q = q.filter(Stock.is_active == True)
    stocks = q.order_by(Stock.code).all()
    return {
        "total": len(stocks),
        "stocks": [
            {
                "code": s.code,
                "name": s.name,
                "market": s.market,
                "industry": s.industry,
                "area": s.area,
                "list_date": s.list_date,
            }
            for s in stocks
        ],
    }


@router.get("/quotes/{code}")
def get_quotes(
    code: str,
    start_date: str | None = Query(None, description="起始日期 YYYY-MM-DD"),
    end_date: str | None = Query(None, description="结束日期 YYYY-MM-DD"),
    limit: int = Query(250, ge=1, le=2000, description="返回条数上限"),
    db: Session = Depends(get_db),
):
    """获取单只股票的日线行情。"""
    q = db.query(DailyQuote).filter(DailyQuote.code == code)
    if start_date:
        q = q.filter(DailyQuote.trade_date >= start_date)
    if end_date:
        q = q.filter(DailyQuote.trade_date <= end_date)
    rows = q.order_by(DailyQuote.trade_date.desc()).limit(limit).all()
    return {
        "code": code,
        "total": len(rows),
        "quotes": [
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
            for r in rows
        ],
    }


@router.get("/quotes/{code}/latest")
def get_latest_quote(code: str, db: Session = Depends(get_db)):
    """获取单只股票最新行情。"""
    row = (
        db.query(DailyQuote)
        .filter(DailyQuote.code == code)
        .order_by(DailyQuote.trade_date.desc())
        .first()
    )
    if not row:
        return {"code": code, "quote": None, "message": "无数据"}
    return {
        "code": code,
        "quote": {
            "trade_date": row.trade_date,
            "open": row.open,
            "high": row.high,
            "low": row.low,
            "close": row.close,
            "volume": row.volume,
            "amount": row.amount,
            "pct_change": row.pct_change,
            "turnover": row.turnover,
        },
    }
