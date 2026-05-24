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


@router.post("/sync")
def trigger_sync(
    codes: str | None = Query(None, description="股票代码逗号分隔，不传则同步全部"),
    days: int = Query(730, description="同步天数回溯"),
    db: Session = Depends(get_db),
):
    """触发数据同步：同步指定股票的日线行情。"""
    import logging
    from ..services.data_pipeline import DataPipeline
    from datetime import date, timedelta

    logger = logging.getLogger(__name__)

    if codes:
        code_list = [c.strip() for c in codes.split(",") if c.strip()]
    else:
        code_list = [r[0] for r in db.query(Stock.code).filter(Stock.is_active == True).all()]

    end_date = date.today().strftime("%Y%m%d")
    start_date = (date.today() - timedelta(days=days)).strftime("%Y%m%d")

    logger.info(f"Syncing {len(code_list)} stocks from {start_date} to {end_date}")

    quotes_data = DataPipeline.fetch_daily_quotes(code_list, start_date, end_date)
    quote_count = DataPipeline.save_daily_quotes(quotes_data, db)

    # Check existing daily quotes for these codes
    from ..models.finance import FinancialIndicator
    fi_codes = [r[0] for r in db.query(FinancialIndicator.code).distinct().all()]
    fi_intersect = [c for c in code_list if c in fi_codes]

    return {
        "stocks_requested": len(code_list),
        "stocks_fetched": len(quotes_data),
        "total_quotes_saved": quote_count,
        "date_range": f"{start_date} ~ {end_date}",
        "stocks_with_financials": len(fi_intersect),
    }


@router.get("/debug/akshare/{code}")
def debug_akshare(code: str):
    """Debug: test AKShare stock_zh_a_hist for a single stock."""
    import traceback
    try:
        import akshare as ak
        from datetime import date, timedelta

        end_date = date.today().strftime("%Y%m%d")
        start_date = (date.today() - timedelta(days=90)).strftime("%Y%m%d")

        df = ak.stock_zh_a_hist(symbol=code, period="daily", start_date=start_date, end_date=end_date, adjust="qfq")

        return {
            "code": code,
            "date_range": f"{start_date} ~ {end_date}",
            "success": True,
            "rows": len(df) if df is not None else 0,
            "columns": list(df.columns) if df is not None and not df.empty else [],
            "first_row": df.iloc[0].to_dict() if df is not None and not df.empty else None,
            "last_row": df.iloc[-1].to_dict() if df is not None and not df.empty else None,
        }
    except Exception as e:
        return {
            "code": code,
            "success": False,
            "error": str(e),
            "error_type": type(e).__name__,
            "traceback": traceback.format_exc(),
        }


@router.get("/debug/connectivity")
def debug_connectivity():
    """Test connectivity to various financial data sources from the server."""
    import traceback
    import requests

    results = {}
    session = requests.Session()
    session.headers.update({"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"})

    tests = {
        "eastmoney_quote": "https://push2.eastmoney.com/api/qt/stock/get?secid=1.000001&fields=f43",
        "eastmoney_hist": "https://datacenter-web.eastmoney.com/api/data/v1/get?reportName=RPT_DAILY_BILLBOARD",
        "sina_finance": "https://hq.sinajs.cn/list=sh000001",
        "netease_finance": "https://api.money.126.net/data/feed/0000001",
        "tencent_finance": "https://qt.gtimg.cn/q=sh000001",
        "eastmoney_www": "https://www.eastmoney.com",
        "sina_www": "https://www.sina.com.cn",
        "baidu": "https://www.baidu.com",
    }

    for name, url in tests.items():
        try:
            r = session.get(url, timeout=10)
            results[name] = {"status": r.status_code, "len": len(r.text), "ok": r.ok}
        except Exception as e:
            results[name] = {"status": "error", "error": type(e).__name__, "msg": str(e)[:150]}

    # Test more AKShare alternative functions
    ak_results = {}
    try:
        import akshare as ak
        from datetime import date, timedelta

        # Test stock list (working)
        try:
            df = ak.stock_info_a_code_name()
            ak_results["stock_info_a_code_name"] = {"ok": True, "rows": len(df)}
        except Exception as e:
            ak_results["stock_info_a_code_name"] = {"ok": False, "error": str(e)[:200]}

        # Test index daily — Sina API (working)
        try:
            df = ak.stock_zh_index_daily(symbol="sh000001")
            ak_results["stock_zh_index_daily"] = {"ok": True, "rows": len(df)}
        except Exception as e:
            ak_results["stock_zh_index_daily"] = {"ok": False, "error": str(e)[:200]}

        # Test stock_zh_a_hist with different symbol formats
        for symbol in ["000001", "sz000001"]:
            try:
                df = ak.stock_zh_a_hist(symbol=symbol, period="daily", start_date="20260501", end_date="20260524", adjust="qfq")
                ak_results[f"stock_zh_a_hist_{symbol}"] = {"ok": True, "rows": len(df)}
            except Exception as e:
                ak_results[f"stock_zh_a_hist_{symbol}"] = {"ok": False, "error": type(e).__name__ + ": " + str(e)[:120]}

        # Test Sina-based individual stock hist (may use different endpoint)
        try:
            df = ak.stock_zh_a_daily(symbol="sz000001", start_date="20260501", end_date="20260524", adjust="qfq")
            ak_results["stock_zh_a_daily_sz000001"] = {"ok": True, "rows": len(df)}
        except Exception as e:
            ak_results["stock_zh_a_daily_sz000001"] = {"ok": False, "error": type(e).__name__ + ": " + str(e)[:120]}

        # Test market PE (working)
        try:
            df = ak.stock_market_pe_lg()
            ak_results["stock_market_pe_lg"] = {"ok": True, "rows": len(df) if df is not None else 0}
        except Exception as e:
            ak_results["stock_market_pe_lg"] = {"ok": False, "error": str(e)[:200]}

        # Test spot (batch real-time data)
        try:
            df = ak.stock_zh_a_spot_em()
            ak_results["stock_zh_a_spot_em"] = {"ok": True, "rows": len(df)}
        except Exception as e:
            ak_results["stock_zh_a_spot_em"] = {"ok": False, "error": type(e).__name__ + ": " + str(e)[:120]}

        # Test Sina individual stock quote API directly
        try:
            import requests
            r = requests.get("https://hq.sinajs.cn/list=sh600519,sz000001", headers={"Referer": "https://finance.sina.com.cn"}, timeout=10)
            ak_results["sina_direct_quote"] = {"ok": r.status_code == 200, "len": len(r.text), "status": r.status_code}
        except Exception as e:
            ak_results["sina_direct_quote"] = {"ok": False, "error": str(e)[:120]}

        # Test Tencent individual stock hist API
        try:
            import requests
            r = requests.get("https://web.ifzq.gtimg.cn/appstock/app/fqkline/get?param=sz000001,day,,,10,qfq", timeout=10)
            ak_results["tencent_kline"] = {"ok": r.status_code == 200, "len": len(r.text)}
        except Exception as e:
            ak_results["tencent_kline"] = {"ok": False, "error": str(e)[:120]}

    except ImportError:
        ak_results["import"] = "akshare not installed"

    return {
        "connectivity": results,
        "akshare_alternatives": ak_results,
    }
