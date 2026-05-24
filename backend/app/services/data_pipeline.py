"""AKShare data pipeline — pull A-share market data into PostgreSQL."""

import logging
from datetime import date, timedelta

import akshare as ak
import pandas as pd
from sqlalchemy import text
from sqlalchemy.orm import Session

from ..core.database import engine, SessionLocal
from ..models.finance import FinancialIndicator
from ..models.market import DailyQuote
from ..models.stock import Stock

logger = logging.getLogger(__name__)


class DataPipeline:
    """Sync A-share stock list, daily quotes, and financial data via AKShare."""

    # ── Stock list ─────────────────────────────────────────────────

    @staticmethod
    def fetch_stock_list() -> pd.DataFrame:
        df = ak.stock_info_a_code_name()
        df.columns = [c.strip() for c in df.columns]
        logger.info(f"Fetched {len(df)} stocks from AKShare")
        return df

    @staticmethod
    def sync_stock_list(db: Session | None = None) -> int:
        if db is None:
            db = SessionLocal()
            close_db = True
        else:
            close_db = False

        try:
            df = DataPipeline.fetch_stock_list()
            count = 0
            for _, row in df.iterrows():
                code = str(row.get("code", "")).strip()
                name = str(row.get("name", "")).strip()
                if not code or not name:
                    continue
                market = "SH" if code.startswith("6") else "SZ"
                if code.startswith(("8", "4")):
                    market = "BJ"

                existing = db.get(Stock, code)
                if existing:
                    existing.name = name
                    existing.market = market
                else:
                    db.add(Stock(code=code, name=name, market=market))
                count += 1

            db.commit()
            logger.info(f"Synced {count} stocks to DB")
            return count
        finally:
            if close_db:
                db.close()

    # ── Daily quotes ───────────────────────────────────────────────

    @staticmethod
    def fetch_daily_quotes(
        codes: list[str] | None = None,
        start_date: str | None = None,
        end_date: str | None = None,
        period: str = "daily",
    ) -> dict[str, pd.DataFrame]:
        if start_date is None:
            start_date = (date.today() - timedelta(days=730)).strftime("%Y%m%d")
        if end_date is None:
            end_date = date.today().strftime("%Y%m%d")

        if codes is None:
            with SessionLocal() as db:
                codes = [r[0] for r in db.query(Stock.code).filter(Stock.is_active == True).all()]

        results = {}
        for i, code in enumerate(codes):
            try:
                df = ak.stock_zh_a_hist(
                    symbol=code, period=period, start_date=start_date, end_date=end_date, adjust="qfq",
                )
                if not df.empty:
                    results[code] = df
                if (i + 1) % 50 == 0:
                    logger.info(f"Fetched {i + 1}/{len(codes)} stocks")
            except Exception:
                logger.debug(f"Failed to fetch {code}, skipping")

        logger.info(f"Fetched daily quotes for {len(results)}/{len(codes)} stocks")
        return results

    @staticmethod
    def save_daily_quotes(data: dict[str, pd.DataFrame], db: Session | None = None) -> int:
        if db is None:
            db = SessionLocal()
            close_db = True
        else:
            close_db = False

        try:
            total = 0
            for code, df in data.items():
                for _, row in df.iterrows():
                    trade_date = str(row.get("日期", "")).strip()
                    if not trade_date:
                        continue

                    stmt = text("DELETE FROM daily_quotes WHERE code = :code AND trade_date = :td")
                    db.execute(stmt, {"code": code, "td": trade_date})

                    db.add(
                        DailyQuote(
                            code=code,
                            trade_date=trade_date,
                            open=_safe_float(row.get("开盘")),
                            high=_safe_float(row.get("最高")),
                            low=_safe_float(row.get("最低")),
                            close=_safe_float(row.get("收盘")),
                            volume=_safe_float(row.get("成交量")),
                            amount=_safe_float(row.get("成交额")),
                            amplitude=_safe_float(row.get("振幅")),
                            pct_change=_safe_float(row.get("涨跌幅")),
                            change=_safe_float(row.get("涨跌额")),
                            turnover=_safe_float(row.get("换手率")),
                        )
                    )
                    total += 1

                db.commit()
            logger.info(f"Saved {total} daily quote rows to DB")
            return total
        finally:
            if close_db:
                db.close()

    # ── Financial indicators ───────────────────────────────────────

    @staticmethod
    def fetch_financial_indicators(code: str) -> pd.DataFrame | None:
        """Fetch financial analysis indicators for a single stock."""
        try:
            df = ak.stock_financial_analysis_indicator(symbol=code, start_year="2023")
            return df
        except Exception:
            logger.debug(f"Failed to fetch financial indicators for {code}")
            return None

    @staticmethod
    def sync_financial_indicators(codes: list[str] | None = None, db: Session | None = None) -> int:
        """Sync latest financial indicators for given stocks."""
        if db is None:
            db = SessionLocal()
            close_db = True
        else:
            close_db = False

        if codes is None:
            codes = [r[0] for r in db.query(Stock.code).filter(Stock.is_active == True).all()]

        try:
            count = 0
            for i, code in enumerate(codes):
                df = DataPipeline.fetch_financial_indicators(code)
                if df is None or df.empty:
                    continue

                # Take the most recent report
                latest = df.iloc[0]
                report_date = str(latest.get("日期", "")).strip()
                if not report_date:
                    continue

                # Upsert
                existing = (
                    db.query(FinancialIndicator)
                    .filter(FinancialIndicator.code == code, FinancialIndicator.report_date == report_date)
                    .first()
                )
                if existing:
                    _update_financial_indicator(existing, latest)
                else:
                    db.add(_create_financial_indicator(code, report_date, latest))

                count += 1
                if (i + 1) % 100 == 0:
                    db.commit()
                    logger.info(f"Financial sync: {i + 1}/{len(codes)}")

            db.commit()
            logger.info(f"Synced financial indicators for {count} stocks")
            return count
        finally:
            if close_db:
                db.close()

    # ── Full sync ──────────────────────────────────────────────────

    @staticmethod
    def full_sync(start_date: str | None = None, end_date: str | None = None) -> dict:
        db = SessionLocal()
        try:
            stock_count = DataPipeline.sync_stock_list(db)
            codes = [r[0] for r in db.query(Stock.code).all()]
            quotes_data = DataPipeline.fetch_daily_quotes(codes, start_date, end_date)
            quote_count = DataPipeline.save_daily_quotes(quotes_data, db)
            return {"stocks_synced": stock_count, "quotes_saved": quote_count}
        finally:
            db.close()

    @staticmethod
    def incremental_sync(lookback_days: int = 7) -> dict:
        from datetime import date, timedelta

        db = SessionLocal()
        try:
            stock_count = DataPipeline.sync_stock_list(db)
            codes = [r[0] for r in db.query(Stock.code).all()]

            end_date = date.today().strftime("%Y%m%d")
            start_date = (date.today() - timedelta(days=lookback_days + 3)).strftime("%Y%m%d")

            quotes_data = DataPipeline.fetch_daily_quotes(codes, start_date, end_date)
            quote_count = DataPipeline.save_daily_quotes(quotes_data, db)
            fin_count = DataPipeline.sync_financial_indicators(codes, db)
            return {
                "stocks_synced": stock_count,
                "quotes_saved": quote_count,
                "financial_synced": fin_count,
                "stocks_with_data": len(quotes_data),
            }
        finally:
            db.close()


# ── Helpers ────────────────────────────────────────────────────────

def _safe_float(val) -> float | None:
    if val is None:
        return None
    try:
        f = float(val)
        return None if pd.isna(f) else f
    except (ValueError, TypeError):
        return None


def _create_financial_indicator(code: str, report_date: str, row: pd.Series) -> FinancialIndicator:
    return FinancialIndicator(
        code=code,
        report_date=report_date,
        roe=_safe_float(row.get("净资产收益率(%)")),
        eps=_safe_float(row.get("摊薄每股收益(元)")),
        gross_margin=_safe_float(row.get("销售毛利率(%)")),
        net_margin=_safe_float(row.get("销售净利率(%)")),
        revenue_growth=_safe_float(row.get("主营业务收入增长率(%)")),
        profit_growth=_safe_float(row.get("净利润增长率(%)")),
        asset_growth=_safe_float(row.get("总资产增长率(%)")),
        debt_ratio=_safe_float(row.get("资产负债率(%)")),
        current_ratio=_safe_float(row.get("流动比率")),
        asset_turnover=_safe_float(row.get("总资产周转率(次)")),
        cf_per_share=_safe_float(row.get("每股经营性现金流(元)")),
    )


def _update_financial_indicator(fi: FinancialIndicator, row: pd.Series):
    fi.roe = _safe_float(row.get("净资产收益率(%)"))
    fi.eps = _safe_float(row.get("摊薄每股收益(元)"))
    fi.gross_margin = _safe_float(row.get("销售毛利率(%)"))
    fi.net_margin = _safe_float(row.get("销售净利率(%)"))
    fi.revenue_growth = _safe_float(row.get("主营业务收入增长率(%)"))
    fi.profit_growth = _safe_float(row.get("净利润增长率(%)"))
    fi.asset_growth = _safe_float(row.get("总资产增长率(%)"))
    fi.debt_ratio = _safe_float(row.get("资产负债率(%)"))
    fi.current_ratio = _safe_float(row.get("流动比率"))
    fi.asset_turnover = _safe_float(row.get("总资产周转率(次)"))
    fi.cf_per_share = _safe_float(row.get("每股经营性现金流(元)"))
