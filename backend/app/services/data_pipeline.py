"""AKShare data pipeline — pull A-share market data into PostgreSQL."""

import logging
from datetime import date, timedelta

import akshare as ak
import pandas as pd
from sqlalchemy import text
from sqlalchemy.orm import Session

from ..core.database import engine, SessionLocal
from ..models.market import DailyQuote
from ..models.stock import Stock

logger = logging.getLogger(__name__)


class DataPipeline:
    """Sync A-share stock list and daily quotes via AKShare."""

    # ── Stock list ─────────────────────────────────────────────────

    @staticmethod
    def fetch_stock_list() -> pd.DataFrame:
        """Fetch all A-share stocks from AKShare. Returns DataFrame."""
        df = ak.stock_info_a_code_name()
        df.columns = [c.strip() for c in df.columns]
        logger.info(f"Fetched {len(df)} stocks from AKShare")
        return df

    @staticmethod
    def sync_stock_list(db: Session | None = None) -> int:
        """Insert or update stock basic info. Returns count of stocks synced."""
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
        """Fetch daily K-line data for given stocks.

        Args:
            codes: List of stock codes. If None, fetches all stocks from DB.
            start_date: Start date YYYYMMDD. Defaults to 2 years ago.
            end_date: End date YYYYMMDD. Defaults to today.
            period: 'daily', 'weekly', or 'monthly'.

        Returns:
            Dict mapping stock code → DataFrame.
        """
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
                    symbol=code,
                    period=period,
                    start_date=start_date,
                    end_date=end_date,
                    adjust="qfq",  # 前复权
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
    def save_daily_quotes(
        data: dict[str, pd.DataFrame], db: Session | None = None
    ) -> int:
        """Upsert daily quotes into DB. Returns total rows saved."""
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

                    # Upsert: delete existing then insert
                    stmt = text(
                        "DELETE FROM daily_quotes WHERE code = :code AND trade_date = :td"
                    )
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

    # ── Full sync ──────────────────────────────────────────────────

    @staticmethod
    def full_sync(start_date: str | None = None, end_date: str | None = None) -> dict:
        """One-shot: sync stock list then daily quotes. Returns summary dict."""
        db = SessionLocal()
        try:
            stock_count = DataPipeline.sync_stock_list(db)
            codes = [r[0] for r in db.query(Stock.code).all()]
            quotes_data = DataPipeline.fetch_daily_quotes(codes, start_date, end_date)
            quote_count = DataPipeline.save_daily_quotes(quotes_data, db)
            return {"stocks_synced": stock_count, "quotes_saved": quote_count}
        finally:
            db.close()

    # ── Incremental sync ───────────────────────────────────────────

    @staticmethod
    def incremental_sync(lookback_days: int = 7) -> dict:
        """Daily-use: sync stock list + fetch only the last N trading days.

        Much faster than full_sync — suitable for cron daily runs.
        """
        from datetime import date, timedelta

        db = SessionLocal()
        try:
            stock_count = DataPipeline.sync_stock_list(db)
            codes = [r[0] for r in db.query(Stock.code).all()]

            end_date = date.today().strftime("%Y%m%d")
            start_date = (date.today() - timedelta(days=lookback_days + 3)).strftime("%Y%m%d")

            quotes_data = DataPipeline.fetch_daily_quotes(codes, start_date, end_date)
            quote_count = DataPipeline.save_daily_quotes(quotes_data, db)
            return {
                "stocks_synced": stock_count,
                "quotes_saved": quote_count,
                "stocks_with_data": len(quotes_data),
            }
        finally:
            db.close()


def _safe_float(val) -> float | None:
    """Convert value to float, return None on failure."""
    if val is None:
        return None
    try:
        f = float(val)
        return None if pd.isna(f) else f
    except (ValueError, TypeError):
        return None
