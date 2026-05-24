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
    def _code_to_sina_symbol(code: str) -> str:
        """Convert pure code to Sina symbol format: sz000001 or sh600519."""
        if code.startswith(("6", "5")):
            return "sh" + code
        return "sz" + code

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
        errors = []
        for i, code in enumerate(codes):
            df = None
            em_error = None
            # Try East Money API first (more complete data)
            try:
                df = ak.stock_zh_a_hist(
                    symbol=code, period=period, start_date=start_date, end_date=end_date, adjust="qfq",
                )
                if df is not None and not df.empty:
                    # Normalize East Money column names
                    df = df.rename(columns={
                        "日期": "date", "开盘": "open", "收盘": "close",
                        "最高": "high", "最低": "low", "成交量": "volume",
                        "成交额": "amount", "振幅": "amplitude",
                        "涨跌幅": "pct_change", "涨跌额": "change", "换手率": "turnover",
                    })
            except Exception as e:
                em_error = e

            # Fallback: Sina API via stock_zh_a_daily
            if df is None or df.empty:
                try:
                    sina_symbol = DataPipeline._code_to_sina_symbol(code)
                    df = ak.stock_zh_a_daily(
                        symbol=sina_symbol, start_date=start_date, end_date=end_date, adjust="qfq",
                    )
                    # Sina API returns English column names already
                    if df is not None and not df.empty and "date" not in df.columns:
                        df = df.rename(columns={
                            "日期": "date", "开盘": "open", "收盘": "close",
                            "最高": "high", "最低": "low", "成交量": "volume",
                            "成交额": "amount", "振幅": "amplitude",
                            "涨跌幅": "pct_change", "涨跌额": "change", "换手率": "turnover",
                        })
                except Exception as e2:
                    errors.append(f"{code}: EM={em_error}, Sina={e2}")
                    if i < 3:
                        logger.warning(f"Failed to fetch {code}: EM={em_error}, Sina={e2}")
                    continue

            if df is not None and not df.empty:
                results[code] = df
            if (i + 1) % 50 == 0:
                logger.info(f"Fetched {i + 1}/{len(codes)} stocks")

        if errors:
            logger.warning(f"Failed to fetch {len(errors)}/{len(codes)} stocks. First errors: {errors[:3]}")
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
                    trade_date = _get_col(row, "date", "日期")
                    if not trade_date:
                        continue

                    stmt = text("DELETE FROM daily_quotes WHERE code = :code AND trade_date = :td")
                    db.execute(stmt, {"code": code, "td": str(trade_date).strip()})

                    db.add(
                        DailyQuote(
                            code=code,
                            trade_date=str(trade_date).strip(),
                            open=_safe_float(_get_col(row, "open", "开盘")),
                            high=_safe_float(_get_col(row, "high", "最高")),
                            low=_safe_float(_get_col(row, "low", "最低")),
                            close=_safe_float(_get_col(row, "close", "收盘")),
                            volume=_safe_float(_get_col(row, "volume", "成交量")),
                            amount=_safe_float(_get_col(row, "amount", "成交额")),
                            amplitude=_safe_float(_get_col(row, "amplitude", "振幅")),
                            pct_change=_safe_float(_get_col(row, "pct_change", "涨跌幅")),
                            change=_safe_float(_get_col(row, "change", "涨跌额")),
                            turnover=_safe_float(_get_col(row, "turnover", "换手率")),
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

    # Mapping from indicator name in stock_financial_abstract to our field
    _FINANCIAL_INDICATOR_MAP = {
        "roe": ["净资产收益率(ROE)", "净资产收益率"],
        "eps": ["基本每股收益", "摊薄每股收益_最新股数"],
        "gross_margin": ["毛利率"],
        "net_margin": ["销售净利率"],
        "revenue_growth": ["营业总收入增长率"],
        "profit_growth": ["归属母公司净利润增长率", "净利润增长率"],
        "asset_growth": ["总资产增长率"],
        "debt_ratio": ["资产负债率"],
        "current_ratio": ["流动比率"],
        "asset_turnover": ["总资产周转率"],
        "cf_per_share": ["每股经营现金流", "每股经营活动现金流量净额"],
        "bv_per_share": ["每股净资产"],
        "revenue_per_share": ["每股营业收入", "每股营业总收入"],
    }

    @staticmethod
    def _extract_indicator(df_abs: pd.DataFrame, names: list[str], latest_period: str) -> float | None:
        """Extract a specific indicator value from stock_financial_abstract DataFrame."""
        for name in names:
            row = df_abs[df_abs["指标"] == name]
            if not row.empty:
                val = row.iloc[0].get(latest_period)
                if val is not None:
                    try:
                        f = float(val)
                        return None if pd.isna(f) else f
                    except (ValueError, TypeError):
                        continue
        return None

    @staticmethod
    def fetch_financial_indicators(code: str) -> dict | None:
        """Fetch financial indicators using stock_financial_abstract (Sina source).
        Returns a dict ready for FinancialIndicator creation, or None."""
        try:
            df = ak.stock_financial_abstract(symbol=code)
            if df is None or df.empty:
                return None

            # Period columns are date strings like "20260331"
            date_cols = [c for c in df.columns if c not in ["选项", "指标"] and str(c).isdigit()]
            if not date_cols:
                return None
            latest_period = date_cols[0]  # First column = most recent

            result = {
                "code": code,
                "report_date": latest_period,
                "roe": DataPipeline._extract_indicator(df, DataPipeline._FINANCIAL_INDICATOR_MAP["roe"], latest_period),
                "eps": DataPipeline._extract_indicator(df, DataPipeline._FINANCIAL_INDICATOR_MAP["eps"], latest_period),
                "gross_margin": DataPipeline._extract_indicator(df, DataPipeline._FINANCIAL_INDICATOR_MAP["gross_margin"], latest_period),
                "net_margin": DataPipeline._extract_indicator(df, DataPipeline._FINANCIAL_INDICATOR_MAP["net_margin"], latest_period),
                "revenue_growth": DataPipeline._extract_indicator(df, DataPipeline._FINANCIAL_INDICATOR_MAP["revenue_growth"], latest_period),
                "profit_growth": DataPipeline._extract_indicator(df, DataPipeline._FINANCIAL_INDICATOR_MAP["profit_growth"], latest_period),
                "asset_growth": DataPipeline._extract_indicator(df, DataPipeline._FINANCIAL_INDICATOR_MAP["asset_growth"], latest_period),
                "debt_ratio": DataPipeline._extract_indicator(df, DataPipeline._FINANCIAL_INDICATOR_MAP["debt_ratio"], latest_period),
                "current_ratio": DataPipeline._extract_indicator(df, DataPipeline._FINANCIAL_INDICATOR_MAP["current_ratio"], latest_period),
                "asset_turnover": DataPipeline._extract_indicator(df, DataPipeline._FINANCIAL_INDICATOR_MAP["asset_turnover"], latest_period),
                "cf_per_share": DataPipeline._extract_indicator(df, DataPipeline._FINANCIAL_INDICATOR_MAP["cf_per_share"], latest_period),
                "bv_per_share": DataPipeline._extract_indicator(df, DataPipeline._FINANCIAL_INDICATOR_MAP["bv_per_share"], latest_period),
                "revenue_per_share": DataPipeline._extract_indicator(df, DataPipeline._FINANCIAL_INDICATOR_MAP["revenue_per_share"], latest_period),
            }
            return result
        except Exception as e:
            logger.warning(f"Failed to fetch financial indicators for {code}: {e}")
            return None

    @staticmethod
    def sync_financial_indicators(codes: list[str] | None = None, db: Session | None = None) -> int:
        """Sync latest financial indicators for given stocks using stock_financial_abstract."""
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
                data = DataPipeline.fetch_financial_indicators(code)
                if data is None:
                    continue

                report_date = data["report_date"]

                # Upsert
                existing = (
                    db.query(FinancialIndicator)
                    .filter(FinancialIndicator.code == code, FinancialIndicator.report_date == report_date)
                    .first()
                )
                if existing:
                    existing.roe = data.get("roe")
                    existing.eps = data.get("eps")
                    existing.gross_margin = data.get("gross_margin")
                    existing.net_margin = data.get("net_margin")
                    existing.revenue_growth = data.get("revenue_growth")
                    existing.profit_growth = data.get("profit_growth")
                    existing.asset_growth = data.get("asset_growth")
                    existing.debt_ratio = data.get("debt_ratio")
                    existing.current_ratio = data.get("current_ratio")
                    existing.asset_turnover = data.get("asset_turnover")
                    existing.cf_per_share = data.get("cf_per_share")
                    existing.bv_per_share = data.get("bv_per_share")
                    existing.revenue_per_share = data.get("revenue_per_share")
                else:
                    db.add(FinancialIndicator(
                        code=code, report_date=report_date,
                        roe=data.get("roe"), eps=data.get("eps"),
                        gross_margin=data.get("gross_margin"), net_margin=data.get("net_margin"),
                        revenue_growth=data.get("revenue_growth"), profit_growth=data.get("profit_growth"),
                        asset_growth=data.get("asset_growth"), debt_ratio=data.get("debt_ratio"),
                        current_ratio=data.get("current_ratio"), asset_turnover=data.get("asset_turnover"),
                        cf_per_share=data.get("cf_per_share"),
                        bv_per_share=data.get("bv_per_share"), revenue_per_share=data.get("revenue_per_share"),
                    ))

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

def _get_col(row, en_key: str, cn_key: str):
    """Get value from row using English or Chinese column name."""
    val = row.get(en_key)
    if val is not None:
        return val
    return row.get(cn_key)


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
