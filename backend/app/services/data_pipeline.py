"""AKShare data pipeline — pull A-share market data into PostgreSQL."""

import logging
from datetime import date, timedelta

import akshare as ak
import pandas as pd
from sqlalchemy import func, text
from sqlalchemy.orm import Session

from ..core.database import engine, SessionLocal
from ..models.finance import FinancialIndicator, FundFlow, MarginTrading, ShareholderCount, DragonTiger, LockupRelease
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
            st_count = 0
            for _, row in df.iterrows():
                code = str(row.get("code", "")).strip()
                name = str(row.get("name", "")).strip()
                if not code or not name:
                    continue
                market = "SH" if code.startswith("6") else "SZ"
                if code.startswith(("8", "4")):
                    market = "BJ"

                # Extract list_date from AKShare (column name varies by version)
                list_date = None
                for col_name in ("list_date", "上市日期", "listing_date", "ipo_date", "首发上市日期"):
                    raw = row.get(col_name)
                    if raw is not None and str(raw).strip():
                        list_date = str(raw).strip()[:10]  # normalize to YYYY-MM-DD
                        break

                # Detect ST / delisted stocks
                is_active = True
                if "ST" in name or "退" in name or "PT" in name:
                    is_active = False
                    st_count += 1

                existing = db.get(Stock, code)
                if existing:
                    existing.name = name
                    existing.market = market
                    existing.is_active = is_active
                    if list_date and not existing.list_date:
                        existing.list_date = list_date
                else:
                    db.add(Stock(code=code, name=name, market=market, is_active=is_active, list_date=list_date))
                count += 1

            db.commit()
            logger.info(f"Synced {count} stocks to DB (ST/delisted: {st_count})")
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
            bulk_buffer: list[DailyQuote] = []
            for code, df in data.items():
                # Batch delete for this code-date range
                all_dates = [str(row.get("date", row.get("日期", ""))).strip() for _, row in df.iterrows()]
                all_dates = [d for d in all_dates if d]
                if all_dates:
                    stmt = text("DELETE FROM daily_quotes WHERE code = :code AND trade_date = ANY(:dates)")
                    db.execute(stmt, {"code": code, "dates": all_dates})

                for _, row in df.iterrows():
                    trade_date = _get_col(row, "date", "日期")
                    if not trade_date:
                        continue

                    bulk_buffer.append(
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

                    # Flush buffer every 500 rows
                    if len(bulk_buffer) >= 500:
                        db.add_all(bulk_buffer)
                        db.commit()
                        bulk_buffer.clear()

            # Flush remaining
            if bulk_buffer:
                db.add_all(bulk_buffer)
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
    def _extract_indicator_fuzzy(df_abs: pd.DataFrame, names: list[str], latest_period: str) -> float | None:
        """Extract indicator with exact match first, then substring match as fallback."""
        # Try exact match
        result = DataPipeline._extract_indicator(df_abs, names, latest_period)
        if result is not None:
            return result

        # Fallback: substring match against all indicator names
        indicators = df_abs["指标"].tolist()
        for indicator in indicators:
            for name in names:
                if name in str(indicator) or str(indicator) in name:
                    row = df_abs[df_abs["指标"] == indicator]
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
    def fetch_financial_indicators_em(code: str) -> dict | None:
        """Fetch financial indicators using East Money API (more reliable, structured data).

        Uses stock_financial_analysis_indicator which returns columns like
        '净资产收益率(%)', '基本每股收益(元)', etc. directly — no Chinese string parsing needed.
        """
        try:
            df = ak.stock_financial_analysis_indicator(symbol=code, start_year="2024")
            if df is None or df.empty:
                return None

            latest = df.iloc[0]
            report_date = str(latest.get("报告期", latest.get("end_date", "")))

            # Only proceed if report_date looks valid (8 digits or YYYY-MM-DD format)
            if not report_date or len(report_date) < 6:
                return None

            result = {
                "code": code,
                "report_date": report_date.replace("-", ""),
                "roe": _safe_float(latest.get("净资产收益率(%)", latest.get("roe", None))),
                "eps": _safe_float(latest.get("基本每股收益(元)", latest.get("eps", None))),
                "gross_margin": _safe_float(latest.get("销售毛利率(%)", None)),
                "net_margin": _safe_float(latest.get("销售净利率(%)", None)),
                "revenue_growth": _safe_float(latest.get("营业收入同比增长率(%)", latest.get("营业总收入增长率(%)", None))),
                "profit_growth": _safe_float(latest.get("净利润同比增长率(%)", latest.get("归属母公司净利润增长率(%)", None))),
                "asset_growth": _safe_float(latest.get("总资产增长率(%)", None)),
                "debt_ratio": _safe_float(latest.get("资产负债率(%)", None)),
                "current_ratio": _safe_float(latest.get("流动比率", None)),
                "asset_turnover": _safe_float(latest.get("总资产周转率(次)", None)),
                "cf_per_share": _safe_float(latest.get("每股经营性现金流(元)", latest.get("每股经营现金流(元)", None))),
                "bv_per_share": _safe_float(latest.get("每股净资产(元)", None)),
                "revenue_per_share": _safe_float(latest.get("每股营业收入(元)", latest.get("每股营业总收入(元)", None))),
                # Valuation (PE/PB/PS) — populated when closing price is available
                "pe": _safe_float(latest.get("市盈率", latest.get("pe", latest.get("PE", None)))),
                "pb": _safe_float(latest.get("市净率", latest.get("pb", latest.get("PB", None)))),
                "ps": _safe_float(latest.get("市销率", latest.get("ps", latest.get("PS", None)))),
            }
            # Return None if all indicator values are None
            financial_keys = ["roe", "eps", "gross_margin", "net_margin", "revenue_growth",
                              "profit_growth", "debt_ratio", "current_ratio", "pe", "pb"]
            if all(result.get(k) is None for k in financial_keys):
                return None
            return result
        except Exception as e:
            logger.debug(f"EM financial indicators failed for {code}: {e}")
            return None

    @staticmethod
    def fetch_financial_indicators(code: str) -> dict | None:
        """Fetch financial indicators with fallback: East Money -> Sina -> None.

        Returns a dict ready for FinancialIndicator creation, or None.
        """
        # Try East Money first (more reliable, structured data)
        result = DataPipeline.fetch_financial_indicators_em(code)
        if result is not None:
            return result

        # Fallback to Sina stock_financial_abstract
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
                "roe": DataPipeline._extract_indicator_fuzzy(df, DataPipeline._FINANCIAL_INDICATOR_MAP["roe"], latest_period),
                "eps": DataPipeline._extract_indicator_fuzzy(df, DataPipeline._FINANCIAL_INDICATOR_MAP["eps"], latest_period),
                "gross_margin": DataPipeline._extract_indicator_fuzzy(df, DataPipeline._FINANCIAL_INDICATOR_MAP["gross_margin"], latest_period),
                "net_margin": DataPipeline._extract_indicator_fuzzy(df, DataPipeline._FINANCIAL_INDICATOR_MAP["net_margin"], latest_period),
                "revenue_growth": DataPipeline._extract_indicator_fuzzy(df, DataPipeline._FINANCIAL_INDICATOR_MAP["revenue_growth"], latest_period),
                "profit_growth": DataPipeline._extract_indicator_fuzzy(df, DataPipeline._FINANCIAL_INDICATOR_MAP["profit_growth"], latest_period),
                "asset_growth": DataPipeline._extract_indicator_fuzzy(df, DataPipeline._FINANCIAL_INDICATOR_MAP["asset_growth"], latest_period),
                "debt_ratio": DataPipeline._extract_indicator_fuzzy(df, DataPipeline._FINANCIAL_INDICATOR_MAP["debt_ratio"], latest_period),
                "current_ratio": DataPipeline._extract_indicator_fuzzy(df, DataPipeline._FINANCIAL_INDICATOR_MAP["current_ratio"], latest_period),
                "asset_turnover": DataPipeline._extract_indicator_fuzzy(df, DataPipeline._FINANCIAL_INDICATOR_MAP["asset_turnover"], latest_period),
                "cf_per_share": DataPipeline._extract_indicator_fuzzy(df, DataPipeline._FINANCIAL_INDICATOR_MAP["cf_per_share"], latest_period),
                "bv_per_share": DataPipeline._extract_indicator_fuzzy(df, DataPipeline._FINANCIAL_INDICATOR_MAP["bv_per_share"], latest_period),
                "revenue_per_share": DataPipeline._extract_indicator_fuzzy(df, DataPipeline._FINANCIAL_INDICATOR_MAP["revenue_per_share"], latest_period),
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
                    existing.pe = data.get("pe")
                    existing.pb = data.get("pb")
                    existing.ps = data.get("ps")
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
                        pe=data.get("pe"), pb=data.get("pb"), ps=data.get("ps"),
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

            # Catch-up: detect last sync date and extend lookback if needed
            last_date = db.query(func.max(DailyQuote.trade_date)).scalar()
            if last_date:
                try:
                    last_dt = date.fromisoformat(last_date)
                    gap_days = (date.today() - last_dt).days
                    if gap_days > lookback_days + 3:
                        logger.info(f"Data gap detected: last sync {last_date} ({gap_days} days ago). Extending lookback.")
                        lookback_days = gap_days + 5
                except (ValueError, TypeError):
                    pass

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


    # ── Fund flow (资金流向) ────────────────────────────────────────

    @staticmethod
    def fetch_fund_flow(code: str, trade_date: str | None = None) -> dict | None:
        """Fetch daily fund flow data for a single stock (主力资金流向).

        Uses AKShare stock_individual_fund_flow which returns:
        - 主力净流入 (super_large + large order net inflow)
        - 超大单/大单/中单/小单 净流入
        """
        try:
            df = ak.stock_individual_fund_flow(stock=code, market="sh" if code.startswith("6") else "sz")
            if df is None or df.empty:
                return None

            # Get the row for the target date (default: latest)
            if trade_date:
                row = df[df["日期"].astype(str) == trade_date]
                if row.empty:
                    return None
                row = row.iloc[0]
            else:
                row = df.iloc[0]

            return {
                "code": code,
                "trade_date": str(row.get("日期", "")),
                "main_net_inflow": _safe_float(row.get("主力净流入", row.get("主力净流入-净额", None))),
                "super_large_net": _safe_float(row.get("超大单净流入", row.get("超大单净流入-净额", None))),
                "large_net": _safe_float(row.get("大单净流入", row.get("大单净流入-净额", None))),
                "medium_net": _safe_float(row.get("中单净流入", row.get("中单净流入-净额", None))),
                "small_net": _safe_float(row.get("小单净流入", row.get("小单净流入-净额", None))),
            }
        except Exception as e:
            logger.debug(f"Fund flow fetch failed for {code}: {e}")
            return None

    @staticmethod
    def fetch_north_bound_flow(trade_date: str | None = None) -> pd.DataFrame | None:
        """Fetch north-bound capital flow (北向资金) market-wide data.

        Returns daily north-bound net inflow for all stocks with connect status.
        """
        try:
            df = ak.stock_hsgt_north_net_flow_in_em()
            if df is not None and not df.empty:
                return df
        except Exception as e:
            logger.warning(f"North-bound flow fetch failed: {e}")
        return None

    @staticmethod
    def sync_fund_flows(codes: list[str] | None = None, db: Session | None = None) -> int:
        """Sync daily fund flow data for given stocks."""
        if db is None:
            db = SessionLocal()
            close_db = True
        else:
            close_db = False

        if codes is None:
            codes = [r[0] for r in db.query(Stock.code).filter(Stock.is_active == True).limit(500).all()]

        try:
            count = 0
            for i, code in enumerate(codes):
                data = DataPipeline.fetch_fund_flow(code)
                if data is None:
                    continue

                # Upsert by code + date
                existing = (
                    db.query(FundFlow)
                    .filter(FundFlow.code == code, FundFlow.trade_date == data["trade_date"])
                    .first()
                )
                if existing:
                    existing.main_net_inflow = data.get("main_net_inflow")
                    existing.super_large_net = data.get("super_large_net")
                    existing.large_net = data.get("large_net")
                    existing.medium_net = data.get("medium_net")
                    existing.small_net = data.get("small_net")
                else:
                    db.add(FundFlow(**data))

                count += 1
                if count % 100 == 0:
                    db.commit()
                    logger.info(f"Fund flow sync: {count}/{len(codes)}")

            db.commit()
            logger.info(f"Synced fund flow for {count} stocks")
            return count
        finally:
            if close_db:
                db.close()

    # ── Multi-period financials ─────────────────────────────────────

    @staticmethod
    def fetch_multi_period_financials(code: str, periods: int = 4) -> list[dict] | None:
        """Fetch multiple periods of financial indicators (last N quarters).

        Uses East Money API which returns time-series by default.
        Returns list of FinancialIndicator dicts sorted by report_date desc.
        """
        try:
            df = ak.stock_financial_analysis_indicator(symbol=code, start_year="2023")
            if df is None or df.empty:
                return None

            results = []
            for i in range(min(periods, len(df))):
                row = df.iloc[i]
                report_date = str(row.get("报告期", row.get("end_date", ""))).replace("-", "")
                if not report_date or len(report_date) < 6:
                    continue

                results.append({
                    "code": code,
                    "report_date": report_date,
                    "roe": _safe_float(row.get("净资产收益率(%)", None)),
                    "eps": _safe_float(row.get("基本每股收益(元)", None)),
                    "gross_margin": _safe_float(row.get("销售毛利率(%)", None)),
                    "net_margin": _safe_float(row.get("销售净利率(%)", None)),
                    "revenue_growth": _safe_float(row.get("营业收入同比增长率(%)", None)),
                    "profit_growth": _safe_float(row.get("净利润同比增长率(%)", None)),
                    "debt_ratio": _safe_float(row.get("资产负债率(%)", None)),
                    "current_ratio": _safe_float(row.get("流动比率", None)),
                    "bv_per_share": _safe_float(row.get("每股净资产(元)", None)),
                    "pe": _safe_float(row.get("市盈率", None)),
                    "pb": _safe_float(row.get("市净率", None)),
                })
            return results if results else None
        except Exception as e:
            logger.debug(f"Multi-period financials failed for {code}: {e}")
            return None

    @staticmethod
    def sync_multi_period_financials(codes: list[str] | None = None, periods: int = 4, db: Session | None = None) -> int:
        """Sync multi-period financial data for given stocks."""
        if db is None:
            db = SessionLocal()
            close_db = True
        else:
            close_db = False

        if codes is None:
            codes = [r[0] for r in db.query(Stock.code).filter(Stock.is_active == True).limit(500).all()]

        try:
            count = 0
            for i, code in enumerate(codes):
                results = DataPipeline.fetch_multi_period_financials(code, periods)
                if not results:
                    continue

                for data in results:
                    existing = (
                        db.query(FinancialIndicator)
                        .filter(FinancialIndicator.code == code, FinancialIndicator.report_date == data["report_date"])
                        .first()
                    )
                    if existing:
                        for key in ["roe", "eps", "gross_margin", "net_margin", "revenue_growth",
                                      "profit_growth", "debt_ratio", "current_ratio", "bv_per_share", "pe", "pb"]:
                            if data.get(key) is not None:
                                setattr(existing, key, data[key])
                    else:
                        db.add(FinancialIndicator(**data))
                    count += 1

                if (i + 1) % 50 == 0:
                    db.commit()
                    logger.info(f"Multi-period financials: {i + 1}/{len(codes)}")

            db.commit()
            logger.info(f"Synced {count} multi-period financial records")
            return count
        finally:
            if close_db:
                db.close()

    # ── Market breadth (市场宽度) ────────────────────────────────────

    @staticmethod
    def fetch_market_breadth(trade_date: str | None = None) -> dict | None:
        """Fetch market breadth indicators (涨跌家数, 涨跌停家数).

        Uses AKShare stock_zh_a_spot_em for real-time snapshot.
        """
        try:
            df = ak.stock_zh_a_spot_em()
            if df is None or df.empty:
                return None

            pct_change_col = None
            for col in df.columns:
                if "涨跌幅" in col:
                    pct_change_col = col
                    break

            if pct_change_col is None:
                return None

            changes = pd.to_numeric(df[pct_change_col], errors="coerce")
            advancing = (changes > 0).sum()
            declining = (changes < 0).sum()
            unchanged = (changes == 0).sum()
            limit_up = (changes >= 9.9).sum()  # ~10%涨停
            limit_down = (changes <= -9.9).sum()

            return {
                "trade_date": trade_date or date.today().isoformat(),
                "total_stocks": len(df),
                "advancing": int(advancing),
                "declining": int(declining),
                "unchanged": int(unchanged),
                "limit_up": int(limit_up),
                "limit_down": int(limit_down),
                "adv_decl_ratio": round(float(advancing) / max(float(declining), 1), 2),
            }
        except Exception as e:
            logger.warning(f"Market breadth fetch failed: {e}")
            return None

    # ── Industry classification ────────────────────────────────────

    @staticmethod
    def sync_industry_data(db: Session | None = None) -> int:
        """Fetch industry classification from East Money and populate Stock.industry field."""
        if db is None:
            db = SessionLocal()
            close_db = True
        else:
            close_db = False

        try:
            df = ak.stock_board_industry_name_em()
            count = 0
            for _, row in df.iterrows():
                board_name = row.get("板块名称", "")
                if not board_name:
                    continue

                try:
                    members = ak.stock_board_industry_cons_em(symbol=board_name)
                    if members is not None and not members.empty:
                        code_col = members.columns[0] if members.columns[0] in ["代码", "code"] else None
                        if code_col is None:
                            for c in members.columns:
                                if "代码" in c or "code" in c.lower():
                                    code_col = c
                                    break
                        codes_list = members[code_col].astype(str).str.strip().tolist() if code_col else []
                        for code in codes_list:
                            existing = db.get(Stock, code)
                            if existing and not existing.industry:
                                existing.industry = board_name
                                count += 1
                except Exception:
                    continue

                if count > 0 and count % 500 == 0:
                    db.commit()
                    logger.info(f"Industry sync: {count} stocks classified")

            db.commit()
            logger.info(f"Updated industry for {count} stocks")
            return count
        finally:
            if close_db:
                db.close()

    # ── Index data ──────────────────────────────────────────────────

    @staticmethod
    def sync_index_constituents(index_code: str = "000300") -> list[str]:
        """Fetch index constituents (e.g. CSI300). Returns list of stock codes."""
        try:
            df = ak.index_stock_cons(index_codes=index_code)
            if df is None or df.empty:
                return []
            for col in df.columns:
                if "品种代码" in col or "code" in col.lower():
                    return [str(c).strip().zfill(6) for c in df[col].tolist()]
            return [str(c).strip().zfill(6) for c in df.iloc[:, 0].tolist()]
        except Exception as e:
            logger.warning(f"Failed to fetch index {index_code} constituents: {e}")
            return []

    @staticmethod
    def fetch_index_quotes(
        index_code: str = "000001",
        start_date: str | None = None,
        end_date: str | None = None,
    ) -> pd.DataFrame | None:
        """Fetch index daily quotes for benchmark comparison (Sina source)."""
        try:
            sina_symbol = f"sh{index_code}" if index_code.startswith(("0", "6", "5")) else f"sz{index_code}"
            df = ak.stock_zh_index_daily(symbol=sina_symbol)
            if df is not None and not df.empty:
                df = df.rename(columns={"date": "trade_date"})
                df["trade_date"] = pd.to_datetime(df["trade_date"]).dt.strftime("%Y-%m-%d")
                if start_date:
                    df = df[df["trade_date"] >= start_date]
                if end_date:
                    df = df[df["trade_date"] <= end_date]
                return df
        except Exception as e:
            logger.warning(f"Failed to fetch index {index_code}: {e}")
        return None


    # ── Margin trading (融资融券) ──────────────────────────────────

    @staticmethod
    def fetch_margin_trading(trade_date: str | None = None) -> pd.DataFrame | None:
        """Fetch margin trading & short selling data for all stocks on a given date.

        Uses AKShare stock_margin_detail_sse + stock_margin_detail_szse.
        Data is typically available the NEXT morning (~9:00), NOT same-day.
        """
        all_data = []
        for exchange, fetcher in [
            ("sh", lambda d: ak.stock_margin_detail_sse(date=d)),
            ("sz", lambda d: ak.stock_margin_detail_szse(date=d)),
        ]:
            try:
                df = fetcher(trade_date or date.today().isoformat().replace("-", ""))
                if df is not None and not df.empty:
                    df["exchange"] = exchange
                    all_data.append(df)
            except Exception as e:
                logger.debug(f"Margin trading fetch failed for {exchange}: {e}")
        if not all_data:
            return None
        return pd.concat(all_data, ignore_index=True)

    @staticmethod
    def sync_margin_trading(db: Session | None = None, lookback_days: int = 3) -> int:
        """Sync margin trading data for recent days."""
        if db is None:
            db = SessionLocal()
            close_db = True
        else:
            close_db = False

        try:
            count = 0
            for d in range(lookback_days):
                td = (date.today() - timedelta(days=d)).isoformat().replace("-", "")
                df = DataPipeline.fetch_margin_trading(td)
                if df is None or df.empty:
                    continue

                for _, row in df.iterrows():
                    code = str(row.get("股票代码", row.get("code", ""))).strip()
                    if not code:
                        continue
                    existing = (
                        db.query(MarginTrading)
                        .filter(MarginTrading.code == code, MarginTrading.trade_date == td)
                        .first()
                    )
                    data = {
                        "code": code, "trade_date": td,
                        "margin_balance": _safe_float(row.get("融资余额", row.get("rzye", None))),
                        "margin_buy": _safe_float(row.get("融资买入额", row.get("rzmre", None))),
                        "margin_repay": _safe_float(row.get("融资偿还额", row.get("rzche", None))),
                        "short_balance": _safe_float(row.get("融券余量", row.get("rqyl", None))),
                        "short_sell": _safe_float(row.get("融券卖出量", row.get("rqmcl", None))),
                        "short_repay": _safe_float(row.get("融券偿还量", row.get("rqchl", None))),
                    }
                    if existing:
                        for k, v in data.items():
                            if k not in ("code", "trade_date") and v is not None:
                                setattr(existing, k, v)
                    else:
                        db.add(MarginTrading(**data))
                    count += 1

                if count > 0:
                    db.commit()
                    logger.info(f"Margin trading synced for {td}: {count} records")
            return count
        finally:
            if close_db:
                db.close()

    # ── Shareholder count (股东户数 — 筹码集中度) ──────────────────

    @staticmethod
    def fetch_shareholder_count(code: str) -> dict | None:
        """Fetch shareholder count trend for a single stock."""
        try:
            df = ak.stock_holder_number(symbol=code)
            if df is None or df.empty:
                return None

            latest = df.iloc[0]
            prev = df.iloc[1] if len(df) > 1 else None

            result = {
                "code": code,
                "end_date": str(latest.get("股东户数统计截止日", latest.get("end_date", ""))),
                "holder_count": _safe_int(latest.get("股东户数", latest.get("holder_num", None))),
                "avg_holding": _safe_float(latest.get("户均持股", latest.get("avg_holding", None))),
            }

            # Compute MoM change
            if prev is not None:
                cur_count = _safe_int(latest.get("股东户数", latest.get("holder_num", None)))
                prev_count = _safe_int(prev.get("股东户数", prev.get("holder_num", None)))
                if cur_count and prev_count and prev_count > 0:
                    result["holder_change_pct"] = round((cur_count - prev_count) / prev_count * 100, 2)

            return result
        except Exception as e:
            logger.debug(f"Shareholder count fetch failed for {code}: {e}")
            return None

    @staticmethod
    def sync_shareholder_counts(codes: list[str] | None = None, db: Session | None = None) -> int:
        """Sync shareholder count data for given stocks."""
        if db is None:
            db = SessionLocal()
            close_db = True
        else:
            close_db = False

        if codes is None:
            codes = [r[0] for r in db.query(Stock.code).filter(Stock.is_active == True).limit(300).all()]

        try:
            count = 0
            for i, code in enumerate(codes):
                data = DataPipeline.fetch_shareholder_count(code)
                if not data:
                    continue

                existing = (
                    db.query(ShareholderCount)
                    .filter(ShareholderCount.code == code, ShareholderCount.end_date == data["end_date"])
                    .first()
                )
                if existing:
                    existing.holder_count = data.get("holder_count")
                    existing.avg_holding = data.get("avg_holding")
                    existing.holder_change_pct = data.get("holder_change_pct")
                else:
                    db.add(ShareholderCount(**data))
                count += 1

                if count % 50 == 0:
                    db.commit()
                    logger.info(f"Shareholder count sync: {count} processed")

            db.commit()
            logger.info(f"Synced shareholder counts for {count} stocks")
            return count
        finally:
            if close_db:
                db.close()

    # ── Dragon Tiger List (龙虎榜) ─────────────────────────────────

    @staticmethod
    def fetch_dragon_tiger(trade_date: str | None = None) -> pd.DataFrame | None:
        """Fetch dragon-tiger list for a specific trading day."""
        try:
            td = trade_date or date.today().isoformat()
            df = ak.stock_lhb_detail_daily(date=td, flag="明细")
            if df is not None and not df.empty:
                return df
        except Exception:
            pass

        # Fallback: try top-list summary
        try:
            td = trade_date or date.today().isoformat()
            df = ak.stock_lhb_stock_statistic_daily(date=td)
            if df is not None and not df.empty:
                return df
        except Exception as e:
            logger.debug(f"Dragon tiger list fetch failed: {e}")
        return None

    @staticmethod
    def sync_dragon_tiger(db: Session | None = None, lookback_days: int = 5) -> int:
        """Sync dragon-tiger list data for recent days."""
        if db is None:
            db = SessionLocal()
            close_db = True
        else:
            close_db = False

        try:
            count = 0
            for d in range(lookback_days):
                td = (date.today() - timedelta(days=d)).isoformat()
                df = DataPipeline.fetch_dragon_tiger(td)
                if df is None or df.empty:
                    continue

                for _, row in df.iterrows():
                    code = _safe_str(row.get("代码", row.get("code", "")))
                    if not code:
                        continue

                    existing = (
                        db.query(DragonTiger)
                        .filter(DragonTiger.code == code, DragonTiger.trade_date == td)
                        .first()
                    )
                    data = {
                        "code": code, "trade_date": td,
                        "reason": _safe_str(row.get("上榜原因", "")),
                        "buy_amount": _safe_float(row.get("买入金额", row.get("buy", None))),
                        "sell_amount": _safe_float(row.get("卖出金额", row.get("sell", None))),
                        "net_amount": _safe_float(row.get("净买入额", row.get("net", None))),
                    }
                    if existing:
                        for k, v in data.items():
                            if k not in ("code", "trade_date") and v is not None:
                                setattr(existing, k, v)
                    else:
                        db.add(DragonTiger(**data))
                    count += 1
            db.commit()
            logger.info(f"Synced {count} dragon-tiger records")
            return count
        finally:
            if close_db:
                db.close()

    # ── Lockup Release (限售解禁) ──────────────────────────────────

    @staticmethod
    def fetch_lockup_release(code: str) -> list[dict] | None:
        """Fetch upcoming lockup release schedule for a stock."""
        try:
            df = ak.stock_restricted_release(symbol=code)
            if df is None or df.empty:
                return None

            results = []
            for _, row in df.iterrows():
                release_date = str(row.get("解禁日期", row.get("release_date", "")))
                if not release_date or release_date < date.today().isoformat():
                    continue
                results.append({
                    "code": code,
                    "release_date": release_date,
                    "release_shares": _safe_float(row.get("解禁数量", row.get("release_shares", None))),
                    "release_ratio": _safe_float(row.get("占总股本比例", row.get("release_ratio", None))),
                    "release_market_value": _safe_float(row.get("解禁市值", row.get("market_value", None))),
                })
            return results if results else None
        except Exception as e:
            logger.debug(f"Lockup release fetch failed for {code}: {e}")
            return None

    @staticmethod
    def sync_lockup_releases(codes: list[str] | None = None, db: Session | None = None) -> int:
        """Sync upcoming lockup releases (next 90 days)."""
        if db is None:
            db = SessionLocal()
            close_db = True
        else:
            close_db = False

        if codes is None:
            codes = [r[0] for r in db.query(Stock.code).filter(Stock.is_active == True).limit(300).all()]

        try:
            count = 0
            for code in codes:
                results = DataPipeline.fetch_lockup_release(code)
                if not results:
                    continue
                for data in results:
                    existing = (
                        db.query(LockupRelease)
                        .filter(LockupRelease.code == code, LockupRelease.release_date == data["release_date"])
                        .first()
                    )
                    if not existing:
                        db.add(LockupRelease(**data))
                        count += 1
            db.commit()
            logger.info(f"Synced {count} upcoming lockup releases")
            return count
        finally:
            if close_db:
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


def _safe_int(val) -> int | None:
    if val is None:
        return None
    try:
        i = int(float(val))
        return None if pd.isna(i) else i
    except (ValueError, TypeError):
        return None


def _safe_str(val) -> str | None:
    if val is None:
        return None
    s = str(val).strip()
    return s if s else None


