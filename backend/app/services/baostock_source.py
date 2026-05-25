"""Baostock data source adapter — free, no-registration A-share data.

Baostock (http://baostock.com) provides:
  - Daily K-line (前复权/后复权/不复权)
  - 5/15/30/60 minute K-line
  - Financial indicators (季报/年报)
  - Shareholder data, index constituents
  - No API key required

Used as secondary/validation source alongside AKShare.
"""

import logging
from datetime import date, timedelta
from typing import Any

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

_HAS_BAOSTOCK = False
try:
    import baostock as bs

    _HAS_BAOSTOCK = True
except ImportError:
    logger.warning("Baostock not installed. Run: pip install baostock")


class BaostockSource:
    """Baostock data adapter with unified interface matching DataPipeline."""

    def __init__(self):
        self._logged_in = False

    @property
    def available(self) -> bool:
        return _HAS_BAOSTOCK

    @property
    def name(self) -> str:
        return "baostock"

    def _ensure_login(self):
        if not self.available:
            return False
        if not self._logged_in:
            try:
                lg = bs.login()
                if lg.error_code == "0":
                    self._logged_in = True
                else:
                    logger.warning(f"Baostock login failed: {lg.error_msg}")
                    return False
            except Exception as e:
                logger.warning(f"Baostock login error: {e}")
                return False
        return self._logged_in

    # ── Daily quotes ──────────────────────────────────────────────

    def fetch_daily_quotes(
        self,
        codes: list[str],
        start_date: str,
        end_date: str,
        adjust: str = "2",  # 1=后复权, 2=前复权, 3=不复权
    ) -> dict[str, pd.DataFrame]:
        """Fetch daily K-line for multiple stocks.

        Baostock requires codes in sh.000001 or sz.000001 format.
        """
        if not self.available or not self._ensure_login():
            return {}

        results = {}
        for code in codes:
            bs_code = _to_baostock_code(code)
            try:
                rs = bs.query_history_k_data_plus(
                    bs_code,
                    "date,open,high,low,close,preclose,volume,amount,"
                    "turn,tradestatus,pctChg,peTTM,pbMRQ,psTTM",
                    start_date=start_date,
                    end_date=end_date,
                    frequency="d",
                    adjustflag=adjust,
                )
                if rs.error_code != "0":
                    continue

                rows = []
                while rs.next():
                    row = rs.get_row_data()
                    rows.append(row)

                if not rows:
                    continue

                df = pd.DataFrame(rows, columns=rs.fields)
                for col in ["open", "high", "low", "close", "preclose", "volume", "amount", "turn", "pctChg"]:
                    df[col] = pd.to_numeric(df[col], errors="coerce")

                df = df.rename(columns={
                    "date": "trade_date", "turn": "turnover",
                    "tradestatus": "trade_status", "pctChg": "pct_change",
                })
                df = df.set_index("trade_date")

                # Filter out suspended/halted days
                if "trade_status" in df.columns:
                    df = df[df["trade_status"] == "1"]

                if not df.empty:
                    results[code] = df
            except Exception as e:
                logger.debug(f"Baostock K-line failed for {code}: {e}")
                continue

        logger.info(f"Baostock: fetched K-lines for {len(results)}/{len(codes)} stocks")
        return results

    # ── Financial data ────────────────────────────────────────────

    def fetch_financials(self, code: str, year: int = 2025, quarter: int = 1) -> dict | None:
        """Fetch quarterly financial indicators from Baostock."""
        if not self.available or not self._ensure_login():
            return None

        bs_code = _to_baostock_code(code)
        try:
            rs = bs.query_growth_data(bs_code, year=year, quarter=quarter)
            if rs.error_code != "0":
                return None

            row_data = {}
            while rs.next():
                row = rs.get_row_data()
                row_data[row[0]] = row[1]

            if not row_data:
                return None

            def _v(key):
                try:
                    return float(row_data.get(key, 0))
                except (ValueError, TypeError):
                    return None

            return {
                "code": code,
                "report_date": f"{year}-{quarter*3:02d}-28",
                "roe": _v("ROE"),
                "eps": _v("ESP"),
                "gross_margin": _v("grossProfitMargin"),
                "net_margin": _v("netProfitMargin"),
                "revenue_growth": _v("YOYOperateIncome"),
                "profit_growth": _v("YOYNI"),
            }
        except Exception as e:
            logger.debug(f"Baostock financials failed for {code}: {e}")
            return None

    # ── Shareholder data ──────────────────────────────────────────

    def fetch_shareholder_count(self, code: str) -> list[dict] | None:
        """Fetch historical shareholder count from Baostock."""
        if not self.available or not self._ensure_login():
            return None

        bs_code = _to_baostock_code(code)
        try:
            rs = bs.query_stock_shareholder(bs_code)
            if rs.error_code != "0":
                return None

            results = []
            while rs.next():
                row = rs.get_row_data()
                results.append({
                    "code": code,
                    "end_date": row[0],
                    "holder_count": int(row[1]) if row[1] else None,
                })
            return results if results else None
        except Exception as e:
            logger.debug(f"Baostock shareholder count failed for {code}: {e}")
            return None


def _to_baostock_code(code: str) -> str:
    """Convert 000001 → sh.000001 or sz.000001."""
    if code.startswith(("6", "5", "9")):
        return f"sh.{code}"
    return f"sz.{code}"
