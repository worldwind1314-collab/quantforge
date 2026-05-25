"""JoinQuant (聚宽) data source adapter.

JQData (https://www.joinquant.com) provides:
  - Comprehensive A-share daily/weekly/minute data
  - Financial indicators with standardized fields
  - Industry/sector classification (SW/CSRC/JQ)
  - Factor data (valuation, momentum, quality, etc.)
  - Index constituents, margin trading

Requires: pip install jqdatasdk + registered account.
Free tier: limited API calls per day, sufficient for daily pipeline.
"""

import logging
import os
from datetime import date, timedelta
from typing import Any

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

_HAS_JQDATASDK = False
try:
    import jqdatasdk as jq

    _HAS_JQDATASDK = True
except ImportError:
    logger.warning("jqdatasdk not installed. Run: pip install jqdatasdk")


class JQDataSource:
    """JoinQuant data adapter with unified interface."""

    def __init__(self, username: str | None = None, password: str | None = None):
        self._username = username or os.environ.get("JQ_USERNAME", "")
        self._password = password or os.environ.get("JQ_PASSWORD", "")
        self._logged_in = False

    @property
    def available(self) -> bool:
        return _HAS_JQDATASDK and bool(self._username)

    @property
    def name(self) -> str:
        return "joinquant"

    def _ensure_login(self) -> bool:
        if not self.available:
            return False
        if not self._logged_in:
            try:
                jq.auth(self._username, self._password)
                self._logged_in = True
                logger.info("JQData authenticated successfully")
            except Exception as e:
                logger.warning(f"JQData auth failed: {e}. Set JQ_USERNAME/JQ_PASSWORD env vars.")
                return False
        return self._logged_in

    # ── Daily quotes ──────────────────────────────────────────────

    def fetch_daily_quotes(
        self,
        codes: list[str],
        start_date: str,
        end_date: str,
    ) -> pd.DataFrame | None:
        """Fetch daily K-line for multiple stocks.

        Returns a multi-index DataFrame (code, date) using JQData's get_price().
        """
        if not self.available or not self._ensure_login():
            return None

        try:
            df = jq.get_price(
                codes,
                start_date=start_date,
                end_date=end_date,
                frequency="daily",
                fields=["open", "high", "low", "close", "volume", "money", "factor"],
                skip_paused=True,
                fq="pre",  # 前复权
            )
            if df is not None and not df.empty:
                df = df.rename(columns={"money": "amount"})
                return df
        except Exception as e:
            logger.warning(f"JQData daily quotes failed: {e}")
        return None

    # ── Financial indicators ──────────────────────────────────────

    def fetch_financials(self, codes: list[str], report_date: str) -> pd.DataFrame | None:
        """Fetch financial indicators using JQData's get_fundamentals().

        JQData provides standardized fields through query objects.
        """
        if not self.available or not self._ensure_login():
            return None

        try:
            q = jq.query(
                jq.indicator
            ).filter(
                jq.indicator.code.in_(codes),
                jq.income.statDate == report_date,
            )
            df = jq.get_fundamentals(q, date=report_date)
            if df is not None and not df.empty:
                return df
        except Exception as e:
            logger.debug(f"JQData financials failed: {e}")
        return None

    # ── Industry classification ───────────────────────────────────

    def fetch_industry(self, codes: list[str]) -> dict[str, str]:
        """Get SW (申万) industry classification for stocks."""
        if not self.available or not self._ensure_login():
            return {}

        try:
            df = jq.get_industry(codes, date=date.today().isoformat())
            if df is not None and not df.empty:
                result = {}
                for _, row in df.iterrows():
                    code = str(row.get("code", "")).strip()
                    industry = str(row.get("sw_l1", {}).get("industry_name", ""))
                    if code and industry:
                        result[code] = industry
                return result
        except Exception as e:
            logger.debug(f"JQData industry fetch failed: {e}")
        return {}

    # ── Index constituents ────────────────────────────────────────

    def fetch_index_stocks(self, index_code: str = "000300.XSHG") -> list[str]:
        """Get current constituents of an index (CSI300, CSI500, etc.)."""
        if not self.available or not self._ensure_login():
            return []

        try:
            stocks = jq.get_index_stocks(index_code, date=date.today().isoformat())
            return [str(s).strip() for s in stocks]
        except Exception as e:
            logger.debug(f"JQData index constituents failed: {e}")
            return []

    # ── Factor data ───────────────────────────────────────────────

    def fetch_factor_values(
        self,
        codes: list[str],
        factors: list[str],
        start_date: str,
        end_date: str,
    ) -> pd.DataFrame | None:
        """Fetch pre-computed factor values from JQData factor library.

        Common JQData factors:
          - valuation: pe_ratio, pb_ratio, ps_ratio, market_cap
          - momentum: momentum_3m, momentum_6m, arron_up_25
          - quality: roe_ttm, roa_ttm, gross_margin_ttm
          - sentiment: inst_holding_pct, short_total
        """
        if not self.available or not self._ensure_login():
            return None

        try:
            from jqdatasdk import finance

            # Use get_ba_factor for single-factor batch query
            all_data = []
            for factor in factors:
                try:
                    df = jq.get_ba_factor(
                        codes, factor, start_date=start_date, end_date=end_date,
                    )
                    if df is not None and not df.empty:
                        df = df.rename(columns={"value": factor})
                        all_data.append(df)
                except Exception:
                    continue

            if all_data:
                result = all_data[0]
                for df in all_data[1:]:
                    result = result.merge(df, on=["code", "date"], how="outer")
                return result
        except Exception as e:
            logger.debug(f"JQData factor values failed: {e}")
        return None
