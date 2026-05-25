"""Multi-source data orchestrator with cross-validation and fallback chains.

Data source priority (configurable):
  1. AKShare (East Money)    — primary, most complete
  2. JoinQuant (聚宽)         — secondary, standardized fields
  3. Baostock                — tertiary, free validation source

Fallback chain per data type:
  - Daily quotes:       AKShare EM → AKShare Sina → Baostock
  - Financials:         AKShare EM → JoinQuant → Baostock → AKShare Sina
  - Industry:           AKShare EM → JoinQuant
  - Index constituents: AKShare → JoinQuant
  - Fund flow:          AKShare EM (only source)
  - Margin trading:     AKShare (only source)

Cross-validation:
  - Compares close prices between sources
  - Flags discrepancies > 2% for investigation
"""

import logging
from datetime import date, datetime
from typing import Any

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


class MultiSourceOrchestrator:
    """Orchestrate multiple data sources with fallback and validation."""

    def __init__(self):
        self._sources: dict[str, Any] = {}
        self._init_sources()

    def _init_sources(self):
        """Lazy-init all available data sources."""
        # AKShare is always available (it's the primary dependency)
        self._sources["akshare"] = "akshare"

        # Baostock
        try:
            from .baostock_source import BaostockSource

            bs = BaostockSource()
            if bs.available:
                self._sources["baostock"] = bs
                logger.info("Baostock source available")
        except Exception as e:
            logger.debug(f"Baostock init skipped: {e}")

        # JoinQuant
        try:
            from .jqdata_source import JQDataSource

            jq_src = JQDataSource()
            if jq_src.available:
                self._sources["joinquant"] = jq_src
                logger.info("JoinQuant source available")
        except Exception as e:
            logger.debug(f"JoinQuant init skipped: {e}")

    @property
    def available_sources(self) -> list[str]:
        return list(self._sources.keys())

    def has_source(self, name: str) -> bool:
        return name in self._sources and self._sources[name] != "akshare"

    # ── Daily quotes with fallback ────────────────────────────────

    def fetch_daily_quotes(
        self, codes: list[str], start_date: str, end_date: str
    ) -> dict[str, pd.DataFrame]:
        """Fetch daily quotes with fallback chain."""
        from .data_pipeline import DataPipeline

        # Primary: AKShare East Money
        results = DataPipeline.fetch_daily_quotes(codes, start_date, end_date)
        covered = set(results.keys())
        missing = [c for c in codes if c not in covered]

        if not missing:
            return results

        # Secondary: Baostock
        bs = self._sources.get("baostock")
        if bs and isinstance(bs, object) and hasattr(bs, "fetch_daily_quotes"):
            logger.info(f"Falling back to Baostock for {len(missing)} stocks")
            bs_results = bs.fetch_daily_quotes(missing, start_date, end_date)
            for code, df in bs_results.items():
                if code not in results:
                    results[code] = df
            covered = set(results.keys())
            missing = [c for c in codes if c not in covered]

        if missing:
            logger.warning(f"No data for {len(missing)} stocks from any source: {missing[:5]}...")

        return results

    # ── Cross-validation ──────────────────────────────────────────

    def validate_close_prices(
        self, codes: list[str], trade_date: str, max_deviation_pct: float = 2.0
    ) -> dict:
        """Cross-validate close prices between available sources.

        Returns:
            {
                "validated": int,    # stocks with matching data
                "discrepancies": [   # stocks with mismatches
                    {"code": str, "akshare": float, "source": str, "value": float, "diff_pct": float}
                ],
                "missing_in_akshare": [str],
                "missing_in_source": [str],
            }
        """
        from .data_pipeline import DataPipeline

        result = {
            "validated": 0,
            "discrepancies": [],
            "missing_in_akshare": [],
            "missing_in_source": [],
        }

        start = (date.fromisoformat(trade_date) - pd.Timedelta(days=5)).isoformat()

        # Fetch from AKShare
        ak_results = DataPipeline.fetch_daily_quotes(codes, start, trade_date)

        # Fetch from validation source (Baostock if available)
        validation_source = None
        bs = self._sources.get("baostock")
        if bs and isinstance(bs, object) and hasattr(bs, "fetch_daily_quotes"):
            validation_source = bs
            source_name = "baostock"
        else:
            return result  # no validation source available

        source_results = validation_source.fetch_daily_quotes(codes, start, trade_date)

        for code in codes:
            ak_df = ak_results.get(code)
            src_df = source_results.get(code)

            if ak_df is None or ak_df.empty:
                if src_df is not None and not src_df.empty:
                    result["missing_in_akshare"].append(code)
                continue

            if src_df is None or src_df.empty:
                result["missing_in_source"].append(code)
                continue

            # Compare latest close
            try:
                ak_close = float(ak_df["close"].iloc[-1])
                src_close = float(src_df["close"].iloc[-1])

                if ak_close <= 0 or src_close <= 0:
                    continue

                diff_pct = abs(ak_close - src_close) / ak_close * 100

                if diff_pct > max_deviation_pct:
                    result["discrepancies"].append({
                        "code": code,
                        "akshare": round(ak_close, 2),
                        "source": source_name,
                        "value": round(src_close, 2),
                        "diff_pct": round(diff_pct, 2),
                    })
                else:
                    result["validated"] += 1
            except (IndexError, KeyError, ValueError):
                continue

        if result["discrepancies"]:
            logger.warning(
                f"Cross-validation: {len(result['discrepancies'])} discrepancies, "
                f"{result['validated']} valid"
            )

        return result

    # ── Data freshness audit ──────────────────────────────────────

    def get_data_freshness(self) -> dict:
        """Check how fresh the data is from each source.

        Returns a dict mapping data_type → {latest_date, age_days, source, is_fresh}.
        """
        from ..core.database import SessionLocal
        from ..models.market import DailyQuote
        from ..models.finance import FactorScore, MLPrediction, MarginTrading, DragonTiger
        from sqlalchemy import func

        db = SessionLocal()
        try:
            today = date.today()
            freshness = {}

            checks = [
                ("daily_quotes", DailyQuote, "trade_date"),
                ("factor_scores", FactorScore, "trade_date"),
                ("ml_predictions", MLPrediction, "trade_date"),
                ("margin_trading", MarginTrading, "trade_date"),
                ("dragon_tiger", DragonTiger, "trade_date"),
            ]

            for name, model, col in checks:
                latest = db.query(func.max(getattr(model, col))).scalar()
                age = (today - date.fromisoformat(latest)).days if latest else 999
                freshness[name] = {
                    "latest_date": latest,
                    "age_days": age,
                    "is_fresh": age <= 2,
                    "status": "fresh" if age <= 1 else ("stale" if age <= 3 else "critical"),
                }

            return freshness
        finally:
            db.close()


# ── Singleton ─────────────────────────────────────────────────────────

_orchestrator: MultiSourceOrchestrator | None = None


def get_orchestrator() -> MultiSourceOrchestrator:
    global _orchestrator
    if _orchestrator is None:
        _orchestrator = MultiSourceOrchestrator()
    return _orchestrator
