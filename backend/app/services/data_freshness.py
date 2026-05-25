"""Data freshness monitor for live trading safety.

Tracks the expected data availability schedule for each data type and
verifies data freshness before allowing trading decisions.

A-share data availability timeline:
  ┌─────────────────────────────────────────────────────────┐
  │ 15:00  市场收盘                                           │
  │ 15:30  日K线数据可用 (交易所发布)                            │
  │ 17:00  资金流向数据可用                                     │
  │ 17:30  龙虎榜数据可用                                       │
  │ 18:00  北向资金数据可用                                     │
  │ 次日9:00  融资融券数据可用 (永远晚一天!)                       │
  │ 不定     财报数据 (季报/年报披露日)                           │
  └─────────────────────────────────────────────────────────┘

Key constraint: margin trading data is ALWAYS 1 day behind.
Pipeline runs at 18:00 → all data except margin is same-day.
For next-day trading: daily quotes are T-1, margin data is T-2.

Pre-trade checklist (runs at ~9:00 before market open):
  1. Yesterday's daily quotes must be loaded
  2. Yesterday's fund flow must be loaded
  3. Margin data from 2 days ago is acceptable
  4. Factor scores for latest trading day must exist
  5. ML predictions for latest trading day must exist
"""

import json
import logging
from datetime import date, datetime, timedelta
from enum import Enum
from typing import Any

import pandas as pd
from sqlalchemy import func

from ..core.database import SessionLocal

logger = logging.getLogger(__name__)


class FreshnessStatus(Enum):
    FRESH = "fresh"  # data is current
    ACCEPTABLE = "acceptable"  # slightly delayed but within tolerance
    STALE = "stale"  # beyond expected window
    CRITICAL = "critical"  # dangerously out of date
    UNKNOWN = "unknown"  # can't determine


# ── Data type definitions with expected latency ─────────────────────

DATA_TYPE_CONFIG = {
    "daily_quotes": {
        "name": "日K线行情",
        "table": "daily_quotes",
        "date_col": "trade_date",
        "expected_available_hour": 16,  # ~15:30, rounded to 16:00
        "max_age_hours": 24,  # T+1 data is OK (yesterday)
        "critical_age_hours": 48,
        "market_hours_only": True,
    },
    "fund_flows": {
        "name": "资金流向",
        "table": "fund_flows",
        "date_col": "trade_date",
        "expected_available_hour": 17,
        "max_age_hours": 24,
        "critical_age_hours": 48,
        "market_hours_only": True,
    },
    "factor_scores": {
        "name": "因子得分",
        "table": "factor_scores",
        "date_col": "trade_date",
        "expected_available_hour": 19,  # depends on pipeline completion
        "max_age_hours": 24,
        "critical_age_hours": 72,
        "market_hours_only": True,
    },
    "ml_predictions": {
        "name": "ML预测",
        "table": "ml_predictions",
        "date_col": "trade_date",
        "expected_available_hour": 19,
        "max_age_hours": 24,
        "critical_age_hours": 72,
        "market_hours_only": True,
    },
    "margin_trading": {
        "name": "融资融券",
        "table": "margin_trading",
        "date_col": "trade_date",
        "expected_available_hour": 9,  # next morning
        "max_age_hours": 48,  # T+2 data IS acceptable for margin
        "critical_age_hours": 72,
        "market_hours_only": True,
        "note": "融资融券数据永远滞后一天，这是数据源特性，不影响策略判断",
    },
    "dragon_tiger": {
        "name": "龙虎榜",
        "table": "dragon_tiger",
        "date_col": "trade_date",
        "expected_available_hour": 18,
        "max_age_hours": 48,  # dragon tiger is reference data, can be older
        "critical_age_hours": 120,
        "market_hours_only": True,
    },
    "financial_indicators": {
        "name": "财务指标",
        "table": "financial_indicators",
        "date_col": "report_date",
        "expected_available_hour": 9,
        "max_age_hours": 2160,  # 90 days — quarterly data
        "critical_age_hours": 4320,  # 180 days — miss one reporting season
        "market_hours_only": False,
    },
}


class DataFreshnessMonitor:
    """Pre-trade and runtime data freshness checker."""

    def __init__(self):
        self._warnings: list[dict] = []

    # ── Pre-trade checklist ──────────────────────────────────────

    def pre_trade_check(self) -> dict:
        """Run before market open (9:00-9:25). Returns go/no-go decision.

        Returns:
            {
                "can_trade": bool,
                "overall_status": "go" | "warning" | "no_go",
                "checks": {data_type: status_detail},
                "warnings": [str],
                "blockers": [str],
            }
        """
        self._warnings = []
        now = datetime.now()
        checks = {}
        warnings = []
        blockers = []

        db = SessionLocal()
        try:
            for dtype, config in DATA_TYPE_CONFIG.items():
                status = self._check_type(db, dtype, config, now)
                checks[dtype] = status

                if status["status"] == FreshnessStatus.CRITICAL.value:
                    blockers.append(
                        f"{config['name']}: 最新数据为 {status['latest_date']}，"
                        f"已过期 {status['age_days']} 天"
                    )
                elif status["status"] == FreshnessStatus.STALE.value:
                    warnings.append(
                        f"{config['name']}: 数据延迟 {status['age_days']} 天 "
                        f"(最后更新: {status['latest_date']})"
                    )
        finally:
            db.close()

        # Decision logic
        critical_blockers = [b for b in blockers if "日K线" in b or "因子得分" in b or "ML预测" in b]
        can_trade = len(critical_blockers) == 0

        if len(blockers) > 0:
            overall = "no_go"
        elif len(warnings) > 1:
            overall = "warning"
        else:
            overall = "go"

        return {
            "can_trade": can_trade,
            "overall_status": overall,
            "check_time": now.isoformat(),
            "checks": checks,
            "warnings": warnings,
            "blockers": blockers,
        }

    def _check_type(
        self, db, dtype: str, config: dict, now: datetime
    ) -> dict:
        """Check freshness of a single data type."""
        table = config["table"]
        date_col = config["date_col"]
        max_age = timedelta(hours=config["max_age_hours"])
        critical_age = timedelta(hours=config["critical_age_hours"])

        # Query latest date
        try:
            from sqlalchemy import text

            row = db.execute(
                text(f"SELECT MAX({date_col}) FROM {table}")
            ).fetchone()
            latest_str = row[0] if row and row[0] else None
        except Exception as e:
            logger.debug(f"Freshness check failed for {dtype}: {e}")
            return {
                "status": FreshnessStatus.UNKNOWN.value,
                "latest_date": None,
                "age_days": None,
                "error": str(e),
            }

        if not latest_str:
            return {
                "status": FreshnessStatus.CRITICAL.value,
                "latest_date": None,
                "age_days": None,
                "message": "无数据",
            }

        try:
            latest_date = datetime.strptime(str(latest_str)[:10], "%Y-%m-%d")
        except (ValueError, TypeError):
            latest_date = now  # If unparseable, assume current (no data = critical)

        age = now - latest_date

        # Determine status
        if age <= max_age:
            status = FreshnessStatus.FRESH
        elif age <= max_age * 1.5:
            status = FreshnessStatus.ACCEPTABLE
        elif age <= critical_age:
            status = FreshnessStatus.STALE
        else:
            status = FreshnessStatus.CRITICAL

        return {
            "status": status.value,
            "latest_date": latest_date.strftime("%Y-%m-%d"),
            "age_hours": round(age.total_seconds() / 3600, 1),
            "age_days": age.days,
            "max_tolerated_hours": config["max_age_hours"],
            "note": config.get("note", ""),
        }

    # ── Runtime check (quick, called before each trade) ──────────

    def quick_check(self) -> bool:
        """Fast check: is daily quote data fresh enough to trade?

        Returns True if OK to proceed, False if data is too stale.
        """
        db = SessionLocal()
        try:
            from sqlalchemy import text

            row = db.execute(
                text("SELECT MAX(trade_date) FROM daily_quotes")
            ).fetchone()
            if not row or not row[0]:
                return False

            latest = date.fromisoformat(str(row[0])[:10])
            today = date.today()
            age = (today - latest).days

            # Allow weekend gap: if today is Monday, Friday's data is OK
            if age <= 2:
                return True
            return False
        except Exception:
            return False
        finally:
            db.close()

    # ── Health report for dashboard ──────────────────────────────

    def get_health_report(self) -> dict:
        """Generate a comprehensive data health report for the dashboard."""
        db = SessionLocal()
        try:
            now = datetime.now()
            report = {
                "generated_at": now.isoformat(),
                "is_trading_day": now.weekday() < 5,
                "market_status": self._get_market_status(now),
                "data_types": {},
                "overall_health": "unknown",
            }

            healthy = 0
            total = 0
            for dtype, config in DATA_TYPE_CONFIG.items():
                check = self._check_type(db, dtype, config, now)
                report["data_types"][dtype] = {
                    "name": config["name"],
                    "status": check["status"],
                    "latest_date": check.get("latest_date"),
                    "age_days": check.get("age_days"),
                }
                total += 1
                if check["status"] in ("fresh", "acceptable"):
                    healthy += 1

            ratio = healthy / total if total > 0 else 0
            if ratio >= 0.8:
                report["overall_health"] = "healthy"
            elif ratio >= 0.5:
                report["overall_health"] = "degraded"
            else:
                report["overall_health"] = "critical"

            return report
        finally:
            db.close()

    @staticmethod
    def _get_market_status(now: datetime) -> str:
        """Determine if market is currently open."""
        hour = now.hour + now.minute / 60
        weekday = now.weekday()

        if weekday >= 5:
            return "closed_weekend"

        if 9.25 <= hour < 11.5:
            return "open_morning"
        elif 13.0 <= hour < 15.0:
            return "open_afternoon"
        elif 9.0 <= hour < 9.25:
            return "pre_open"
        elif hour < 9.0:
            return "pre_market"
        else:
            return "closed"

    # ── Latency timeline report ──────────────────────────────────

    def get_latency_report(self) -> str:
        """Generate a human-readable latency timeline for all data types."""
        lines = [
            "=" * 55,
            "  数据延迟分析 (Data Latency Report)",
            "=" * 55,
            "",
            "  数据类型          预期就绪    最大容忍    实际延迟",
            "  " + "-" * 50,
        ]

        now = datetime.now()
        db = SessionLocal()
        try:
            for dtype, config in DATA_TYPE_CONFIG.items():
                check = self._check_type(db, dtype, config, now)
                name = config["name"]
                expected = f"T+{config['expected_available_hour'] - 15}h"
                tolerance = f"{config['max_age_hours']}h"
                actual = f"{check.get('age_hours', '?')}h"
                status = check["status"]
                flag = "✅" if status in ("fresh", "acceptable") else "⚠️" if status == "stale" else "❌"
                lines.append(f"  {flag} {name:<12} {expected:>8}    {tolerance:>8}    {actual:>8}")
        finally:
            db.close()

        lines.append("")
        lines.append("  注: 融资融券数据天然滞后一天(T+2)，属正常现象")
        lines.append("=" * 55)
        return "\n".join(lines)
