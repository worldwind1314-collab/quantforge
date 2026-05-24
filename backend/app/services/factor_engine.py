"""Factor computation engine — transform raw data into z-scored factor scores."""

import logging
from datetime import date, timedelta

import numpy as np
import pandas as pd
from sqlalchemy.orm import Session

from ..core.database import SessionLocal
from ..models.finance import FactorScore
from ..models.market import DailyQuote
from ..models.stock import Stock

logger = logging.getLogger(__name__)


class FactorEngine:
    """Compute daily multi-factor scores for all stocks."""

    def __init__(self, db: Session | None = None):
        self._db = db

    def _get_db(self) -> Session:
        return self._db or SessionLocal()

    # ── Factor computation ─────────────────────────────────────────

    def compute_all_factors(self, trade_date: str, codes: list[str] | None = None) -> pd.DataFrame:
        """Compute all factors for stocks on a given date. If codes is None, processes all active stocks.
        Returns DataFrame indexed by code."""
        db = self._get_db()

        if codes is None:
            codes = [r[0] for r in db.query(Stock.code).filter(Stock.is_active == True).all()]

        factors = {
            "code": [],
            "value_score": [],
            "quality_score": [],
            "momentum_score": [],
            "volatility_score": [],
        }

        # Load price data for momentum/volatility computation
        end = trade_date
        start = (date.fromisoformat(trade_date) - timedelta(days=365)).isoformat()
        price_data = self._load_price_data(db, codes, start, end)

        for code in codes:
            if code not in price_data or price_data[code].empty:
                factors["code"].append(code)
                for k in ["value_score", "quality_score", "momentum_score", "volatility_score"]:
                    factors[k].append(None)
                continue

            df = price_data[code]
            factors["code"].append(code)

            # Get current price for PE/PB/PS computation
            current_price = None
            if trade_date in df.index:
                current_price = df.loc[trade_date, "close"]
                if pd.isna(current_price):
                    current_price = None

            # Value: negative PE/PB → higher score (cheaper)
            value = self._value_factor(code, trade_date, current_price)
            factors["value_score"].append(value)

            # Quality: ROE + margin + growth
            quality = self._quality_factor(code, trade_date)
            factors["quality_score"].append(quality)

            # Momentum: multi-period returns
            momentum = self._momentum_factor(df, trade_date)
            factors["momentum_score"].append(momentum)

            # Volatility: negative volatility → higher score
            vol = self._volatility_factor(df, trade_date)
            factors["volatility_score"].append(vol)

        result = pd.DataFrame(factors).set_index("code")

        # Replace None with np.nan for proper numeric operations
        result = result.replace({None: np.nan}).astype(float)

        # Cross-sectional z-score normalization
        for col in result.columns:
            valid = result[col].dropna()
            if len(valid) > 10:
                mean, std = valid.mean(), valid.std()
                if std > 0:
                    result[col] = (result[col] - mean) / std

        # Composite: equal-weighted sum of normalized scores (skip NaN)
        result["composite_score"] = result.sum(axis=1, skipna=True)

        return result

    # ── Individual factors ─────────────────────────────────────────

    def _value_factor(self, code: str, trade_date: str, current_price: float | None = None) -> float | None:
        """Lower PE/PB/PS → higher value score. Computes PE/PB/PS from per-share data if needed."""
        db = self._get_db()
        from ..models.finance import FinancialIndicator

        fi = (
            db.query(FinancialIndicator)
            .filter(FinancialIndicator.code == code)
            .order_by(FinancialIndicator.report_date.desc())
            .first()
        )
        if not fi:
            return None

        # Get PE/PB/PS: use stored values, or compute from per-share data + price
        pe = fi.pe
        pb = fi.pb
        ps = fi.ps

        if current_price and current_price > 0:
            if pe is None and fi.eps and fi.eps > 0:
                pe = current_price / fi.eps
            if pb is None and fi.bv_per_share and fi.bv_per_share > 0:
                pb = current_price / fi.bv_per_share
            if ps is None and fi.revenue_per_share and fi.revenue_per_share > 0:
                ps = current_price / fi.revenue_per_share

        score = 0.0
        w = 0

        if pe and pe > 0:
            score += -pe  # negative PE → positive contribution
            w += 1
        if pb and pb > 0:
            score += -pb * 10
            w += 1
        if ps and ps > 0:
            score += -ps * 5
            w += 1

        return score / w if w > 0 else None

    def _quality_factor(self, code: str, trade_date: str) -> float | None:
        """Higher ROE, margins, growth → higher quality score."""
        db = self._get_db()
        from ..models.finance import FinancialIndicator

        fi = (
            db.query(FinancialIndicator)
            .filter(FinancialIndicator.code == code)
            .order_by(FinancialIndicator.report_date.desc())
            .first()
        )
        if not fi:
            return None

        score = 0.0
        w = 0

        if fi.roe is not None:
            score += fi.roe
            w += 1
        if fi.net_margin is not None:
            score += fi.net_margin
            w += 1
        if fi.revenue_growth is not None:
            score += fi.revenue_growth
            w += 1
        if fi.profit_growth is not None:
            score += fi.profit_growth
            w += 1

        return score / w if w > 0 else None

    def _momentum_factor(self, df: pd.DataFrame, trade_date: str) -> float | None:
        """Multi-period momentum: weighted average of 1M/3M/6M/12M returns."""
        close = df["close"]
        if trade_date not in close.index:
            return None

        idx = close.index.get_loc(trade_date)
        current = close.iloc[idx]
        if pd.isna(current) or current <= 0:
            return None

        periods = {
            21: 0.35,   # 1 month
            63: 0.30,   # 3 months
            126: 0.20,  # 6 months
            252: 0.15,  # 12 months
        }

        score = 0.0
        w = 0
        for days, weight in periods.items():
            if idx >= days:
                past = close.iloc[idx - days]
                if past > 0 and not pd.isna(past):
                    ret = (current - past) / past * 100
                    score += ret * weight
                    w += weight

        return score / w if w > 0 else None

    def _volatility_factor(self, df: pd.DataFrame, trade_date: str) -> float | None:
        """Lower volatility → higher score (inverse). Use 20-day daily returns std."""
        close = df["close"]
        if trade_date not in close.index:
            return None

        idx = close.index.get_loc(trade_date)
        if idx < 20:
            return None

        daily_rets = close.iloc[idx - 19 : idx + 1].pct_change().dropna()
        if len(daily_rets) < 10:
            return None

        vol = daily_rets.std() * 100  # daily return volatility in %
        return -vol  # negative: lower vol = higher score

    # ── Persistence ────────────────────────────────────────────────

    def save_factors(self, factors: pd.DataFrame, trade_date: str, db: Session | None = None) -> int:
        """Save factor scores to DB. Returns count of rows."""
        if db is None:
            db = self._get_db()
            close_db = True
        else:
            close_db = False

        # Delete existing for this date
        db.query(FactorScore).filter(FactorScore.trade_date == trade_date).delete()

        count = 0
        for code, row in factors.iterrows():
            composite = row.get("composite_score")
            if composite is None or (isinstance(composite, float) and np.isnan(composite)):
                continue
            db.add(
                FactorScore(
                    code=str(code),
                    trade_date=trade_date,
                    value_score=_nativize(row.get("value_score")),
                    quality_score=_nativize(row.get("quality_score")),
                    momentum_score=_nativize(row.get("momentum_score")),
                    volatility_score=_nativize(row.get("volatility_score")),
                    composite_score=_nativize(composite),
                )
            )
            count += 1

        db.commit()
        if close_db:
            db.close()
        return count

    # ── Helpers ────────────────────────────────────────────────────

    @staticmethod
    def _load_price_data(db: Session, codes: list[str], start: str, end: str) -> dict[str, pd.DataFrame]:
        """Load price data for factor computation."""
        rows = (
            db.query(DailyQuote)
            .filter(DailyQuote.code.in_(codes), DailyQuote.trade_date >= start, DailyQuote.trade_date <= end)
            .order_by(DailyQuote.code, DailyQuote.trade_date)
            .all()
        )
        data: dict[str, list[dict]] = {}
        for r in rows:
            if r.code not in data:
                data[r.code] = []
            data[r.code].append({"trade_date": r.trade_date, "close": r.close})

        result = {}
        for code, records in data.items():
            df = pd.DataFrame(records).sort_values("trade_date").set_index("trade_date")
            result[code] = df
        return result


def _nativize(val) -> float | None:
    """Convert numpy/pandas types to Python native float or None."""
    if val is None:
        return None
    try:
        if isinstance(val, (np.floating, np.integer)):
            return float(val.item())
        f = float(val)
        return None if np.isnan(f) else f
    except (ValueError, TypeError):
        return None
