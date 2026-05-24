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

    # ── Granular factor computation (17 factors for ML) ──────────────

    def compute_granular_factors(
        self, trade_date: str, codes: list[str] | None = None
    ) -> pd.DataFrame:
        """Compute 17 granular factors for ML training.

        Price-based factors (14) work for ALL stocks with price history.
        Financial-based factors (3) only for stocks with financial indicators.

        Returns DataFrame indexed by code with 17 factor columns.
        """
        db = self._get_db()

        if codes is None:
            codes = [r[0] for r in db.query(Stock.code).filter(Stock.is_active == True).all()]

        # Load 1 year of price data for all codes
        end = trade_date
        start = (date.fromisoformat(trade_date) - timedelta(days=365)).isoformat()
        price_data = self._load_price_data(db, codes, start, end)

        # Load financial indicators for value/quality factors
        from ..models.finance import FinancialIndicator

        fi_rows = (
            db.query(FinancialIndicator)
            .filter(FinancialIndicator.code.in_(codes))
            .all()
        )
        fi_map: dict[str, FinancialIndicator] = {}
        for fi in fi_rows:
            if fi.code not in fi_map:  # keep most recent per code
                fi_map[fi.code] = fi

        rows = []
        for code in codes:
            factors = self._compute_stock_factors(code, trade_date, price_data.get(code), fi_map.get(code))
            factors["code"] = code
            rows.append(factors)

        result = pd.DataFrame(rows).set_index("code")

        # Cross-sectional z-score normalization (per factor, across all stocks)
        for col in result.columns:
            valid = result[col].dropna()
            if len(valid) > 30:
                mean, std = valid.mean(), valid.std()
                if std > 0 and not np.isnan(std):
                    result[col] = (result[col] - mean) / std
                    # Clip extreme values
                    result[col] = result[col].clip(-5, 5)

        return result

    def _compute_stock_factors(
        self,
        code: str,
        trade_date: str,
        df: pd.DataFrame | None,
        fi,  # FinancialIndicator | None
    ) -> dict:
        """Compute all 17 factors for a single stock."""
        f = {
            # Momentum factors
            "mom_5d": None, "mom_10d": None, "mom_20d": None,
            "mom_60d": None, "mom_120d": None,
            # Volatility factors
            "vol_5d": None, "vol_20d": None, "vol_60d": None,
            # Volume / liquidity
            "turnover_5d": None, "turnover_20d": None, "vol_ratio_5_20": None,
            # Technical
            "rsi_14": None, "price_position_20d": None,
            "max_dd_20d": None, "max_dd_60d": None,
            # Financial
            "value_score": None, "quality_score": None,
        }

        if df is None or df.empty:
            return f

        close = df.get("close")
        turnover = df.get("turnover")
        high = df.get("high")
        low = df.get("low")
        if close is None:
            return f

        if trade_date not in df.index:
            return f

        idx = df.index.get_loc(trade_date)
        cur_close = float(close.iloc[idx])
        if pd.isna(cur_close) or cur_close <= 0:
            return f

        # ── Momentum: N-day returns ──
        for label, n in [("mom_5d", 5), ("mom_10d", 10), ("mom_20d", 20),
                          ("mom_60d", 60), ("mom_120d", 120)]:
            if idx >= n:
                past = close.iloc[idx - n]
                if not pd.isna(past) and past > 0:
                    f[label] = (cur_close - past) / past * 100

        # ── Volatility: std of daily returns ──
        if idx >= 5:
            rets = close.iloc[max(0, idx - 4):idx + 1].pct_change().dropna()
            if len(rets) >= 3:
                f["vol_5d"] = float(rets.std() * 100)
        if idx >= 20:
            rets = close.iloc[max(0, idx - 19):idx + 1].pct_change().dropna()
            if len(rets) >= 10:
                f["vol_20d"] = float(rets.std() * 100)
        if idx >= 60:
            rets = close.iloc[max(0, idx - 59):idx + 1].pct_change().dropna()
            if len(rets) >= 30:
                f["vol_60d"] = float(rets.std() * 100)

        # ── Turnover ──
        if turnover is not None:
            def _safe_mean(series, start, end):
                window = series.iloc[start:end].dropna()
                return float(window.mean()) if len(window) > 0 else None

            if idx >= 4:
                f["turnover_5d"] = _safe_mean(turnover, idx - 4, idx + 1)
            if idx >= 19:
                f["turnover_20d"] = _safe_mean(turnover, idx - 19, idx + 1)
            if idx >= 19:
                t5 = _safe_mean(turnover, idx - 4, idx + 1)
                t20 = _safe_mean(turnover, idx - 19, idx + 1)
                if t5 and t20 and t20 > 0:
                    f["vol_ratio_5_20"] = t5 / t20

        # ── RSI (14-day) ──
        if idx >= 14:
            window = close.iloc[idx - 13:idx + 1]
            deltas = window.diff().dropna()
            gains = deltas.clip(lower=0).mean()
            losses = (-deltas.clip(upper=0)).mean()
            if losses > 0:
                rs = gains / losses
                f["rsi_14"] = 100.0 - (100.0 / (1.0 + rs))
            elif gains > 0:
                f["rsi_14"] = 100.0

        # ── Price position in 20-day range ──
        if high is not None and low is not None and idx >= 20:
            h20 = high.iloc[idx - 19:idx + 1].max()
            l20 = low.iloc[idx - 19:idx + 1].min()
            if not pd.isna(h20) and not pd.isna(l20) and h20 > l20:
                f["price_position_20d"] = (cur_close - l20) / (h20 - l20)

        # ── Max drawdown ──
        if idx >= 20:
            window = close.iloc[idx - 19:idx + 1]
            f["max_dd_20d"] = self._max_drawdown(window)
        if idx >= 60:
            window = close.iloc[idx - 59:idx + 1]
            f["max_dd_60d"] = self._max_drawdown(window)

        # ── Value factor (needs financial data) ──
        if fi is not None:
            f["value_score"] = self._calc_value_factor(fi, cur_close)
            f["quality_score"] = self._calc_quality_factor(fi)

        return f

    @staticmethod
    def _max_drawdown(prices: pd.Series) -> float:
        """Compute maximum drawdown as a negative percentage."""
        prices = prices.dropna()
        if len(prices) < 5:
            return 0.0
        peak = prices.expanding().max()
        dd = (prices - peak) / peak
        return float(dd.min() * 100)  # negative number

    @staticmethod
    def _calc_value_factor(fi, current_price: float) -> float | None:
        """Lower PE/PB/PS → higher value score."""
        score = 0.0
        w = 0

        pe = fi.pe
        pb = fi.pb
        ps = fi.ps

        if current_price > 0:
            if pe is None and fi.eps and fi.eps > 0:
                pe = current_price / fi.eps
            if pb is None and fi.bv_per_share and fi.bv_per_share > 0:
                pb = current_price / fi.bv_per_share
            if ps is None and fi.revenue_per_share and fi.revenue_per_share > 0:
                ps = current_price / fi.revenue_per_share

        if pe and pe > 0:
            score += -pe
            w += 1
        if pb and pb > 0:
            score += -pb * 10
            w += 1
        if ps and ps > 0:
            score += -ps * 5
            w += 1

        return score / w if w > 0 else None

    @staticmethod
    def _calc_quality_factor(fi) -> float | None:
        """Higher ROE, margins, growth → higher quality score."""
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

    def compute_granular_factors_from_prices(
        self, trade_date: str, codes: list[str], price_data: dict[str, pd.DataFrame]
    ) -> pd.DataFrame:
        """Compute granular factors using pre-loaded price data (avoids DB queries).

        Args:
            trade_date: The snapshot date.
            codes: Stock codes to compute factors for.
            price_data: {code: DataFrame} with trade_date index, pre-filtered to lookback window.

        Returns DataFrame indexed by code with 17 factor columns.
        """
        db = self._get_db()

        # Load financial indicators
        from ..models.finance import FinancialIndicator

        fi_rows = (
            db.query(FinancialIndicator)
            .filter(FinancialIndicator.code.in_(codes))
            .all()
        )
        fi_map: dict = {}
        for fi in fi_rows:
            if fi.code not in fi_map:
                fi_map[fi.code] = fi

        rows = []
        for code in codes:
            factors = self._compute_stock_factors(
                code, trade_date, price_data.get(code), fi_map.get(code)
            )
            factors["code"] = code
            rows.append(factors)

        result = pd.DataFrame(rows).set_index("code")

        # Cross-sectional z-score normalization
        for col in result.columns:
            valid = result[col].dropna()
            if len(valid) > 30:
                mean, std = valid.mean(), valid.std()
                if std > 0 and not np.isnan(std):
                    result[col] = (result[col] - mean) / std
                    result[col] = result[col].clip(-5, 5)

        return result

    # ── Legacy factor computation (for dashboard / backward compat) ──

    def compute_all_factors(self, trade_date: str, codes: list[str] | None = None) -> pd.DataFrame:
        """Compute composite factors for all stocks. Returns DataFrame indexed by code."""
        db = self._get_db()

        if codes is None:
            codes = [r[0] for r in db.query(Stock.code).filter(Stock.is_active == True).all()]

        # Use granular factor computation then aggregate
        granular = self.compute_granular_factors(trade_date, codes)

        result = pd.DataFrame(index=granular.index)
        result["value_score"] = granular.get("value_score", np.nan)
        result["quality_score"] = granular.get("quality_score", np.nan)

        # Aggregate momentum from sub-factors
        mom_cols = ["mom_5d", "mom_10d", "mom_20d", "mom_60d", "mom_120d"]
        available_mom = [c for c in mom_cols if c in granular.columns]
        if available_mom:
            result["momentum_score"] = granular[available_mom].mean(axis=1, skipna=True)
            # Normalize
            valid = result["momentum_score"].dropna()
            if len(valid) > 10:
                m, s = valid.mean(), valid.std()
                if s > 0:
                    result["momentum_score"] = (result["momentum_score"] - m) / s

        # Aggregate volatility from sub-factors (invert: lower vol = higher score)
        vol_cols = ["vol_5d", "vol_20d", "vol_60d"]
        available_vol = [c for c in vol_cols if c in granular.columns]
        if available_vol:
            raw_vol = granular[available_vol].mean(axis=1, skipna=True)
            valid = raw_vol.dropna()
            if len(valid) > 10:
                m, s = valid.mean(), valid.std()
                if s > 0:
                    raw_vol = (raw_vol - m) / s
            result["volatility_score"] = -raw_vol

        # Composite: equal-weighted sum of normalized scores
        score_cols = ["value_score", "quality_score", "momentum_score", "volatility_score"]
        result["composite_score"] = result[score_cols].mean(axis=1, skipna=True)

        return result

    # ── Persistence ────────────────────────────────────────────────

    def save_factors(self, factors: pd.DataFrame, trade_date: str, db: Session | None = None) -> int:
        """Save factor scores to DB. Returns count of rows."""
        if db is None:
            db = self._get_db()
            close_db = True
        else:
            close_db = False

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
            data[r.code].append({
                "trade_date": r.trade_date, "open": r.open, "high": r.high,
                "low": r.low, "close": r.close, "volume": r.volume,
                "amount": r.amount, "turnover": r.turnover,
            })

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
