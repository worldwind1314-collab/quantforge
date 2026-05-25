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

    # ── Granular factor computation (~65 factors for ML) ──────────────

    def compute_granular_factors(
        self, trade_date: str, codes: list[str] | None = None
    ) -> pd.DataFrame:
        """Compute ~65 granular factors for ML training (Alpha158-inspired).

        Price/volume-based factors (~63) work for ALL stocks with price history.
        Financial-based factors (2) only for stocks with financial indicators.

        Returns DataFrame indexed by code with ~65 factor columns.
        """
        db = self._get_db()

        if codes is None:
            codes = [r[0] for r in db.query(Stock.code).filter(Stock.is_active == True).all()]

        # Load 1 year of price data for all codes
        end = trade_date
        start = (date.fromisoformat(trade_date) - timedelta(days=365)).isoformat()
        price_data = self._load_price_data(db, codes, start, end)

        # Load financial indicators — Point-in-Time: only data available on or before trade_date
        from ..models.finance import FinancialIndicator

        fi_rows = (
            db.query(FinancialIndicator)
            .filter(FinancialIndicator.code.in_(codes), FinancialIndicator.report_date <= trade_date)
            .order_by(FinancialIndicator.code, FinancialIndicator.report_date.desc())
            .all()
        )
        fi_map: dict[str, FinancialIndicator] = {}
        for fi in fi_rows:
            if fi.code not in fi_map:  # first is most recent (ordered desc)
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

    # ── Factor metadata registry ─────────────────────────────────
    # Maps factor_name -> category for documentation, not used in computation
    FACTOR_CATEGORIES = {
        "mom": ["mom_5d", "mom_10d", "mom_20d", "mom_60d", "mom_120d",
                "roc_6", "roc_14", "roc_30"],
        "volatility": ["vol_5d", "vol_10d", "vol_20d", "vol_60d"],
        "turnover": ["turnover_5d", "turnover_20d", "vol_ratio_5_20",
                     "turn_ma5_ratio", "turn_ma10_ratio", "turn_ma20_ratio"],
        "volume": ["vma5_ratio", "vma10_ratio", "vma20_ratio", "vstd5", "vstd20",
                   "amount_ma5_ratio", "amount_ma10_ratio"],
        "kline": ["k_mid", "k_len", "k_up", "k_down", "k_sft", "k_ym1"],
        "deviation": ["ma5_ratio", "ma10_ratio", "ma20_ratio", "ma60_ratio",
                      "std5_ratio", "std10_ratio", "std20_ratio",
                      "max5_ratio", "max20_ratio", "min5_ratio", "min20_ratio",
                      "price_position_60d"],
        "distance": ["imax5", "imax20", "imax60", "imin5", "imin20", "imin60"],
        "technical": ["rsv_9", "rsv_14", "rsv_20", "rsi_14", "price_position_20d",
                      "max_dd_20d", "max_dd_60d",
                      "macd_dif", "macd_dea", "macd_hist",
                      "kdj_k", "kdj_d", "kdj_j",
                      "boll_pos", "boll_width", "wr_14", "cci_14",
                      "amplitude_5d", "amplitude_20d"],
        "fundamental": ["value_score", "quality_score"],
    }

    def _compute_stock_factors(
        self,
        code: str,
        trade_date: str,
        df: pd.DataFrame | None,
        fi,  # FinancialIndicator | None
    ) -> dict:
        """Compute ~65 factors for a single stock (Alpha158-inspired expansion)."""
        f = {
            # Momentum (8)
            "mom_5d": None, "mom_10d": None, "mom_20d": None,
            "mom_60d": None, "mom_120d": None,
            "roc_6": None, "roc_14": None, "roc_30": None,
            # Volatility (4)
            "vol_5d": None, "vol_10d": None, "vol_20d": None, "vol_60d": None,
            # Turnover (6)
            "turnover_5d": None, "turnover_20d": None, "vol_ratio_5_20": None,
            "turn_ma5_ratio": None, "turn_ma10_ratio": None, "turn_ma20_ratio": None,
            # Volume (7)
            "vma5_ratio": None, "vma10_ratio": None, "vma20_ratio": None,
            "vstd5": None, "vstd20": None,
            "amount_ma5_ratio": None, "amount_ma10_ratio": None,
            # K-line pattern (6)
            "k_mid": None, "k_len": None, "k_up": None, "k_down": None,
            "k_sft": None, "k_ym1": None,
            # Price deviation (12)
            "ma5_ratio": None, "ma10_ratio": None, "ma20_ratio": None, "ma60_ratio": None,
            "std5_ratio": None, "std10_ratio": None, "std20_ratio": None,
            "max5_ratio": None, "max20_ratio": None,
            "min5_ratio": None, "min20_ratio": None,
            "price_position_60d": None,
            # Distance from high/low (6)
            "imax5": None, "imax20": None, "imax60": None,
            "imin5": None, "imin20": None, "imin60": None,
            # RSV / Technical (18)
            "rsv_9": None, "rsv_14": None, "rsv_20": None,
            "rsi_14": None, "price_position_20d": None,
            "max_dd_20d": None, "max_dd_60d": None,
            "macd_dif": None, "macd_dea": None, "macd_hist": None,
            "kdj_k": None, "kdj_d": None, "kdj_j": None,
            "boll_pos": None, "boll_width": None,
            "wr_14": None, "cci_14": None,
            "amplitude_5d": None, "amplitude_20d": None,
            # Fundamental (2)
            "value_score": None, "quality_score": None,
        }

        if df is None or df.empty:
            return f

        close = df.get("close")
        high = df.get("high")
        low = df.get("low")
        open_ = df.get("open")
        volume = df.get("volume")
        amount = df.get("amount")
        turnover = df.get("turnover")
        if close is None:
            return f

        if trade_date not in df.index:
            return f

        idx = df.index.get_loc(trade_date)
        cur_close = float(close.iloc[idx])
        if pd.isna(cur_close) or cur_close <= 0:
            return f

        cur_open = _safe_val(open_, idx)
        cur_high = _safe_val(high, idx)
        cur_low = _safe_val(low, idx)

        # ── Pre-compute rolling series (reused across factors) ──
        returns = close.pct_change()
        sma5 = close.rolling(5).mean()
        sma10 = close.rolling(10).mean()
        sma20 = close.rolling(20).mean()
        sma60 = close.rolling(60).mean()
        std5 = returns.rolling(5).std()
        std10 = returns.rolling(10).std()
        std20 = returns.rolling(20).std()
        std60 = returns.rolling(60).std()
        high5 = high.rolling(5).max()
        high20 = high.rolling(20).max()
        high60 = high.rolling(60).max()
        low5 = low.rolling(5).min()
        low20 = low.rolling(20).min()
        low60 = low.rolling(60).min()
        vma5 = volume.rolling(5).mean() if volume is not None else None
        vma10 = volume.rolling(10).mean() if volume is not None else None
        vma20 = volume.rolling(20).mean() if volume is not None else None
        vstd5_s = volume.rolling(5).std() if volume is not None else None
        vstd20_s = volume.rolling(20).std() if volume is not None else None
        ama5 = amount.rolling(5).mean() if amount is not None else None
        ama10 = amount.rolling(10).mean() if amount is not None else None
        tma5 = turnover.rolling(5).mean() if turnover is not None else None
        tma10 = turnover.rolling(10).mean() if turnover is not None else None
        tma20 = turnover.rolling(20).mean() if turnover is not None else None

        # ── Momentum: N-day returns ──
        for label, n in [("mom_5d", 5), ("mom_10d", 10), ("mom_20d", 20),
                          ("mom_60d", 60), ("mom_120d", 120)]:
            if idx >= n:
                past = close.iloc[idx - n]
                if not pd.isna(past) and past > 0:
                    f[label] = (cur_close - past) / past * 100

        # ── ROC (Rate of Change, equivalent to mom but common in Alpha158) ──
        for label, n in [("roc_6", 6), ("roc_14", 14), ("roc_30", 30)]:
            if idx >= n:
                past = close.iloc[idx - n]
                if not pd.isna(past) and past > 0:
                    f[label] = (cur_close - past) / past * 100

        # ── Volatility: std of daily returns ──
        for label, n, series in [("vol_5d", 5, std5), ("vol_10d", 10, std10),
                                  ("vol_20d", 20, std20), ("vol_60d", 60, std60)]:
            val = _idx_val(series, idx)
            if val is not None and not pd.isna(val):
                f[label] = float(val * 100)

        # ── K-line pattern factors (Alpha158 K-series) ──
        if cur_open and cur_open > 0:
            f["k_mid"] = (cur_close - cur_open) / cur_open
            if cur_high and cur_low:
                f["k_len"] = (cur_high - cur_low) / cur_open
                upper_shadow = cur_high - max(cur_open, cur_close)
                lower_shadow = min(cur_open, cur_close) - cur_low
                f["k_up"] = upper_shadow / cur_open
                f["k_down"] = lower_shadow / cur_open
                hl_range = cur_high - cur_low
                if hl_range > 0:
                    f["k_sft"] = (cur_close - cur_open) / hl_range
            # Open gap from yesterday
            if idx >= 1:
                yest_close = _safe_val(close, idx - 1)
                if yest_close and yest_close > 0:
                    f["k_ym1"] = (cur_open - yest_close) / yest_close

        # ── Price deviation: close relative to MA ──
        for label, sma in [("ma5_ratio", sma5), ("ma10_ratio", sma10),
                            ("ma20_ratio", sma20), ("ma60_ratio", sma60)]:
            m = _idx_val(sma, idx)
            if m and m > 0:
                f[label] = cur_close / m

        # ── Std ratio: volatility / mean volatility ──
        for label, n, s_series in [("std5_ratio", 5, std5), ("std10_ratio", 10, std10),
                                     ("std20_ratio", 20, std20)]:
            s_val = _idx_val(s_series, idx)
            if s_val and s_val > 0 and idx >= n:
                mean_std = s_series.iloc[max(0, idx - n * 3):idx + 1].mean()
                if mean_std and mean_std > 0:
                    f[label] = s_val / mean_std

        # ── Close / N-day high & low ──
        for label, h_series in [("max5_ratio", high5), ("max20_ratio", high20)]:
            h = _idx_val(h_series, idx)
            if h and h > 0:
                f[label] = cur_close / h
        for label, l_series in [("min5_ratio", low5), ("min20_ratio", low20)]:
            l = _idx_val(l_series, idx)
            if l and l > 0:
                f[label] = cur_close / l

        # ── Distance from N-day high/low (IMAX/IMIN) ──
        for label, h_series in [("imax5", high5), ("imax20", high20), ("imax60", high60)]:
            h = _idx_val(h_series, idx)
            if h and h > 0:
                f[label] = (cur_close - h) / h  # negative = below recent high
        for label, l_series in [("imin5", low5), ("imin20", low20), ("imin60", low60)]:
            l = _idx_val(l_series, idx)
            if l and l > 0:
                f[label] = (cur_close - l) / l  # positive = above recent low

        # ── RSV (Raw Stochastic Value — KDJ sub-component) ──
        for label, n, h_s, l_s in [("rsv_9", 9, high.rolling(9).max(), low.rolling(9).min()),
                                     ("rsv_14", 14, high.rolling(14).max(), low.rolling(14).min()),
                                     ("rsv_20", 20, high.rolling(20).max(), low.rolling(20).min())]:
            if idx >= n - 1:
                h_n = _idx_val(h_s, idx)
                l_n = _idx_val(l_s, idx)
                if h_n is not None and l_n is not None and h_n > l_n:
                    f[label] = (cur_close - l_n) / (h_n - l_n) * 100

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

        # ── Price position in N-day range ──
        for label, n, h_s, l_s in [("price_position_20d", 20, high20, low20),
                                     ("price_position_60d", 60, high60, low60)]:
            if idx >= n - 1:
                h_n = _idx_val(h_s, idx)
                l_n = _idx_val(l_s, idx)
                if h_n is not None and l_n is not None and h_n > l_n:
                    f[label] = (cur_close - l_n) / (h_n - l_n)

        # ── Max drawdown ──
        if idx >= 20:
            window = close.iloc[idx - 19:idx + 1]
            f["max_dd_20d"] = self._max_drawdown(window)
        if idx >= 60:
            window = close.iloc[idx - 59:idx + 1]
            f["max_dd_60d"] = self._max_drawdown(window)

        # ── MACD (12/26/9) ──
        if idx >= 33:
            ema12 = close.ewm(span=12, adjust=False).mean()
            ema26 = close.ewm(span=26, adjust=False).mean()
            dif = ema12 - ema26
            dea = dif.ewm(span=9, adjust=False).mean()
            macd_hist_series = (dif - dea) * 2
            f["macd_dif"] = _idx_val(dif, idx)
            f["macd_dea"] = _idx_val(dea, idx)
            f["macd_hist"] = _idx_val(macd_hist_series, idx)

        # ── KDJ (9/3/3) ──
        if idx >= 8:
            h9 = high.rolling(9).max()
            l9 = low.rolling(9).min()
            rsv9 = (close - l9) / (h9 - l9 + 1e-12) * 100
            k_val = rsv9.ewm(alpha=1/3, adjust=False).mean()
            d_val = k_val.ewm(alpha=1/3, adjust=False).mean()
            j_val = 3 * k_val - 2 * d_val
            f["kdj_k"] = _idx_val(k_val, idx)
            f["kdj_d"] = _idx_val(d_val, idx)
            f["kdj_j"] = _idx_val(j_val, idx)

        # ── Bollinger Bands (20, 2) ──
        if idx >= 19:
            bb_mid = sma20
            bb_std = close.rolling(20).std()
            bb_upper = bb_mid + 2 * bb_std
            bb_lower = bb_mid - 2 * bb_std
            upper_v = _idx_val(bb_upper, idx)
            lower_v = _idx_val(bb_lower, idx)
            mid_v = _idx_val(bb_mid, idx)
            if upper_v is not None and lower_v is not None and upper_v > lower_v:
                f["boll_pos"] = (cur_close - lower_v) / (upper_v - lower_v)
            if mid_v and mid_v > 0:
                f["boll_width"] = (upper_v - lower_v) / mid_v if upper_v and lower_v else None

        # ── Williams %R (14) ──
        if idx >= 13:
            h14 = high.iloc[idx - 13:idx + 1].max()
            l14 = low.iloc[idx - 13:idx + 1].min()
            if not pd.isna(h14) and not pd.isna(l14) and h14 > l14:
                f["wr_14"] = (h14 - cur_close) / (h14 - l14) * -100

        # ── CCI (14) ──
        if idx >= 13:
            tp = (high.iloc[idx - 13:idx + 1] + low.iloc[idx - 13:idx + 1] + close.iloc[idx - 13:idx + 1]) / 3
            tp_ma = tp.mean()
            tp_md = (tp - tp_ma).abs().mean()
            cur_tp = (cur_high + cur_low + cur_close) / 3 if cur_high and cur_low else None
            if cur_tp is not None and tp_md > 0:
                f["cci_14"] = (cur_tp - tp_ma) / (0.015 * tp_md)

        # ── Amplitude (日内振幅) ──
        if high is not None and low is not None and close is not None:
            amp = (high - low) / close
            if idx >= 4:
                f["amplitude_5d"] = float(amp.iloc[idx - 4:idx + 1].mean())
            if idx >= 19:
                f["amplitude_20d"] = float(amp.iloc[idx - 19:idx + 1].mean())

        # ── Turnover ──
        if turnover is not None:
            if idx >= 4:
                f["turnover_5d"] = _idx_val(tma5, idx)
            if idx >= 19:
                f["turnover_20d"] = _idx_val(tma20, idx)
            if idx >= 19:
                t5 = _idx_val(tma5, idx)
                t20 = _idx_val(tma20, idx)
                if t5 and t20 and t20 > 0:
                    f["vol_ratio_5_20"] = t5 / t20
            for label, tma in [("turn_ma5_ratio", tma5), ("turn_ma10_ratio", tma10),
                                 ("turn_ma20_ratio", tma20)]:
                ma_val = _idx_val(tma, idx)
                cur_t = _safe_val(turnover, idx)
                if ma_val and ma_val > 0 and cur_t:
                    f[label] = cur_t / ma_val

        # ── Volume factors ──
        if volume is not None:
            cur_vol = _safe_val(volume, idx)
            for label, vma, n_days in [("vma5_ratio", vma5, 5), ("vma10_ratio", vma10, 10),
                                         ("vma20_ratio", vma20, 20)]:
                vma_val = _idx_val(vma, idx)
                if vma_val and vma_val > 0 and cur_vol:
                    f[label] = cur_vol / vma_val
            for label, vstd_series in [("vstd5", vstd5_s), ("vstd20", vstd20_s)]:
                vstd_v = _idx_val(vstd_series, idx)
                if vstd_v:
                    f[label] = vstd_v

        # ── Amount factors ──
        if amount is not None:
            cur_amt = _safe_val(amount, idx)
            for label, ama in [("amount_ma5_ratio", ama5), ("amount_ma10_ratio", ama10)]:
                ama_val = _idx_val(ama, idx)
                if ama_val and ama_val > 0 and cur_amt:
                    f[label] = cur_amt / ama_val

        # ── Value / Quality (needs financial data) ──
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

        if pe is not None and 0 < pe < 500:  # exclude negative PE and extreme outliers
            score += -pe
            w += 1
        if pb is not None and 0 < pb < 50:
            score += -pb * 10
            w += 1
        if ps is not None and 0 < ps < 30:
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

        Returns DataFrame indexed by code with ~65 factor columns.
        """
        db = self._get_db()

        # Load financial indicators — Point-in-Time: only data available on or before trade_date
        from ..models.finance import FinancialIndicator

        fi_rows = (
            db.query(FinancialIndicator)
            .filter(FinancialIndicator.code.in_(codes), FinancialIndicator.report_date <= trade_date)
            .order_by(FinancialIndicator.code, FinancialIndicator.report_date.desc())
            .all()
        )
        fi_map: dict = {}
        for fi in fi_rows:
            if fi.code not in fi_map:  # first is most recent (ordered desc)
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


def _safe_val(series: pd.Series | None, idx: int) -> float | None:
    """Safely extract a float value from a series at a given positional index."""
    if series is None or idx < 0 or idx >= len(series):
        return None
    try:
        v = series.iloc[idx]
        if pd.isna(v):
            return None
        return float(v)
    except (ValueError, TypeError):
        return None


def _idx_val(series: pd.Series | None, idx: int) -> float | None:
    """Extract a pre-computed rolling/transformed series value at positional index."""
    if series is None or idx < 0 or idx >= len(series):
        return None
    try:
        v = series.iloc[idx]
        if pd.isna(v):
            return None
        return float(v)
    except (ValueError, TypeError):
        return None
