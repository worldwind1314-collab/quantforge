"""ML pipeline — LightGBM model for cross-sectional stock ranking."""

import json
import logging
from datetime import date, timedelta

import numpy as np
import pandas as pd
from sqlalchemy.orm import Session

from ..core.database import SessionLocal
from ..models.finance import FactorScore, MLPrediction
from ..models.market import DailyQuote
from ..models.stock import Stock

logger = logging.getLogger(__name__)

# Feature columns produced by FactorEngine.compute_granular_factors()
FEATURE_COLS = [
    "mom_5d", "mom_10d", "mom_20d", "mom_60d", "mom_120d",
    "vol_5d", "vol_20d", "vol_60d",
    "turnover_5d", "turnover_20d", "vol_ratio_5_20",
    "rsi_14", "price_position_20d", "max_dd_20d", "max_dd_60d",
    "value_score", "quality_score",
]


class MLPipeline:
    """LightGBM-based stock ranking pipeline.

    Trains on granular factors + cross-sectional rank targets,
    predicts next-period relative ranking for all stocks.
    """

    def __init__(self, db: Session | None = None):
        self._db = db
        self._model = None
        self._feature_names = None

    def _get_db(self) -> Session:
        return self._db or SessionLocal()

    # ── Data preparation ───────────────────────────────────────────

    def prepare_training_data(
        self, start_date: str, end_date: str
    ) -> tuple[np.ndarray, np.ndarray]:
        """Build training dataset from granular factors for ALL stocks with price data.

        Target: cross-sectional rank percentile of forward 20-day return.

        Optimizations:
        - One bulk DB query for all price data (not per-stock)
        - Forward returns computed in-memory from pre-loaded prices
        - Bi-weekly snapshots to balance sample count vs computation
        """
        db = self._get_db()

        # Get ALL stocks with price data
        quote_codes = sorted(
            r[0] for r in db.query(DailyQuote.code).distinct().all()
        )
        if not quote_codes:
            raise ValueError("No stocks with price data")

        logger.info(f"Training universe: {len(quote_codes)} stocks")

        # Load ALL price data in one bulk query
        # Need: start_date - 1 year (for factor lookback) to end_date + 30 days (for forward return)
        data_start = (date.fromisoformat(start_date) - timedelta(days=400)).isoformat()
        data_end = (date.fromisoformat(end_date) + timedelta(days=40)).isoformat()

        logger.info(f"Bulk-loading price data {data_start} ~ {data_end}...")
        rows = (
            db.query(DailyQuote)
            .filter(
                DailyQuote.code.in_(quote_codes),
                DailyQuote.trade_date >= data_start,
                DailyQuote.trade_date <= data_end,
            )
            .order_by(DailyQuote.code, DailyQuote.trade_date)
            .all()
        )
        logger.info(f"Loaded {len(rows)} quote rows")

        # Build per-stock DataFrames
        price_data: dict[str, pd.DataFrame] = {}
        for r in rows:
            if r.code not in price_data:
                price_data[r.code] = []
            price_data[r.code].append({
                "trade_date": r.trade_date, "open": r.open, "high": r.high,
                "low": r.low, "close": r.close, "volume": r.volume,
                "amount": r.amount, "turnover": r.turnover,
            })

        for code in price_data:
            df = pd.DataFrame(price_data[code])
            df = df.sort_values("trade_date").set_index("trade_date")
            price_data[code] = df

        logger.info(f"Built DataFrames for {len(price_data)} stocks")

        # Collect all unique trading dates from price data within the training range
        all_trading_dates = set()
        for df in price_data.values():
            all_trading_dates.update(df.index.tolist())
        trading_dates = sorted(
            d for d in all_trading_dates
            if start_date <= d <= end_date
        )
        # Take every 10th trading day (approximately bi-weekly)
        snapshot_dates = trading_dates[::10]
        if not snapshot_dates:
            raise ValueError("No trading dates found in training range")

        logger.info(f"Computing factors for {len(snapshot_dates)} snapshots...")

        from .factor_engine import FactorEngine

        all_samples = []
        engine = FactorEngine(db)

        for i, snap_date in enumerate(snapshot_dates):
            try:
                # Build price data subset for this snapshot
                # (1 year lookback from snap_date)
                lookback_start = (date.fromisoformat(snap_date) - timedelta(days=365)).isoformat()
                codes_for_snapshot = []
                snapshot_prices = {}

                for code, df in price_data.items():
                    if snap_date in df.index:
                        # Filter to 1-year lookback window
                        mask = (df.index >= lookback_start) & (df.index <= snap_date)
                        subset = df.loc[mask]
                        if len(subset) >= 10:  # need min data for factors
                            snapshot_prices[code] = subset
                            codes_for_snapshot.append(code)

                if len(codes_for_snapshot) < 50:
                    continue

                # Compute granular factors using pre-loaded prices
                factors_df = engine.compute_granular_factors_from_prices(
                    snap_date, codes_for_snapshot, snapshot_prices
                )

                if factors_df.empty:
                    continue

                # Compute forward returns from pre-loaded prices
                forward_rets = {}
                for code in factors_df.index:
                    df = price_data.get(code)
                    if df is None:
                        continue
                    ret = self._forward_return_from_prices(df, snap_date, horizon=20)
                    if ret is not None:
                        forward_rets[code] = ret

                if len(forward_rets) < 50:
                    continue

                # Cross-sectional rank percentile
                ret_series = pd.Series(forward_rets)
                ranks = ret_series.rank(pct=True)

                for code in factors_df.index:
                    if code not in forward_rets:
                        continue
                    feats = {}
                    for col in FEATURE_COLS:
                        val = factors_df.loc[code, col] if col in factors_df.columns else None
                        feats[col] = val if not (isinstance(val, float) and np.isnan(val)) else None
                    feats["target"] = ranks[code]
                    feats["code"] = code
                    feats["trade_date"] = snap_date
                    all_samples.append(feats)

                if (i + 1) % 5 == 0:
                    logger.info(f"  Snapshot {i+1}/{len(snapshot_dates)}: {snap_date}, "
                                f"{len(forward_rets)} stocks, {len(all_samples)} total samples")

            except Exception as e:
                logger.warning(f"Snapshot {snap_date} failed: {e}")
                continue

        if len(all_samples) < 100:
            raise ValueError(f"Not enough training samples: {len(all_samples)}")

        df = pd.DataFrame(all_samples)
        X = df[FEATURE_COLS].values.astype(np.float32)
        y = df["target"].values.astype(np.float32)

        logger.info(f"Training data: {len(X)} samples, {len(FEATURE_COLS)} features")
        logger.info(f"Target stats: mean={y.mean():.3f}, std={y.std():.3f}, "
                    f"min={y.min():.3f}, max={y.max():.3f}")

        self._feature_names = FEATURE_COLS
        return X, y

    # ── Model training ─────────────────────────────────────────────

    def train(self, start_date: str = "2024-01-01", end_date: str = "2026-05-01") -> dict:
        """Train LightGBM model. Returns training metrics."""
        import lightgbm as lgb

        X, y = self.prepare_training_data(start_date, end_date)

        # Time-series split: last 20% for validation
        split = int(len(X) * 0.8)
        X_train, X_val = X[:split], X[split:]
        y_train, y_val = y[:split], y[split:]

        logger.info(f"Train: {len(X_train)} samples, Val: {len(X_val)} samples")

        # LightGBM with ranking-aware parameters
        n_est = min(500, max(100, len(X_train) // 5))
        num_leaves = min(63, max(15, len(X_train) // 200))

        self._model = lgb.LGBMRegressor(
            n_estimators=n_est,
            num_leaves=num_leaves,
            learning_rate=0.03,
            subsample=0.7,
            subsample_freq=1,
            colsample_bytree=0.7,
            reg_alpha=0.1,
            reg_lambda=0.5,
            min_child_samples=20,
            random_state=42,
            n_jobs=-1,
            verbose=-1,
        )
        self._model.fit(
            X_train, y_train,
            eval_set=[(X_val, y_val)],
            eval_metric="rmse",
            callbacks=[lgb.early_stopping(50, verbose=False), lgb.log_evaluation(20)],
        )

        # Evaluate
        pred_train = self._model.predict(X_train)
        pred_val = self._model.predict(X_val)

        import scipy.stats as stats

        train_ic = stats.spearmanr(y_train, pred_train)[0]
        val_ic = stats.spearmanr(y_val, pred_val)[0]

        # Also compute IC using raw forward returns (more interpretable)
        # For ranking target, IC on the ranks is what matters
        val_mse = np.mean((y_val - pred_val) ** 2)
        val_mae = np.mean(np.abs(y_val - pred_val))

        # Feature importance
        importance = dict(
            zip(
                FEATURE_COLS,
                self._model.feature_importances_.tolist(),
            )
        )
        # Sort by importance
        importance = dict(sorted(importance.items(), key=lambda x: x[1], reverse=True))

        # Compute IC by date (within each cross-section)
        # We don't have dates in the split arrays, so skip for now

        metrics = {
            "train_samples": len(X_train),
            "val_samples": len(X_val),
            "train_ic": round(train_ic, 4),
            "val_ic": round(val_ic, 4),
            "val_mse": round(val_mse, 4),
            "val_mae": round(val_mae, 4),
            "feature_importance": importance,
            "n_features": len(FEATURE_COLS),
            "best_iteration": self._model.best_iteration_,
        }

        logger.info(f"Model trained: IC={val_ic:.4f}, best_iter={self._model.best_iteration_}")
        return metrics

    # ── Prediction ─────────────────────────────────────────────────

    def predict(self, trade_date: str, top_n: int = 30) -> list[dict]:
        """Generate predictions for all stocks on a given date. Returns top-N ranked."""
        if self._model is None:
            raise RuntimeError("Model not trained. Call train() first.")

        db = self._get_db()

        from .factor_engine import FactorEngine

        engine = FactorEngine(db)

        # Get all stocks with price data
        codes = sorted(
            r[0] for r in db.query(DailyQuote.code)
            .filter(DailyQuote.trade_date == trade_date)
            .distinct().all()
        )
        if not codes:
            # Fallback: get all active stocks
            codes = sorted(
                r[0] for r in db.query(Stock.code).filter(Stock.is_active == True).all()
            )

        logger.info(f"Computing factors for {len(codes)} stocks on {trade_date}")
        factors_df = engine.compute_granular_factors(trade_date, codes=codes)

        # Build feature matrix
        X = np.zeros((len(factors_df), len(FEATURE_COLS)), dtype=np.float32)
        valid_codes = []
        for i, code in enumerate(factors_df.index):
            feats = []
            for col in FEATURE_COLS:
                val = factors_df.loc[code, col] if col in factors_df.columns else np.nan
                feats.append(val if not (isinstance(val, float) and np.isnan(val)) else np.nan)
            X[i] = feats
            valid_codes.append(code)

        # Fill NaN with 0 for prediction (LightGBM handles NaN but some versions don't)
        X = np.nan_to_num(X, nan=0.0)
        preds = self._model.predict(X)

        # Build results
        results = []
        for code, pred in zip(valid_codes, preds):
            results.append({
                "code": code,
                "trade_date": trade_date,
                "predicted_return": round(float(pred), 4),
                "confidence": 0.5,  # simplified
            })

        results.sort(key=lambda x: x["predicted_return"], reverse=True)
        for rank, r in enumerate(results, 1):
            r["prediction_rank"] = rank

        self._save_predictions(db, results)
        return results[:top_n]

    # ── Market regime detection ────────────────────────────────────

    def detect_market_regime(self) -> dict:
        """Detect current market regime using CSI 000001 as proxy."""
        db = self._get_db()
        try:
            rows = (
                db.query(DailyQuote)
                .filter(DailyQuote.code == "000001", DailyQuote.trade_date >= "2025-01-01")
                .order_by(DailyQuote.trade_date.desc())
                .limit(120)
                .all()
            )
            if len(rows) < 60:
                return {"regime": "unknown", "confidence": 0}

            closes = [r.close for r in rows if r.close]
            if len(closes) < 60:
                return {"regime": "unknown", "confidence": 0}

            ma20 = np.mean(closes[:20])
            ma60 = np.mean(closes[:60])
            ma120 = np.mean(closes[:120]) if len(closes) >= 120 else ma60

            current = closes[0]
            above_short = current > ma20
            above_mid = ma20 > ma60
            above_long = ma60 > ma120 if len(closes) >= 120 else True

            score = sum([above_short, above_mid, above_long])

            if score == 3:
                regime, confidence = "bull", 0.8
            elif score == 2:
                regime, confidence = "bullish", 0.6
            elif score == 0:
                regime, confidence = "bear", 0.8
            elif score == 1:
                regime, confidence = "bearish", 0.6
            else:
                regime, confidence = "neutral", 0.5

            return {
                "regime": regime,
                "confidence": confidence,
                "current_price": current,
                "ma20": round(ma20, 2),
                "ma60": round(ma60, 2),
                "signal": "buy" if regime in ("bull", "bullish") else (
                    "sell" if regime in ("bear", "bearish") else "hold"
                ),
            }
        except Exception as e:
            logger.warning(f"Regime detection failed: {e}")
            return {"regime": "unknown", "confidence": 0, "error": str(e)}

    # ── Helpers ────────────────────────────────────────────────────

    @staticmethod
    def _forward_return_from_prices(df: pd.DataFrame, trade_date: str, horizon: int = 20) -> float | None:
        """Compute forward N-day return from pre-loaded price DataFrame."""
        if df is None or trade_date not in df.index:
            return None

        close = df["close"]
        idx = close.index.get_loc(trade_date)

        if idx + horizon >= len(close):
            return None

        cur = close.iloc[idx]
        fut = close.iloc[idx + horizon]

        if pd.isna(cur) or pd.isna(fut) or cur <= 0 or fut <= 0:
            return None

        return (fut - cur) / cur * 100

    @staticmethod
    def _get_forward_return(db: Session, code: str, trade_date: str, horizon: int = 20) -> float | None:
        """Compute forward N-trading-day return for a stock."""
        future = (
            db.query(DailyQuote)
            .filter(DailyQuote.code == code, DailyQuote.trade_date > trade_date)
            .order_by(DailyQuote.trade_date)
            .limit(horizon + 1)
            .all()
        )
        if len(future) < horizon:
            return None

        current = (
            db.query(DailyQuote)
            .filter(DailyQuote.code == code, DailyQuote.trade_date == trade_date)
            .first()
        )
        if not current or not current.close or current.close <= 0:
            return None

        target = future[horizon - 1]
        if not target.close or target.close <= 0:
            return None

        return (target.close - current.close) / current.close * 100

    @staticmethod
    def _save_predictions(db: Session, results: list[dict]):
        """Save predictions to DB."""
        for r in results:
            db.query(MLPrediction).filter(
                MLPrediction.code == r["code"],
                MLPrediction.trade_date == r["trade_date"],
            ).delete()

        for r in results:
            db.add(
                MLPrediction(
                    code=r["code"],
                    trade_date=r["trade_date"],
                    predicted_return=r["predicted_return"],
                    prediction_rank=r["prediction_rank"],
                    confidence=r["confidence"],
                )
            )
        db.commit()
