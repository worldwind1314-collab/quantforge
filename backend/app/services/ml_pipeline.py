"""ML pipeline — XGBoost model training, prediction, stock ranking."""

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


class MLPipeline:
    """XGBoost-based stock ranking pipeline.

    Trains on factor scores + future returns, predicts next-period returns,
    and ranks stocks for portfolio selection.
    """

    def __init__(self, db: Session | None = None):
        self._db = db
        self._model = None

    def _get_db(self) -> Session:
        return self._db or SessionLocal()

    # ── Data preparation ───────────────────────────────────────────

    def prepare_training_data(
        self, start_date: str, end_date: str
    ) -> tuple[np.ndarray, np.ndarray]:
        """Build training dataset: compute factors from daily_quotes,
        using weekly snapshots to reduce computation.

        Returns (X, y) as numpy arrays.
        """
        db = self._get_db()

        # Always compute factors from daily_quotes to ensure consistent coverage.
        # FactorScore table may have sparse/inconsistent data.
        try:
            return self._build_from_quotes(db, start_date, end_date)
        except ValueError:
            # Fallback: try pre-computed FactorScore
            factors = (
                db.query(FactorScore)
                .filter(FactorScore.trade_date >= start_date, FactorScore.trade_date <= end_date)
                .order_by(FactorScore.trade_date, FactorScore.code)
                .all()
            )
            if len(factors) >= 50:
                return self._build_from_factors(db, factors)
            raise ValueError(f"Not enough factor data: {len(factors)} rows")

    def _build_from_factors(self, db: Session, factors: list) -> tuple[np.ndarray, np.ndarray]:
        factor_data = []
        for f in factors:
            factor_data.append({
                "code": f.code, "trade_date": f.trade_date,
                "value": f.value_score or 0, "quality": f.quality_score or 0,
                "momentum": f.momentum_score or 0, "volatility": f.volatility_score or 0,
                "composite": f.composite_score or 0,
            })
        df = pd.DataFrame(factor_data)
        df["trade_date"] = pd.to_datetime(df["trade_date"])
        return self._build_features_labels(db, df)

    def _build_from_quotes(self, db: Session, start_date: str, end_date: str) -> tuple[np.ndarray, np.ndarray]:
        """Compute factors directly from daily_quotes using weekly snapshots."""
        from .factor_engine import FactorEngine

        engine = FactorEngine(db)

        # Get stocks that have price data + financial data
        from ..models.finance import FinancialIndicator

        fi_codes = set(r[0] for r in db.query(FinancialIndicator.code).distinct().all())
        quote_codes = set(r[0] for r in db.query(DailyQuote.code).distinct().all())
        codes = list(fi_codes & quote_codes)

        if not codes:
            raise ValueError("No stocks with both financial and price data")

        # Use weekly snapshots to maximize training samples from limited stock universe
        from datetime import date, timedelta

        all_data = []
        current = date.fromisoformat(end_date)
        start = date.fromisoformat(start_date)
        while current >= start:
            date_str = current.isoformat()
            try:
                factors = engine.compute_all_factors(date_str, codes=codes)
                engine.save_factors(factors, date_str, db)
                for code, row in factors.iterrows():
                    if row.get("composite_score") is None:
                        continue
                    all_data.append({
                        "code": str(code), "trade_date": date_str,
                        "value": _nativize_ml(row.get("value_score")) or 0,
                        "quality": _nativize_ml(row.get("quality_score")) or 0,
                        "momentum": _nativize_ml(row.get("momentum_score")) or 0,
                        "volatility": _nativize_ml(row.get("volatility_score")) or 0,
                        "composite": _nativize_ml(row.get("composite_score")) or 0,
                    })
            except Exception as e:
                logger.debug(f"No factor data for {date_str}: {e}")
            current -= timedelta(days=7)

        if len(all_data) < 20:
            raise ValueError(f"Not enough training samples: {len(all_data)}")

        df = pd.DataFrame(all_data)
        df["trade_date"] = pd.to_datetime(df["trade_date"])
        logger.info(f"Built {len(df)} training samples from daily_quotes")
        return self._build_features_labels(db, df)

    def _build_features_labels(self, db: Session, df: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
        labels, features = [], []
        for _, row in df.iterrows():
            forward_ret = self._get_forward_return(
                db, row["code"], row["trade_date"].strftime("%Y-%m-%d"), horizon=20
            )
            if forward_ret is not None:
                features.append([
                    row["value"], row["quality"], row["momentum"],
                    row["volatility"], row["composite"],
                ])
                labels.append(forward_ret)

        if len(features) < 20:
            raise ValueError(f"Not enough training samples: {len(features)}")

        y = np.array(labels)
        lo, hi = np.percentile(y, [1, 99])
        mask = (y >= lo) & (y <= hi)
        X, y = np.array(features)[mask], y[mask]

        logger.info(f"Training data: {len(X)} samples, 5 features")
        return X, y

    # ── Model training ─────────────────────────────────────────────

    def train(self, start_date: str = "2024-01-01", end_date: str = "2026-05-01") -> dict:
        """Train XGBoost model on historical factor data. Returns training metrics."""
        X, y = self.prepare_training_data(start_date, end_date)

        # Train/validation split (time-series, last 20% for validation)
        split = int(len(X) * 0.8)
        X_train, X_val = X[:split], X[split:]
        y_train, y_val = y[:split], y[split:]

        import xgboost as xgb

        # Adjust model complexity to dataset size
        n_est = min(200, max(50, len(X_train) // 2))
        max_d = min(5, max(2, len(X_train) // 40))

        self._model = xgb.XGBRegressor(
            n_estimators=n_est,
            max_depth=max_d,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            reg_alpha=0.1,
            reg_lambda=1.0,
            random_state=42,
            n_jobs=-1,
        )
        self._model.fit(
            X_train, y_train,
            eval_set=[(X_val, y_val)],
            verbose=False,
        )

        # Evaluate
        pred_train = self._model.predict(X_train)
        pred_val = self._model.predict(X_val)

        from sklearn.metrics import mean_squared_error, r2_score
        import scipy.stats as stats

        train_ic = stats.spearmanr(y_train, pred_train)[0]
        val_ic = stats.spearmanr(y_val, pred_val)[0]
        val_rmse = np.sqrt(mean_squared_error(y_val, pred_val))
        val_r2 = r2_score(y_val, pred_val)

        # Feature importance
        importance = dict(
            zip(
                ["value", "quality", "momentum", "volatility", "composite"],
                self._model.feature_importances_.tolist(),
            )
        )

        metrics = {
            "train_samples": len(X_train),
            "val_samples": len(X_val),
            "train_ic": round(train_ic, 4),
            "val_ic": round(val_ic, 4),
            "val_rmse": round(val_rmse, 4),
            "val_r2": round(val_r2, 4),
            "feature_importance": importance,
        }

        logger.info(f"Model trained: IC={val_ic:.4f}, R²={val_r2:.4f}")
        return metrics

    # ── Prediction ─────────────────────────────────────────────────

    def predict(self, trade_date: str, top_n: int = 30) -> list[dict]:
        """Generate predictions for all stocks on a given date. Returns top-N ranked."""
        if self._model is None:
            raise RuntimeError("Model not trained. Call train() first.")

        db = self._get_db()
        factors = (
            db.query(FactorScore)
            .filter(FactorScore.trade_date == trade_date)
            .all()
        )

        if not factors:
            raise ValueError(f"No factor data for {trade_date}")

        codes = [f.code for f in factors]
        X = np.array(
            [[f.value_score or 0, f.quality_score or 0, f.momentum_score or 0, f.volatility_score or 0, f.composite_score or 0] for f in factors]
        )

        preds = self._model.predict(X)

        # Build ranking
        results = []
        for i, (code, pred) in enumerate(zip(codes, preds)):
            results.append(
                {
                    "code": code,
                    "trade_date": trade_date,
                    "predicted_return": round(float(pred), 4),
                    "confidence": self._estimate_confidence(factors[i]),
                }
            )

        results.sort(key=lambda x: x["predicted_return"], reverse=True)

        # Assign ranks
        for rank, r in enumerate(results, 1):
            r["prediction_rank"] = rank

        # Save to DB
        self._save_predictions(db, results)

        return results[:top_n]

    # ── Market regime detection ────────────────────────────────────

    def detect_market_regime(self) -> dict:
        """Detect current market regime: bull, bear, or neutral.

        Uses 60-day moving average of CSI 000001 (SZ) as proxy for A-share market.
        """
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

            # Regime detection
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
                "signal": "buy" if regime in ("bull", "bullish") else ("sell" if regime in ("bear", "bearish") else "hold"),
            }
        except Exception as e:
            logger.warning(f"Regime detection failed: {e}")
            return {"regime": "unknown", "confidence": 0, "error": str(e)}

    # ── Helpers ────────────────────────────────────────────────────

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

        # Get close from trade_date
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
    def _estimate_confidence(factor: FactorScore) -> float:
        """Estimate prediction confidence based on factor availability."""
        available = sum(
            1
            for x in [
                factor.value_score,
                factor.quality_score,
                factor.momentum_score,
                factor.volatility_score,
            ]
            if x is not None
        )
        return min(available / 4, 1.0)

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


def _nativize_ml(val) -> float | None:
    """Convert numpy types to Python native float or None."""
    if val is None:
        return None
    try:
        if hasattr(val, "item"):
            return float(val.item())
        f = float(val)
        return None if np.isnan(f) else f
    except (ValueError, TypeError):
        return None
