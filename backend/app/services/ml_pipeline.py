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

# Feature columns produced by FactorEngine.compute_granular_factors() (~65 Alpha158-inspired factors)
FEATURE_COLS = [
    # Momentum (8)
    "mom_5d", "mom_10d", "mom_20d", "mom_60d", "mom_120d",
    "roc_6", "roc_14", "roc_30",
    # Volatility (4)
    "vol_5d", "vol_10d", "vol_20d", "vol_60d",
    # Turnover (6)
    "turnover_5d", "turnover_20d", "vol_ratio_5_20",
    "turn_ma5_ratio", "turn_ma10_ratio", "turn_ma20_ratio",
    # Volume (7)
    "vma5_ratio", "vma10_ratio", "vma20_ratio",
    "vstd5", "vstd20",
    "amount_ma5_ratio", "amount_ma10_ratio",
    # K-line pattern (6)
    "k_mid", "k_len", "k_up", "k_down", "k_sft", "k_ym1",
    # Price deviation (12)
    "ma5_ratio", "ma10_ratio", "ma20_ratio", "ma60_ratio",
    "std5_ratio", "std10_ratio", "std20_ratio",
    "max5_ratio", "max20_ratio", "min5_ratio", "min20_ratio",
    "price_position_60d",
    # Distance from high/low (6)
    "imax5", "imax20", "imax60",
    "imin5", "imin20", "imin60",
    # RSV / Technical (18)
    "rsv_9", "rsv_14", "rsv_20",
    "rsi_14", "price_position_20d", "max_dd_20d", "max_dd_60d",
    "macd_dif", "macd_dea", "macd_hist",
    "kdj_k", "kdj_d", "kdj_j",
    "boll_pos", "boll_width", "wr_14", "cci_14",
    "amplitude_5d", "amplitude_20d",
    # Fundamental (2)
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

    @staticmethod
    def _load_industry_map(db: Session, codes: list[str]) -> dict[str, str]:
        """Load industry classification for a set of stock codes."""
        rows = db.query(Stock.code, Stock.industry).filter(Stock.code.in_(codes)).all()
        return {r.code: (r.industry or "未知") for r in rows}

    @staticmethod
    def _get_top_industries(industry_map: dict[str, str], top_n: int = 15) -> list[str]:
        """Return the top-N most common industries."""
        from collections import Counter
        counts = Counter(industry_map.values())
        return [ind for ind, _ in counts.most_common(top_n)]

    def prepare_training_data(
        self, start_date: str, end_date: str, max_stocks: int = 500
    ) -> tuple[np.ndarray, np.ndarray, list[str]]:
        """Build training dataset from granular factors for ALL stocks with price data.

        Target: average of cross-sectional rank percentiles of forward 5d, 10d, 20d returns.
        Multi-horizon average reduces single-period noise and produces more stable rankings.

        max_stocks: limit number of stocks to sample for training (memory-constrained servers).
                    Stocks are selected by liquidity (highest average turnover).
        """
        db = self._get_db()

        # Get ALL stocks with price data
        all_quote_codes = sorted(
            r[0] for r in db.query(DailyQuote.code).distinct().all()
        )
        if not all_quote_codes:
            raise ValueError("No stocks with price data")

        logger.info(f"Total stocks with price data: {len(all_quote_codes)}")

        # Sample top-N by liquidity to stay within memory budget
        if max_stocks and len(all_quote_codes) > max_stocks:
            from ..models.market import DailyQuote as DQ
            recent_window = (
                date.fromisoformat(end_date) - timedelta(days=60)
            ).isoformat()
            liquidity = (
                db.query(DQ.code, func.avg(DQ.turnover).label("avg_turn"))
                .filter(
                    DQ.code.in_(all_quote_codes),
                    DQ.trade_date >= recent_window,
                    DQ.trade_date <= end_date,
                )
                .group_by(DQ.code)
                .order_by(func.avg(DQ.turnover).desc())
                .limit(max_stocks)
                .all()
            )
            quote_codes = sorted(r[0] for r in liquidity)
            logger.info(f"Sampled {len(quote_codes)}/{len(all_quote_codes)} stocks by liquidity "
                        f"(avg turnover range: {liquidity[-1][1]:.2f}% ~ {liquidity[0][1]:.2f}%)")
        else:
            quote_codes = all_quote_codes

        logger.info(f"Training universe: {len(quote_codes)} stocks")

        # Load price data in bulk
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

                # Multi-horizon forward returns (5d, 10d, 20d) for noise-reduced target
                forward_rets_5d = {}
                forward_rets_10d = {}
                forward_rets_20d = {}
                for code in factors_df.index:
                    df = price_data.get(code)
                    if df is None:
                        continue
                    r5 = self._forward_return_from_prices(df, snap_date, horizon=5)
                    r10 = self._forward_return_from_prices(df, snap_date, horizon=10)
                    r20 = self._forward_return_from_prices(df, snap_date, horizon=20)
                    if r5 is not None:
                        forward_rets_5d[code] = r5
                    if r10 is not None:
                        forward_rets_10d[code] = r10
                    if r20 is not None:
                        forward_rets_20d[code] = r20

                # Use intersection of codes that have all 3 horizons
                valid_codes = set(forward_rets_5d) & set(forward_rets_10d) & set(forward_rets_20d)
                if len(valid_codes) < 50:
                    continue

                # Multi-horizon target: average of 5d, 10d, 20d cross-sectional rank percentiles
                rank_5d = pd.Series({c: forward_rets_5d[c] for c in valid_codes}).rank(pct=True)
                rank_10d = pd.Series({c: forward_rets_10d[c] for c in valid_codes}).rank(pct=True)
                rank_20d = pd.Series({c: forward_rets_20d[c] for c in valid_codes}).rank(pct=True)
                avg_rank = (rank_5d + rank_10d + rank_20d) / 3.0

                for code in factors_df.index:
                    if code not in valid_codes:
                        continue
                    feats = {}
                    for col in FEATURE_COLS:
                        val = factors_df.loc[code, col] if col in factors_df.columns else None
                        feats[col] = val if not (isinstance(val, float) and np.isnan(val)) else None
                    feats["target"] = avg_rank[code]
                    feats["code"] = code
                    feats["trade_date"] = snap_date
                    all_samples.append(feats)

                if (i + 1) % 5 == 0:
                    logger.info(f"  Snapshot {i+1}/{len(snapshot_dates)}: {snap_date}, "
                                f"{len(valid_codes)} stocks, {len(all_samples)} total samples")

            except Exception as e:
                logger.warning(f"Snapshot {snap_date} failed: {e}")
                continue

        if len(all_samples) < 100:
            raise ValueError(f"Not enough training samples: {len(all_samples)}")

        df = pd.DataFrame(all_samples)

        # ── Industry one-hot encoding ──
        all_codes = df["code"].unique().tolist()
        industry_map = self._load_industry_map(db, all_codes)
        top_industries = self._get_top_industries(industry_map)
        logger.info(f"Top industries: {top_industries}")

        industry_dummies = pd.DataFrame(0, index=df.index, columns=[f"ind_{ind}" for ind in top_industries])
        for idx, code in zip(df.index, df["code"]):
            ind = industry_map.get(code, "未知")
            col_name = f"ind_{ind}"
            if col_name in industry_dummies.columns:
                industry_dummies.loc[idx, col_name] = 1

        self._industry_cols = list(industry_dummies.columns)
        self._top_industries = top_industries

        base_cols = [c for c in FEATURE_COLS if c in df.columns]
        X_base = df[base_cols].values.astype(np.float32)
        X_ind = industry_dummies.values.astype(np.float32)
        X = np.concatenate([X_base, X_ind], axis=1)
        y = df["target"].values.astype(np.float32)

        all_feature_names = base_cols + list(industry_dummies.columns)
        logger.info(f"Training data: {len(X)} samples, {len(all_feature_names)} features "
                     f"({len(base_cols)} factors + {len(industry_dummies.columns)} industries)")
        logger.info(f"Target stats: mean={y.mean():.3f}, std={y.std():.3f}, "
                    f"min={y.min():.3f}, max={y.max():.3f}")

        self._feature_names = all_feature_names
        return X, y, all_feature_names

    # ── Model training ─────────────────────────────────────────────

    def train(self, start_date: str = "2024-01-01", end_date: str = "2026-05-01", max_stocks: int = 800) -> dict:
        """Train LightGBM model with multi-horizon target and industry features.

        Uses lower learning rate + more trees for stable convergence,
        stronger L2 regularization to prevent overfitting on noisy financial data.
        """
        import lightgbm as lgb

        X, y, feature_names = self.prepare_training_data(start_date, end_date, max_stocks=max_stocks)

        # Time-series split: last 20% for validation
        split = int(len(X) * 0.8)
        X_train, X_val = X[:split], X[split:]
        y_train, y_val = y[:split], y[split:]

        logger.info(f"Train: {len(X_train)} samples, Val: {len(X_val)} samples")

        n_est = min(1000, max(300, len(X_train) // 4))
        num_leaves = min(31, max(15, len(X_train) // 300))

        self._model = lgb.LGBMRegressor(
            n_estimators=n_est,
            num_leaves=num_leaves,
            learning_rate=0.01,
            subsample=0.6,
            subsample_freq=1,
            colsample_bytree=0.55,
            reg_alpha=0.3,
            reg_lambda=2.0,
            min_child_samples=50,
            random_state=42,
            n_jobs=-1,
            verbose=-1,
        )
        self._model.fit(
            X_train, y_train,
            eval_set=[(X_val, y_val)],
            eval_metric="rmse",
            callbacks=[lgb.early_stopping(120, verbose=False), lgb.log_evaluation(50)],
        )

        # Compute per-sample prediction variance for confidence calibration
        self._calibrate_confidence(X_val)

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
                feature_names,
                self._model.feature_importances_.tolist(),
            )
        )
        importance = dict(sorted(importance.items(), key=lambda x: x[1], reverse=True)[:20])

        metrics = {
            "train_samples": len(X_train),
            "val_samples": len(X_val),
            "train_ic": round(train_ic, 4),
            "val_ic": round(val_ic, 4),
            "val_mse": round(val_mse, 4),
            "val_mae": round(val_mae, 4),
            "feature_importance": importance,
            "n_features": len(feature_names),
            "best_iteration": self._model.best_iteration_,
        }

        logger.info(f"Model trained: IC={val_ic:.4f}, best_iter={self._model.best_iteration_}")
        return metrics

    # ── Confidence calibration ─────────────────────────────────────

    def _calibrate_confidence(self, X_val: np.ndarray):
        """Compute prediction std from per-tree leaf variance on validation set.

        Uses the raw booster to get per-tree predictions, then computes
        the standard deviation across trees as an uncertainty measure.
        Stores the 95th percentile of std as the normalization factor.
        """
        try:
            booster = self._model.booster_
            # Get per-tree predictions for validation set
            tree_preds = []
            for tidx in range(booster.num_trees()):
                tpred = booster.predict(X_val, start_iteration=tidx, num_iteration=1)
                tree_preds.append(tpred)
            tree_preds = np.column_stack(tree_preds)
            # Std across trees = prediction uncertainty
            stds = np.std(tree_preds, axis=1)
            self._conf_p95 = float(np.percentile(stds, 95))
            self._conf_trained = True
            logger.info(f"Confidence calibrated: p95_std={self._conf_p95:.4f}")
        except Exception as e:
            logger.warning(f"Confidence calibration failed, using fallback: {e}")
            self._conf_p95 = 0.1
            self._conf_trained = False

    def _compute_confidence(self, X: np.ndarray) -> np.ndarray:
        """Compute per-sample confidence from tree prediction variance.

        Lower variance across trees = higher confidence.
        Maps std → [0.3, 1.0] range via exponential decay.
        """
        if not getattr(self, '_conf_trained', False) or self._model is None:
            return np.full(len(X), 0.5)

        try:
            booster = self._model.booster_
            tree_preds = []
            for tidx in range(booster.num_trees()):
                tpred = booster.predict(X, start_iteration=tidx, num_iteration=1)
                tree_preds.append(tpred)
            tree_preds = np.column_stack(tree_preds)
            stds = np.std(tree_preds, axis=1)
            # Normalize: lower std → higher confidence
            norm_factor = max(self._conf_p95, 1e-6)
            confidence = np.exp(-stds / norm_factor)
            return np.clip(confidence, 0.3, 1.0)
        except Exception:
            return np.full(len(X), 0.5)

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
            codes = sorted(
                r[0] for r in db.query(Stock.code).filter(Stock.is_active == True).all()
            )

        logger.info(f"Computing factors for {len(codes)} stocks on {trade_date}")
        factors_df = engine.compute_granular_factors(trade_date, codes=codes)

        # Feature names from training (base factors only, exclude industry dummies)
        industry_cols = getattr(self, '_industry_cols', [])
        base_feature_names = [c for c in self._feature_names if c in FEATURE_COLS]

        # Build base feature matrix
        X_base = np.zeros((len(factors_df), len(base_feature_names)), dtype=np.float32)
        valid_codes = []
        for i, code in enumerate(factors_df.index):
            feats = []
            for col in base_feature_names:
                val = factors_df.loc[code, col] if col in factors_df.columns else np.nan
                feats.append(val if not (isinstance(val, float) and np.isnan(val)) else np.nan)
            X_base[i] = feats
            valid_codes.append(code)

        X_base = np.nan_to_num(X_base, nan=0.0)

        # Industry one-hot features
        if industry_cols:
            industry_map = self._load_industry_map(db, valid_codes)
            X_ind = np.zeros((len(valid_codes), len(industry_cols)), dtype=np.float32)
            for i, code in enumerate(valid_codes):
                ind = industry_map.get(code, "未知")
                col_name = f"ind_{ind}"
                if col_name in industry_cols:
                    j = industry_cols.index(col_name)
                    X_ind[i, j] = 1.0
            X = np.concatenate([X_base, X_ind], axis=1)
        else:
            X = X_base

        preds = self._model.predict(X)
        confidences = self._compute_confidence(X)

        # Build results
        results = []
        for code, pred, conf in zip(valid_codes, preds, confidences):
            results.append({
                "code": code,
                "trade_date": trade_date,
                "predicted_return": round(float(pred), 4),
                "confidence": round(float(conf), 4),
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

    # ── Bulk prediction (for rolling training) ────────────────────

    def predict_for_range(self, start_date: str, end_date: str, interval_days: int = 5) -> pd.DataFrame:
        """Generate predictions for all trading dates in a range.

        Returns DataFrame with columns: code, trade_date, predicted_return.
        Used by RollingTrainer for out-of-sample prediction stitching.
        """
        db = self._get_db()

        trading_dates = sorted(
            r[0] for r in db.query(DailyQuote.trade_date)
            .filter(DailyQuote.trade_date >= start_date, DailyQuote.trade_date <= end_date)
            .distinct()
            .all()
        )

        all_preds = []
        for td in trading_dates[::interval_days]:
            try:
                preds = self.predict(td, top_n=500)
                all_preds.extend(preds)
            except Exception as e:
                logger.warning(f"Prediction for {td} failed: {e}")
                continue

        if not all_preds:
            return pd.DataFrame()

        return pd.DataFrame(all_preds)

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
