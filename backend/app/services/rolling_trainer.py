"""Rolling training framework — sliding windows + periodic retraining.

Inspired by Qlib's rolling training:
  - Train on overlapping windows (e.g., 12-month train, 1-month test)
  - Stitch predictions from each window
  - Track model stability across time periods
"""

import logging
from datetime import date, timedelta
from typing import Any

import numpy as np
import pandas as pd
from dateutil.relativedelta import relativedelta
from sqlalchemy.orm import Session

from ..core.database import SessionLocal

logger = logging.getLogger(__name__)


class RollingTrainer:
    """Sliding-window training with prediction stitching.

    Usage:
        trainer = RollingTrainer(train_window=12, test_window=1, step=1)
        results = trainer.run(start_date="2024-01-01", end_date="2026-05-01")
        # results["stitched_predictions"] is a DataFrame of all predictions
    """

    def __init__(
        self,
        db: Session | None = None,
        train_window: int = 12,  # months
        test_window: int = 1,  # months
        step: int = 1,  # months to slide per iteration
    ):
        self._db = db
        self.train_window = train_window
        self.test_window = test_window
        self.step = step

    def _get_db(self) -> Session:
        return self._db or SessionLocal()

    def run(
        self,
        start_date: str = "2024-01-01",
        end_date: str = "2026-05-01",
        min_train_samples: int = 500,
        model_type: str = "lightgbm",  # "lightgbm" or "alstm"
    ) -> dict:
        """Execute rolling training across the date range.

        Returns:
            {
                "windows": [window_metrics, ...],
                "stitched_predictions": DataFrame,
                "aggregate_ic": float,
                "window_ics": [float, ...],
                "n_windows": int,
            }
        """
        start_dt = date.fromisoformat(start_date)
        end_dt = date.fromisoformat(end_date)

        train_start = start_dt
        all_ics = []
        all_predictions = []
        window_metrics = []

        window_idx = 0
        while train_start < end_dt:
            # Compute window boundaries
            train_end = train_start + relativedelta(months=self.train_window)
            test_end = train_end + relativedelta(months=self.test_window)

            if train_end > end_dt:
                break
            if test_end > end_dt:
                test_end = end_dt

            train_start_s = train_start.isoformat()
            train_end_s = train_end.isoformat()
            test_end_s = test_end.isoformat()

            logger.info(
                f"Window #{window_idx+1}: train={train_start_s}~{train_end_s}, "
                f"test={train_end_s}~{test_end_s}"
            )

            try:
                if model_type == "lightgbm":
                    metrics, predictions = self._train_lgb_window(
                        train_start_s, train_end_s, test_end_s
                    )
                else:
                    metrics, predictions = self._train_dl_window(
                        train_start_s, train_end_s, test_end_s
                    )

                if metrics and predictions is not None and not predictions.empty:
                    all_ics.append(metrics.get("val_ic", 0))
                    all_predictions.append(predictions)
                    metrics["window"] = window_idx + 1
                    metrics["train_start"] = train_start_s
                    metrics["train_end"] = train_end_s
                    window_metrics.append(metrics)

            except Exception as e:
                logger.warning(f"Window #{window_idx+1} failed: {e}")

            # Slide
            train_start += relativedelta(months=self.step)
            window_idx += 1

        # Stitch predictions: for overlapping periods, use most recent model
        if all_predictions:
            stitched = self._stitch_predictions(all_predictions)
            ic = self._compute_stitched_ic(stitched)
        else:
            stitched = pd.DataFrame()
            ic = 0.0

        return {
            "windows": window_metrics,
            "stitched_predictions": stitched,
            "aggregate_ic": round(ic, 4),
            "window_ics": [round(ic, 4) for ic in all_ics],
            "mean_ic": round(float(np.mean(all_ics)), 4) if all_ics else 0,
            "ic_std": round(float(np.std(all_ics)), 4) if all_ics else 0,
            "ic_trend": _trend_label(all_ics),
            "n_windows": len(window_metrics),
        }

    def _train_lgb_window(
        self, train_start: str, train_end: str, test_end: str
    ) -> tuple[dict | None, pd.DataFrame | None]:
        """Train LightGBM on a single window."""
        from .ml_pipeline import MLPipeline

        db = self._get_db()
        pipeline = MLPipeline(db)

        # Train on (train_start, train_end), predict on (train_end, test_end]
        metrics = pipeline.train(train_start, train_end)
        predictions = pipeline.predict_for_range(train_end, test_end, interval_days=5)

        return metrics, predictions

    def _train_dl_window(
        self, train_start: str, train_end: str, test_end: str
    ) -> tuple[dict | None, pd.DataFrame | None]:
        """Train deep learning model on a single window."""
        from .pytorch_models import DeepLearningPipeline, build_sequences, _HAS_TORCH

        if not _HAS_TORCH:
            return None, None

        # Placeholder — DL training requires sequence construction
        # which is more involved. Returns empty for now.
        logger.info("DL window training not yet implemented for rolling trainer")
        return None, None

    # ── Prediction stitching ─────────────────────────────────────

    @staticmethod
    def _stitch_predictions(predictions: list[pd.DataFrame]) -> pd.DataFrame:
        """Stitch predictions from multiple windows.

        For overlapping periods, keeps the most recent model's predictions
        (under the assumption that recent models are better adapted).
        """
        # Each DataFrame should have columns: code, trade_date, predicted_return
        if not predictions:
            return pd.DataFrame()

        combined = pd.concat(predictions, ignore_index=True)
        combined = combined.drop_duplicates(
            subset=["code", "trade_date"], keep="last"
        )
        combined = combined.sort_values(["trade_date", "code"]).reset_index(drop=True)
        return combined

    def _compute_stitched_ic(self, predictions: pd.DataFrame) -> float:
        """Compute Spearman IC of stitched predictions vs forward 20-day returns."""
        from scipy.stats import spearmanr

        if predictions.empty:
            return 0.0

        # Compute forward returns for each prediction
        predictions = self._attach_forward_returns(predictions)

        if "actual_return" not in predictions.columns:
            return 0.0

        mask = predictions["predicted_return"].notna() & predictions["actual_return"].notna()
        if mask.sum() < 30:
            return 0.0

        ic, _ = spearmanr(
            predictions.loc[mask, "predicted_return"],
            predictions.loc[mask, "actual_return"],
        )
        return float(ic) if not np.isnan(ic) else 0.0

    def _attach_forward_returns(self, predictions: pd.DataFrame) -> pd.DataFrame:
        """Attach actual 20-day forward returns to predictions DataFrame."""
        from ..models.market import DailyQuote

        db = self._get_db()
        preds = predictions.copy()

        # Get all unique (code, trade_date) pairs
        pairs = list(preds[["code", "trade_date"]].drop_duplicates().itertuples(index=False))

        forward_returns = {}
        for code, td in pairs:
            future = (
                db.query(DailyQuote.close)
                .filter(DailyQuote.code == code, DailyQuote.trade_date > td)
                .order_by(DailyQuote.trade_date)
                .limit(21)
                .all()
            )
            if len(future) < 20:
                continue
            current = (
                db.query(DailyQuote.close)
                .filter(DailyQuote.code == code, DailyQuote.trade_date == td)
                .scalar()
            )
            if current and current > 0 and future[19][0] and future[19][0] > 0:
                forward_returns[(code, td)] = (future[19][0] - current) / current * 100

        preds["actual_return"] = preds.apply(
            lambda r: forward_returns.get((r["code"], r["trade_date"])), axis=1
        )
        return preds


def _trend_label(ics: list[float]) -> str:
    """Classify IC trend across windows: improving, stable, or degrading."""
    if len(ics) < 3:
        return "insufficient_data"

    first_half = np.mean(ics[:len(ics) // 2])
    second_half = np.mean(ics[len(ics) // 2:])

    diff = second_half - first_half
    if diff > 0.02:
        return "improving"
    elif diff < -0.02:
        return "degrading"
    else:
        return "stable"


def generate_rolling_report(trainer_results: dict) -> str:
    """Generate a human-readable rolling training report."""
    lines = [
        "=" * 50,
        "  Rolling Training Report",
        "=" * 50,
        f"  Windows completed: {trainer_results['n_windows']}",
        f"  Mean IC: {trainer_results['mean_ic']:.4f}",
        f"  IC Std: {trainer_results['ic_std']:.4f}",
        f"  IC Trend: {trainer_results['ic_trend']}",
        f"  Aggregate IC: {trainer_results['aggregate_ic']:.4f}",
        "",
        "  Per-window ICs:",
    ]
    for i, ic in enumerate(trainer_results.get("window_ics", [])):
        marker = " *" if i == len(trainer_results.get("window_ics", [])) - 1 else ""
        lines.append(f"    Window {i+1}: IC={ic:.4f}{marker}")
    lines.append("=" * 50)
    return "\n".join(lines)
