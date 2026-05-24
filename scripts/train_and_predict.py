"""QuantForge ML training + prediction step. Called by run_pipeline.sh."""
import logging
import pickle
import sys
from datetime import date

sys.path.insert(0, ".")

from app.core.database import SessionLocal
from app.services.ml_pipeline import MLPipeline
from app.models.market import DailyQuote
from sqlalchemy import func

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("ml_pipeline")

MODEL_FILE = "/var/www/quantforge/backend/model.pkl"

db = SessionLocal()
pipeline = MLPipeline(db)

# ── Train ──
end_date = date.today().isoformat()
start_date = date.today().replace(year=date.today().year - 2).isoformat()
logger.info("Training model: %s ~ %s", start_date, end_date)

metrics = pipeline.train(start_date, end_date)
train_samples = metrics.get("train_samples")
val_ic = metrics.get("val_ic")
feature_importance = metrics.get("feature_importance", {})

logger.info("Train done: samples=%s, val_ic=%s", train_samples, val_ic)
logger.info("Feature importance: %s", feature_importance)

# ── Save model to disk ──
with open(MODEL_FILE, "wb") as f:
    pickle.dump(
        {
            "model": pipeline._model,
            "scaler": getattr(pipeline, "_scaler", None),
            "feature_names": getattr(pipeline, "_feature_names", None),
            "metrics": metrics,
        },
        f,
    )
logger.info("Model saved to %s", MODEL_FILE)

# ── Predict ──
latest_date = db.query(func.max(DailyQuote.trade_date)).scalar()
logger.info("Generating predictions for %s", latest_date)

predictions = pipeline.predict(latest_date, top_n=100)
logger.info("Generated %d predictions", len(predictions))

# Print summary
for p in predictions[:10]:
    logger.info(
        "  #%d %s return=%+.4f conf=%.2f",
        p.get("prediction_rank", 0),
        p.get("code", "?"),
        p.get("predicted_return", 0),
        p.get("confidence", 0),
    )

db.close()
logger.info("Train + predict complete!")

# Output key results for pipeline log
print(f"TRAIN_SAMPLES={train_samples}")
print(f"VAL_IC={val_ic}")
print(f"PREDICT_DATE={latest_date}")
print(f"PREDICT_COUNT={len(predictions)}")
