"""QuantForge ML training + prediction step. Called by run_pipeline.sh."""
import json
import logging
import sys
from datetime import date

sys.path.insert(0, ".")

from app.core.database import SessionLocal
from app.services.ml_pipeline import MLPipeline, FEATURE_COLS
from app.models.market import DailyQuote
from sqlalchemy import func

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("ml_pipeline")

MODEL_FILE = "/var/www/quantforge/backend/model.txt"
META_FILE = "/var/www/quantforge/backend/model_meta.json"

db = SessionLocal()
pipeline = MLPipeline(db)

# ── Train ──
end_date = date.today().isoformat()
start_date = date.today().replace(year=date.today().year - 2).isoformat()
logger.info("Training model: %s ~ %s", start_date, end_date)

metrics = pipeline.train(start_date, end_date)
train_samples = metrics.get("train_samples")
val_ic = metrics.get("val_ic")
val_mse = metrics.get("val_mse")
feature_importance = metrics.get("feature_importance", {})
n_features = metrics.get("n_features", 0)
best_iter = metrics.get("best_iteration", 0)

logger.info("Train done: samples=%s, val_ic=%s, val_mse=%s, n_features=%s, best_iter=%s",
            train_samples, val_ic, val_mse, n_features, best_iter)
logger.info("Top features: %s",
            dict(list(feature_importance.items())[:6]) if feature_importance else {})

# ── Save model using LightGBM native format ──
pipeline._model.booster_.save_model(MODEL_FILE)
with open(META_FILE, "w") as f:
    json.dump({
        "feature_names": pipeline._feature_names or FEATURE_COLS,
        "metrics": {k: v for k, v in metrics.items() if k != "feature_importance"},
        "feature_importance": feature_importance,
    }, f, indent=2)
logger.info("Model saved to %s (meta: %s)", MODEL_FILE, META_FILE)

# ── Predict ──
latest_date = db.query(func.max(DailyQuote.trade_date)).scalar()
logger.info("Generating predictions for %s", latest_date)

predictions = pipeline.predict(latest_date, top_n=100)
logger.info("Generated %d predictions", len(predictions))

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

print(f"TRAIN_SAMPLES={train_samples}")
print(f"VAL_IC={val_ic}")
print(f"VAL_MSE={val_mse}")
print(f"N_FEATURES={n_features}")
print(f"PREDICT_DATE={latest_date}")
print(f"PREDICT_COUNT={len(predictions)}")
