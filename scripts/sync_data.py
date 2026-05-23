"""Standalone data sync script — run via cron/systemd timer."""

import logging
import sys
import os

# Allow running from repo root
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "backend"))

from app.services.data_pipeline import DataPipeline

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("sync_data")

if __name__ == "__main__":
    logger.info("Starting QuantForge data sync...")
    result = DataPipeline.full_sync()
    logger.info(f"Sync complete: {result}")
