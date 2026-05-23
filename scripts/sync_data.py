"""Standalone data sync script — run via cron/systemd timer.

Usage:
    python scripts/sync_data.py            # incremental (last 7 days)
    python scripts/sync_data.py --full     # full historical sync
    python scripts/sync_data.py --days 3   # incremental with custom lookback
"""

import argparse
import logging
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "backend"))

from app.services.data_pipeline import DataPipeline

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("sync_data")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="QuantForge data sync")
    parser.add_argument("--full", action="store_true", help="Full historical sync")
    parser.add_argument("--days", type=int, default=7, help="Lookback days for incremental sync")
    args = parser.parse_args()

    logger.info("Starting QuantForge data sync...")
    if args.full:
        result = DataPipeline.full_sync()
    else:
        result = DataPipeline.incremental_sync(lookback_days=args.days)
    logger.info(f"Sync complete: {result}")
