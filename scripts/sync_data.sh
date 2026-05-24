#!/bin/bash
# QuantForge daily data sync script
# Runs incremental sync: stock list + daily quotes (7d) + financial indicators

set -e

LOG_FILE="/var/log/quantforge-sync.log"
VENV_PYTHON="/var/www/quantforge/backend/venv/bin/python3"
WORKDIR="/var/www/quantforge/backend"

echo "[$(date '+%Y-%m-%d %H:%M:%S')] Starting scheduled data sync..." >> "$LOG_FILE"

cd "$WORKDIR"
$VENV_PYTHON -c "
import logging, sys
logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s')
logger = logging.getLogger('sync')

sys.path.insert(0, '.')
from app.services.data_pipeline import DataPipeline

logger.info('Running incremental sync (7-day lookback)...')
try:
    result = DataPipeline.incremental_sync(lookback_days=7)
    logger.info(f'Sync complete: {result}')
    print(result)
except Exception as e:
    logger.error(f'Sync failed: {e}', exc_info=True)
    sys.exit(1)
" >> "$LOG_FILE" 2>&1

echo "[$(date '+%Y-%m-%d %H:%M:%S')] Sync finished." >> "$LOG_FILE"
