#!/bin/bash
# QuantForge 全自动流水线 — 数据同步 → 因子计算 → ML训练 → 预测生成
# 每日定时运行，无需手动干预
set -e

LOG_FILE="/var/log/quantforge-pipeline.log"
VENV_PYTHON="/var/www/quantforge/backend/venv/bin/python3"
WORKDIR="/var/www/quantforge/backend"

log() { echo "[$(date '+%m-%d %H:%M:%S')] $1" | tee -a "$LOG_FILE"; }

log "========== Pipeline Start =========="

cd "$WORKDIR"

# ── Step 1: 数据同步 (分批处理，边拉边存) ──
log "Step 1/4: Data sync..."

$VENV_PYTHON -c "
import logging, sys
logging.basicConfig(level=logging.WARNING, format='%(asctime)s %(levelname)s %(message)s')
logger = logging.getLogger('pipeline')

sys.path.insert(0, '.')
from app.core.database import SessionLocal
from app.models.stock import Stock
from app.services.data_pipeline import DataPipeline
from datetime import date, timedelta

db = SessionLocal()

# Sync stock list first
stock_count = DataPipeline.sync_stock_list(db)
codes = [r[0] for r in db.query(Stock.code).filter(Stock.is_active == True).all()]
logger.info(f'Stock list synced: {stock_count} stocks, {len(codes)} active')

# Batch fetch settings
BATCH_SIZE = 200
end_date = date.today().strftime('%Y%m%d')
start_date = (date.today() - timedelta(days=10)).strftime('%Y%m%d')

total_quotes = 0
total_stocks_with_data = 0
batches = [codes[i:i+BATCH_SIZE] for i in range(0, len(codes), BATCH_SIZE)]

for batch_idx, batch_codes in enumerate(batches):
    try:
        logger.info(f'Batch {batch_idx+1}/{len(batches)}: {len(batch_codes)} stocks')
        quotes_data = DataPipeline.fetch_daily_quotes(batch_codes, start_date, end_date)
        if quotes_data:
            saved = DataPipeline.save_daily_quotes(quotes_data, db)
            total_quotes += saved
            total_stocks_with_data += len(quotes_data)
            logger.info(f'  Saved {saved} quotes for {len(quotes_data)} stocks')
    except Exception as e:
        logger.warning(f'  Batch {batch_idx+1} failed: {e}')
        continue

# Sync financials for stocks that have quotes
from app.models.market import DailyQuote
from sqlalchemy import func
quote_codes = [r[0] for r in db.query(DailyQuote.code).distinct().all()]
logger.info(f'Syncing financials for {len(quote_codes)} stocks with quotes...')
fin_count = DataPipeline.sync_financial_indicators(quote_codes, db)

db.close()
print(f'QUOTES_SAVED={total_quotes}')
print(f'STOCKS_WITH_DATA={total_stocks_with_data}')
print(f'FINANCIAL_SYNCED={fin_count}')
" >> "$LOG_FILE" 2>&1

log "Step 1/4: Done."

# ── Step 2: 因子计算 ──
log "Step 2/4: Factor computation..."

$VENV_PYTHON -c "
import sys, logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s')
logger = logging.getLogger('factors')

sys.path.insert(0, '.')
from app.core.database import SessionLocal
from app.services.factor_engine import FactorEngine
from app.models.market import DailyQuote
from sqlalchemy import func

db = SessionLocal()

# Get latest trading date from daily quotes
latest_date = db.query(func.max(DailyQuote.trade_date)).scalar()
if not latest_date:
    logger.error('No daily quotes found!')
    sys.exit(1)

# Get stocks that have data on this date
codes = [r[0] for r in db.query(DailyQuote.code).filter(DailyQuote.trade_date == latest_date).distinct().all()]
logger.info(f'Computing factors for {len(codes)} stocks on {latest_date}')

engine = FactorEngine(db)
factors = engine.compute_all_factors(latest_date, codes)
count = engine.save_factors(factors, latest_date, db)
logger.info(f'Saved {count} factor scores')
db.close()
print(f'FACTOR_DATE={latest_date}')
print(f'FACTOR_COUNT={count}')
" >> "$LOG_FILE" 2>&1

log "Step 2/4: Done."

# ── Step 3+4: ML 训练 + 预测 (combined to share model in memory) ──
log "Step 3/4: ML training + prediction..."

$VENV_PYTHON "$WORKDIR/../scripts/train_and_predict.py" >> "$LOG_FILE" 2>&1

log "Step 3/4: Done."
log "========== Pipeline Complete =========="
