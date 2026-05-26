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

# Sync industry classification
logger.info('Syncing industry classification...')
industry_count = DataPipeline.sync_industry_data(db)

# Sync CSI300 and CSI500 index constituents (for benchmark)
logger.info('Syncing index constituents...')
csi300 = DataPipeline.sync_index_constituents('000300')
csi500 = DataPipeline.sync_index_constituents('000905')
logger.info(f'CSI300: {len(csi300)} stocks, CSI500: {len(csi500)} stocks')

# Sync fund flow for top 200 liquid stocks
logger.info('Syncing fund flows for top stocks...')
from app.services.universe import get_liquid_stocks
liquid_codes = get_liquid_stocks(db, top_n=200)
fund_flow_count = DataPipeline.sync_fund_flows(liquid_codes, db)

# Sync multi-period financials for same stocks (4 quarters of history)
logger.info('Syncing multi-period financials...')
mp_fin_count = DataPipeline.sync_multi_period_financials(liquid_codes, periods=4, db=db)

# Sync margin trading (融资融券 — may be T+1, gracefully handles no data)
logger.info('Syncing margin trading...')
margin_count = DataPipeline.sync_margin_trading(db, lookback_days=3)

# Sync shareholder count changes (股东户数 — 筹码集中度)
logger.info('Syncing shareholder counts...')
shareholder_count = DataPipeline.sync_shareholder_counts(liquid_codes, db)

# Sync dragon tiger list (龙虎榜)
logger.info('Syncing dragon tiger list...')
dt_count = DataPipeline.sync_dragon_tiger(db, lookback_days=5)

# Sync upcoming lockup releases
logger.info('Syncing lockup releases...')
lockup_count = DataPipeline.sync_lockup_releases(liquid_codes, db)

# Repair: backfill pct_change for existing quotes where it's null
logger.info('Backfilling pct_change for existing quotes...')
from sqlalchemy import text
repair_result = db.execute(text("""
    WITH ordered AS (
        SELECT id, code, trade_date, close,
               LAG(close) OVER (PARTITION BY code ORDER BY trade_date) AS prev_close
        FROM daily_quotes
        WHERE pct_change IS NULL
    )
    UPDATE daily_quotes dq
    SET pct_change = ROUND((o.close - o.prev_close) / NULLIF(o.prev_close, 0) * 100, 4),
        change = ROUND((o.close - o.prev_close)::numeric, 4)
    FROM ordered o
    WHERE dq.id = o.id AND o.prev_close IS NOT NULL AND o.prev_close > 0
"""))
db.commit()
repair_count = repair_result.rowcount if repair_result else 0
logger.info(f'Backfilled pct_change for {repair_count} quotes')

db.close()
print(f'QUOTES_SAVED={total_quotes}')
print(f'STOCKS_WITH_DATA={total_stocks_with_data}')
print(f'FINANCIAL_SYNCED={fin_count}')
print(f'INDUSTRY_SYNCED={industry_count}')
print(f'FUND_FLOW_SYNCED={fund_flow_count}')
print(f'MULTI_PERIOD_FIN={mp_fin_count}')
print(f'MARGIN_SYNCED={margin_count}')
print(f'SHAREHOLDER_SYNCED={shareholder_count}')
print(f'DRAGON_TIGER_SYNCED={dt_count}')
print(f'LOCKUP_SYNCED={lockup_count}')
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

# ── Step 5: Backtesting ──
log "Step 5/5: Backtesting..."

$VENV_PYTHON -c "
import sys, logging, json
logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s')
logger = logging.getLogger('backtest')
sys.path.insert(0, '.')
from app.core.database import SessionLocal
from app.models.market import DailyQuote
from app.models.trading import BacktestResult
from app.services.backtest_engine import BacktestEngine, save_backtest_result
from app.services.strategies.ma_crossover import MACrossoverStrategy
from app.services.strategies.momentum_breakout import MomentumBreakoutStrategy
from app.services.strategies.mean_reversion import MeanReversionStrategy
from app.services.strategy_engine import StrategyConfig
from app.services.data_pipeline import DataPipeline
from app.services.universe import get_universe
from sqlalchemy import func
from datetime import date, timedelta

db = SessionLocal()

latest_date = db.query(func.max(DailyQuote.trade_date)).scalar()
if not latest_date:
    logger.error('No daily quotes found — skipping backtest')
    db.close()
    sys.exit(0)

logger.info(f'=== Backtest through {latest_date} ===')

universe = get_universe(db, name='liquid_100')
logger.info(f'Backtest universe: {len(universe)} stocks')

engine = BacktestEngine(db)

# Daily: run all 3 strategies with default params
daily_strategies = {
    'ma_crossover': MACrossoverStrategy(fast=10, slow=30),
    'momentum_breakout': MomentumBreakoutStrategy(lookback=20),
    'mean_reversion': MeanReversionStrategy(bb_window=20),
}

for name, strat in daily_strategies.items():
    try:
        report = engine.run(strat, universe, '2025-01-01', latest_date, 100000)
        rid = save_backtest_result(db, report)
        logger.info(f'{name}: return={report.total_return:.1f}% sharpe={report.sharpe_ratio:.2f} max_dd={report.max_drawdown:.1f}% trades={report.total_trades}')
    except Exception as e:
        logger.warning(f'Daily backtest {name} failed: {e}')

is_friday = (date.today().weekday() == 4)

if is_friday:
    logger.info('=== Friday: Running full grid search ===')
    grid_configs = {
        'ma_crossover': [
            {'strategy': MACrossoverStrategy(fast=f, slow=s), 'params': f'fast={f},slow={s}'}
            for f in [5, 10, 20] for s in [20, 30, 60] if f < s
        ],
        'momentum_breakout': [
            {'strategy': MomentumBreakoutStrategy(lookback=lb), 'params': f'lookback={lb}'}
            for lb in [10, 20, 30, 60]
        ],
        'mean_reversion': [
            {'strategy': MeanReversionStrategy(bb_window=w), 'params': f'bb_window={w}'}
            for w in [10, 20, 30]
        ],
    }

    best_results = {}
    for strategy_name, grid in grid_configs.items():
        best_sharpe = -999
        best_entry = None
        for g in grid:
            try:
                report = engine.run(g['strategy'], universe, '2025-01-01', latest_date, 100000)
                rid = save_backtest_result(db, report)
                entry = {
                    'id': rid, 'params': g['params'],
                    'return': report.total_return, 'sharpe': report.sharpe_ratio,
                    'max_dd': report.max_drawdown, 'win_rate': report.win_rate,
                    'trades': report.total_trades,
                }
                if report.sharpe_ratio > best_sharpe:
                    best_sharpe = report.sharpe_ratio
                    best_entry = entry
                    result_row = db.query(BacktestResult).filter(BacktestResult.id == rid).first()
                    if result_row:
                        result_row.strategy_name = f'{strategy_name}_BEST'
                        db.commit()
            except Exception as e:
                logger.warning(f'Grid {strategy_name}/{g[\"params\"]} failed: {e}')

        if best_entry:
            best_results[strategy_name] = best_entry
            logger.info(f'Best {strategy_name}: {best_entry[\"params\"]} sharpe={best_entry[\"sharpe\"]:.2f} return={best_entry[\"return\"]:.1f}%')

    # Benchmark comparison
    benchmark_return = None
    try:
        idx_df = DataPipeline.fetch_index_quotes('000001', '2025-01-01', latest_date)
        if idx_df is not None and not idx_df.empty:
            idx_first = float(idx_df.iloc[0].get('close', 0))
            idx_last = float(idx_df.iloc[-1].get('close', 0))
            if idx_first > 0:
                benchmark_return = round((idx_last - idx_first) / idx_first * 100, 2)
    except Exception as e:
        logger.warning(f'Benchmark fetch failed: {e}')

    for name, br in best_results.items():
        alpha = br['return'] - benchmark_return if benchmark_return is not None else None
        logger.info(f'{name}: alpha={alpha:.1f}%' if alpha else f'{name}: benchmark N/A')

    summary = {
        'date': date.today().isoformat(),
        'latest_trade_date': latest_date,
        'universe_size': len(universe),
        'benchmark_return': benchmark_return,
        'best_results': best_results,
    }
    with open('/var/log/quantforge-weekly-summary.json', 'w') as f:
        json.dump(summary, f, indent=2, default=str)
    logger.info('Weekly grid search complete')
else:
    logger.info('Not Friday — daily backtest complete (grid search skipped)')

db.close()
logger.info('Backtesting done')
" >> "$LOG_FILE" 2>&1

log "Step 5/5: Done."
log "========== Pipeline Complete =========="
