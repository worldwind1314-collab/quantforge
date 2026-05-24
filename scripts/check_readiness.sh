#!/bin/bash
# QuantForge 实盘就绪度检查脚本
# 每天自动运行，记录关键指标到 /var/log/quantforge-readiness.log

LOG_FILE="/var/log/quantforge-readiness.log"
VENV_PYTHON="/var/www/quantforge/backend/venv/bin/python3"
WORKDIR="/var/www/quantforge/backend"

cd "$WORKDIR"
$VENV_PYTHON -c "
import json, sys, logging
from datetime import date, timedelta
from sqlalchemy import func

sys.path.insert(0, '.')
from app.core.database import SessionLocal
from app.models.market import DailyQuote
from app.models.stock import Stock
from app.models.finance import FinancialIndicator, FactorScore, MLPrediction
from app.models.trading import BacktestResult

db = SessionLocal()

# ── 数据覆盖 ──
total_stocks = db.query(func.count(Stock.code)).scalar()
quote_codes = db.query(func.count(func.distinct(DailyQuote.code))).scalar()
fi_codes = db.query(func.count(func.distinct(FinancialIndicator.code))).scalar()
latest_quote = db.query(func.max(DailyQuote.trade_date)).scalar()
latest_ml = db.query(func.max(MLPrediction.trade_date)).scalar()
fs_dates = db.query(func.count(func.distinct(FactorScore.trade_date))).scalar()
ml_dates = db.query(func.count(func.distinct(MLPrediction.trade_date))).scalar()

# ── 数据新鲜度 ──
freshness = 'STALE'
if latest_quote:
    try:
        d = date.fromisoformat(latest_quote)
        diff = (date.today() - d).days
        freshness = 'FRESH' if diff <= 2 else f'{diff}d old'
    except: pass

# ── 最新模型表现 ──
latest_bt = db.query(BacktestResult).order_by(BacktestResult.created_at.desc()).first()
sharpe = latest_bt.sharpe_ratio if latest_bt else None
total_ret = latest_bt.total_return if latest_bt else None
ic_mean = latest_bt.ic_mean if latest_bt else None
max_dd = latest_bt.max_drawdown if latest_bt else None

# ── 就绪度评分 ──
score = 0
checks = []

# 1. 行情覆盖 >= 500 只 (阈值: 500)
q_ok = quote_codes >= 500
score += 25 if q_ok else (quote_codes / 500 * 25)
checks.append({'check': '行情覆盖≥500', 'value': quote_codes, 'ok': q_ok})

# 2. 财务覆盖 >= 200 只 (阈值: 200)
f_ok = fi_codes >= 200
score += 20 if f_ok else (fi_codes / 200 * 20)
checks.append({'check': '财务覆盖≥200', 'value': fi_codes, 'ok': f_ok})

# 3. 因子覆盖天数 >= 60 (阈值: 60)
fs_ok = fs_dates >= 60
score += 15 if fs_ok else (fs_dates / 60 * 15)
checks.append({'check': '因子天数≥60', 'value': fs_dates, 'ok': fs_ok})

# 4. IC > 0.03 或 Sharpe > 0.3 (阈值)
model_ok = (ic_mean is not None and ic_mean > 0.03) or (sharpe is not None and sharpe > 0.3)
score += 20 if model_ok else 10 if ic_mean is not None else 0
checks.append({'check': '模型IC>0.03', 'value': round(ic_mean, 4) if ic_mean else None, 'ok': model_ok})

# 5. 数据新鲜度 (2天内)
fresh_ok = freshness == 'FRESH'
score += 10 if fresh_ok else 0
checks.append({'check': '数据2天内', 'value': freshness, 'ok': fresh_ok})

# 6. ML预测覆盖
ml_ok = ml_dates >= 2
score += 10 if ml_ok else 0
checks.append({'check': 'ML预测≥2天', 'value': ml_dates, 'ok': ml_ok})

# ── 评级 ──
if score >= 90:
    grade = 'A+ READY'
elif score >= 75:
    grade = 'A ALMOST'
elif score >= 60:
    grade = 'B PROGRESS'
elif score >= 40:
    grade = 'C EARLY'
else:
    grade = 'D START'

result = {
    'ts': str(date.today()),
    'grade': grade,
    'score': round(score, 1),
    'data': {
        'stocks_total': total_stocks,
        'quote_codes': quote_codes,
        'fi_codes': fi_codes,
        'fs_dates': fs_dates,
        'ml_dates': ml_dates,
        'latest_quote': latest_quote,
        'freshness': freshness,
    },
    'model': {
        'sharpe': round(sharpe, 3) if sharpe else None,
        'total_return': round(total_ret, 3) if total_ret else None,
        'ic_mean': round(ic_mean, 4) if ic_mean else None,
        'max_dd': round(max_dd, 3) if max_dd else None,
    },
    'checks': checks,
}

print(json.dumps(result, ensure_ascii=False, indent=2))
db.close()
" >> "$LOG_FILE" 2>&1

# 只保留最近 90 条记录
tail -n 90 "$LOG_FILE" > "${LOG_FILE}.tmp" 2>/dev/null && mv "${LOG_FILE}.tmp" "$LOG_FILE"
