"""Trading API — backtesting, paper trading, strategy management."""

from fastapi import APIRouter, Depends, Query
from sqlalchemy.orm import Session

from ..core.database import get_db
from ..models.stock import Stock
from ..models.trading import BacktestResult as BacktestResultModel
from ..services.backtest_engine import BacktestEngine, save_backtest_result
from ..services.paper_trading import PaperTradingService
from ..services.strategies.ma_crossover import MACrossoverStrategy
from ..services.strategies.mean_reversion import MeanReversionStrategy
from ..services.strategies.momentum_breakout import MomentumBreakoutStrategy
from ..services.strategy_engine import StrategyConfig

router = APIRouter(prefix="/trading", tags=["trading"])

STRATEGY_MAP = {
    "ma_crossover": MACrossoverStrategy,
    "momentum_breakout": MomentumBreakoutStrategy,
    "mean_reversion": MeanReversionStrategy,
}


# ── Strategy info ──────────────────────────────────────────────────

@router.get("/strategies")
def list_strategies():
    return {
        "strategies": [
            {
                "id": "ma_crossover",
                "name": "均线交叉",
                "description": "金叉买入，死叉卖出。经典趋势跟踪策略。",
                "params": {"fast": 10, "slow": 30},
            },
            {
                "id": "momentum_breakout",
                "name": "动量突破",
                "description": "价格突破N日最高点时买入，跌破95%时卖出。",
                "params": {"lookback": 20},
            },
            {
                "id": "mean_reversion",
                "name": "均值回归",
                "description": "布林带下轨+RSI超卖买入，上轨+RSI超买卖出。",
                "params": {"bb_window": 20, "rsi_window": 14, "rsi_oversold": 30, "rsi_overbought": 70},
            },
        ]
    }


# ── Backtesting ────────────────────────────────────────────────────

@router.post("/backtest")
def run_backtest(
    strategy_id: str = Query(..., description="策略ID"),
    codes: str = Query("000001,600519,002594,300750,000858", description="股票代码 逗号分隔"),
    start_date: str = Query("2025-01-01", description="起始日期"),
    end_date: str = Query("2026-05-23", description="结束日期"),
    initial_capital: float = Query(100_000, description="初始资金"),
    fast: int = Query(10, description="MA快线"),
    slow: int = Query(30, description="MA慢线"),
    lookback: int = Query(20, description="突破/回归窗口"),
    db: Session = Depends(get_db),
):
    """运行回测并返回报告。结果自动保存到数据库。"""
    strategy_cls = STRATEGY_MAP.get(strategy_id)
    if not strategy_cls:
        return {"error": f"Unknown strategy: {strategy_id}", "available": list(STRATEGY_MAP.keys())}

    code_list = [c.strip() for c in codes.split(",") if c.strip()]
    if not code_list:
        code_list = [r[0] for r in db.query(Stock.code).filter(Stock.is_active == True).limit(10).all()]

    config = StrategyConfig(
        name=strategy_id,
        initial_capital=initial_capital,
        max_position_pct=0.2,
        max_positions=5,
    )

    if strategy_id == "ma_crossover":
        strategy = MACrossoverStrategy(fast=fast, slow=slow, config=config)
    elif strategy_id == "momentum_breakout":
        strategy = MomentumBreakoutStrategy(lookback=lookback, config=config)
    elif strategy_id == "mean_reversion":
        strategy = MeanReversionStrategy(bb_window=lookback, config=config)
    else:
        return {"error": f"Strategy not configured: {strategy_id}"}

    engine = BacktestEngine(db)
    report = engine.run(strategy, code_list, start_date, end_date, initial_capital)

    # Save to DB
    result_id = save_backtest_result(db, report)

    return {
        "id": result_id,
        "strategy": report.strategy_name,
        "period": f"{report.start_date} ~ {report.end_date}",
        "performance": {
            "initial_capital": report.initial_capital,
            "final_value": report.final_value,
            "total_return_pct": report.total_return,
            "annual_return_pct": report.annual_return,
            "sharpe_ratio": report.sharpe_ratio,
            "max_drawdown_pct": report.max_drawdown,
            "win_rate_pct": report.win_rate,
            "total_trades": report.total_trades,
            "profit_factor": report.profit_factor,
        },
        "trades": report.trades[:20],
        "equity_curve": report.daily_values,
    }


@router.get("/backtest/history")
def list_backtests(
    limit: int = Query(20, ge=1, le=100),
    db: Session = Depends(get_db),
):
    """列出历史回测记录。"""
    rows = (
        db.query(BacktestResultModel)
        .order_by(BacktestResultModel.created_at.desc())
        .limit(limit)
        .all()
    )
    return {
        "total": len(rows),
        "results": [
            {
                "id": r.id,
                "strategy": r.strategy_name,
                "period": f"{r.start_date} ~ {r.end_date}",
                "return_pct": r.total_return,
                "sharpe": r.sharpe_ratio,
                "max_dd_pct": r.max_drawdown,
                "win_rate_pct": r.win_rate,
                "trades": r.total_trades,
                "created_at": r.created_at.isoformat() if r.created_at else None,
            }
            for r in rows
        ],
    }


@router.get("/backtest/{result_id}")
def get_backtest(result_id: int, db: Session = Depends(get_db)):
    """获取单个回测结果的完整数据（含权益曲线和交易记录）。"""
    r = db.query(BacktestResultModel).filter(BacktestResultModel.id == result_id).first()
    if not r:
        return {"error": "Result not found"}

    import json

    return {
        "id": r.id,
        "strategy": r.strategy_name,
        "period": f"{r.start_date} ~ {r.end_date}",
        "performance": {
            "initial_capital": r.initial_capital,
            "final_value": r.final_value,
            "total_return_pct": r.total_return,
            "annual_return_pct": r.annual_return,
            "sharpe_ratio": r.sharpe_ratio,
            "max_drawdown_pct": r.max_drawdown,
            "win_rate_pct": r.win_rate,
            "total_trades": r.total_trades,
            "profit_factor": r.profit_factor,
        },
        "equity_curve": json.loads(r.daily_values_json) if r.daily_values_json else [],
        "trades": json.loads(r.trade_log_json) if r.trade_log_json else [],
    }


# ── Paper Trading ──────────────────────────────────────────────────

@router.post("/paper/account")
def create_paper_account(
    name: str = Query("default"),
    initial_capital: float = Query(100_000),
    db: Session = Depends(get_db),
):
    """创建或重置纸交易账户。"""
    svc = PaperTradingService(db)
    account = svc.get_or_create_account(name, initial_capital)
    return svc.get_account_summary(account.id)


@router.get("/paper/account/{account_id}")
def get_paper_account(account_id: int, db: Session = Depends(get_db)):
    """获取纸交易账户状态。"""
    svc = PaperTradingService(db)
    return svc.get_account_summary(account_id)


@router.get("/paper/orders/{account_id}")
def get_paper_orders(account_id: int, limit: int = 50, db: Session = Depends(get_db)):
    """获取纸交易订单历史。"""
    svc = PaperTradingService(db)
    return {"orders": svc.get_orders(account_id, limit=limit)}
