"""Research API — AI review, trading memory, factor mining, rolling training."""

from datetime import date

from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy.orm import Session

from ..core.database import get_db
from ..models.finance import FactorScore, FinancialIndicator
from ..models.market import DailyQuote
from ..models.stock import Stock

router = APIRouter(prefix="/research", tags=["research"])


# ── AI Stock Review ─────────────────────────────────────────────────

@router.get("/review/{code}")
def review_stock(
    code: str,
    trade_date: str | None = Query(None, description="交易日期 YYYY-MM-DD，默认最新"),
    db: Session = Depends(get_db),
):
    """AI-powered stock review with bull/bear debate analysis."""
    from ..services.ai_review import AIReviewer

    reviewer = AIReviewer(db)
    return reviewer.review_stock(code, trade_date)


@router.post("/review/batch")
def batch_review_stocks(
    codes: str = Query(..., description="逗号分隔的股票代码，最多10只"),
    trade_date: str | None = Query(None),
    db: Session = Depends(get_db),
):
    """Batch AI review for multiple stocks."""
    code_list = [c.strip() for c in codes.split(",") if c.strip()][:10]
    if not code_list:
        raise HTTPException(status_code=400, detail="No valid stock codes")

    from ..services.ai_review import AIReviewer

    reviewer = AIReviewer(db)
    return reviewer.batch_review(code_list, trade_date)


@router.get("/review/market-environment")
def assess_market(db: Session = Depends(get_db)):
    """AI assessment of current market environment."""
    from ..services.ai_review import AIReviewer

    reviewer = AIReviewer(db)
    return reviewer.assess_market_environment(date.today().isoformat())


# ── Trading Memory ──────────────────────────────────────────────────

@router.get("/memory/log")
def get_decision_log(
    limit: int = Query(50, ge=1, le=200),
    db: Session = Depends(get_db),
):
    """Get recent trading decision log."""
    from ..services.memory import MemorySystem

    ms = MemorySystem(db)
    return {"decisions": ms.get_decisions(limit)}


@router.get("/memory/lessons")
def get_lessons(
    limit: int = Query(20, ge=1, le=100),
    db: Session = Depends(get_db),
):
    """Get learned lessons from past trading decisions (5-day deferred)."""
    from ..services.memory import MemorySystem

    ms = MemorySystem(db)
    ms.review_pending()
    return {"lessons": ms.get_lessons(limit=limit)}


@router.get("/memory/stats")
def get_memory_stats(db: Session = Depends(get_db)):
    """Get trading memory statistics."""
    from ..services.memory import MemorySystem

    ms = MemorySystem(db)
    return ms.get_stats()


# ── Factor Mining ───────────────────────────────────────────────────

@router.post("/factors/mine")
def mine_factors(
    trade_date: str = Query(..., description="基准日期 YYYY-MM-DD"),
    n_attempts: int = Query(10, ge=1, le=30, description="最大尝试次数"),
    min_ic: float = Query(0.02, ge=0.0, le=0.1, description="最小IC阈值"),
    db: Session = Depends(get_db),
):
    """Run automated factor mining (LLM-driven hypothesis → test → refine)."""
    from ..services.factor_miner import FactorMiner

    miner = FactorMiner(db)
    results = miner.mine(trade_date, n_attempts=n_attempts, min_ic=min_ic)
    return {"iterations": len(results), "results": results}


# ── Rolling Training ────────────────────────────────────────────────

@router.post("/ml/rolling-train")
def rolling_train(
    start_date: str = Query("2024-01-01"),
    end_date: str = Query("2026-05-01"),
    train_months: int = Query(12, ge=6, le=24, description="训练窗口(月)"),
    test_months: int = Query(1, ge=1, le=6, description="测试窗口(月)"),
    step_months: int = Query(1, ge=1, le=6, description="滑动步长(月)"),
    model_type: str = Query("lightgbm", description="模型类型: lightgbm, alstm"),
    db: Session = Depends(get_db),
):
    """Run rolling window ML training with out-of-sample prediction stitching."""
    from ..services.rolling_trainer import RollingTrainer

    trainer = RollingTrainer(db, train_window=train_months, test_window=test_months, step=step_months)
    report = trainer.run(start_date, end_date, model_type=model_type)
    return report


# ── Deep Learning ───────────────────────────────────────────────────

@router.post("/ml/dl-train")
def dl_train(
    model_type: str = Query("alstm", description="模型类型: alstm, gru, transformer"),
    start_date: str = Query("2024-01-01"),
    end_date: str = Query("2026-05-01"),
    epochs: int = Query(50, ge=10, le=200),
    db: Session = Depends(get_db),
):
    """Train a deep learning model (ALSTM/GRU/Transformer) for return prediction."""
    from ..services.pytorch_models import DeepLearningPipeline, MODEL_REGISTRY

    if model_type not in MODEL_REGISTRY:
        raise HTTPException(status_code=400, detail=f"Unknown model type: {model_type}. Available: {list(MODEL_REGISTRY.keys())}")

    dl = DeepLearningPipeline(model_name=model_type)
    metrics = dl.train(start_date, end_date, epochs=epochs)
    return {"model_type": model_type, "metrics": metrics}
