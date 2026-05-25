"""QuantForge 熔量 — FastAPI application entry point."""

from contextlib import asynccontextmanager

from pathlib import Path

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

from .api import dashboard, market_data, research, trading
from .core.config import settings
from .core.database import Base, engine
from .models import DailyQuote, Stock, PaperAccount, PaperPosition, PaperOrder, BacktestResult, FinancialIndicator, FactorScore, FundFlow, MLPrediction, MarginTrading, ShareholderCount, DragonTiger, LockupRelease, TradeMemory  # noqa: F401


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Startup: create tables; shutdown: clean up."""
    Base.metadata.create_all(bind=engine)

    # Lightweight migrations — add new columns if they don't exist
    from sqlalchemy import text
    with engine.connect() as conn:
        migrations = [
            ("financial_indicators", "bv_per_share", "FLOAT"),
            ("financial_indicators", "revenue_per_share", "FLOAT"),
            ("backtest_results", "feature_importance_json", "TEXT"),
            ("backtest_results", "ic_mean", "FLOAT"),
        ]
        for table, col, col_type in migrations:
            try:
                conn.execute(text(f"ALTER TABLE {table} ADD COLUMN IF NOT EXISTS {col} {col_type}"))
                conn.commit()
            except Exception:
                conn.rollback()

        # Column type migrations (ALTER COLUMN TYPE)
        type_migrations = [
            "ALTER TABLE stocks ALTER COLUMN industry TYPE VARCHAR(100)",
        ]
        for sql in type_migrations:
            try:
                conn.execute(text(sql))
                conn.commit()
            except Exception:
                conn.rollback()

        # Performance indexes — CREATE INDEX IF NOT EXISTS
        indexes = [
            "CREATE INDEX IF NOT EXISTS idx_daily_quotes_trade_date ON daily_quotes(trade_date)",
            "CREATE INDEX IF NOT EXISTS idx_factor_scores_trade_date ON factor_scores(trade_date)",
            "CREATE INDEX IF NOT EXISTS idx_ml_predictions_trade_date ON ml_predictions(trade_date)",
            "CREATE INDEX IF NOT EXISTS idx_factor_scores_code_date ON factor_scores(code, trade_date)",
            "CREATE INDEX IF NOT EXISTS idx_ml_predictions_code_date ON ml_predictions(code, trade_date)",
        ]
        for idx_sql in indexes:
            try:
                conn.execute(text(idx_sql))
                conn.commit()
            except Exception:
                conn.rollback()
    yield


app = FastAPI(
    title=f"{settings.PROJECT_NAME}（{settings.CHINESE_NAME}）",
    version=settings.VERSION,
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.CORS_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(dashboard.router)
app.include_router(market_data.router, prefix="/api")
app.include_router(trading.router, prefix="/api")
app.include_router(research.router, prefix="/api")

_static_dir = Path(__file__).parent / "static"
_static_dir.mkdir(exist_ok=True)
app.mount("/static", StaticFiles(directory=str(_static_dir)), name="static")


@app.get("/api/health")
def health():
    return {
        "project": settings.PROJECT_NAME,
        "chinese_name": settings.CHINESE_NAME,
        "version": settings.VERSION,
        "status": "ok",
    }
