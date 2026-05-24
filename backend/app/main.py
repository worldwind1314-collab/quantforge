"""QuantForge 熔量 — FastAPI application entry point."""

from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from .api import dashboard, market_data, trading
from .core.config import settings
from .core.database import Base, engine
from .models import DailyQuote, Stock, PaperAccount, PaperPosition, PaperOrder, BacktestResult, FinancialIndicator, FactorScore, MLPrediction  # noqa: F401


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


@app.get("/api/health")
def health():
    return {
        "project": settings.PROJECT_NAME,
        "chinese_name": settings.CHINESE_NAME,
        "version": settings.VERSION,
        "status": "ok",
    }
