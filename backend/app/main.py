"""QuantForge 熔量 — FastAPI application entry point."""

from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from .api import market_data, trading
from .core.config import settings
from .core.database import Base, engine
from .models import DailyQuote, Stock, PaperAccount, PaperPosition, PaperOrder, BacktestResult, FinancialIndicator, FactorScore, MLPrediction  # noqa: F401


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Startup: create tables; shutdown: clean up."""
    Base.metadata.create_all(bind=engine)

    # Add new columns if they don't exist (lightweight migration)
    from sqlalchemy import text
    with engine.connect() as conn:
        for col, col_type in [("bv_per_share", "FLOAT"), ("revenue_per_share", "FLOAT")]:
            try:
                conn.execute(text(f"ALTER TABLE financial_indicators ADD COLUMN IF NOT EXISTS {col} {col_type}"))
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
