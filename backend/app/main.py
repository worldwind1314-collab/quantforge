"""QuantForge 熔量 — FastAPI application entry point."""

from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from .api import market_data
from .core.config import settings
from .core.database import Base, engine
from .models import DailyQuote, Stock  # noqa: F401 ensure models registered


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Startup: create tables; shutdown: clean up."""
    Base.metadata.create_all(bind=engine)
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


@app.get("/api/health")
def health():
    return {
        "project": settings.PROJECT_NAME,
        "chinese_name": settings.CHINESE_NAME,
        "version": settings.VERSION,
        "status": "ok",
    }
