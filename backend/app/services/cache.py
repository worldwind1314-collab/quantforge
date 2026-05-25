"""Three-tier caching system for QuantForge.

Tier 1: In-memory LRU with TTL (fastest, per-process)
Tier 2: Disk cache via JSON/Parquet (survives restarts)
Tier 3: Pre-loading warmup on startup

Inspired by Qlib's three-tier cache for factor/data serving.
"""

import json
import logging
import os
import pickle
import threading
import time
from collections import OrderedDict
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

DEFAULT_CACHE_DIR = Path("/tmp/quantforge_cache")


# ── Tier 1: In-memory LRU cache ─────────────────────────────────────

class LRUCache:
    """Thread-safe LRU cache with per-key TTL expiration."""

    def __init__(self, max_size: int = 256, default_ttl: int = 300):
        self._max_size = max_size
        self._default_ttl = default_ttl  # seconds
        self._store: OrderedDict[str, tuple[Any, float]] = OrderedDict()
        self._lock = threading.Lock()

    def get(self, key: str) -> Any | None:
        with self._lock:
            self._evict_expired()
            if key in self._store:
                value, expires_at = self._store[key]
                if time.time() < expires_at:
                    self._store.move_to_end(key)  # bump to MRU
                    return value
                del self._store[key]
            return None

    def set(self, key: str, value: Any, ttl: int | None = None):
        with self._lock:
            self._evict_expired()
            if key in self._store:
                del self._store[key]
            elif len(self._store) >= self._max_size:
                self._store.popitem(last=False)  # evict LRU
            expires_at = time.time() + (ttl if ttl is not None else self._default_ttl)
            self._store[key] = (value, expires_at)

    def delete(self, key: str):
        with self._lock:
            self._store.pop(key, None)

    def clear(self):
        with self._lock:
            self._store.clear()

    def _evict_expired(self):
        now = time.time()
        expired = [k for k, (_, exp) in self._store.items() if now >= exp]
        for k in expired:
            del self._store[k]

    def __len__(self) -> int:
        with self._lock:
            self._evict_expired()
            return len(self._store)

    def keys(self) -> list[str]:
        with self._lock:
            self._evict_expired()
            return list(self._store.keys())


# ── Tier 2: Disk cache ──────────────────────────────────────────────

class DiskCache:
    """File-based cache for larger objects that survive restarts.

    Uses pickle for arbitrary Python objects. Each key maps to a file.
    Automatically cleans expired entries on read.
    """

    def __init__(self, cache_dir: Path | str = DEFAULT_CACHE_DIR):
        self._dir = Path(cache_dir)
        self._dir.mkdir(parents=True, exist_ok=True)

    def get(self, key: str) -> Any | None:
        path = self._key_path(key)
        if not path.exists():
            return None

        try:
            with open(path, "rb") as f:
                meta = pickle.load(f)
            if time.time() >= meta.get("expires_at", 0):
                path.unlink(missing_ok=True)
                return None
            return meta["value"]
        except (pickle.PickleError, EOFError, KeyError) as e:
            logger.debug(f"Disk cache read error for {key}: {e}")
            path.unlink(missing_ok=True)
            return None

    def set(self, key: str, value: Any, ttl: int = 3600):
        path = self._key_path(key)
        try:
            with open(path, "wb") as f:
                pickle.dump({
                    "value": value,
                    "expires_at": time.time() + ttl,
                    "created_at": time.time(),
                }, f)
        except (pickle.PickleError, OSError) as e:
            logger.warning(f"Disk cache write error for {key}: {e}")

    def delete(self, key: str):
        self._key_path(key).unlink(missing_ok=True)

    def clear(self):
        for path in self._dir.glob("qc_*.cache"):
            path.unlink(missing_ok=True)

    def size(self) -> int:
        return len(list(self._dir.glob("qc_*.cache")))

    def _key_path(self, key: str) -> Path:
        safe = "".join(c if c.isalnum() or c in "_-" else "_" for c in key)
        return self._dir / f"qc_{safe}.cache"


# ── Tier 3: Pre-loading warmup ──────────────────────────────────────

class CacheWarmer:
    """Pre-loads frequently accessed data into memory cache on startup.

    Typical usage:
        warmer = CacheWarmer(mem_cache)
        warmer.warmup_factors()
        warmer.warmup_stock_list()
    """

    def __init__(self, mem_cache: LRUCache, disk_cache: DiskCache | None = None):
        self._mem = mem_cache
        self._disk = disk_cache

    def warmup_stock_list(self):
        """Load active stock codes into cache."""
        from ..core.database import SessionLocal
        from ..models.stock import Stock

        try:
            db = SessionLocal()
            codes = [
                r[0] for r in db.query(Stock.code)
                .filter(Stock.is_active == True)
                .order_by(Stock.code).all()
            ]
            db.close()

            self._mem.set("stock:active_codes", codes, ttl=3600)
            logger.info(f"Warmed up stock list: {len(codes)} active codes")
        except Exception as e:
            logger.warning(f"Stock list warmup failed: {e}")

    def warmup_latest_dates(self):
        """Pre-load latest trading dates for key tables."""
        from ..core.database import SessionLocal
        from ..models.market import DailyQuote
        from ..models.finance import FactorScore, MLPrediction
        from sqlalchemy import func

        try:
            db = SessionLocal()
            for name, model in [("quote", DailyQuote), ("factor", FactorScore), ("pred", MLPrediction)]:
                latest = db.query(func.max(model.trade_date)).scalar()
                if latest:
                    self._mem.set(f"latest:{name}_date", latest, ttl=600)
            db.close()
            logger.info("Warmed up latest dates")
        except Exception as e:
            logger.warning(f"Latest dates warmup failed: {e}")

    def warmup_factor_cache(self, codes: list[str] | None = None, trade_date: str | None = None):
        """Pre-compute and cache factor scores for recent date."""
        from ..core.database import SessionLocal
        from ..models.market import DailyQuote
        from ..models.finance import FactorScore
        from sqlalchemy import func

        try:
            db = SessionLocal()
            if trade_date is None:
                trade_date = db.query(func.max(FactorScore.trade_date)).scalar()
            if not trade_date:
                db.close()
                return

            scores = (
                db.query(FactorScore)
                .filter(FactorScore.trade_date == trade_date)
                .all()
            )
            if scores:
                score_map = {
                    s.code: {
                        "composite": s.composite_score,
                        "momentum": s.momentum_score,
                        "value": s.value_score,
                        "quality": s.quality_score,
                        "volatility": s.volatility_score,
                    }
                    for s in scores
                }
                self._mem.set(f"factors:{trade_date}", score_map, ttl=3600)
                logger.info(f"Warmed up {len(score_map)} factor scores for {trade_date}")
            db.close()
        except Exception as e:
            logger.warning(f"Factor warmup failed: {e}")


# ── Singleton cache instance ─────────────────────────────────────────

# Global LRU and Disk cache instances
_mem_cache: LRUCache | None = None
_disk_cache: DiskCache | None = None


def get_cache(cache_dir: Path | str = DEFAULT_CACHE_DIR) -> tuple[LRUCache, DiskCache]:
    """Get or create the global cache instances."""
    global _mem_cache, _disk_cache
    if _mem_cache is None:
        _mem_cache = LRUCache(max_size=512, default_ttl=300)
    if _disk_cache is None:
        _disk_cache = DiskCache(cache_dir)
    return _mem_cache, _disk_cache


# ── Decorator for function result caching ────────────────────────────

def cached(ttl: int = 300, key_prefix: str = "", use_disk: bool = False):
    """Decorator: cache function return value with TTL.

    Usage:
        @cached(ttl=600, key_prefix="data")
        def fetch_expensive_data(code: str) -> dict:
            ...
    """
    def decorator(func):
        def wrapper(*args, **kwargs):
            mem, disk = get_cache()
            key_parts = [key_prefix, func.__name__]
            key_parts.extend(str(a) for a in args)
            key_parts.extend(f"{k}={v}" for k, v in sorted(kwargs.items()))
            cache_key = ":".join(key_parts)

            # Try memory first
            result = mem.get(cache_key)
            if result is not None:
                return result

            # Try disk if enabled
            if use_disk and disk:
                result = disk.get(cache_key)
                if result is not None:
                    mem.set(cache_key, result, ttl)
                    return result

            # Compute and cache
            result = func(*args, **kwargs)
            mem.set(cache_key, result, ttl)
            if use_disk and disk:
                disk.set(cache_key, result, ttl * 2)
            return result

        return wrapper
    return decorator
