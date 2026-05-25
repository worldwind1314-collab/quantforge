"""Trading memory — trade decision log with deferred reflection.

Every trade decision is logged with context. After 5 trading days, the system
auto-reviews the outcome: was the decision correct? What can we learn?
"""

import json
import logging
from datetime import date, datetime, timedelta

from sqlalchemy.orm import Session

from ..core.database import Base, SessionLocal
from ..models.market import DailyQuote
from ..models.stock import Stock
from .ai_review import AIReviewer

logger = logging.getLogger(__name__)

from sqlalchemy import DateTime, Float, Integer, String, Text, func
from sqlalchemy.orm import Mapped, mapped_column


class TradeMemory(Base):
    """Append-only trade decision log with deferred outcome reflection."""

    __tablename__ = "trade_memories"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    code: Mapped[str] = mapped_column(String(12), nullable=False, index=True)
    decision_date: Mapped[str] = mapped_column(String(10), nullable=False)
    decision: Mapped[str] = mapped_column(String(10), nullable=False)  # buy/add/hold/reduce/sell
    rating: Mapped[int] = mapped_column(Integer, default=3)  # 1-5 AI rating
    confidence: Mapped[float] = mapped_column(Float, default=0.5)
    entry_price: Mapped[float | None] = mapped_column(Float, nullable=True)
    position_pct: Mapped[float | None] = mapped_column(Float, nullable=True)
    reason: Mapped[str | None] = mapped_column(Text, nullable=True)
    context_json: Mapped[str | None] = mapped_column(Text, nullable=True)  # full AI review context

    # Deferred outcome (filled 5 days later)
    outcome_pnl_pct: Mapped[float | None] = mapped_column(Float, nullable=True)
    outcome_reviewed: Mapped[bool] = mapped_column(default=False)
    outcome_date: Mapped[str | None] = mapped_column(String(10), nullable=True)
    lesson_learned: Mapped[str | None] = mapped_column(Text, nullable=True)
    was_correct: Mapped[bool | None] = mapped_column(default=None)  # True if direction was right

    created_at: Mapped[datetime] = mapped_column(DateTime, server_default=func.now())


class MemorySystem:
    """Manages the trading memory lifecycle.

    Flow:
      1. log_decision() — record decision at entry time
      2. review_pending() — after 5 trading days, compare predicted vs actual
      3. get_lessons() — retrieve accumulated lessons for a stock or strategy
    """

    def __init__(self, db: Session | None = None):
        self._db = db
        self._reviewer = AIReviewer(db)

    def _get_db(self) -> Session:
        return self._db or SessionLocal()

    # ── Decision logging ───────────────────────────────────────────

    def log_decision(
        self,
        code: str,
        decision_date: str,
        decision: str,
        rating: int = 3,
        confidence: float = 0.5,
        entry_price: float | None = None,
        position_pct: float | None = None,
        reason: str | None = None,
        context: dict | None = None,
    ) -> int:
        """Record a trading decision. Returns the memory ID."""
        db = self._get_db()

        memory = TradeMemory(
            code=code,
            decision_date=decision_date,
            decision=decision,
            rating=rating,
            confidence=confidence,
            entry_price=entry_price,
            position_pct=position_pct,
            reason=reason,
            context_json=json.dumps(context, ensure_ascii=False, default=str) if context else None,
        )
        db.add(memory)
        db.commit()
        db.refresh(memory)
        logger.info(f"Memory #{memory.id}: {decision} {code} on {decision_date} (rating={rating})")
        return memory.id

    # ── Deferred reflection (5-day auto review) ───────────────────

    def review_pending(self, lookback_days: int = 10) -> list[dict]:
        """Review all decisions that are 5+ trading days old and unreviewed.

        For each pending memory:
          1. Calculate actual 5-day forward return
          2. Determine if the decision direction was correct
          3. Generate a lesson learned
        """
        db = self._get_db()
        cutoff = (date.today() - timedelta(days=6)).isoformat()

        pending = (
            db.query(TradeMemory)
            .filter(
                TradeMemory.outcome_reviewed == False,  # noqa: E712
                TradeMemory.decision_date <= cutoff,
            )
            .all()
        )

        results = []
        for mem in pending:
            try:
                outcome = self._compute_outcome(mem)
                mem.outcome_pnl_pct = outcome["pnl_pct"]
                mem.outcome_date = outcome["outcome_date"]
                mem.was_correct = outcome["was_correct"]
                mem.lesson_learned = self._generate_lesson(mem, outcome)
                mem.outcome_reviewed = True
                db.commit()
                results.append({
                    "id": mem.id,
                    "code": mem.code,
                    "decision": mem.decision,
                    "decision_date": mem.decision_date,
                    "pnl_pct": outcome["pnl_pct"],
                    "was_correct": outcome["was_correct"],
                    "lesson": mem.lesson_learned,
                })
                logger.info(
                    f"Memory #{mem.id} reviewed: {mem.decision} {mem.code} → "
                    f"{outcome['pnl_pct']:+.2f}%, correct={outcome['was_correct']}"
                )
            except Exception as e:
                logger.warning(f"Failed to review memory #{mem.id}: {e}")
                continue

        return results

    def _compute_outcome(self, mem: TradeMemory) -> dict:
        """Compute the 5-day forward outcome for a memory entry."""
        db = self._get_db()

        quotes = (
            db.query(DailyQuote.trade_date, DailyQuote.close)
            .filter(
                DailyQuote.code == mem.code,
                DailyQuote.trade_date >= mem.decision_date,
            )
            .order_by(DailyQuote.trade_date)
            .limit(6)  # decision day + 5 forward days
            .all()
        )

        if len(quotes) < 2:
            return {"pnl_pct": 0.0, "outcome_date": mem.decision_date, "was_correct": None}

        entry_price = mem.entry_price or quotes[0].close or 0
        exit_price = quotes[-1].close or entry_price

        if entry_price <= 0:
            return {"pnl_pct": 0.0, "outcome_date": quotes[-1].trade_date, "was_correct": None}

        pnl_pct = (exit_price - entry_price) / entry_price * 100

        # Determine if direction was correct
        was_correct = None
        if mem.decision in ("buy", "add") and pnl_pct > 0:
            was_correct = True
        elif mem.decision in ("buy", "add") and pnl_pct < 0:
            was_correct = False
        elif mem.decision in ("sell", "reduce") and pnl_pct < 0:
            was_correct = True  # correctly avoided losses
        elif mem.decision in ("sell", "reduce") and pnl_pct > 0:
            was_correct = False  # missed gains
        # "hold" decisions: correct if no extreme move in either direction
        elif mem.decision == "hold":
            was_correct = abs(pnl_pct) < 5.0

        return {
            "pnl_pct": round(pnl_pct, 2),
            "outcome_date": quotes[-1].trade_date,
            "was_correct": was_correct,
        }

    def _generate_lesson(self, mem: TradeMemory, outcome: dict) -> str:
        """Generate a concise lesson from a reviewed decision."""
        decision_cn = {"buy": "买入", "add": "加仓", "hold": "持有",
                       "reduce": "减仓", "sell": "卖出"}

        if outcome["was_correct"] is True:
            return (
                f"{mem.decision_date} {decision_cn.get(mem.decision, mem.decision)} {mem.code}: "
                f"决策正确，5日后收益{outcome['pnl_pct']:+.2f}%。"
                f"评分{mem.rating}/5，置信度{mem.confidence:.0%}。"
            )
        elif outcome["was_correct"] is False:
            return (
                f"{mem.decision_date} {decision_cn.get(mem.decision, mem.decision)} {mem.code}: "
                f"决策失误，5日后收益{outcome['pnl_pct']:+.2f}%。"
                f"评分{mem.rating}/5可能偏高，需审视判断依据。"
            )
        else:
            return (
                f"{mem.decision_date} {decision_cn.get(mem.decision, mem.decision)} {mem.code}: "
                f"5日后收益{outcome['pnl_pct']:+.2f}%，方向判断不确定。"
            )

    # ── Knowledge retrieval ───────────────────────────────────────

    def get_decisions(self, limit: int = 50) -> list[dict]:
        """Get recent trading decisions (both reviewed and pending)."""
        db = self._get_db()
        memories = (
            db.query(TradeMemory)
            .order_by(TradeMemory.decision_date.desc())
            .limit(limit)
            .all()
        )
        return [
            {
                "id": m.id,
                "code": m.code,
                "decision_date": m.decision_date,
                "decision": m.decision,
                "rating": m.rating,
                "confidence": m.confidence,
                "reason": m.reason,
                "outcome_pnl_pct": m.outcome_pnl_pct,
                "was_correct": m.was_correct,
                "outcome_reviewed": m.outcome_reviewed,
            }
            for m in memories
        ]

    def get_lessons(
        self, code: str | None = None, limit: int = 20, correct_only: bool = False
    ) -> list[dict]:
        """Retrieve accumulated lessons, optionally filtered by stock."""
        db = self._get_db()

        q = db.query(TradeMemory).filter(TradeMemory.outcome_reviewed == True)  # noqa: E712
        if code:
            q = q.filter(TradeMemory.code == code)
        if correct_only:
            q = q.filter(TradeMemory.was_correct == True)  # noqa: E712

        memories = q.order_by(TradeMemory.decision_date.desc()).limit(limit).all()

        return [
            {
                "id": m.id,
                "code": m.code,
                "decision_date": m.decision_date,
                "decision": m.decision,
                "rating": m.rating,
                "pnl_pct": m.outcome_pnl_pct,
                "was_correct": m.was_correct,
                "lesson": m.lesson_learned,
            }
            for m in memories
        ]

    def get_stats(self) -> dict:
        """Get aggregate memory statistics."""
        db = self._get_db()
        from sqlalchemy import func as sql_func

        total = db.query(sql_func.count(TradeMemory.id)).scalar() or 0
        reviewed = (
            db.query(sql_func.count(TradeMemory.id))
            .filter(TradeMemory.outcome_reviewed == True)  # noqa: E712
            .scalar() or 0
        )
        correct = (
            db.query(sql_func.count(TradeMemory.id))
            .filter(TradeMemory.was_correct == True)  # noqa: E712
            .scalar() or 0
        )

        avg_pnl = (
            db.query(sql_func.avg(TradeMemory.outcome_pnl_pct))
            .filter(TradeMemory.outcome_reviewed == True)  # noqa: E712
            .scalar()
        )

        return {
            "total_decisions": total,
            "reviewed": reviewed,
            "pending_review": total - reviewed,
            "correct_decisions": correct,
            "accuracy": round(correct / reviewed * 100, 1) if reviewed > 0 else 0,
            "avg_5d_pnl": round(float(avg_pnl), 2) if avg_pnl else 0,
        }
