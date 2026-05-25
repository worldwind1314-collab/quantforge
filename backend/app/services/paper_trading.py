"""Paper trading service — simulated order execution, position tracking, P&L."""

import logging
from datetime import date, datetime

from sqlalchemy.orm import Session

from ..core.database import SessionLocal
from ..models.trading import PaperAccount, PaperOrder, PaperPosition
from .strategy_engine import Signal, SignalType

logger = logging.getLogger(__name__)


class PaperTradingService:
    """Manage virtual accounts, execute signals, track P&L."""

    def __init__(self, db: Session | None = None):
        self._db = db

    def _db_session(self) -> Session:
        return self._db or SessionLocal()

    # ── Account ────────────────────────────────────────────────────

    def get_or_create_account(self, name: str = "default", initial_capital: float = 100_000) -> PaperAccount:
        db = self._db_session()
        account = db.query(PaperAccount).filter(PaperAccount.name == name).first()
        if not account:
            account = PaperAccount(name=name, initial_capital=initial_capital, cash=initial_capital, total_value=initial_capital)
            db.add(account)
            db.commit()
            db.refresh(account)
        return account

    def get_account_summary(self, account_id: int) -> dict:
        db = self._db_session()
        account = db.query(PaperAccount).filter(PaperAccount.id == account_id).first()
        if not account:
            return {"error": "Account not found"}

        positions = db.query(PaperPosition).filter(PaperPosition.account_id == account_id).all()
        total_equity = sum(p.market_value for p in positions)
        total_value = account.cash + total_equity
        total_pnl = total_value - account.initial_capital

        return {
            "account_id": account.id,
            "name": account.name,
            "initial_capital": account.initial_capital,
            "cash": account.cash,
            "equity": total_equity,
            "total_value": total_value,
            "total_pnl": round(total_pnl, 2),
            "total_pnl_pct": round(total_pnl / account.initial_capital * 100, 2),
            "positions": [
                {
                    "code": p.code,
                    "quantity": p.quantity,
                    "avg_cost": p.avg_cost,
                    "current_price": p.current_price,
                    "market_value": p.market_value,
                    "unrealized_pnl": p.unrealized_pnl,
                    "unrealized_pnl_pct": round(p.unrealized_pnl / (p.avg_cost * p.quantity) * 100, 2) if p.quantity > 0 else 0,
                }
                for p in positions
                if p.quantity > 0
            ],
            "updated_at": account.updated_at.isoformat() if account.updated_at else None,
        }

    # ── Signal execution ───────────────────────────────────────────

    def execute_signals(
        self, account_id: int, signals: list[Signal], prices: dict[str, float] | None = None
    ) -> list[dict]:
        """Execute a batch of signals against a paper account.

        Args:
            account_id: Paper account ID
            signals: List of signals to execute
            prices: {code: current_price} — if None, signal.price is used
        """
        db = self._db_session()
        account = db.query(PaperAccount).filter(PaperAccount.id == account_id).first()
        if not account:
            return [{"error": "Account not found"}]

        results = []
        for sig in signals:
            price = (prices or {}).get(sig.code, sig.price)
            if not price or price <= 0:
                results.append({"code": sig.code, "error": "No valid price", "signal": sig.signal_type.value})
                continue

            if sig.signal_type == SignalType.BUY:
                result = self._execute_buy(db, account, sig, price)
            elif sig.signal_type == SignalType.SELL:
                result = self._execute_sell(db, account, sig, price)
            else:
                continue
            results.append(result)

        # Recalculate total value
        self._update_account_value(db, account)
        db.commit()
        return results

    def _execute_buy(self, db: Session, account: PaperAccount, sig: Signal, price: float) -> dict:
        """Execute a buy order. Deduct cash, create/update position."""
        qty = sig.quantity or self._calc_buy_qty(account.cash, price)
        if qty <= 0:
            return {"code": sig.code, "error": "Quantity too small", "price": price}

        cost = price * qty + self._trade_cost(price, qty, "BUY")
        if cost > account.cash:
            return {"code": sig.code, "error": f"Insufficient cash: need {cost:.2f}, have {account.cash:.2f}"}

        account.cash -= cost

        # Create order
        order = PaperOrder(
            account_id=account.id,
            code=sig.code,
            direction="BUY",
            quantity=qty,
            price=price,
            status="filled",
            strategy=sig.reason,
        )
        db.add(order)

        # Update or create position
        pos = db.query(PaperPosition).filter(
            PaperPosition.account_id == account.id,
            PaperPosition.code == sig.code,
        ).first()

        if pos:
            total_qty = pos.quantity + qty
            total_cost = pos.avg_cost * pos.quantity + price * qty
            pos.quantity = total_qty
            pos.avg_cost = round(total_cost / total_qty, 4) if total_qty > 0 else 0
            pos.current_price = price
            pos.market_value = total_qty * price
            pos.unrealized_pnl = (price - pos.avg_cost) * total_qty
        else:
            pos = PaperPosition(
                account_id=account.id,
                code=sig.code,
                quantity=qty,
                avg_cost=price,
                current_price=price,
                market_value=qty * price,
                unrealized_pnl=0.0,
            )
            db.add(pos)

        return {
            "code": sig.code,
            "action": "BUY",
            "quantity": qty,
            "price": price,
            "cost": round(cost, 2),
            "reason": sig.reason,
        }

    def _execute_sell(self, db: Session, account: PaperAccount, sig: Signal, price: float) -> dict:
        """Execute a sell order. Add proceeds, reduce position."""
        pos = db.query(PaperPosition).filter(
            PaperPosition.account_id == account.id,
            PaperPosition.code == sig.code,
        ).first()

        if not pos or pos.quantity <= 0:
            return {"code": sig.code, "error": "No position to sell"}

        qty = sig.quantity if sig.quantity > 0 else pos.quantity

        proceeds = price * qty - self._trade_cost(price, qty, "SELL")
        account.cash += proceeds

        # Create order
        order = PaperOrder(
            account_id=account.id,
            code=sig.code,
            direction="SELL",
            quantity=qty,
            price=price,
            status="filled",
            strategy=sig.reason,
        )
        db.add(order)

        # Update position
        pos.quantity -= qty
        if pos.quantity <= 0:
            db.delete(pos)
        else:
            pos.current_price = price
            pos.market_value = pos.quantity * price
            pos.unrealized_pnl = (price - pos.avg_cost) * pos.quantity

        realized_pnl = (price - pos.avg_cost) * qty  # consistent PnL calculation
        return {
            "code": sig.code,
            "action": "SELL",
            "quantity": qty,
            "price": price,
            "proceeds": round(proceeds, 2),
            "pnl": round(realized_pnl, 2),
            "reason": sig.reason,
        }

    @staticmethod
    def _calc_buy_qty(cash: float, price: float, max_pct: float = 0.2) -> int:
        max_value = cash * max_pct
        qty = int(max_value / price // 100) * 100
        return max(qty, 0)

    @staticmethod
    def _trade_cost(price: float, quantity: int, direction: str) -> float:
        from .backtest_engine import trade_cost
        return trade_cost(price, quantity, direction)

    @staticmethod
    def _update_account_value(db: Session, account: PaperAccount):
        positions = db.query(PaperPosition).filter(
            PaperPosition.account_id == account.id,
            PaperPosition.quantity > 0,
        ).all()
        total_equity = sum(p.market_value for p in positions)
        account.total_value = account.cash + total_equity
        account.updated_at = datetime.now()

    # ── Order history ──────────────────────────────────────────────

    def get_orders(self, account_id: int, limit: int = 50) -> list[dict]:
        db = self._db_session()
        orders = (
            db.query(PaperOrder)
            .filter(PaperOrder.account_id == account_id)
            .order_by(PaperOrder.filled_at.desc())
            .limit(limit)
            .all()
        )
        return [
            {
                "id": o.id,
                "code": o.code,
                "direction": o.direction,
                "quantity": o.quantity,
                "price": o.price,
                "status": o.status,
                "strategy": o.strategy,
                "filled_at": o.filled_at.isoformat() if o.filled_at else None,
            }
            for o in orders
        ]

    # ── Daily mark-to-market ───────────────────────────────────────

    def mark_to_market(self, account_id: int, prices: dict[str, float]):
        """Update position prices to current market. Call this daily."""
        db = self._db_session()
        positions = db.query(PaperPosition).filter(
            PaperPosition.account_id == account_id,
            PaperPosition.quantity > 0,
        ).all()

        for pos in positions:
            price = prices.get(pos.code)
            if price and price > 0:
                pos.current_price = price
                pos.market_value = pos.quantity * price
                pos.unrealized_pnl = (price - pos.avg_cost) * pos.quantity

        account = db.query(PaperAccount).filter(PaperAccount.id == account_id).first()
        if account:
            self._update_account_value(db, account)

        db.commit()
