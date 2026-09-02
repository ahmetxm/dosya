"""In-memory paper broker. No wallet, no live orders."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from typing import Iterable

from .opportunities import Leg, Opportunity


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


@dataclass
class Position:
    token_id: str
    market_id: str
    question: str
    outcome: str
    shares: float
    avg_price: float
    realized_pnl: float = 0.0

    def mark(self, mid: float) -> float:
        return self.shares * mid


@dataclass
class Trade:
    ts: str
    opportunity_kind: str
    locked: bool
    reason: str
    expected_pnl: float
    cash_delta: float
    legs: list[dict]


class PaperBroker:
    def __init__(self, starting_cash: float = 10_000.0) -> None:
        self.starting_cash = starting_cash
        self.cash = starting_cash
        self.positions: dict[str, Position] = {}
        self.trades: list[Trade] = []
        self.marks: dict[str, float] = {}
        self.rejected: list[dict] = []

    def reset(self, starting_cash: float | None = None) -> None:
        self.starting_cash = starting_cash if starting_cash is not None else self.starting_cash
        self.cash = self.starting_cash
        self.positions.clear()
        self.trades.clear()
        self.marks.clear()
        self.rejected.clear()

    def update_marks(self, mids: dict[str, float]) -> None:
        self.marks.update({key: value for key, value in mids.items() if value is not None})

    def equity(self) -> float:
        marked = 0.0
        for token_id, position in self.positions.items():
            mid = self.marks.get(token_id)
            if mid is None:
                mid = position.avg_price
            marked += position.mark(mid)
        return self.cash + marked

    def execute(self, opportunity: Opportunity, max_notional_frac: float = 0.2) -> Trade | None:
        if not opportunity.legs:
            return None
        scale = self._affordable_scale(opportunity.legs, max_notional_frac)
        if scale <= 0:
            self.rejected.append(
                {
                    "ts": _now_iso(),
                    "kind": opportunity.kind,
                    "reason": "insufficient paper cash or size",
                    "question": opportunity.legs[0].question,
                }
            )
            return None
        legs = [_scale_leg(leg, scale) for leg in opportunity.legs]
        cash_delta = 0.0
        for leg in legs:
            cash_delta += self._apply_leg(leg)
        trade = Trade(
            ts=_now_iso(),
            opportunity_kind=opportunity.kind,
            locked=opportunity.locked,
            reason=opportunity.reason,
            expected_pnl=opportunity.expected_pnl * scale,
            cash_delta=cash_delta,
            legs=[asdict(leg) for leg in legs],
        )
        self.trades.append(trade)
        return trade

    def _affordable_scale(self, legs: Iterable[Leg], max_notional_frac: float) -> float:
        debit = sum(max(0.0, -leg.cash) for leg in legs)
        budget = min(self.cash * 0.98, self.starting_cash * max_notional_frac)
        if debit <= 0:
            return 1.0
        if budget < 1e-9:
            return 0.0
        scale = min(1.0, budget / debit)
        shares = min(leg.shares for leg in legs) * scale
        if shares + 1e-9 < 5:
            return 0.0
        return scale

    def _apply_leg(self, leg: Leg) -> float:
        signed = leg.shares if leg.side == "buy" else -leg.shares
        existing = self.positions.get(leg.token_id)
        self.cash += leg.cash
        if existing is None:
            avg = leg.price if signed > 0 else leg.price
            self.positions[leg.token_id] = Position(
                token_id=leg.token_id,
                market_id=leg.market_id,
                question=leg.question,
                outcome=leg.outcome,
                shares=signed,
                avg_price=avg,
            )
            return leg.cash

        new_shares = existing.shares + signed
        if abs(new_shares) < 1e-9:
            # Flat: realize vs average.
            existing.realized_pnl += _realize(existing, signed, leg.price)
            del self.positions[leg.token_id]
            return leg.cash
        if existing.shares * signed > 0:
            total = abs(existing.shares) + abs(signed)
            existing.avg_price = (
                existing.avg_price * abs(existing.shares) + leg.price * abs(signed)
            ) / total
            existing.shares = new_shares
            return leg.cash

        existing.realized_pnl += _realize(existing, signed, leg.price)
        existing.shares = new_shares
        existing.avg_price = leg.price
        return leg.cash

    def snapshot(self) -> dict:
        positions = []
        for token_id, position in self.positions.items():
            mid = self.marks.get(token_id, position.avg_price)
            positions.append(
                {
                    **asdict(position),
                    "mid": mid,
                    "unrealized": position.shares * (mid - position.avg_price),
                    "mtm": position.mark(mid),
                }
            )
        return {
            "starting_cash": self.starting_cash,
            "cash": self.cash,
            "equity": self.equity(),
            "pnl": self.equity() - self.starting_cash,
            "open_positions": len(self.positions),
            "trade_count": len(self.trades),
            "positions": positions,
            "trades": [asdict(trade) for trade in self.trades[-80:]],
            "rejected": self.rejected[-20:],
        }


def _realize(position: Position, signed_delta: float, price: float) -> float:
    closed = min(abs(position.shares), abs(signed_delta))
    direction = 1.0 if position.shares > 0 else -1.0
    return closed * (price - position.avg_price) * direction


def _scale_leg(leg: Leg, scale: float) -> Leg:
    if scale == 1.0:
        return leg
    return Leg(
        token_id=leg.token_id,
        outcome=leg.outcome,
        side=leg.side,
        shares=leg.shares * scale,
        price=leg.price,
        fee=leg.fee * scale,
        cash=leg.cash * scale,
        market_id=leg.market_id,
        question=leg.question,
    )
