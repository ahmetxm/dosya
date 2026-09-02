"""Detect locked or high-confidence crypto price gaps on Polymarket."""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Iterable

from .books import Fill, Level, available_size, best_ask, best_bid, walk_asks, walk_bids
from .fees import buy_cost, fee_rate_from_market, sell_proceeds

STRIKE_RE = re.compile(
    r"(?P<kind>reach|hit|be above|above|dip to)\s+\$?(?P<strike>[0-9][0-9,]*(?:\.[0-9]+)?)",
    re.IGNORECASE,
)
ASSET_RE = re.compile(r"\b(bitcoin|btc|ethereum|eth|solana|sol)\b", re.IGNORECASE)
ASSET_ALIASES = {
    "bitcoin": "BTC",
    "btc": "BTC",
    "ethereum": "ETH",
    "eth": "ETH",
    "solana": "SOL",
    "sol": "SOL",
}
MIN_SHARES = 5.0
MIN_EDGE = 0.004
MAX_SHARE_CAP = 250.0
SPOT_BUFFER = 0.012
NEAR_CERTAIN_ASK = 0.985


@dataclass(frozen=True)
class Leg:
    token_id: str
    outcome: str
    side: str  # buy | sell
    shares: float
    price: float
    fee: float
    cash: float
    market_id: str
    question: str


@dataclass
class Opportunity:
    kind: str
    edge_per_share: float
    locked: bool
    reason: str
    event_title: str
    shares: float
    expected_pnl: float
    legs: list[Leg] = field(default_factory=list)
    extra: dict = field(default_factory=dict)

    @property
    def score(self) -> float:
        return self.expected_pnl


@dataclass
class QuotedMarket:
    event_id: str
    event_title: str
    market: dict
    question: str
    yes_id: str
    no_id: str
    yes_bids: list[Level]
    yes_asks: list[Level]
    no_bids: list[Level]
    no_asks: list[Level]
    fee_rate: float
    min_size: float
    asset: str | None
    kind: str | None
    strike: float | None
    end_date: str | None

    @property
    def market_id(self) -> str:
        return str(self.market.get("id") or self.market.get("conditionId") or self.question)

    @property
    def yes_ask(self) -> float | None:
        return best_ask(self.yes_asks)

    @property
    def yes_bid(self) -> float | None:
        return best_bid(self.yes_bids)

    @property
    def no_ask(self) -> float | None:
        return best_ask(self.no_asks)

    @property
    def no_bid(self) -> float | None:
        return best_bid(self.no_bids)


def parse_market_meta(question: str, event_title: str = "") -> tuple[str | None, str | None, float | None]:
    text = f"{question} {event_title}"
    asset_match = ASSET_RE.search(text)
    asset = ASSET_ALIASES.get(asset_match.group(1).lower()) if asset_match else None
    strike_match = STRIKE_RE.search(question)
    if not strike_match:
        return asset, None, None
    raw_kind = strike_match.group("kind").lower()
    kind = "dip" if "dip" in raw_kind else "above"
    strike = float(strike_match.group("strike").replace(",", ""))
    return asset, kind, strike


def _cap_shares(*fills: Fill, min_size: float, cap: float = MAX_SHARE_CAP) -> float:
    sizes = [fill.shares for fill in fills if fill.shares > 0]
    if not sizes:
        return 0.0
    shares = min(min(sizes), cap)
    floor = max(min_size, MIN_SHARES)
    if shares + 1e-9 < floor:
        return 0.0
    return shares


def _buy_leg(quoted: QuotedMarket, outcome: str, token_id: str, asks: list[Level], shares: float) -> Leg | None:
    fill = walk_asks(asks, shares)
    if fill.shares + 1e-9 < shares:
        return None
    cash = buy_cost(shares, fill.vwap, quoted.fee_rate)
    fee = cash - shares * fill.vwap
    return Leg(
        token_id=token_id,
        outcome=outcome,
        side="buy",
        shares=shares,
        price=fill.vwap,
        fee=fee,
        cash=-cash,
        market_id=quoted.market_id,
        question=quoted.question,
    )


def _sell_leg(quoted: QuotedMarket, outcome: str, token_id: str, bids: list[Level], shares: float) -> Leg | None:
    fill = walk_bids(bids, shares)
    if fill.shares + 1e-9 < shares:
        return None
    cash = sell_proceeds(shares, fill.vwap, quoted.fee_rate)
    fee = shares * fill.vwap - cash
    return Leg(
        token_id=token_id,
        outcome=outcome,
        side="sell",
        shares=shares,
        price=fill.vwap,
        fee=fee,
        cash=cash,
        market_id=quoted.market_id,
        question=quoted.question,
    )


def find_binary_completeness(quoted: QuotedMarket) -> list[Opportunity]:
    found: list[Opportunity] = []
    min_size = max(float(quoted.min_size or MIN_SHARES), MIN_SHARES)

    yes_ask_size = available_size(quoted.yes_asks)
    no_ask_size = available_size(quoted.no_asks)
    buy_shares = _cap_shares(
        Fill(shares=yes_ask_size, vwap=0, notional=0, exhausted=False),
        Fill(shares=no_ask_size, vwap=0, notional=0, exhausted=False),
        min_size=min_size,
    )
    if buy_shares:
        yes_leg = _buy_leg(quoted, "Yes", quoted.yes_id, quoted.yes_asks, buy_shares)
        no_leg = _buy_leg(quoted, "No", quoted.no_id, quoted.no_asks, buy_shares)
        if yes_leg and no_leg:
            cost = -(yes_leg.cash + no_leg.cash)
            edge = 1.0 - cost / buy_shares
            if edge >= MIN_EDGE:
                found.append(
                    Opportunity(
                        kind="binary_buy_pair",
                        edge_per_share=edge,
                        locked=True,
                        reason="YES ask + NO ask + fees < $1, so a matched pair locks $1 at resolution.",
                        event_title=quoted.event_title,
                        shares=buy_shares,
                        expected_pnl=edge * buy_shares,
                        legs=[yes_leg, no_leg],
                        extra={"pair_cost": cost, "payout": buy_shares},
                    )
                )

    yes_bid_size = available_size(quoted.yes_bids)
    no_bid_size = available_size(quoted.no_bids)
    sell_shares = _cap_shares(
        Fill(shares=yes_bid_size, vwap=0, notional=0, exhausted=False),
        Fill(shares=no_bid_size, vwap=0, notional=0, exhausted=False),
        min_size=min_size,
    )
    if sell_shares:
        yes_leg = _sell_leg(quoted, "Yes", quoted.yes_id, quoted.yes_bids, sell_shares)
        no_leg = _sell_leg(quoted, "No", quoted.no_id, quoted.no_bids, sell_shares)
        if yes_leg and no_leg:
            proceeds = yes_leg.cash + no_leg.cash
            edge = proceeds / sell_shares - 1.0
            if edge >= MIN_EDGE:
                found.append(
                    Opportunity(
                        kind="binary_sell_pair",
                        edge_per_share=edge,
                        locked=True,
                        reason="YES bid + NO bid − fees > $1, so shorting both locks the overround.",
                        event_title=quoted.event_title,
                        shares=sell_shares,
                        expected_pnl=edge * sell_shares,
                        legs=[yes_leg, no_leg],
                        extra={"pair_proceeds": proceeds, "payout_owed": sell_shares},
                    )
                )
    return found


def find_strike_monotonicity(quoted_markets: list[QuotedMarket]) -> list[Opportunity]:
    groups: dict[tuple[str, str, str], list[QuotedMarket]] = {}
    for quoted in quoted_markets:
        if quoted.asset and quoted.kind in {"above", "dip"} and quoted.strike:
            groups.setdefault((quoted.event_id, quoted.asset, quoted.kind), []).append(quoted)

    found: list[Opportunity] = []
    for (_, _, kind), members in groups.items():
        members = [item for item in members if item.strike is not None]
        members.sort(key=lambda item: item.strike or 0.0)
        for left, right in zip(members, members[1:]):
            easier, harder = (right, left) if kind == "dip" else (left, right)
            if not easier.yes_asks or not harder.yes_bids:
                continue
            min_size = max(easier.min_size, harder.min_size, MIN_SHARES)
            shares = _cap_shares(
                Fill(shares=available_size(easier.yes_asks), vwap=0, notional=0, exhausted=False),
                Fill(shares=available_size(harder.yes_bids), vwap=0, notional=0, exhausted=False),
                min_size=min_size,
            )
            if not shares:
                continue
            buy_leg = _buy_leg(easier, "Yes", easier.yes_id, easier.yes_asks, shares)
            sell_leg = _sell_leg(harder, "Yes", harder.yes_id, harder.yes_bids, shares)
            if not buy_leg or not sell_leg:
                continue
            net = buy_leg.cash + sell_leg.cash
            edge = net / shares
            if edge >= MIN_EDGE:
                found.append(
                    Opportunity(
                        kind="strike_monotonicity",
                        edge_per_share=edge,
                        locked=True,
                        reason=(
                            "Inverted strike ladder: the harder YES is bid above the easier YES ask, "
                            "so buy-easy / sell-hard cannot lose at resolution."
                        ),
                        event_title=easier.event_title,
                        shares=shares,
                        expected_pnl=net,
                        legs=[buy_leg, sell_leg],
                        extra={
                            "easier_strike": easier.strike,
                            "harder_strike": harder.strike,
                            "kind": kind,
                        },
                    )
                )
    return found


def _parse_end(end_date: str | None) -> datetime | None:
    if not end_date:
        return None
    text = end_date.replace("Z", "+00:00")
    try:
        parsed = datetime.fromisoformat(text)
    except ValueError:
        return None
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return parsed


def find_spot_certainty(
    quoted_markets: Iterable[QuotedMarket],
    spots: dict[str, float],
    now: datetime | None = None,
) -> list[Opportunity]:
    """High-confidence, not fully locked: strike already decided vs live spot near expiry."""
    now = now or datetime.now(timezone.utc)
    found: list[Opportunity] = []
    for quoted in quoted_markets:
        if quoted.kind != "above" or not quoted.asset or not quoted.strike:
            continue
        spot = spots.get(quoted.asset)
        if not spot:
            continue
        end = _parse_end(quoted.end_date or quoted.market.get("endDate") or quoted.market.get("endDateIso"))
        if end is None:
            continue
        hours_left = (end - now).total_seconds() / 3600.0
        if hours_left > 48 or hours_left < -6:
            continue
        min_size = max(quoted.min_size, MIN_SHARES)

        if spot >= quoted.strike * (1 + SPOT_BUFFER) and quoted.yes_ask and quoted.yes_ask <= NEAR_CERTAIN_ASK:
            shares = _cap_shares(
                Fill(shares=available_size(quoted.yes_asks), vwap=0, notional=0, exhausted=False),
                min_size=min_size,
            )
            leg = _buy_leg(quoted, "Yes", quoted.yes_id, quoted.yes_asks, shares) if shares else None
            if leg:
                edge = 1.0 + leg.cash / shares
                if edge >= MIN_EDGE:
                    found.append(
                        Opportunity(
                            kind="spot_yes_certainty",
                            edge_per_share=edge,
                            locked=False,
                            reason=(
                                f"{quoted.asset} spot {spot:,.2f} is already above {quoted.strike:,.0f} "
                                f"with {hours_left:.1f}h left, but YES ask is {quoted.yes_ask:.3f}."
                            ),
                            event_title=quoted.event_title,
                            shares=shares,
                            expected_pnl=edge * shares,
                            legs=[leg],
                            extra={"spot": spot, "strike": quoted.strike, "hours_left": hours_left},
                        )
                    )
        elif spot <= quoted.strike * (1 - SPOT_BUFFER) and quoted.no_ask and quoted.no_ask <= NEAR_CERTAIN_ASK:
            shares = _cap_shares(
                Fill(shares=available_size(quoted.no_asks), vwap=0, notional=0, exhausted=False),
                min_size=min_size,
            )
            leg = _buy_leg(quoted, "No", quoted.no_id, quoted.no_asks, shares) if shares else None
            if leg:
                edge = 1.0 + leg.cash / shares
                if edge >= MIN_EDGE:
                    found.append(
                        Opportunity(
                            kind="spot_no_certainty",
                            edge_per_share=edge,
                            locked=False,
                            reason=(
                                f"{quoted.asset} spot {spot:,.2f} is still below {quoted.strike:,.0f} "
                                f"with {hours_left:.1f}h left, but NO ask is {quoted.no_ask:.3f}."
                            ),
                            event_title=quoted.event_title,
                            shares=shares,
                            expected_pnl=edge * shares,
                            legs=[leg],
                            extra={"spot": spot, "strike": quoted.strike, "hours_left": hours_left},
                        )
                    )
    return found


def snapshot_quotes(quoted: QuotedMarket) -> dict:
    return {
        "market_id": quoted.market_id,
        "event_title": quoted.event_title,
        "question": quoted.question,
        "yes_bid": quoted.yes_bid,
        "yes_ask": quoted.yes_ask,
        "no_bid": quoted.no_bid,
        "no_ask": quoted.no_ask,
        "pair_ask": (quoted.yes_ask + quoted.no_ask) if quoted.yes_ask and quoted.no_ask else None,
        "pair_bid": (quoted.yes_bid + quoted.no_bid) if quoted.yes_bid and quoted.no_bid else None,
        "asset": quoted.asset,
        "kind": quoted.kind,
        "strike": quoted.strike,
        "volume24hr": quoted.market.get("volume24hr"),
        "liquidity": quoted.market.get("liquidityNum") or quoted.market.get("liquidity"),
    }


def rank_near_misses(quoted_markets: Iterable[QuotedMarket], limit: int = 12) -> list[dict]:
    rows = []
    for quoted in quoted_markets:
        if quoted.yes_ask is None or quoted.no_ask is None:
            continue
        pair_ask = quoted.yes_ask + quoted.no_ask
        rows.append({**snapshot_quotes(quoted), "gap_to_arb": pair_ask - 1.0})
    rows.sort(key=lambda row: row["gap_to_arb"])
    return rows[:limit]
