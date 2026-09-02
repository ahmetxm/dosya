"""Live scan loop: public prices in, paper fills out."""

from __future__ import annotations

import logging
import threading
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Callable

from .books import best_ask, best_bid
from .clients import PublicClient, book_levels, token_map
from .fees import fee_rate_from_market
from .opportunities import (
    QuotedMarket,
    find_binary_completeness,
    find_spot_certainty,
    find_strike_monotonicity,
    parse_market_meta,
    rank_near_misses,
    snapshot_quotes,
)
from .paper import PaperBroker

LOGGER = logging.getLogger("polymarket_sim")
MAX_MARKETS_PER_CYCLE = 70
MIN_VOLUME = 200.0


@dataclass
class EngineConfig:
    starting_cash: float = 10_000.0
    event_limit: int = 10
    interval_sec: float = 10.0
    allow_unlocked: bool = True
    max_notional_frac: float = 0.15


@dataclass
class EngineState:
    running: bool = False
    last_error: str | None = None
    last_cycle_at: str | None = None
    cycle: int = 0
    scanned_markets: int = 0
    spots: dict[str, float] = field(default_factory=dict)
    opportunities: list[dict] = field(default_factory=list)
    near_misses: list[dict] = field(default_factory=list)
    quotes: list[dict] = field(default_factory=list)
    log: list[str] = field(default_factory=list)


class LiveEngine:
    def __init__(self, client: PublicClient | None = None, config: EngineConfig | None = None) -> None:
        self.client = client or PublicClient()
        self.config = config or EngineConfig()
        self.broker = PaperBroker(self.config.starting_cash)
        self.state = EngineState()
        self._lock = threading.RLock()
        self._thread: threading.Thread | None = None
        self._stop = threading.Event()

    def snapshot(self) -> dict:
        with self._lock:
            return {
                "running": self.state.running,
                "last_error": self.state.last_error,
                "last_cycle_at": self.state.last_cycle_at,
                "cycle": self.state.cycle,
                "scanned_markets": self.state.scanned_markets,
                "spots": dict(self.state.spots),
                "opportunities": list(self.state.opportunities),
                "near_misses": list(self.state.near_misses),
                "quotes": list(self.state.quotes[:40]),
                "log": list(self.state.log[-40:]),
                "paper": self.broker.snapshot(),
                "mode": "paper",
                "note": "Simulation only. No Polymarket orders are placed.",
            }

    def start(self) -> None:
        with self._lock:
            if self.state.running:
                return
            self._stop.clear()
            self.state.running = True
            self._note("live scan started")
            self._thread = threading.Thread(target=self._loop, name="arb-engine", daemon=True)
            self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        with self._lock:
            self.state.running = False
            self._note("scan stopped")

    def reset(self) -> None:
        with self._lock:
            self.broker.reset(self.config.starting_cash)
            self.state.opportunities.clear()
            self.state.near_misses.clear()
            self.state.quotes.clear()
            self.state.cycle = 0
            self.state.last_error = None
            self._note("paper book reset")

    def run_cycle(self) -> dict:
        """One live scan + paper execution. Safe to call from tests with a fake client."""
        spots = self.client.fetch_spot()
        events = self.client.list_crypto_events(self.config.event_limit)
        markets = _select_markets(events)
        token_ids: list[str] = []
        prepared: list[tuple[dict, dict, dict[str, str]]] = []
        for event, market in markets:
            tokens = token_map(market)
            if "yes" not in tokens or "no" not in tokens:
                continue
            if not market.get("acceptingOrders", True) or not market.get("enableOrderBook", True):
                continue
            prepared.append((event, market, tokens))
            token_ids.extend([tokens["yes"], tokens["no"]])

        books = self.client.fetch_books(token_ids)
        quoted = [_to_quoted(event, market, tokens, books) for event, market, tokens in prepared]
        quoted = [item for item in quoted if item is not None]

        opportunities = []
        for item in quoted:
            opportunities.extend(find_binary_completeness(item))
        opportunities.extend(find_strike_monotonicity(quoted))
        if self.config.allow_unlocked:
            opportunities.extend(find_spot_certainty(quoted, spots))
        opportunities.sort(key=lambda item: item.score, reverse=True)

        mids: dict[str, float] = {}
        for item in quoted:
            if item.yes_bid is not None and item.yes_ask is not None:
                mids[item.yes_id] = (item.yes_bid + item.yes_ask) / 2
            elif item.yes_ask is not None:
                mids[item.yes_id] = item.yes_ask
            if item.no_bid is not None and item.no_ask is not None:
                mids[item.no_id] = (item.no_bid + item.no_ask) / 2
            elif item.no_ask is not None:
                mids[item.no_id] = item.no_ask

        executed = []
        with self._lock:
            self.broker.update_marks(mids)
            seen_tokens: set[str] = set()
            for opportunity in opportunities:
                tokens = {leg.token_id for leg in opportunity.legs}
                if tokens & seen_tokens:
                    continue
                if not opportunity.locked:
                    extra = opportunity.extra or {}
                    hours = float(extra.get("hours_left") or 99)
                    if not self.config.allow_unlocked or hours > 12 or opportunity.edge_per_share < 0.03:
                        continue
                trade = self.broker.execute(opportunity, self.config.max_notional_frac)
                if trade:
                    executed.append(trade)
                    seen_tokens |= tokens
                    self._note(
                        f"PAPER {opportunity.kind} edge={opportunity.edge_per_share:.4f} "
                        f"pnl≈{trade.expected_pnl:.2f} {opportunity.legs[0].question[:80]}"
                    )
            self.state.spots = spots
            self.state.scanned_markets = len(quoted)
            self.state.opportunities = [_opp_dict(item) for item in opportunities[:30]]
            self.state.near_misses = rank_near_misses(quoted)
            self.state.quotes = [snapshot_quotes(item) for item in quoted]
            self.state.cycle += 1
            self.state.last_cycle_at = datetime.now(timezone.utc).isoformat()
            self.state.last_error = None
            paper = self.broker.snapshot()

        return {
            "spots": spots,
            "scanned": len(quoted),
            "opportunities": len(opportunities),
            "executed": len(executed),
            "paper": paper,
        }

    def run_for(self, seconds: float, on_cycle: Callable[[dict], None] | None = None) -> dict:
        deadline = time.time() + seconds
        last: dict = {}
        while time.time() < deadline and not self._stop.is_set():
            last = self.run_cycle()
            if on_cycle:
                on_cycle(last)
            remaining = deadline - time.time()
            if remaining <= 0:
                break
            self._stop.wait(min(self.config.interval_sec, remaining))
        return last or self.snapshot()

    def _loop(self) -> None:
        while not self._stop.is_set():
            try:
                self.run_cycle()
            except Exception as exc:  # noqa: BLE001
                LOGGER.exception("cycle failed")
                with self._lock:
                    self.state.last_error = str(exc)
                    self._note(f"cycle error: {exc}")
            self._stop.wait(self.config.interval_sec)

    def _note(self, message: str) -> None:
        stamp = datetime.now(timezone.utc).strftime("%H:%M:%S")
        self.state.log.append(f"{stamp}  {message}")
        if len(self.state.log) > 200:
            self.state.log = self.state.log[-200:]
        LOGGER.info(message)


def _select_markets(events: list[dict]) -> list[tuple[dict, dict]]:
    pairs: list[tuple[float, dict, dict]] = []
    for event in events:
        for market in event.get("markets") or []:
            if market.get("closed") or not market.get("active", True):
                continue
            volume = float(market.get("volume24hr") or 0)
            if volume < MIN_VOLUME:
                continue
            pairs.append((volume, event, market))
    pairs.sort(key=lambda item: item[0], reverse=True)
    return [(event, market) for _, event, market in pairs[:MAX_MARKETS_PER_CYCLE]]


def _to_quoted(event: dict, market: dict, tokens: dict[str, str], books: dict[str, dict]) -> QuotedMarket | None:
    yes_book = books.get(tokens["yes"])
    no_book = books.get(tokens["no"])
    if not yes_book or not no_book:
        return None
    yes_bids, yes_asks = book_levels(yes_book)
    no_bids, no_asks = book_levels(no_book)
    if best_ask(yes_asks) is None and best_bid(yes_bids) is None:
        return None
    question = str(market.get("question") or "")
    title = str(event.get("title") or "")
    asset, kind, strike = parse_market_meta(question, title)
    try:
        min_size = float(market.get("orderMinSize") or 5)
    except (TypeError, ValueError):
        min_size = 5.0
    return QuotedMarket(
        event_id=str(event.get("id") or title),
        event_title=title,
        market=market,
        question=question,
        yes_id=tokens["yes"],
        no_id=tokens["no"],
        yes_bids=yes_bids,
        yes_asks=yes_asks,
        no_bids=no_bids,
        no_asks=no_asks,
        fee_rate=fee_rate_from_market(market),
        min_size=min_size,
        asset=asset,
        kind=kind,
        strike=strike,
        end_date=market.get("endDate") or market.get("endDateIso") or event.get("endDate"),
    )


def _opp_dict(opportunity) -> dict:
    return {
        "kind": opportunity.kind,
        "edge_per_share": opportunity.edge_per_share,
        "locked": opportunity.locked,
        "reason": opportunity.reason,
        "event_title": opportunity.event_title,
        "shares": opportunity.shares,
        "expected_pnl": opportunity.expected_pnl,
        "question": opportunity.legs[0].question if opportunity.legs else "",
        "legs": [
            {
                "outcome": leg.outcome,
                "side": leg.side,
                "shares": leg.shares,
                "price": leg.price,
                "fee": leg.fee,
                "question": leg.question,
            }
            for leg in opportunity.legs
        ],
        "extra": opportunity.extra,
    }
