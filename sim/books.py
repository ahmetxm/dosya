"""Order-book walking for executable size and VWAP."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Level:
    price: float
    size: float


@dataclass(frozen=True)
class Fill:
    shares: float
    vwap: float
    notional: float
    exhausted: bool


def parse_levels(raw_levels: list | None) -> list[Level]:
    levels: list[Level] = []
    for item in raw_levels or []:
        try:
            price = float(item["price"])
            size = float(item["size"])
        except (KeyError, TypeError, ValueError):
            continue
        if price > 0 and size > 0:
            levels.append(Level(price=price, size=size))
    return levels


def best_bid(levels: list[Level]) -> float | None:
    return max((level.price for level in levels), default=None)


def best_ask(levels: list[Level]) -> float | None:
    return min((level.price for level in levels), default=None)


def walk_asks(levels: list[Level], shares: float) -> Fill:
    """Take asks from lowest price up until `shares` are filled."""
    return _walk(levels, shares, reverse=False)


def walk_bids(levels: list[Level], shares: float) -> Fill:
    """Hit bids from highest price down until `shares` are filled."""
    return _walk(levels, shares, reverse=True)


def _walk(levels: list[Level], shares: float, reverse: bool) -> Fill:
    if shares <= 0 or not levels:
        return Fill(shares=0.0, vwap=0.0, notional=0.0, exhausted=True)
    ordered = sorted(levels, key=lambda level: level.price, reverse=reverse)
    remaining = shares
    filled = 0.0
    notional = 0.0
    for level in ordered:
        take = min(remaining, level.size)
        if take <= 0:
            continue
        filled += take
        notional += take * level.price
        remaining -= take
        if remaining <= 1e-12:
            break
    if filled <= 0:
        return Fill(shares=0.0, vwap=0.0, notional=0.0, exhausted=True)
    return Fill(
        shares=filled,
        vwap=notional / filled,
        notional=notional,
        exhausted=remaining > 1e-9,
    )


def available_size(levels: list[Level]) -> float:
    return sum(level.size for level in levels)
