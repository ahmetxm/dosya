"""Polymarket taker fees.

Official formula from https://docs.polymarket.com/trading/fees :

    fee = C × feeRate × p × (1 - p)

rounded to 5 decimal places. Crypto markets use feeRate = 0.07 unless the
market metadata supplies a different schedule rate.
"""

from __future__ import annotations

DEFAULT_CRYPTO_FEE_RATE = 0.07
FEE_DECIMALS = 5
MIN_FEE = 0.00001


def taker_fee(shares: float, price: float, fee_rate: float = DEFAULT_CRYPTO_FEE_RATE) -> float:
    """USDC taker fee for buying or selling `shares` at `price`."""
    if shares <= 0 or fee_rate <= 0:
        return 0.0
    if price <= 0.0 or price >= 1.0:
        return 0.0
    raw = shares * fee_rate * price * (1.0 - price)
    rounded = round(raw, FEE_DECIMALS)
    if 0 < rounded < MIN_FEE:
        return MIN_FEE
    return rounded


def buy_cost(shares: float, price: float, fee_rate: float = DEFAULT_CRYPTO_FEE_RATE) -> float:
    """Cash spent to take the ask."""
    return shares * price + taker_fee(shares, price, fee_rate)


def sell_proceeds(shares: float, price: float, fee_rate: float = DEFAULT_CRYPTO_FEE_RATE) -> float:
    """Cash received when taking the bid."""
    return shares * price - taker_fee(shares, price, fee_rate)


def fee_rate_from_market(market: dict | None) -> float:
    if not market:
        return DEFAULT_CRYPTO_FEE_RATE
    schedule = market.get("feeSchedule") or {}
    rate = schedule.get("rate")
    if rate is None:
        return DEFAULT_CRYPTO_FEE_RATE
    try:
        value = float(rate)
    except (TypeError, ValueError):
        return DEFAULT_CRYPTO_FEE_RATE
    return value if value >= 0 else DEFAULT_CRYPTO_FEE_RATE
