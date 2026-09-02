"""Public Polymarket Gamma/CLOB and Coinbase spot clients. No auth, no orders."""

from __future__ import annotations

import json
import logging
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any

import requests
from requests.adapters import HTTPAdapter

from .books import Level, parse_levels

LOGGER = logging.getLogger("polymarket_sim")

GAMMA = "https://gamma-api.polymarket.com"
CLOB = "https://clob.polymarket.com"
COINBASE = "https://api.exchange.coinbase.com"
CRYPTO_TAG_ID = "21"
USER_AGENT = "polymarket-paper-arb-sim/1.0"


class PublicClient:
    def __init__(self, timeout: float = 12.0, max_workers: int = 16) -> None:
        self.timeout = timeout
        self.max_workers = max_workers
        self.session = requests.Session()
        self.session.headers.update({"User-Agent": USER_AGENT, "Accept": "application/json"})
        adapter = HTTPAdapter(pool_connections=32, pool_maxsize=32, max_retries=1)
        self.session.mount("https://", adapter)
        self.session.mount("http://", adapter)

    def _get(self, url: str, params: dict | None = None) -> Any:
        response = self.session.get(url, params=params, timeout=self.timeout)
        response.raise_for_status()
        return response.json()

    def list_crypto_events(self, limit: int = 12) -> list[dict]:
        data = self._get(
            f"{GAMMA}/events",
            {
                "active": "true",
                "closed": "false",
                "tag_id": CRYPTO_TAG_ID,
                "limit": str(limit),
                "order": "volume24hr",
                "ascending": "false",
            },
        )
        return data if isinstance(data, list) else []

    def fetch_book(self, token_id: str) -> dict:
        return self._get(f"{CLOB}/book", {"token_id": token_id})

    def fetch_books(self, token_ids: list[str]) -> dict[str, dict]:
        unique = [token for token in dict.fromkeys(token_ids) if token]
        books: dict[str, dict] = {}
        if not unique:
            return books
        with ThreadPoolExecutor(max_workers=self.max_workers) as pool:
            futures = {pool.submit(self.fetch_book, token): token for token in unique}
            for future in as_completed(futures):
                token = futures[future]
                try:
                    books[token] = future.result()
                except Exception as exc:  # noqa: BLE001 — keep scan alive on one bad book
                    LOGGER.warning("book fetch failed for %s: %s", token[:16], exc)
        return books

    def fetch_spot(self) -> dict[str, float]:
        spots: dict[str, float] = {}
        for product, key in (("BTC-USD", "BTC"), ("ETH-USD", "ETH")):
            try:
                ticker = self._get(f"{COINBASE}/products/{product}/ticker")
                spots[key] = float(ticker["price"])
            except Exception as exc:  # noqa: BLE001
                LOGGER.warning("spot fetch failed for %s: %s", product, exc)
        return spots


def decode_json_field(value: Any) -> Any:
    if isinstance(value, str):
        try:
            return json.loads(value)
        except json.JSONDecodeError:
            return value
    return value


def token_map(market: dict) -> dict[str, str]:
    outcomes = decode_json_field(market.get("outcomes")) or []
    token_ids = decode_json_field(market.get("clobTokenIds")) or []
    mapping: dict[str, str] = {}
    for name, token_id in zip(outcomes, token_ids):
        if name and token_id:
            mapping[str(name).strip().lower()] = str(token_id)
    return mapping


def book_levels(book: dict | None) -> tuple[list[Level], list[Level]]:
    if not book:
        return [], []
    return parse_levels(book.get("bids")), parse_levels(book.get("asks"))
