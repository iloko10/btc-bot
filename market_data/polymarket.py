"""
market_data/polymarket.py — Polymarket REST + WebSocket client.

Wraps the Gamma API (market metadata) and CLOB API (live prices/orderbook).
All market data flows through this module.
"""
import urllib3
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)
import time
import json
import threading
from typing import Optional, Callable
from dataclasses import dataclass, field

import requests
import websocket
from loguru import logger

from config import PolymarketConfig


# ── Data models ──────────────────────────────────────────────────────────────

@dataclass
class Market:
    """A single Polymarket binary market."""
    condition_id: str
    question: str
    category: str
    end_date: str
    yes_token_id: str
    no_token_id: str
    best_bid: float = 0.0      # Highest bid for YES (implied prob)
    best_ask: float = 0.0      # Lowest ask for YES
    mid_price: float = 0.0
    volume_24h: float = 0.0
    liquidity_usd: float = 0.0
    is_active: bool = True

    @property
    def spread(self) -> float:
        return self.best_ask - self.best_bid

    @property
    def implied_probability(self) -> float:
        """Mid-price is the market's implied probability for YES."""
        return self.mid_price


@dataclass
class OrderBook:
    """Snapshot of the order book for a token."""
    token_id: str
    timestamp: float = field(default_factory=time.time)
    bids: list[tuple[float, float]] = field(default_factory=list)  # (price, size)
    asks: list[tuple[float, float]] = field(default_factory=list)

    @property
    def best_bid(self) -> float:
        return self.bids[0][0] if self.bids else 0.0

    @property
    def best_ask(self) -> float:
        return self.asks[0][0] if self.asks else 1.0

    @property
    def mid(self) -> float:
        return (self.best_bid + self.best_ask) / 2

    def liquidity_within(self, pct: float = 0.02) -> float:
        """Total USD liquidity within `pct` of mid price."""
        ref = self.mid
        total = 0.0
        for price, size in self.bids + self.asks:
            if abs(price - ref) / ref <= pct:
                total += price * size
        return total


# ── REST client ──────────────────────────────────────────────────────────────

class PolymarketClient:
    """
    Thin REST wrapper for Polymarket's Gamma API (market data)
    and CLOB API (live prices & order placement).
    """

    def __init__(self, cfg: PolymarketConfig):
        self.cfg = cfg
        self._session = requests.Session()
        self._session.headers.update({"Accept": "application/json"})

    # ── Market discovery ─────────────────────────────────────────────────────

    def get_markets(
        self,
        active_only: bool = True,
        limit: int = 100,
        offset: int = 0,
        category: Optional[str] = None,
    ) -> list[Market]:
        """
        Fetch markets from the Gamma API.

        Example response fields used:
          conditionId, question, category, endDate,
          clobTokenIds, volume24Hours, liquidityNum
        """
        params = {
            "active": str(active_only).lower(),
            "limit": limit,
            "offset": offset,
            "closed": "false",
            "order": "volume24Hours",
            "ascending": "false",
        }
        if category:
            params["category"] = category

        try:
            resp = self._session.get(
                f"{self.cfg.gamma_api_url}/markets",
                params=params,
                timeout=10,
                verify=False,
            )
            resp.raise_for_status()
        except requests.RequestException as e:
            logger.error(f"Gamma API error: {e}")
            return []

        markets = []
        for raw in resp.json():
            try:
                token_ids = raw.get("clobTokenIds", [None, None])
                market = Market(
                    condition_id=raw["conditionId"],
                    question=raw.get("question", ""),
                    category=raw.get("category", ""),
                    end_date=raw.get("endDate", ""),
                    yes_token_id=token_ids[0] if token_ids else "",
                    no_token_id=token_ids[1] if len(token_ids) > 1 else "",
                    volume_24h=float(raw.get("volume24Hours", 0)),
                    liquidity_usd=float(raw.get("liquidityNum", 0)),
                )
                markets.append(market)
            except (KeyError, TypeError, ValueError) as e:
                logger.warning(f"Skipping malformed market: {e}")

        return markets

    def get_market(self, condition_id: str) -> Optional[Market]:
        """Fetch a single market by condition ID."""
        markets = self.get_markets(limit=1)
        # In production: use /markets/{conditionId} endpoint directly
        try:
            resp = self._session.get(
                f"{self.cfg.gamma_api_url}/markets/{condition_id}",
                timeout=10,
                verify=False,
            )
            resp.raise_for_status()
            raw = resp.json()
            token_ids = raw.get("clobTokenIds", [None, None])
            return Market(
                condition_id=raw["conditionId"],
                question=raw.get("question", ""),
                category=raw.get("category", ""),
                end_date=raw.get("endDate", ""),
                yes_token_id=token_ids[0] if token_ids else "",
                no_token_id=token_ids[1] if len(token_ids) > 1 else "",
                volume_24h=float(raw.get("volume24Hours", 0)),
                liquidity_usd=float(raw.get("liquidityNum", 0)),
            )
        except Exception as e:
            logger.error(f"Failed to fetch market {condition_id}: {e}")
            return None

    # ── Prices ───────────────────────────────────────────────────────────────

    def get_price(self, token_id: str, side: str = "buy") -> float:
        """
        Get the best buy or sell price for a token from the CLOB.
        side: 'buy' | 'sell'
        """
        try:
            resp = self._session.get(
                f"{self.cfg.clob_api_url}/price",
                params={"token_id": token_id, "side": side},
                timeout=5,
            )
            resp.raise_for_status()
            return float(resp.json().get("price", 0))
        except Exception as e:
            logger.warning(f"Price fetch failed for {token_id}: {e}")
            return 0.0

    def get_orderbook(self, token_id: str) -> OrderBook:
        """Fetch the full order book for a token."""
        try:
            resp = self._session.get(
                f"{self.cfg.clob_api_url}/book",
                params={"token_id": token_id},
                timeout=5,
            )
            resp.raise_for_status()
            data = resp.json()
            book = OrderBook(token_id=token_id)
            book.bids = [
                (float(lvl["price"]), float(lvl["size"]))
                for lvl in data.get("bids", [])
            ]
            book.asks = [
                (float(lvl["price"]), float(lvl["size"]))
                for lvl in data.get("asks", [])
            ]
            return book
        except Exception as e:
            logger.warning(f"Orderbook fetch failed for {token_id}: {e}")
            return OrderBook(token_id=token_id)

    def get_price_history(self, market_id: str, interval: str = "1h", limit: int = 100) -> list[dict]:
        """
        Fetch OHLCV-style price history from Gamma API.
        Returns list of {t: timestamp, p: price} dicts.
        """
        try:
            params = {
                "market_id": market_id,
                "interval": interval,
                "limit": limit,
            }
            resp = self._session.get(
                f"{self.cfg.gamma_api_url}/markets/{market_id}/history",
                params=params,
                timeout=10,
                verify=False,
            )
            resp.raise_for_status()
            return resp.json().get("history", [])
        except Exception as e:
            logger.warning(f"History fetch failed for {market_id}: {e}")
            return []

    def enrich_market_prices(self, market: Market) -> Market:
        """Add live bid/ask/mid to a Market object in-place."""
        book = self.get_orderbook(market.yes_token_id)
        market.best_bid = book.best_bid
        market.best_ask = book.best_ask
        market.mid_price = book.mid
        return market


# ── WebSocket feed ────────────────────────────────────────────────────────────

class MarketFeed:
    """
    Subscribes to Polymarket's WebSocket for real-time price updates.
    Calls `on_price_update(token_id, price)` on each tick.
    """

    def __init__(self, cfg: PolymarketConfig, on_price_update: Callable):
        self.cfg = cfg
        self.on_price_update = on_price_update
        self._subscribed_tokens: set[str] = set()
        self._ws: Optional[websocket.WebSocketApp] = None
        self._thread: Optional[threading.Thread] = None
        self._running = False

    def subscribe(self, token_ids: list[str]) -> None:
        self._subscribed_tokens.update(token_ids)
        if self._ws:
            self._send_subscribe(token_ids)

    def start(self) -> None:
        self._running = True
        self._ws = websocket.WebSocketApp(
            self.cfg.ws_url,
            on_open=self._on_open,
            on_message=self._on_message,
            on_error=self._on_error,
            on_close=self._on_close,
        )
        self._thread = threading.Thread(target=self._ws.run_forever, daemon=True)
        self._thread.start()
        logger.info("WebSocket feed started")

    def stop(self) -> None:
        self._running = False
        if self._ws:
            self._ws.close()

    def _on_open(self, ws) -> None:
        logger.info("WebSocket connected")
        if self._subscribed_tokens:
            self._send_subscribe(list(self._subscribed_tokens))

    def _send_subscribe(self, token_ids: list[str]) -> None:
        msg = {
            "auth": {},
            "type": "Market",
            "assets_ids": token_ids,
        }
        self._ws.send(json.dumps(msg))

    def _on_message(self, ws, message: str) -> None:
        try:
            data = json.loads(message)
            # CLOB WS emits list of price updates
            if isinstance(data, list):
                for event in data:
                    token_id = event.get("asset_id")
                    price = float(event.get("price", 0))
                    if token_id and price:
                        self.on_price_update(token_id, price)
        except Exception as e:
            logger.debug(f"WS parse error: {e}")

    def _on_error(self, ws, error) -> None:
        logger.error(f"WebSocket error: {error}")
        if self._running:
            time.sleep(5)
            self.start()  # Reconnect

    def _on_close(self, ws, code, msg) -> None:
        logger.warning(f"WebSocket closed: {code} {msg}")
