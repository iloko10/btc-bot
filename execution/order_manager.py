"""
execution/order_manager.py

Handles order placement, cancellation, and fill tracking.

In paper mode: simulates fills at mid-price with a configurable slippage.
In live mode:  submits signed orders to Polymarket's CLOB via py-clob-client.
"""

import time
import uuid
from dataclasses import dataclass, field
from enum import Enum
from typing import Optional

from loguru import logger

from config import BotConfig
from market_data.polymarket import Market
from strategies.base import Signal, SignalType


class OrderStatus(Enum):
    PENDING = "PENDING"
    FILLED = "FILLED"
    PARTIAL = "PARTIAL"
    CANCELLED = "CANCELLED"
    REJECTED = "REJECTED"


@dataclass
class Order:
    """Represents a single order sent (or simulated)."""
    id: str = field(default_factory=lambda: str(uuid.uuid4())[:8])
    condition_id: str = ""
    token_id: str = ""
    side: str = ""              # "BUY"
    outcome: str = ""           # "YES" | "NO"
    size_usd: float = 0.0
    limit_price: float = 0.0
    fill_price: float = 0.0
    status: OrderStatus = OrderStatus.PENDING
    timestamp: float = field(default_factory=time.time)
    error: str = ""


class OrderManager:
    """
    Routes orders to either the paper simulator or Polymarket CLOB.

    Usage:
        om = OrderManager(config)
        order = om.place(signal, size_usd)
        if order.status == OrderStatus.FILLED:
            ...
    """

    def __init__(self, cfg: BotConfig):
        self.cfg = cfg
        self.mode = cfg.mode
        self._orders: list[Order] = []

        # Only import + init CLOB client in live mode
        self._clob_client = None
        if self.mode == "live":
            self._init_clob_client()

    # ── Public API ────────────────────────────────────────────────────────────

    def place(self, signal: Signal, size_usd: float) -> Order:
        """Place (or simulate) an order for the given signal."""
        market = signal.market
        is_yes = signal.signal_type == SignalType.BUY_YES

        token_id = market.yes_token_id if is_yes else market.no_token_id
        limit_price = market.best_ask if is_yes else (1.0 - market.best_bid)
        outcome = "YES" if is_yes else "NO"

        order = Order(
            condition_id=market.condition_id,
            token_id=token_id,
            side="BUY",
            outcome=outcome,
            size_usd=size_usd,
            limit_price=limit_price,
        )

        if self.mode == "paper":
            self._simulate_fill(order, market)
        elif self.mode == "live":
            self._submit_live_order(order)
        else:
            logger.warning(f"Unknown mode '{self.mode}' — treating as paper")
            self._simulate_fill(order, market)

        self._orders.append(order)
        self._log_order(order)
        return order

    def cancel_all(self, condition_id: str) -> None:
        """Cancel all pending orders for a market (live mode only)."""
        if self.mode == "live" and self._clob_client:
            try:
                self._clob_client.cancel_market_orders(
                    market=condition_id, asset_id=None
                )
                logger.info(f"Cancelled orders for {condition_id[:8]}...")
            except Exception as e:
                logger.error(f"Cancel failed: {e}")

    @property
    def filled_orders(self) -> list[Order]:
        return [o for o in self._orders if o.status == OrderStatus.FILLED]

    @property
    def open_orders(self) -> list[Order]:
        return [o for o in self._orders if o.status == OrderStatus.PENDING]

    # ── Paper trading ─────────────────────────────────────────────────────────

    def _simulate_fill(self, order: Order, market: Market) -> None:
        """
        Simulate an immediate fill at the ask price + a small slippage.
        In reality Polymarket fills are near-instant for market orders.
        """
        slippage = 0.002  # 0.2% simulated slippage
        order.fill_price = min(order.limit_price * (1 + slippage), 0.99)
        order.status = OrderStatus.FILLED
        logger.debug(f"[PAPER] Simulated fill @ {order.fill_price:.4f}")

    # ── Live trading ──────────────────────────────────────────────────────────

    def _init_clob_client(self) -> None:
        """Initialise the py-clob-client with L2 auth (API key + wallet)."""
        try:
            from py_clob_client.client import ClobClient
            from py_clob_client.clob_types import ApiCreds

            creds = ApiCreds(
                api_key=self.cfg.polymarket.api_key,
                api_secret=self.cfg.polymarket.api_secret,
                api_passphrase=self.cfg.polymarket.api_passphrase,
            )
            self._clob_client = ClobClient(
                host=self.cfg.polymarket.clob_api_url,
                chain_id=self.cfg.polymarket.chain_id,
                key=self.cfg.polymarket.private_key,
                creds=creds,
            )
            logger.info("CLOB client initialised (live mode)")
        except ImportError:
            logger.error("py-clob-client not installed — falling back to paper mode")
            self.mode = "paper"
        except Exception as e:
            logger.error(f"CLOB init failed: {e} — falling back to paper mode")
            self.mode = "paper"

    def _submit_live_order(self, order: Order) -> None:
        """
        Submit a signed market order to Polymarket.
        Uses py-clob-client's create_and_post_order.
        """
        if not self._clob_client:
            logger.error("CLOB client not available")
            order.status = OrderStatus.REJECTED
            order.error = "No CLOB client"
            return

        try:
            from py_clob_client.clob_types import MarketOrderArgs, OrderType

            size_shares = order.size_usd / order.limit_price

            args = MarketOrderArgs(
                token_id=order.token_id,
                amount=size_shares,
            )
            signed = self._clob_client.create_market_order(args)
            resp = self._clob_client.post_order(signed, OrderType.FOK)

            if resp and resp.get("status") == "matched":
                order.fill_price = float(resp.get("price", order.limit_price))
                order.status = OrderStatus.FILLED
            else:
                order.status = OrderStatus.REJECTED
                order.error = str(resp)

        except Exception as e:
            order.status = OrderStatus.REJECTED
            order.error = str(e)
            logger.error(f"Live order failed: {e}")

    # ── Logging ───────────────────────────────────────────────────────────────

    def _log_order(self, order: Order) -> None:
        status_icon = "✅" if order.status == OrderStatus.FILLED else "❌"
        logger.info(
            f"{status_icon} Order [{order.id}] "
            f"{order.side} {order.outcome} "
            f"${order.size_usd:.2f} @ {order.fill_price:.4f} | "
            f"Status: {order.status.value}"
            + (f" | Error: {order.error}" if order.error else "")
        )
