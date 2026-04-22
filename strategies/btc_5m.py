"""
strategies/btc_5m.py

Στρατηγική για τα Polymarket BTC UP/DOWN 5-λεπτα markets.

Λογική:
  - Κοιτάει τα τελευταία N candles του BTC price (από Binance public API)
  - Αν momentum είναι ανοδικό → BUY UP token
  - Αν momentum είναι καθοδικό → BUY DOWN token
  - Συνδυάζει RSI + price momentum για confirmation
"""

import time
import requests
import numpy as np
from loguru import logger

from market_data.polymarket import Market
from strategies.base import BaseStrategy, Signal, SignalType


class BTC5mStrategy(BaseStrategy):
    """
    Momentum strategy για BTC UP/DOWN 5-λεπτα markets.

    Χρησιμοποιεί Binance public API (δεν χρειάζεται auth)
    για να πάρει τα τελευταία BTC/USDT 1m candles.
    """

    name = "btc_5m"

    def __init__(self, params: dict):
        super().__init__(params)
        self.lookback = params.get("lookback", 10)        # candles για momentum
        self.rsi_period = params.get("rsi_period", 7)
        self.momentum_threshold = params.get("momentum_threshold", 0.001)  # 0.1% move
        self.binance_url = "https://api.binance.com/api/v3/klines"
        self._last_btc_fetch = 0
        self._cached_candles = []

    def generate_signal(self, market: Market, history: list[dict]) -> Signal:
        # Πάρε BTC candles από Binance
        candles = self._get_btc_candles()
        if not candles:
            return self.hold(market, "Could not fetch BTC price data")

        closes = np.array([float(c[4]) for c in candles])  # close prices

        # ── Momentum ──────────────────────────────────────────────────────
        recent = closes[-self.lookback:]
        momentum = (recent[-1] - recent[0]) / recent[0]  # % change over window

        # ── RSI ───────────────────────────────────────────────────────────
        rsi = self._rsi(closes, self.rsi_period)

        # ── Volume trend ──────────────────────────────────────────────────
        volumes = np.array([float(c[5]) for c in candles[-self.lookback:]])
        vol_trend = volumes[-1] > np.mean(volumes)  # above average volume

        current_price = market.mid_price

        logger.info(
            f"BTC momentum={momentum:+.4f} RSI={rsi:.1f} "
            f"vol_above_avg={vol_trend} market_price={current_price:.3f}"
        )

        # ── Signal logic ──────────────────────────────────────────────────

        # Strong upward momentum → BUY UP
        if (momentum > self.momentum_threshold
                and rsi < 70          # not overbought
                and vol_trend):       # volume confirms

            confidence = min(abs(momentum) * 50, 0.85)
            fair_value = min(current_price + abs(momentum) * 2, 0.90)

            return Signal(
                market=market,
                signal_type=SignalType.BUY_YES,  # YES = UP
                confidence=confidence,
                fair_value=fair_value,
                target_size_usd=self.params.get("max_position", 30.0),
                reason=f"BTC momentum {momentum:+.4f}, RSI {rsi:.1f}, vol confirmed",
            )

        # Strong downward momentum → BUY DOWN (BUY NO)
        if (momentum < -self.momentum_threshold
                and rsi > 30          # not oversold
                and vol_trend):

            confidence = min(abs(momentum) * 50, 0.85)
            fair_value = min(1 - current_price + abs(momentum) * 2, 0.90)

            return Signal(
                market=market,
                signal_type=SignalType.BUY_NO,   # NO = DOWN
                confidence=confidence,
                fair_value=fair_value,
                target_size_usd=self.params.get("max_position", 30.0),
                reason=f"BTC momentum {momentum:+.4f}, RSI {rsi:.1f}, vol confirmed",
            )

        return self.hold(
            market,
            f"No clear momentum: {momentum:+.4f} (threshold ±{self.momentum_threshold})"
        )

    # ── Helpers ───────────────────────────────────────────────────────────────

    def _get_btc_candles(self) -> list:
        """
        Fetch BTC/USDT 1m candles from Binance.
        Cached for 30 seconds to avoid hammering the API.
        """
        now = time.time()
        if now - self._last_btc_fetch < 30 and self._cached_candles:
            return self._cached_candles

        try:
            resp = requests.get(
                self.binance_url,
                params={
                    "symbol": "BTCUSDT",
                    "interval": "1m",
                    "limit": 20,
                },
                timeout=5,
            )
            resp.raise_for_status()
            self._cached_candles = resp.json()
            self._last_btc_fetch = now
            logger.debug(f"BTC price: ${float(self._cached_candles[-1][4]):,.2f}")
            return self._cached_candles
        except Exception as e:
            logger.warning(f"Binance fetch failed: {e}")
            return self._cached_candles  # return stale if available

    def _rsi(self, closes: np.ndarray, period: int) -> float:
        """Calculate RSI."""
        if len(closes) < period + 1:
            return 50.0
        deltas = np.diff(closes)
        gains = np.where(deltas > 0, deltas, 0)
        losses = np.where(deltas < 0, -deltas, 0)
        avg_gain = np.mean(gains[-period:])
        avg_loss = np.mean(losses[-period:])
        if avg_loss == 0:
            return 100.0
        rs = avg_gain / avg_loss
        return 100 - (100 / (1 + rs))
