"""
strategies/mean_reversion.py

Core idea: Prediction market prices are probability estimates that tend to
be noisy in the short term. When a market price deviates significantly from
its recent mean (measured in standard deviations), it is likely to revert.

Entry: z-score of current price vs rolling window crosses a threshold.
Exit:  z-score returns toward zero.

This is NOT fundamental analysis — it's purely technical/statistical.
Best suited for high-volume markets with lots of liquidity.
"""

import numpy as np
from loguru import logger

from market_data.polymarket import Market
from strategies.base import BaseStrategy, Signal, SignalType


class MeanReversionStrategy(BaseStrategy):
    """
    Z-score mean reversion on implied probabilities.

    Parameters (all passed via params dict):
        lookback_n    : Number of candles to compute rolling stats over (default 48).
        z_entry       : Z-score threshold to open a trade (default 1.5).
        z_exit        : Z-score threshold to close a trade (default 0.3).
        max_position  : Max USD per trade before risk scaling (default 50).
    """

    name = "mean_reversion"

    def __init__(self, params: dict):
        super().__init__(params)
        self.lookback_n = params.get("lookback_n", 48)
        self.z_entry = params.get("z_entry", 1.5)
        self.z_exit = params.get("z_exit", 0.3)
        self.max_position = params.get("max_position", 50.0)

    def generate_signal(self, market: Market, history: list[dict]) -> Signal:
        """
        Compute rolling z-score and emit BUY / HOLD signal.
        """
        prices = self._extract_prices(history)

        if len(prices) < self.lookback_n:
            return self.hold(market, f"Insufficient history ({len(prices)}/{self.lookback_n})")

        window = np.array(prices[-self.lookback_n:])
        current_price = market.mid_price

        mean = np.mean(window)
        std = np.std(window)

        if std < 1e-6:
            return self.hold(market, "Zero volatility — market may be near resolution")

        z_score = (current_price - mean) / std

        logger.debug(
            f"{market.question[:50]}... | "
            f"price={current_price:.3f} mean={mean:.3f} z={z_score:+.2f}"
        )

        # ── Entry logic ──────────────────────────────────────────────────────

        # Price is significantly BELOW mean → expect upward reversion → BUY YES
        if z_score <= -self.z_entry:
            fair_value = mean  # Our estimate: price will revert to mean
            confidence = self._z_to_confidence(abs(z_score))
            return Signal(
                market=market,
                signal_type=SignalType.BUY_YES,
                confidence=confidence,
                fair_value=fair_value,
                target_size_usd=self.max_position,
                reason=f"Price {current_price:.3f} is {abs(z_score):.2f}σ below mean {mean:.3f}",
            )

        # Price is significantly ABOVE mean → expect downward reversion → BUY NO
        if z_score >= self.z_entry:
            fair_value = 1.0 - mean
            confidence = self._z_to_confidence(abs(z_score))
            return Signal(
                market=market,
                signal_type=SignalType.BUY_NO,
                confidence=confidence,
                fair_value=1.0 - mean,
                target_size_usd=self.max_position,
                reason=f"Price {current_price:.3f} is {z_score:.2f}σ above mean {mean:.3f}",
            )

        # ── Exit logic ───────────────────────────────────────────────────────
        if abs(z_score) <= self.z_exit:
            return Signal(
                market=market,
                signal_type=SignalType.CLOSE,
                confidence=1.0,
                fair_value=current_price,
                target_size_usd=0.0,
                reason=f"Z-score reverted to {z_score:+.2f} — take profit",
            )

        return self.hold(market, f"Z-score {z_score:+.2f} within neutral band")

    # ── Helpers ──────────────────────────────────────────────────────────────

    def _extract_prices(self, history: list[dict]) -> list[float]:
        """Pull price values from history in chronological order."""
        return [float(h["p"]) for h in sorted(history, key=lambda x: x["t"])]

    def _z_to_confidence(self, abs_z: float) -> float:
        """
        Map z-score magnitude to a [0, 1] confidence score using a
        sigmoid-style transform. z=1.5 → ~0.5, z=3.0 → ~0.9.
        """
        return float(np.clip(1 - np.exp(-0.5 * (abs_z - 1.0)), 0.0, 0.99))
