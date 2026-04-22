"""
strategies/news_sentiment.py

Uses Polymarket's own market metadata and external search to assign a
directional signal when we detect that market pricing hasn't yet reflected
recent news developments.

In a full implementation you'd plug in a news API (NewsAPI, GDELT, etc.)
and run a language model or sentiment classifier. This module shows the
pattern with a pluggable `news_scorer` callable.
"""

import time
from typing import Callable, Optional
from loguru import logger

from market_data.polymarket import Market
from strategies.base import BaseStrategy, Signal, SignalType


# Type alias: takes a market question string, returns (-1.0 to 1.0) sentiment score
NewsScorerFn = Callable[[str], float]


def _default_scorer(question: str) -> float:
    """
    Placeholder scorer — replace with a real news API + NLP pipeline.
    Returns 0.0 (neutral) for all markets until you wire in real data.
    """
    logger.warning("Using placeholder news scorer — returns 0.0 for all markets.")
    return 0.0


class NewsSentimentStrategy(BaseStrategy):
    """
    Trades when news sentiment strongly diverges from the market price.

    The idea:
      - Fetch a sentiment score S in [-1, +1] for the market question.
      - Positive S → news favours YES outcome.
      - If market price (implied_prob) is well below our news-implied probability,
        we have an edge → BUY YES.
      - If market price is well above our news-implied probability → BUY NO.

    Parameters:
        sentiment_threshold  : Minimum |score| to act (default 0.6).
        edge_threshold       : Minimum gap between news prob and market prob (default 0.05).
        score_decay_hours    : Treat scores older than N hours as stale (default 4).
    """

    name = "news_sentiment"

    def __init__(self, params: dict, news_scorer: Optional[NewsScorerFn] = None):
        super().__init__(params)
        self.sentiment_threshold = params.get("sentiment_threshold", 0.60)
        self.edge_threshold = params.get("edge_threshold", 0.05)
        self.score_decay_hours = params.get("score_decay_hours", 4)
        self._scorer = news_scorer or _default_scorer
        self._score_cache: dict[str, tuple[float, float]] = {}  # {condition_id: (score, timestamp)}

    def generate_signal(self, market: Market, history: list[dict]) -> Signal:
        # ── Get (possibly cached) sentiment score ─────────────────────────
        score = self._get_score(market)

        if abs(score) < self.sentiment_threshold:
            return self.hold(market, f"Sentiment score {score:.2f} below threshold")

        # Convert score to a probability in [0.05, 0.95]
        # score=+1.0 → news_prob≈0.95, score=-1.0 → news_prob≈0.05
        news_prob = 0.5 + score * 0.45

        market_prob = market.implied_probability
        edge = news_prob - market_prob

        logger.info(
            f"[NewsStrategy] {market.question[:50]}... | "
            f"sentiment={score:+.2f} news_prob={news_prob:.3f} "
            f"market_prob={market_prob:.3f} edge={edge:+.3f}"
        )

        if edge >= self.edge_threshold:
            return Signal(
                market=market,
                signal_type=SignalType.BUY_YES,
                confidence=min(abs(score), 0.9),
                fair_value=news_prob,
                target_size_usd=30.0,
                reason=f"News bullish (score={score:+.2f}), market underpriced by {edge:.1%}",
            )

        if edge <= -self.edge_threshold:
            return Signal(
                market=market,
                signal_type=SignalType.BUY_NO,
                confidence=min(abs(score), 0.9),
                fair_value=news_prob,
                target_size_usd=30.0,
                reason=f"News bearish (score={score:+.2f}), market overpriced by {abs(edge):.1%}",
            )

        return self.hold(market, f"News/market gap {edge:+.3f} below edge threshold")

    def _get_score(self, market: Market) -> float:
        """Return a cached or freshly computed sentiment score."""
        cid = market.condition_id
        now = time.time()

        if cid in self._score_cache:
            score, ts = self._score_cache[cid]
            if (now - ts) / 3600 < self.score_decay_hours:
                return score

        score = self._scorer(market.question)
        self._score_cache[cid] = (score, now)
        return score
