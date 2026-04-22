from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import Enum
from typing import Optional

from market_data.polymarket import Market


class SignalType(Enum):
    BUY_YES = "BUY_YES"
    BUY_NO = "BUY_NO"
    CLOSE = "CLOSE"
    HOLD = "HOLD"


@dataclass
class Signal:
    market: Market
    signal_type: SignalType
    confidence: float
    fair_value: float
    target_size_usd: float
    reason: str = ""

    @property
    def edge(self) -> float:
        if self.signal_type == SignalType.BUY_YES:
            return self.fair_value - self.market.best_ask
        elif self.signal_type == SignalType.BUY_NO:
            return (1 - self.fair_value) - self.market.best_ask
        return 0.0

    def is_actionable(self, min_edge: float = 0.03) -> bool:
        return (
            self.signal_type not in (SignalType.HOLD, SignalType.CLOSE)
            and self.edge >= min_edge
            and 0 < self.confidence < 1
        )


class BaseStrategy(ABC):
    name: str = "base"

    def __init__(self, params: dict):
        self.params = params

    @abstractmethod
    def generate_signal(self, market: Market, history: list[dict]) -> Signal:
        ...

    def hold(self, market: Market, reason: str = "") -> Signal:
        return Signal(
            market=market,
            signal_type=SignalType.HOLD,
            confidence=0.0,
            fair_value=market.mid_price,
            target_size_usd=0.0,
            reason=reason,
        )