"""
risk/risk_manager.py

Handles all pre-trade and portfolio-level risk controls:
  - Kelly criterion position sizing
  - Max drawdown circuit breaker
  - Per-market and portfolio exposure caps
  - Liquidity and spread filters
"""

from dataclasses import dataclass, field
from typing import Optional
from loguru import logger

from config import RiskConfig
from strategies.base import Signal, SignalType


@dataclass
class Position:
    """An open position in a market."""
    condition_id: str
    token_id: str
    side: str                     # "YES" | "NO"
    shares: float
    avg_entry_price: float
    current_price: float = 0.0
    realized_pnl: float = 0.0

    @property
    def cost_basis(self) -> float:
        return self.shares * self.avg_entry_price

    @property
    def market_value(self) -> float:
        return self.shares * self.current_price

    @property
    def unrealized_pnl(self) -> float:
        return self.market_value - self.cost_basis

    @property
    def total_pnl(self) -> float:
        return self.unrealized_pnl + self.realized_pnl


@dataclass
class PortfolioState:
    """Snapshot of the bot's portfolio."""
    bankroll: float
    peak_bankroll: float
    positions: dict[str, Position] = field(default_factory=dict)
    total_trades: int = 0
    winning_trades: int = 0

    @property
    def open_exposure_usd(self) -> float:
        return sum(p.cost_basis for p in self.positions.values())

    @property
    def exposure_pct(self) -> float:
        return self.open_exposure_usd / self.bankroll if self.bankroll > 0 else 0.0

    @property
    def drawdown_pct(self) -> float:
        if self.peak_bankroll <= 0:
            return 0.0
        return max(0.0, (self.peak_bankroll - self.bankroll) / self.peak_bankroll)

    @property
    def win_rate(self) -> float:
        return self.winning_trades / self.total_trades if self.total_trades > 0 else 0.0


class RiskManager:
    """
    Evaluates signals against risk rules and returns an approved position size.

    Call `approve(signal, portfolio)` before every trade.
    Returns (approved: bool, size_usd: float, reason: str).
    """

    def __init__(self, cfg: RiskConfig):
        self.cfg = cfg

    def approve(
        self,
        signal: Signal,
        portfolio: PortfolioState,
    ) -> tuple[bool, float, str]:
        """
        Run all risk checks. Returns (approved, final_size_usd, reason).
        If not approved, size_usd is 0.
        """

        market = signal.market

        # ── 1. Circuit breaker: max drawdown ─────────────────────────────
        if portfolio.drawdown_pct >= self.cfg.max_drawdown_pct:
            return False, 0.0, (
                f"Circuit breaker: drawdown {portfolio.drawdown_pct:.1%} "
                f">= limit {self.cfg.max_drawdown_pct:.1%}"
            )

        # ── 2. Portfolio exposure cap ─────────────────────────────────────
        if portfolio.exposure_pct >= self.cfg.max_portfolio_exposure:
            return False, 0.0, (
                f"Portfolio fully exposed: {portfolio.exposure_pct:.1%} "
                f">= {self.cfg.max_portfolio_exposure:.1%}"
            )

        # ── 3. Liquidity filter ───────────────────────────────────────────
        if market.liquidity_usd < self.cfg.min_liquidity_usd:
            return False, 0.0, (
                f"Insufficient liquidity: ${market.liquidity_usd:,.0f} "
                f"< ${self.cfg.min_liquidity_usd:,.0f}"
            )

        # ── 4. Price range filter ─────────────────────────────────────────
        price = market.best_ask if signal.signal_type == SignalType.BUY_YES else (
            1 - market.best_bid
        )
        if price > self.cfg.max_odds or price < self.cfg.min_odds:
            return False, 0.0, f"Price {price:.3f} outside tradeable range"

        # ── 5. Minimum edge ───────────────────────────────────────────────
        if signal.edge < self.cfg.min_edge:
            return False, 0.0, (
                f"Edge {signal.edge:.3f} < minimum {self.cfg.min_edge}"
            )

        # ── 6. Kelly criterion sizing ─────────────────────────────────────
        kelly_size = self._kelly_size(signal, portfolio.bankroll)

        # Cap at max_position_pct of bankroll
        max_size = portfolio.bankroll * self.cfg.max_position_pct

        # Can't exceed remaining exposure room
        room = portfolio.bankroll * self.cfg.max_portfolio_exposure - portfolio.open_exposure_usd
        final_size = min(kelly_size, max_size, room, signal.target_size_usd)

        if final_size < 1.0:
            return False, 0.0, f"Position size ${final_size:.2f} too small (< $1)"

        logger.debug(
            f"Risk approved: kelly=${kelly_size:.2f} "
            f"max=${max_size:.2f} room=${room:.2f} → final=${final_size:.2f}"
        )

        return True, final_size, "All checks passed"

    def _kelly_size(self, signal: Signal, bankroll: float) -> float:
        """
        Fractional Kelly criterion.

        Kelly fraction f* = (bp - q) / b
            b = odds received (1/price - 1 for binary markets)
            p = our estimated win probability
            q = 1 - p

        We use `kelly_fraction` (e.g. 0.25) × f* to reduce volatility.
        """
        if signal.signal_type == SignalType.BUY_YES:
            price = signal.market.best_ask
            p = signal.fair_value
        elif signal.signal_type == SignalType.BUY_NO:
            price = 1.0 - signal.market.best_bid
            p = 1.0 - signal.fair_value
        else:
            return 0.0

        if price <= 0 or price >= 1:
            return 0.0

        b = (1.0 / price) - 1.0   # decimal odds
        q = 1.0 - p
        kelly = (b * p - q) / b

        if kelly <= 0:
            return 0.0

        return bankroll * kelly * self.cfg.kelly_fraction

    def update_portfolio_after_fill(
        self,
        portfolio: PortfolioState,
        signal: Signal,
        size_usd: float,
        fill_price: float,
    ) -> None:
        """Update portfolio state after a successful order fill."""
        cid = signal.market.condition_id
        side = "YES" if signal.signal_type == SignalType.BUY_YES else "NO"
        token_id = (
            signal.market.yes_token_id
            if side == "YES"
            else signal.market.no_token_id
        )
        shares = size_usd / fill_price if fill_price > 0 else 0.0

        if cid in portfolio.positions:
            pos = portfolio.positions[cid]
            total_cost = pos.cost_basis + size_usd
            total_shares = pos.shares + shares
            pos.avg_entry_price = total_cost / total_shares if total_shares > 0 else fill_price
            pos.shares = total_shares
        else:
            portfolio.positions[cid] = Position(
                condition_id=cid,
                token_id=token_id,
                side=side,
                shares=shares,
                avg_entry_price=fill_price,
                current_price=fill_price,
            )

        portfolio.total_trades += 1
        portfolio.bankroll -= size_usd

        # Update peak for drawdown calculation
        nav = portfolio.bankroll + sum(
            p.market_value for p in portfolio.positions.values()
        )
        if nav > portfolio.peak_bankroll:
            portfolio.peak_bankroll = nav

        logger.info(
            f"Position updated: {cid[:8]}... {side} "
            f"${size_usd:.2f} @ {fill_price:.3f} | "
            f"Bankroll: ${portfolio.bankroll:.2f}"
        )
