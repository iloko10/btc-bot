"""
config.py — Centralized configuration with Pydantic validation.
All secrets come from environment variables / .env file.
"""

import os
from dataclasses import dataclass, field
from typing import Optional
from dotenv import load_dotenv

load_dotenv()


@dataclass
class PolymarketConfig:
    """Polymarket API credentials and endpoints."""
    api_key: str = field(default_factory=lambda: os.getenv("POLY_API_KEY", ""))
    api_secret: str = field(default_factory=lambda: os.getenv("POLY_API_SECRET", ""))
    api_passphrase: str = field(default_factory=lambda: os.getenv("POLY_PASSPHRASE", ""))
    private_key: str = field(default_factory=lambda: os.getenv("POLY_PRIVATE_KEY", ""))
    wallet_address: str = field(default_factory=lambda: os.getenv("POLY_WALLET", ""))

    # Endpoints
    gamma_api_url: str = "https://gamma-api.polymarket.com"
    clob_api_url: str = "https://clob.polymarket.com"
    ws_url: str = "wss://ws-subscriptions-clob.polymarket.com/ws/market"

    # Chain (137 = Polygon mainnet)
    chain_id: int = 137


@dataclass
class RiskConfig:
    """Risk management parameters."""
    max_position_pct: float = 0.05       # Max 5% of bankroll per trade
    max_portfolio_exposure: float = 0.40  # Max 40% total exposure
    kelly_fraction: float = 0.25          # Quarter-Kelly for safety
    min_edge: float = 0.03                # Min 3% edge to trade
    max_drawdown_pct: float = 0.15        # Halt trading at 15% drawdown
    min_liquidity_usd: float = 1000.0     # Skip illiquid markets
    max_odds: float = 0.95                # Don't bet heavy favourites
    min_odds: float = 0.05                # Don't touch longshots


@dataclass
class StrategyConfig:
    """Strategy-specific parameters."""
    # Mean reversion
    lookback_hours: int = 48
    z_score_entry: float = 1.5      # Enter when price deviates 1.5 SD
    z_score_exit: float = 0.3       # Exit when price reverts to 0.3 SD
    
    # Sentiment
    sentiment_threshold: float = 0.65  # Confidence threshold for news signals

    # General
    poll_interval_seconds: int = 30
    order_timeout_seconds: int = 60


@dataclass
class BotConfig:
    """Top-level bot configuration."""
    mode: str = "paper"          # paper | live | backtest
    strategy: str = "mean_reversion"
    bankroll_usd: float = float(os.getenv("BANKROLL_USD", "1000"))
    log_level: str = os.getenv("LOG_LEVEL", "INFO")
    log_file: str = "logs/bot.log"
    db_path: str = "data/trades.db"

    polymarket: PolymarketConfig = field(default_factory=PolymarketConfig)
    risk: RiskConfig = field(default_factory=RiskConfig)
    strategy_params: StrategyConfig = field(default_factory=StrategyConfig)

    def validate(self) -> None:
        """Raise if config is missing required fields for live mode."""
        if self.mode == "live":
            missing = [
                k for k, v in {
                    "POLY_API_KEY": self.polymarket.api_key,
                    "POLY_PRIVATE_KEY": self.polymarket.private_key,
                    "POLY_WALLET": self.polymarket.wallet_address,
                }.items() if not v
            ]
            if missing:
                raise EnvironmentError(f"Missing env vars for live mode: {missing}")


# Singleton
config = BotConfig()
