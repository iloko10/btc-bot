"""
main.py — Entry point for the prediction market trading bot.

Modes:
  backtest  — Replay historical data through the strategy
  paper     — Live market data, simulated trades (default)
  live      — Real money on Polymarket (requires API keys)

Examples:
  python main.py --mode paper --strategy mean_reversion
  python main.py --mode backtest --strategy mean_reversion --days 30
  python main.py --mode live --strategy mean_reversion --bankroll 500
"""

import sys
import time
import signal as os_signal

import click
from loguru import logger
from rich.console import Console
from rich.live import Live
from rich.table import Table

from config import config, BotConfig
from market_data.polymarket import PolymarketClient, Market
from strategies.mean_reversion import MeanReversionStrategy
from strategies.news_sentiment import NewsSentimentStrategy
from strategies.base import BaseStrategy, SignalType
from risk.risk_manager import RiskManager, PortfolioState
from execution.order_manager import OrderManager, OrderStatus

console = Console()


# ── Strategy registry ─────────────────────────────────────────────────────────

STRATEGIES: dict[str, type[BaseStrategy]] = {
    "mean_reversion": MeanReversionStrategy,
    "news_sentiment": NewsSentimentStrategy,
}


def build_strategy(name: str, cfg: BotConfig) -> BaseStrategy:
    if name not in STRATEGIES:
        logger.error(f"Unknown strategy '{name}'. Available: {list(STRATEGIES)}")
        sys.exit(1)
    params = {
        "lookback_n": cfg.strategy_params.lookback_hours,
        "z_entry": cfg.strategy_params.z_score_entry,
        "z_exit": cfg.strategy_params.z_score_exit,
        "max_position": cfg.bankroll_usd * cfg.risk.max_position_pct,
        "sentiment_threshold": cfg.strategy_params.sentiment_threshold,
    }
    return STRATEGIES[name](params)


# ── Bot loop ──────────────────────────────────────────────────────────────────

class TradingBot:
    """
    Core bot loop.
    1. Fetch active markets with enough liquidity.
    2. For each market, pull price history and generate a signal.
    3. Pass signals through the risk manager.
    4. Submit approved orders via the order manager.
    """

    def __init__(self, cfg: BotConfig, strategy: BaseStrategy):
        self.cfg = cfg
        self.strategy = strategy
        self.client = PolymarketClient(cfg.polymarket)
        self.risk = RiskManager(cfg.risk)
        self.orders = OrderManager(cfg)
        self.portfolio = PortfolioState(
            bankroll=cfg.bankroll_usd,
            peak_bankroll=cfg.bankroll_usd,
        )
        self._running = True

        # Graceful shutdown on SIGINT / SIGTERM
        os_signal.signal(os_signal.SIGINT, self._shutdown)
        os_signal.signal(os_signal.SIGTERM, self._shutdown)

    def run(self) -> None:
        logger.info(
            f"Starting bot | mode={self.cfg.mode} "
            f"strategy={self.strategy.name} "
            f"bankroll=${self.portfolio.bankroll:,.2f}"
        )

        while self._running:
            try:
                self._tick()
                self._print_status()
                time.sleep(self.cfg.strategy_params.poll_interval_seconds)
            except KeyboardInterrupt:
                break
            except Exception as e:
                logger.exception(f"Unhandled error in tick: {e}")
                time.sleep(10)

        logger.info("Bot stopped.")

    def _tick(self) -> None:
        import numpy as np
        return

    def _passes_prefilter(self, market: Market) -> bool:
        """Quick pre-filters before fetching full data."""
        return (
            market.liquidity_usd >= self.cfg.risk.min_liquidity_usd
            and market.volume_24h > 0
            and market.is_active
        )

    def _print_status(self) -> None:
        """Print a live portfolio status table."""
        table = Table(title="Portfolio Status", show_header=True, header_style="bold cyan")
        table.add_column("Metric")
        table.add_column("Value", style="green")
        table.add_row("Bankroll",    f"${self.portfolio.bankroll:,.2f}")
        table.add_row("Exposure",    f"{self.portfolio.exposure_pct:.1%}")
        table.add_row("Drawdown",    f"{self.portfolio.drawdown_pct:.1%}")
        table.add_row("Open Pos.",   str(len(self.portfolio.positions)))
        table.add_row("Total Trades", str(self.portfolio.total_trades))
        console.print(table)

    def _shutdown(self, *_) -> None:
        logger.info("Shutdown signal received — stopping after current tick...")
        self._running = False


# ── Backtest runner ───────────────────────────────────────────────────────────

def run_backtest(cfg: BotConfig, strategy: BaseStrategy, days: int) -> None:
    from backtesting.backtest import Backtest
    import numpy as np

    console.print(f"[cyan]Generating synthetic data for {days}-day backtest...[/cyan]")

    def make_history(n=days*24):
        """Simulate a noisy probability time series."""
        prices = [0.5]
        for _ in range(n):
            change = np.random.normal(0, 0.02)
            new_price = np.clip(prices[-1] + change, 0.05, 0.95)
            prices.append(new_price)
        base_time = 1700000000
        return [{"t": base_time + i*3600, "p": p} for i, p in enumerate(prices)]

    markets_history = []
    questions = [
        "Will BTC exceed $100k by end of 2025?",
        "Will the Fed cut rates in June 2025?",
        "Will Arsenal win the Premier League?",
        "Will GPT-5 be released before July 2025?",
        "Will inflation drop below 3% by Q3 2025?",
    ]
    for i, q in enumerate(questions):
        market = Market(
            condition_id=f"synthetic_{i:04d}",
            question=q,
            category="Synthetic",
            end_date="2025-12-31",
            yes_token_id=f"yes_{i}",
            no_token_id=f"no_{i}",
            liquidity_usd=50000.0,
            volume_24h=10000.0,
        )
        markets_history.append({"market": market, "history": make_history()})

    console.print(f"Running backtest on {len(markets_history)} synthetic markets...")

    bt = Backtest(
        strategy=strategy,
        starting_bankroll=cfg.bankroll_usd,
        risk_cfg=cfg.risk,
    )
    results = bt.run(markets_history)
    bt.print_report(results)
    bt.plot(results)


# ── CLI ───────────────────────────────────────────────────────────────────────

@click.command()
@click.option("--mode", default="paper", type=click.Choice(["paper", "live", "backtest"]))
@click.option("--strategy", default="mean_reversion", type=click.Choice(list(STRATEGIES)))
@click.option("--bankroll", default=None, type=float, help="Override bankroll (USD)")
@click.option("--days", default=30, type=int, help="Days of history for backtest")
@click.option("--log-level", default="INFO", type=click.Choice(["DEBUG", "INFO", "WARNING"]))
def main(mode: str, strategy: str, bankroll: float, days: int, log_level: str):
    # Configure logger
    logger.remove()
    logger.add(sys.stderr, level=log_level, format="<green>{time:HH:mm:ss}</green> | <level>{level}</level> | {message}")
    logger.add(config.log_file, rotation="10 MB", level="DEBUG")

    # Override config from CLI flags
    config.mode = mode
    if bankroll:
        config.bankroll_usd = bankroll

    # Validate config for live mode
    if mode == "live":
        try:
            config.validate()
        except EnvironmentError as e:
            console.print(f"[red]Config error: {e}[/red]")
            sys.exit(1)

    strat = build_strategy(strategy, config)

    console.print(f"[bold green]Prediction Market Bot[/bold green]")
    console.print(f"Mode: [cyan]{mode}[/cyan] | Strategy: [cyan]{strategy}[/cyan]")

    if mode == "backtest":
        run_backtest(config, strat, days)
    else:
        bot = TradingBot(config, strat)
        bot.run()


if __name__ == "__main__":
    main()
