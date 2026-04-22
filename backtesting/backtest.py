"""
backtesting/backtest.py

Walk-forward backtesting engine.

Replays historical price data through any strategy and risk manager,
tracking simulated fills, P&L, and producing a performance report.

Usage:
    from backtesting.backtest import Backtest
    from strategies.mean_reversion import MeanReversionStrategy

    bt = Backtest(
        strategy=MeanReversionStrategy(params={...}),
        starting_bankroll=1000,
    )
    results = bt.run(markets_history)
    bt.print_report(results)
    bt.plot(results)       # Opens interactive Plotly chart
"""

import time
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd
from loguru import logger

from config import RiskConfig
from market_data.polymarket import Market
from risk.risk_manager import RiskManager, PortfolioState, Position
from strategies.base import BaseStrategy, Signal, SignalType


@dataclass
class TradeRecord:
    """Single completed trade in the backtest."""
    condition_id: str
    question: str
    outcome: str
    entry_time: float
    exit_time: float
    entry_price: float
    exit_price: float
    size_usd: float
    pnl_usd: float
    pnl_pct: float
    reason: str


@dataclass
class BacktestResults:
    """Aggregated results from a backtest run."""
    trades: list[TradeRecord]
    equity_curve: pd.Series          # Portfolio NAV over time
    starting_bankroll: float
    final_bankroll: float
    total_return_pct: float
    sharpe_ratio: float
    max_drawdown_pct: float
    win_rate: float
    avg_trade_pnl: float
    n_trades: int
    n_winning: int


class Backtest:
    """
    Walk-forward backtester.

    `markets_history` should be a list of dicts, each containing:
        {
          "market":  Market object (prices populated from history),
          "history": list of {t: timestamp, p: price} dicts
        }
    """

    def __init__(
        self,
        strategy: BaseStrategy,
        starting_bankroll: float = 1000.0,
        risk_cfg: RiskConfig = None,
        slippage_pct: float = 0.002,
        trading_fee_pct: float = 0.02,     # Polymarket charges ~2% maker fee
    ):
        self.strategy = strategy
        self.starting_bankroll = starting_bankroll
        self.risk = RiskManager(risk_cfg or RiskConfig())
        self.slippage_pct = slippage_pct
        self.trading_fee_pct = trading_fee_pct

    def run(self, markets_history: list[dict[str, Any]]) -> BacktestResults:
        """
        Simulate trading across all market history.

        Each entry in markets_history is processed chronologically.
        After the last candle, all positions are closed at that final price.
        """
        portfolio = PortfolioState(
            bankroll=self.starting_bankroll,
            peak_bankroll=self.starting_bankroll,
        )

        trades: list[TradeRecord] = []
        equity_timestamps: list[float] = []
        equity_values: list[float] = []

        # Open position tracker: {condition_id: (entry_price, size_usd, side, entry_time, question)}
        open_positions: dict[str, tuple] = {}

        for item in markets_history:
            market: Market = item["market"]
            history: list[dict] = item["history"]

            if len(history) < 10:
                continue

            # Step through history candle by candle
            for i in range(10, len(history)):
                candle = history[i]
                current_time = candle["t"]
                current_price = candle["p"]

                # Update market mid price for this candle
                market.mid_price = current_price
                market.best_ask = current_price + 0.005
                market.best_bid = current_price - 0.005

                # Get strategy signal
                signal: Signal = self.strategy.generate_signal(
                    market, history[:i]
                )

                cid = market.condition_id

                # ── Exit logic ──────────────────────────────────────────────
                if cid in open_positions and signal.signal_type == SignalType.CLOSE:
                    entry_price, size_usd, side, entry_time, question = open_positions[cid]
                    exit_price = current_price * (1 - self.slippage_pct)
                    fee = size_usd * self.trading_fee_pct

                    shares = size_usd / entry_price
                    proceeds = shares * exit_price - fee

                    pnl_usd = proceeds - size_usd
                    pnl_pct = pnl_usd / size_usd if size_usd > 0 else 0.0

                    portfolio.bankroll += (size_usd + pnl_usd)

                    trades.append(TradeRecord(
                        condition_id=cid,
                        question=question,
                        outcome=side,
                        entry_time=entry_time,
                        exit_time=current_time,
                        entry_price=entry_price,
                        exit_price=exit_price,
                        size_usd=size_usd,
                        pnl_usd=pnl_usd,
                        pnl_pct=pnl_pct,
                        reason=signal.reason,
                    ))
                    del open_positions[cid]

                # ── Entry logic ─────────────────────────────────────────────
                elif cid not in open_positions and signal.is_actionable(self.risk.cfg.min_edge):
                    approved, size_usd, reason = self.risk.approve(signal, portfolio)

                    if approved and size_usd > 0:
                        fill_price = market.best_ask * (1 + self.slippage_pct)
                        fee = size_usd * self.trading_fee_pct
                        actual_cost = size_usd + fee

                        if actual_cost <= portfolio.bankroll:
                            portfolio.bankroll -= actual_cost
                            side = "YES" if signal.signal_type == SignalType.BUY_YES else "NO"
                            open_positions[cid] = (
                                fill_price, size_usd, side, current_time, market.question
                            )

                # Record NAV
                open_value = sum(
                    (size_usd / ep) * current_price
                    for ep, size_usd, *_ in open_positions.values()
                )
                nav = portfolio.bankroll + open_value
                equity_timestamps.append(current_time)
                equity_values.append(nav)

        # ── Force-close all remaining positions at last known price ──────────
        for cid, (entry_price, size_usd, side, entry_time, question) in open_positions.items():
            last_price = markets_history[-1]["history"][-1]["p"] if markets_history else entry_price
            pnl_usd = (size_usd / entry_price) * last_price - size_usd
            trades.append(TradeRecord(
                condition_id=cid,
                question=question,
                outcome=side,
                entry_time=entry_time,
                exit_time=time.time(),
                entry_price=entry_price,
                exit_price=last_price,
                size_usd=size_usd,
                pnl_usd=pnl_usd,
                pnl_pct=pnl_usd / size_usd if size_usd > 0 else 0.0,
                reason="Force-closed at end of backtest",
            ))
            portfolio.bankroll += size_usd + pnl_usd

        # ── Compute summary metrics ──────────────────────────────────────────
        equity = pd.Series(equity_values, index=pd.to_datetime(equity_timestamps, unit="s"))
        final_bankroll = equity.iloc[-1] if len(equity) > 0 else portfolio.bankroll

        return BacktestResults(
            trades=trades,
            equity_curve=equity,
            starting_bankroll=self.starting_bankroll,
            final_bankroll=final_bankroll,
            total_return_pct=(final_bankroll - self.starting_bankroll) / self.starting_bankroll,
            sharpe_ratio=self._sharpe(equity),
            max_drawdown_pct=self._max_drawdown(equity),
            win_rate=sum(1 for t in trades if t.pnl_usd > 0) / len(trades) if trades else 0.0,
            avg_trade_pnl=float(np.mean([t.pnl_usd for t in trades])) if trades else 0.0,
            n_trades=len(trades),
            n_winning=sum(1 for t in trades if t.pnl_usd > 0),
        )

    # ── Reporting ─────────────────────────────────────────────────────────────

    def print_report(self, r: BacktestResults) -> None:
        """Print a concise performance summary to the terminal."""
        from rich.console import Console
        from rich.table import Table

        console = Console()
        table = Table(title=f"Backtest Results — {self.strategy.name}", show_header=True)
        table.add_column("Metric", style="cyan")
        table.add_column("Value", style="green")

        rows = [
            ("Starting Bankroll", f"${r.starting_bankroll:,.2f}"),
            ("Final Bankroll",    f"${r.final_bankroll:,.2f}"),
            ("Total Return",      f"{r.total_return_pct:+.2%}"),
            ("Sharpe Ratio",      f"{r.sharpe_ratio:.2f}"),
            ("Max Drawdown",      f"{r.max_drawdown_pct:.2%}"),
            ("Total Trades",      str(r.n_trades)),
            ("Win Rate",          f"{r.win_rate:.1%}"),
            ("Avg Trade P&L",     f"${r.avg_trade_pnl:+.2f}"),
        ]
        for metric, value in rows:
            table.add_row(metric, value)

        console.print(table)

        if r.trades:
            console.print("\n[bold]Top 5 Trades:[/bold]")
            for t in sorted(r.trades, key=lambda x: x.pnl_usd, reverse=True)[:5]:
                icon = "🟢" if t.pnl_usd > 0 else "🔴"
                console.print(
                    f"  {icon} {t.question[:60]}... | "
                    f"{t.outcome} | P&L: ${t.pnl_usd:+.2f} ({t.pnl_pct:+.1%})"
                )

    def plot(self, r: BacktestResults) -> None:
        """Open an interactive Plotly chart of the equity curve."""
        try:
            import plotly.graph_objects as go
            from plotly.subplots import make_subplots

            fig = make_subplots(
                rows=2, cols=1,
                subplot_titles=("Portfolio NAV", "Trade P&L Distribution"),
                row_heights=[0.7, 0.3],
            )

            fig.add_trace(
                go.Scatter(
                    x=r.equity_curve.index,
                    y=r.equity_curve.values,
                    name="NAV",
                    line=dict(color="royalblue", width=2),
                    fill="tozeroy",
                    fillcolor="rgba(65, 105, 225, 0.1)",
                ),
                row=1, col=1,
            )

            pnls = [t.pnl_usd for t in r.trades]
            colors = ["green" if p > 0 else "red" for p in pnls]
            fig.add_trace(
                go.Bar(
                    x=list(range(len(pnls))),
                    y=pnls,
                    marker_color=colors,
                    name="Trade P&L",
                ),
                row=2, col=1,
            )

            fig.update_layout(
                title=f"Backtest: {self.strategy.name} | Return: {r.total_return_pct:+.2%}",
                showlegend=False,
                height=700,
            )
            fig.show()
        except ImportError:
            logger.warning("plotly not installed — pip install plotly")

    # ── Helpers ───────────────────────────────────────────────────────────────

    def _sharpe(self, equity: pd.Series, risk_free: float = 0.05) -> float:
        if len(equity) < 2:
            return 0.0
        returns = equity.pct_change().dropna()
        excess = returns - risk_free / 252
        if excess.std() == 0:
            return 0.0
        return float(excess.mean() / excess.std() * np.sqrt(252))

    def _max_drawdown(self, equity: pd.Series) -> float:
        if len(equity) < 2:
            return 0.0
        roll_max = equity.cummax()
        drawdown = (equity - roll_max) / roll_max
        return float(drawdown.min()) * -1
