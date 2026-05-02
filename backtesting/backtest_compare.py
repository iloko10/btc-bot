"""
backtest_compare.py — Replays OLD vs NEW signal logic over historical
Binance 1m BTCUSDT candles, with the same 5-minute binary settle rule
your live bot uses.

Run:
    python backtesting/backtest_compare.py
    python backtesting/backtest_compare.py --days 30

Output: per-strategy summary (trades, win rate, P&L, max drawdown).
This is a paper backtest. The fill model is simplified — real Polymarket
fills will differ. Treat results as directional, not as profit guarantees.
"""

import argparse
import time
from datetime import datetime
from typing import Callable

import numpy as np
import requests


# ── Data fetch ────────────────────────────────────────────────────────────────

BINANCE_KLINES = "https://api.binance.com/api/v3/klines"
INTERVAL_MS = 60_000  # 1 minute


def fetch_klines(symbol: str, days: int) -> list[list]:
    """Fetch 1m klines for the last `days` days. Paginates 1000 at a time."""
    end = int(time.time() * 1000)
    start = end - days * 24 * 60 * 60 * 1000
    out: list[list] = []
    cur = start
    while cur < end:
        r = requests.get(
            BINANCE_KLINES,
            params={
                "symbol": symbol,
                "interval": "1m",
                "startTime": cur,
                "limit": 1000,
            },
            timeout=15,
        )
        r.raise_for_status()
        batch = r.json()
        if not batch:
            break
        out.extend(batch)
        cur = batch[-1][0] + INTERVAL_MS
        time.sleep(0.05)  # be nice
    return out


# ── Indicators (must match bot.py) ────────────────────────────────────────────

def calc_rsi(closes: np.ndarray, p: int = 7) -> float:
    if len(closes) < p + 1:
        return 50.0
    d = np.diff(closes)
    g = float(np.mean(np.where(d > 0, d, 0)[-p:]))
    l = float(np.mean(np.where(d < 0, -d, 0)[-p:]))
    return 100.0 if l == 0 else 100 - (100 / (1 + g / l))


def calc_ema(prices: np.ndarray, period: int) -> float:
    if len(prices) < period:
        return float(np.mean(prices))
    k = 2 / (period + 1)
    ema = float(np.mean(prices[:period]))
    for x in prices[period:]:
        ema = float(x) * k + ema * (1 - k)
    return ema


def calc_atr(highs: np.ndarray, lows: np.ndarray, closes: np.ndarray, p: int = 14) -> float:
    if len(closes) < p + 1:
        return 0.0
    tr = []
    for i in range(1, len(closes)):
        h, l, pc = highs[i], lows[i], closes[i - 1]
        tr.append(max(h - l, abs(h - pc), abs(l - pc)))
    return float(np.mean(tr[-p:]))


# ── Strategies ────────────────────────────────────────────────────────────────
# Each strategy gets a window of recent bars and returns "UP", "DOWN", or "HOLD".
# It also takes seconds-since-last-loss so it can apply a cooldown.

def strategy_old(highs, lows, closes, since_loss: float) -> str:
    if len(closes) < 10:
        return "HOLD"
    mom = (closes[-1] - closes[-8]) / closes[-8]
    r = calc_rsi(closes)
    if mom > 0.0001 and r < 85:
        return "UP"
    if mom < -0.0001 and r > 15:
        return "DOWN"
    return "HOLD"


LOSS_COOLDOWN_SEC = 180


def strategy_new(highs, lows, closes, since_loss: float) -> str:
    if len(closes) < 30:
        return "HOLD"
    mom = (closes[-1] - closes[-8]) / closes[-8]
    r = calc_rsi(closes)
    ema9 = calc_ema(closes, 9)
    ema21 = calc_ema(closes, 21)
    atr_now = calc_atr(highs, lows, closes, 14)

    # Volatility regime: percentile over last 60 bars
    atr_window = []
    for i in range(max(15, len(closes) - 60), len(closes)):
        atr_window.append(calc_atr(highs[: i + 1], lows[: i + 1], closes[: i + 1], 14))
    atr_pct = (
        sum(1 for x in atr_window if x <= atr_now) / len(atr_window)
        if atr_window else 0.5
    )

    sig = "HOLD"
    if mom > 0.0001 and r < 70 and ema9 > ema21:
        sig = "UP"
    elif mom < -0.0001 and r > 30 and ema9 < ema21:
        sig = "DOWN"

    if sig != "HOLD":
        if atr_pct < 0.30 or atr_pct > 0.95:
            sig = "HOLD"
        elif since_loss < LOSS_COOLDOWN_SEC:
            sig = "HOLD"

    return sig


# ── Backtest engine ───────────────────────────────────────────────────────────

def run_backtest(name: str, strategy: Callable, klines: list[list]) -> dict:
    """Replay every minute. One trade open at a time, 5min binary settle."""
    closes_all = np.array([float(k[4]) for k in klines])
    highs_all = np.array([float(k[2]) for k in klines])
    lows_all = np.array([float(k[3]) for k in klines])

    bankroll = 1000.0
    initial = bankroll
    wins = losses = 0
    pending = None  # {"side", "size", "fill", "entry", "settle_idx"}
    last_loss_idx = -10_000
    equity = [bankroll]
    trades_log = []

    for i in range(30, len(closes_all)):
        # Settle if due
        if pending and i >= pending["settle_idx"]:
            went_up = closes_all[i] > pending["entry"]
            won = (pending["side"] == "UP" and went_up) or \
                  (pending["side"] == "DOWN" and not went_up)
            if won:
                shares = pending["size"] / pending["fill"]
                bankroll += shares * 1.0
                pnl = shares * 1.0 - pending["size"]
                wins += 1
            else:
                pnl = -pending["size"]
                losses += 1
                last_loss_idx = i
            trades_log.append((pending["side"], pnl, won))
            pending = None

        # Pick a window of recent bars (need 60+ for ATR pct)
        window_start = max(0, i - 90)
        ch = closes_all[window_start: i + 1]
        hh = highs_all[window_start: i + 1]
        lh = lows_all[window_start: i + 1]

        since_loss = (i - last_loss_idx) * 60  # minutes -> seconds
        sig = strategy(hh, lh, ch, since_loss)

        # Open a trade if no pending
        if sig != "HOLD" and pending is None:
            mom = (ch[-1] - ch[-8]) / ch[-8]
            kelly = bankroll * min(abs(mom) * 5, 0.05)
            size = round(max(min(kelly, 30.0), 1.0), 2)
            mkt_price = float(np.clip(0.5 + mom * 3, 0.10, 0.90))
            fill = round(mkt_price * 1.002, 4)
            if size <= bankroll:
                bankroll -= size
                pending = {
                    "side": sig,
                    "size": size,
                    "fill": fill,
                    "entry": ch[-1],
                    "settle_idx": i + 5,
                }

        equity.append(bankroll + (pending["size"] if pending else 0))

    # Max drawdown
    peak = equity[0]
    max_dd = 0.0
    for v in equity:
        peak = max(peak, v)
        dd = (peak - v) / peak if peak else 0
        max_dd = max(max_dd, dd)

    n = wins + losses
    return {
        "name": name,
        "trades": n,
        "wins": wins,
        "losses": losses,
        "win_rate": wins / n if n else 0.0,
        "final_bankroll": round(equity[-1], 2),
        "pnl": round(equity[-1] - initial, 2),
        "pnl_pct": round((equity[-1] - initial) / initial * 100, 2),
        "max_drawdown_pct": round(max_dd * 100, 2),
    }


def fmt_row(label: str, r: dict) -> str:
    return (
        f"{label:6s}  trades={r['trades']:4d}  "
        f"win%={r['win_rate']*100:5.1f}  "
        f"P&L=${r['pnl']:>+8.2f} ({r['pnl_pct']:>+5.1f}%)  "
        f"MaxDD={r['max_drawdown_pct']:5.2f}%  "
        f"final=${r['final_bankroll']:.2f}"
    )


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--days", type=int, default=30, help="Backtest window in days")
    ap.add_argument("--symbol", default="BTCUSDT")
    args = ap.parse_args()

    print(f"Fetching {args.days} days of {args.symbol} 1m candles from Binance...")
    klines = fetch_klines(args.symbol, args.days)
    if not klines:
        print("No data returned. Exiting.")
        return
    start_ts = datetime.utcfromtimestamp(klines[0][0] / 1000)
    end_ts = datetime.utcfromtimestamp(klines[-1][0] / 1000)
    print(f"Got {len(klines):,} bars  ({start_ts:%Y-%m-%d %H:%M} -> {end_ts:%Y-%m-%d %H:%M} UTC)")
    print()

    old = run_backtest("OLD", strategy_old, klines)
    new = run_backtest("NEW", strategy_new, klines)

    print(fmt_row("OLD ", old))
    print(fmt_row("NEW ", new))
    print()
    if new["trades"] == 0:
        print("NEW took zero trades — filters may be too strict for this window.")
    else:
        delta_pnl = new["pnl"] - old["pnl"]
        delta_wr = (new["win_rate"] - old["win_rate"]) * 100
        print(f"Delta:  P&L {delta_pnl:+.2f}  win-rate {delta_wr:+.1f} pp")


if __name__ == "__main__":
    main()
