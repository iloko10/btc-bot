"""
btc_dashboard.py — BTC 5m trading dashboard (redesigned).
Run: python btc_dashboard.py
Open: http://localhost:5000
"""

import time
import threading
import requests
import numpy as np
from flask import Flask, render_template_string
from collections import deque

app = Flask(__name__)

state = {
    "bankroll": 1000.0,
    "trades": [],
    "equity_history": [{"t": time.time(), "v": 1000.0}],
    "log": deque(maxlen=12),
    "btc_price": 0.0,
    "btc_change": 0.0,
    "btc_candles": [],
    "momentum": 0.0,
    "rsi": 50.0,
    "last_signal": "HOLD",
    "running": True,
    "next_trade_in": 300,
    "cycle": 0,
}


def get_btc_candles(limit=40):
    try:
        r = requests.get(
            "https://api.binance.com/api/v3/klines",
            params={"symbol": "BTCUSDT", "interval": "1m", "limit": limit},
            timeout=5,
        )
        return r.json()
    except:
        return []


def calc_rsi(closes, period=7):
    if len(closes) < period + 1:
        return 50.0
    deltas = np.diff(closes)
    gains = np.where(deltas > 0, deltas, 0)
    losses = np.where(deltas < 0, -deltas, 0)
    avg_gain = np.mean(gains[-period:])
    avg_loss = np.mean(losses[-period:])
    if avg_loss == 0:
        return 100.0
    return 100 - (100 / (1 + avg_gain / avg_loss))


def trading_loop():
    import sys, os
    sys.path.insert(0, os.path.dirname(__file__))

    from config import config
    from strategies.btc_5m import BTC5mStrategy
    from risk.risk_manager import RiskManager, PortfolioState
    from execution.order_manager import OrderManager, OrderStatus
    from strategies.base import SignalType
    from market_data.polymarket import Market

    config.mode = "paper"
    config.risk.min_edge = 0.01
    config.risk.min_liquidity_usd = 100.0

    strategy = BTC5mStrategy({
        "lookback": 8,
        "rsi_period": 7,
        "momentum_threshold": 0.0005,
        "max_position": 30.0,
    })
    risk = RiskManager(config.risk)
    orders = OrderManager(config)
    portfolio = PortfolioState(bankroll=1000.0, peak_bankroll=1000.0)
    cycle = 0

    while state["running"]:
        cycle += 1
        candles = get_btc_candles(40)
        if not candles:
            time.sleep(10)
            continue

        closes = np.array([float(c[4]) for c in candles])
        btc_price = closes[-1]
        btc_prev = closes[-6]
        btc_change = (btc_price - btc_prev) / btc_prev * 100
        momentum = (closes[-1] - closes[-8]) / closes[-8]
        rsi = calc_rsi(closes)

        state["btc_price"] = round(btc_price, 2)
        state["btc_change"] = round(btc_change, 4)
        state["btc_candles"] = [round(float(c[4]), 2) for c in candles[-30:]]
        state["momentum"] = round(momentum * 100, 4)
        state["rsi"] = round(rsi, 1)
        state["cycle"] = cycle

        base_price = 0.5 + momentum * 3
        market_price = float(np.clip(base_price, 0.10, 0.90))

        market = Market(
            condition_id="btc_5m_synthetic",
            question="Will BTC be UP in the next 5 minutes?",
            category="Crypto",
            end_date="2025-12-31",
            yes_token_id="btc_up",
            no_token_id="btc_down",
            liquidity_usd=50000.0,
            volume_24h=100000.0,
            mid_price=market_price,
            best_ask=market_price + 0.005,
            best_bid=market_price - 0.005,
        )

        history = [{"t": int(c[0]) // 1000, "p": float(np.clip(0.5 + (float(c[4]) - closes[0]) / closes[0] * 10, 0.05, 0.95))}
                   for c in candles]

        signal = strategy.generate_signal(market, history)
        state["last_signal"] = signal.signal_type.name

        if cycle % 5 == 1 and signal.signal_type != SignalType.HOLD:
            approved, size_usd, reason = risk.approve(signal, portfolio)
            if approved:
                order = orders.place(signal, size_usd)
                if order.status == OrderStatus.FILLED:
                    risk.update_portfolio_after_fill(portfolio, signal, size_usd, order.fill_price)
                    side = "UP" if signal.signal_type == SignalType.BUY_YES else "DOWN"
                    state["trades"].append({
                        "time": time.strftime("%H:%M:%S"),
                        "side": side,
                        "size": round(size_usd, 2),
                        "price": round(order.fill_price, 4),
                        "btc": round(btc_price, 0),
                        "momentum": f"{momentum*100:+.3f}%",
                        "rsi": round(rsi, 1),
                    })
                    state["log"].appendleft(
                        f"{time.strftime('%H:%M:%S')}  {side}  ${size_usd:.0f}  BTC ${btc_price:,.0f}"
                    )
            else:
                state["log"].appendleft(f"{time.strftime('%H:%M:%S')}  skipped — {reason[:40]}")
        else:
            d = "up" if momentum > 0 else "down"
            state["log"].appendleft(
                f"{time.strftime('%H:%M:%S')}  {d}  {momentum*100:+.3f}%  RSI {rsi:.1f}"
            )

        state["bankroll"] = round(portfolio.bankroll, 2)
        state["equity_history"].append({"t": time.time(), "v": round(portfolio.bankroll, 2)})
        if len(state["equity_history"]) > 300:
            state["equity_history"].pop(0)
        state["next_trade_in"] = (5 - (cycle % 5)) * 60
        time.sleep(60)


HTML = open("/home/claude/btc_template.html").read()


@app.route("/")
def index():
    eq_vals = [h["v"] for h in state["equity_history"][-60:]]
    pnl = state["bankroll"] - 1000.0
    mp = float(np.clip(0.5 + state["momentum"] / 100 * 3, 0.10, 0.90))
    nxt = state["next_trade_in"]

    d = type("D", (), {
        "bankroll": state["bankroll"],
        "pnl": round(pnl, 2),
        "n_trades": len(state["trades"]),
        "trades": state["trades"],
        "log": list(state["log"]),
        "btc_price": state["btc_price"],
        "btc_change": state["btc_change"],
        "btc_candles": state["btc_candles"],
        "equity_vals": eq_vals,
        "momentum": state["momentum"],
        "rsi": state["rsi"],
        "last_signal": state["last_signal"],
        "next_trade_in": nxt,
        "next_m": nxt // 60,
        "next_s": nxt % 60,
        "market_price": round(mp, 3),
        "time": time.strftime("%H:%M:%S"),
    })()

    return render_template_string(HTML, d=d)


if __name__ == "__main__":
    t = threading.Thread(target=trading_loop, daemon=True)
    t.start()
    print("\n  BTC 5m Dashboard -> http://localhost:5000\n")
    app.run(debug=False, port=5000)