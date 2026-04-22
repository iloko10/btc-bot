"""
btc_dashboard.py — Live dashboard ειδικά για BTC 5-λεπτα UP/DOWN markets.

Τρέξε: python btc_dashboard.py
Άνοιξε: http://localhost:5000
"""

import json
import time
import threading
import requests
import numpy as np
from flask import Flask, render_template_string
from collections import deque

app = Flask(__name__)

# ── Shared state ──────────────────────────────────────────────────────────────
state = {
    "bankroll": 1000.0,
    "trades": [],
    "equity_history": [{"t": time.time(), "v": 1000.0}],
    "log": deque(maxlen=15),
    "btc_price": 0.0,
    "btc_change": 0.0,
    "btc_candles": [],
    "momentum": 0.0,
    "rsi": 50.0,
    "last_signal": "—",
    "running": True,
    "next_trade_in": 300,
}


def get_btc_candles(limit=20):
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

        # ── Fetch BTC data ────────────────────────────────────────────────
        candles = get_btc_candles(20)
        if not candles:
            state["log"].appendleft(f"{time.strftime('%H:%M:%S')} — ⚠️ Binance fetch failed")
            time.sleep(10)
            continue

        closes = np.array([float(c[4]) for c in candles])
        btc_price = closes[-1]
        btc_prev = closes[-6]  # 5 candles ago
        btc_change = (btc_price - btc_prev) / btc_prev * 100
        momentum = (closes[-1] - closes[-8]) / closes[-8]
        rsi = calc_rsi(closes)

        state["btc_price"] = round(btc_price, 2)
        state["btc_change"] = round(btc_change, 4)
        state["btc_candles"] = [float(c[4]) for c in candles[-30:]]
        state["momentum"] = round(momentum * 100, 4)
        state["rsi"] = round(rsi, 1)

        # ── Synthetic Polymarket BTC 5m market ───────────────────────────
        # In production: fetch from gamma-api.polymarket.com with VPN
        # Market price reflects current momentum
        base_price = 0.5 + momentum * 3
        market_price = float(np.clip(base_price, 0.10, 0.90))

        market = Market(
            condition_id="btc_5m_synthetic",
            question=f"Will BTC be UP in the next 5 minutes? (Current: ${btc_price:,.0f})",
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

        # Every 5 minutes (cycle 1, 6, 11...) attempt a trade
        if cycle % 5 == 1 and signal.signal_type != SignalType.HOLD:
            approved, size_usd, reason = risk.approve(signal, portfolio)
            if approved:
                order = orders.place(signal, size_usd)
                if order.status == OrderStatus.FILLED:
                    risk.update_portfolio_after_fill(portfolio, signal, size_usd, order.fill_price)
                    side = "🟢 UP" if signal.signal_type == SignalType.BUY_YES else "🔴 DOWN"
                    state["trades"].append({
                        "time": time.strftime("%H:%M:%S"),
                        "side": "UP" if signal.signal_type == SignalType.BUY_YES else "DOWN",
                        "size": round(size_usd, 2),
                        "price": round(order.fill_price, 4),
                        "btc": f"${btc_price:,.0f}",
                        "momentum": f"{momentum*100:+.3f}%",
                        "rsi": round(rsi, 1),
                    })
                    state["log"].appendleft(
                        f"{time.strftime('%H:%M:%S')} — {side} ${size_usd:.0f} @ {order.fill_price:.3f} | BTC ${btc_price:,.0f}"
                    )
            else:
                state["log"].appendleft(f"{time.strftime('%H:%M:%S')} — Skipped: {reason[:50]}")
        else:
            direction = "📈 UP" if momentum > 0 else "📉 DOWN"
            state["log"].appendleft(
                f"{time.strftime('%H:%M:%S')} — {direction} momentum {momentum*100:+.3f}% | RSI {rsi:.1f}"
            )

        state["bankroll"] = round(portfolio.bankroll, 2)
        state["equity_history"].append({"t": time.time(), "v": round(portfolio.bankroll, 2)})
        if len(state["equity_history"]) > 300:
            state["equity_history"].pop(0)

        state["next_trade_in"] = (5 - (cycle % 5)) * 60

        time.sleep(60)  # check every 1 minute, trade every 5


# ── HTML ──────────────────────────────────────────────────────────────────────

HTML = """
<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta http-equiv="refresh" content="10">
<title>BTC 5m Bot</title>
<link href="https://fonts.googleapis.com/css2?family=Space+Mono:wght@400;700&family=DM+Sans:wght@300;400;600&display=swap" rel="stylesheet">
<style>
:root {
  --bg: #080c14;
  --panel: #0d1421;
  --border: #162032;
  --green: #00e676;
  --red: #ff1744;
  --blue: #29b6f6;
  --orange: #ff9100;
  --text: #cdd9e5;
  --muted: #4a6080;
}
* { margin:0; padding:0; box-sizing:border-box; }
body { background:var(--bg); color:var(--text); font-family:'DM Sans',sans-serif; padding:20px; min-height:100vh; }

header { display:flex; justify-content:space-between; align-items:center; margin-bottom:24px; padding-bottom:14px; border-bottom:1px solid var(--border); }
.logo { font-family:'Space Mono',monospace; font-size:15px; color:var(--orange); letter-spacing:3px; }
.btc-price { font-family:'Space Mono',monospace; font-size:22px; font-weight:700; color:var(--blue); }
.btc-change { font-size:13px; margin-left:10px; }

.grid4 { display:grid; grid-template-columns:repeat(4,1fr); gap:14px; margin-bottom:20px; }
.card { background:var(--panel); border:1px solid var(--border); border-radius:10px; padding:18px; }
.card .lbl { font-family:'Space Mono',monospace; font-size:10px; text-transform:uppercase; letter-spacing:1.5px; color:var(--muted); margin-bottom:8px; }
.card .val { font-family:'Space Mono',monospace; font-size:24px; font-weight:700; }
.green { color:var(--green); }
.red   { color:var(--red); }
.blue  { color:var(--blue); }
.orange{ color:var(--orange); }

.main-grid { display:grid; grid-template-columns:1.8fr 1fr; gap:16px; margin-bottom:16px; }
.panel { background:var(--panel); border:1px solid var(--border); border-radius:10px; padding:18px; }
.panel h3 { font-family:'Space Mono',monospace; font-size:10px; text-transform:uppercase; letter-spacing:2px; color:var(--muted); margin-bottom:14px; }

/* BTC Chart */
svg.btcchart { width:100%; height:120px; }

/* Momentum bar */
.mom-bar-wrap { margin:14px 0 6px; height:8px; background:#162032; border-radius:4px; overflow:hidden; }
.mom-bar { height:100%; border-radius:4px; transition:width .5s; }

/* RSI */
.rsi-wrap { position:relative; height:6px; background:#162032; border-radius:3px; margin:10px 0 4px; }
.rsi-fill { height:100%; border-radius:3px; }
.rsi-marker { position:absolute; top:-4px; width:14px; height:14px; border-radius:50%; border:2px solid var(--bg); transform:translateX(-50%); }

/* Signal badge */
.signal { display:inline-block; padding:4px 14px; border-radius:20px; font-family:'Space Mono',monospace; font-size:12px; font-weight:700; }
.signal.UP   { background:rgba(0,230,118,0.15); color:var(--green); border:1px solid var(--green); }
.signal.DOWN { background:rgba(255,23,68,0.15);  color:var(--red);   border:1px solid var(--red); }
.signal.HOLD { background:rgba(74,96,128,0.2);   color:var(--muted); border:1px solid var(--muted); }

/* Trades */
table { width:100%; border-collapse:collapse; font-size:12px; }
th { font-family:'Space Mono',monospace; font-size:9px; text-transform:uppercase; letter-spacing:1px; color:var(--muted); padding:6px 8px; border-bottom:1px solid var(--border); text-align:left; }
td { padding:8px 8px; border-bottom:1px solid #0f1c2e; }
tr:last-child td { border:none; }

/* Log */
.log-line { font-family:'Space Mono',monospace; font-size:10px; color:var(--muted); padding:5px 0; border-bottom:1px solid #0f1c2e; }
.log-line:first-child { color:var(--text); }
.log-line:last-child { border:none; }

.footer { font-family:'Space Mono',monospace; font-size:10px; color:var(--muted); text-align:center; margin-top:14px; }

.next-trade { font-family:'Space Mono',monospace; font-size:11px; color:var(--orange); }
</style>
</head>
<body>

<header>
  <div class="logo">⬡ BTC 5M TRADING BOT</div>
  <div style="display:flex;align-items:center">
    <div class="btc-price">${{ "{:,.2f}".format(d.btc_price) }}</div>
    <div class="btc-change {{ 'green' if d.btc_change >= 0 else 'red' }}">
      {{ "+" if d.btc_change >= 0 else "" }}{{ "%.3f"|format(d.btc_change) }}%
    </div>
  </div>
  <div class="next-trade">Next trade check: ~{{ d.next_trade_in // 60 }}m {{ d.next_trade_in % 60 }}s</div>
</header>

<div class="grid4">
  <div class="card">
    <div class="lbl">Bankroll</div>
    <div class="val blue">${{ "%.2f"|format(d.bankroll) }}</div>
  </div>
  <div class="card">
    <div class="lbl">P&L</div>
    <div class="val {{ 'green' if d.pnl >= 0 else 'red' }}">{{ "+" if d.pnl >= 0 else "" }}${{ "%.2f"|format(d.pnl) }}</div>
  </div>
  <div class="card">
    <div class="lbl">Signal</div>
    <div style="margin-top:4px">
      <span class="signal {{ 'UP' if 'YES' in d.last_signal else ('DOWN' if 'NO' in d.last_signal else 'HOLD') }}">
        {{ '📈 UP' if 'YES' in d.last_signal else ('📉 DOWN' if 'NO' in d.last_signal else '— HOLD') }}
      </span>
    </div>
  </div>
  <div class="card">
    <div class="lbl">Trades</div>
    <div class="val orange">{{ d.n_trades }}</div>
  </div>
</div>

<div class="main-grid">
  <div class="panel">
    <h3>BTC Price (last 30 candles)</h3>
    {% if d.btc_candles|length > 2 %}
    <svg class="btcchart" viewBox="0 0 600 110" preserveAspectRatio="none">
      <defs>
        <linearGradient id="btcgrad" x1="0" y1="0" x2="0" y2="1">
          <stop offset="0%" stop-color="{{ '#00e676' if d.btc_change >= 0 else '#ff1744' }}" stop-opacity="0.25"/>
          <stop offset="100%" stop-color="{{ '#00e676' if d.btc_change >= 0 else '#ff1744' }}" stop-opacity="0"/>
        </linearGradient>
      </defs>
      <polygon points="{{ d.btc_area }}" fill="url(#btcgrad)"/>
      <polyline points="{{ d.btc_line }}" fill="none" stroke="{{ '#00e676' if d.btc_change >= 0 else '#ff1744' }}" stroke-width="2"/>
    </svg>
    {% endif %}

    <h3 style="margin-top:16px">Momentum ({{ d.momentum }}%)</h3>
    {% set mom_pct = [[(d.momentum / 2 + 50)|round|int, 5]|max, 95]|min %}
    <div class="mom-bar-wrap">
      <div class="mom-bar" style="width:{{ mom_pct }}%;background:{{ '#00e676' if d.momentum >= 0 else '#ff1744' }}"></div>
    </div>
    <div style="display:flex;justify-content:space-between;font-size:10px;color:var(--muted);font-family:monospace">
      <span>BEAR</span><span>NEUTRAL</span><span>BULL</span>
    </div>

    <h3 style="margin-top:16px">RSI ({{ d.rsi }})</h3>
    {% set rsi_pct = d.rsi|round|int %}
    <div class="rsi-wrap">
      <div class="rsi-fill" style="width:{{ rsi_pct }}%;background:{{ '#ff1744' if d.rsi > 70 else ('#00e676' if d.rsi < 30 else '#29b6f6') }}"></div>
      <div class="rsi-marker" style="left:{{ rsi_pct }}%;background:{{ '#ff1744' if d.rsi > 70 else ('#00e676' if d.rsi < 30 else '#29b6f6') }}"></div>
    </div>
    <div style="display:flex;justify-content:space-between;font-size:10px;color:var(--muted);font-family:monospace;margin-top:6px">
      <span>0 Oversold</span><span>50</span><span>100 Overbought</span>
    </div>

    <h3 style="margin-top:18px">Recent Trades</h3>
    <table>
      <tr><th>Time</th><th>Side</th><th>Size</th><th>Price</th><th>BTC</th><th>Mom.</th><th>RSI</th></tr>
      {% for t in d.trades[-6:]|reverse %}
      <tr>
        <td style="color:var(--muted);font-family:monospace">{{ t.time }}</td>
        <td class="{{ 'green' if t.side == 'UP' else 'red' }}" style="font-weight:700;font-family:monospace">
          {{ '📈 UP' if t.side == 'UP' else '📉 DOWN' }}
        </td>
        <td style="font-family:monospace">${{ t.size }}</td>
        <td style="font-family:monospace">{{ t.price }}</td>
        <td style="font-family:monospace;color:var(--blue)">{{ t.btc }}</td>
        <td style="font-family:monospace;color:{{ 'var(--green)' if '+' in t.momentum else 'var(--red)' }}">{{ t.momentum }}</td>
        <td style="font-family:monospace">{{ t.rsi }}</td>
      </tr>
      {% else %}
      <tr><td colspan="7" style="text-align:center;color:var(--muted);padding:16px">Waiting for next 5-minute cycle...</td></tr>
      {% endfor %}
    </table>
  </div>

  <div class="panel">
    <h3>Activity Log</h3>
    {% for line in d.log %}
    <div class="log-line">{{ line }}</div>
    {% else %}
    <div class="log-line">Initializing...</div>
    {% endfor %}
  </div>
</div>

<div class="footer">
  Auto-refresh 10s &nbsp;|&nbsp; Paper Trading Mode &nbsp;|&nbsp; {{ d.time }} &nbsp;|&nbsp;
  <span style="color:var(--orange)">⚠️ Connect VPN for live Polymarket data</span>
</div>

</body>
</html>
"""


def build_svg(values, W=600, H=100):
    if len(values) < 2:
        return "", ""
    n = len(values)
    mn, mx = min(values), max(values)
    rng = mx - mn if mx != mn else 1

    def pt(i, v):
        x = i / (n - 1) * W
        y = H - ((v - mn) / rng) * (H - 4) - 2
        return f"{x:.1f},{y:.1f}"

    line = " ".join(pt(i, v) for i, v in enumerate(values))
    area = f"0,{H} " + line + f" {W},{H}"
    return line, area


@app.route("/")
def index():
    btc_line, btc_area = build_svg(state["btc_candles"]) if state["btc_candles"] else ("", "")
    pnl = state["bankroll"] - 1000.0

    d = type("D", (), {
        "bankroll": state["bankroll"],
        "pnl": round(pnl, 2),
        "pnl_pct": round(pnl / 10, 2),
        "n_trades": len(state["trades"]),
        "trades": state["trades"],
        "log": list(state["log"]),
        "btc_price": state["btc_price"],
        "btc_change": state["btc_change"],
        "btc_candles": state["btc_candles"],
        "btc_line": btc_line,
        "btc_area": btc_area,
        "momentum": state["momentum"],
        "rsi": state["rsi"],
        "last_signal": state["last_signal"],
        "next_trade_in": state["next_trade_in"],
        "time": time.strftime("%H:%M:%S"),
    })()

    return render_template_string(HTML, d=d)


if __name__ == "__main__":
    t = threading.Thread(target=trading_loop, daemon=True)
    t.start()
    print("\n🚀 BTC 5m Dashboard: http://localhost:5000\n")
    print("📊 Fetching live BTC price from Binance...")
    print("⚠️  Polymarket data needs VPN — running in paper mode\n")
    app.run(debug=False, port=5000)