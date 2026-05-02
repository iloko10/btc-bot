"""
bot.py — BTC 5m trading bot with proper settlement
Run: python bot.py
Open: http://localhost:5000
"""

import time, threading, requests, numpy as np
import sqlite3, json

DB = "trades.db"

def init_db():
    conn = sqlite3.connect(DB)
    conn.execute("""CREATE TABLE IF NOT EXISTS state (
        key TEXT PRIMARY KEY, value TEXT)""")
    conn.execute("""CREATE TABLE IF NOT EXISTS trades (
        id INTEGER PRIMARY KEY, data TEXT)""")
    conn.commit()
    conn.close()

def save_state(bankroll, wins, losses, equity):
    conn = sqlite3.connect(DB)
    conn.execute("INSERT OR REPLACE INTO state VALUES ('bankroll', ?)", (str(bankroll),))
    conn.execute("INSERT OR REPLACE INTO state VALUES ('wins', ?)", (str(wins),))
    conn.execute("INSERT OR REPLACE INTO state VALUES ('losses', ?)", (str(losses),))
    conn.execute("INSERT OR REPLACE INTO state VALUES ('equity', ?)", (json.dumps(equity),))
    conn.commit()
    conn.close()

def load_state():
    conn = sqlite3.connect(DB)
    rows = {r[0]: r[1] for r in conn.execute("SELECT key, value FROM state")}
    trades = [json.loads(r[0]) for r in conn.execute("SELECT data FROM trades ORDER BY id")]
    conn.close()
    return (
        float(rows.get("bankroll", 1000.0)),
        int(rows.get("wins", 0)),
        int(rows.get("losses", 0)),
        json.loads(rows.get("equity", "[1000.0]")),
        trades,
    )

def save_trade(trade):
    conn = sqlite3.connect(DB)
    conn.execute("INSERT OR REPLACE INTO trades VALUES (?, ?)", (trade["id"], json.dumps(trade)))
    conn.commit()
    conn.close()

init_db()
from flask import Flask, render_template_string
from collections import deque

app = Flask(__name__)

state = {
    "bankroll": 1000.0,
    "trades": [],
    "pending": [],
    "log": deque(maxlen=15),
    "btc_candles": [],
    "btc_price": 0.0,
    "btc_change": 0.0,
    "momentum": 0.0,
    "rsi": 50.0,
    "signal": "HOLD",
    "equity": [1000.0],
    "wins": 0,
    "losses": 0,
}

def get_candles():
    """Try Binance first, fall back to Coinbase if blocked (e.g. on Render)."""
    # 1. Binance — best on local/EU networks
    try:
        r = requests.get("https://api.binance.com/api/v3/klines",
            params={"symbol":"BTCUSDT","interval":"1m","limit":120}, timeout=5)
        if r.status_code == 200:
            data = r.json()
            if isinstance(data, list) and len(data) >= 30:
                return data
    except Exception:
        pass
    # 2. Coinbase Exchange — works from cloud servers Binance blocks
    try:
        r = requests.get(
            "https://api.exchange.coinbase.com/products/BTC-USD/candles",
            params={"granularity": 60}, timeout=5,
            headers={"User-Agent": "btc-bot/1.0"},
        )
        if r.status_code == 200:
            raw = r.json()  # [[time, low, high, open, close, volume], ...] newest first
            out = []
            for row in reversed(raw):
                t, lo, hi, op, cl, vol = row
                out.append([int(t)*1000, op, hi, lo, cl, vol,
                            int(t)*1000+59999, 0, 0, 0, 0, 0])
            return out
    except Exception:
        pass
    return []

def calc_rsi(closes, p=7):
    if len(closes) < p+1: return 50.0
    d = np.diff(closes)
    g = np.mean(np.where(d>0,d,0)[-p:])
    l = np.mean(np.where(d<0,-d,0)[-p:])
    return 100 if l==0 else 100-(100/(1+g/l))

def calc_ema(prices, period):
    """Standard EMA; falls back to SMA when not enough data."""
    if len(prices) < period:
        return float(np.mean(prices))
    k = 2 / (period + 1)
    ema = float(np.mean(prices[:period]))
    for x in prices[period:]:
        ema = x * k + ema * (1 - k)
    return ema

def calc_atr(highs, lows, closes, p=14):
    """Average True Range over the last p bars."""
    if len(closes) < p + 1:
        return 0.0
    tr = []
    for i in range(1, len(closes)):
        h, l, pc = highs[i], lows[i], closes[i-1]
        tr.append(max(h - l, abs(h - pc), abs(l - pc)))
    return float(np.mean(tr[-p:]))

def trade_loop():
    bankroll, wins, losses, equity, saved_trades = load_state()
    state["bankroll"] = bankroll
    state["wins"] = wins
    state["losses"] = losses
    state["equity"] = equity
    state["trades"] = saved_trades
   

    last_loss_at = 0.0           # cooldown after a losing settle
    LOSS_COOLDOWN_SEC = 180      # 3 minutes (~3 cycles)

    while True:
        candles = get_candles()
        if not candles:
            time.sleep(10)
            continue

        valid  = [c for c in candles if len(c) > 4]
        closes = np.array([float(c[4]) for c in valid])
        highs  = np.array([float(c[2]) for c in valid])
        lows   = np.array([float(c[3]) for c in valid])
        if len(closes) < 30:
            time.sleep(10)
            continue

        price = closes[-1]
        mom   = (closes[-1] - closes[-8]) / closes[-8]
        r     = calc_rsi(closes)
        chg   = (closes[-1] - closes[-6]) / closes[-6] * 100
        ema9  = calc_ema(closes, 9)
        ema21 = calc_ema(closes, 21)
        atr   = calc_atr(highs, lows, closes, 14)

        # ATR percentile over last 60 bars (volatility regime)
        atr_window = []
        for i in range(max(15, len(closes) - 60), len(closes)):
            atr_window.append(calc_atr(highs[:i+1], lows[:i+1], closes[:i+1], 14))
        atr_pct = (
            float(sum(1 for x in atr_window if x <= atr) / len(atr_window))
            if atr_window else 0.5
        )

        state["btc_price"]   = round(price, 2)
        state["btc_change"]  = round(chg, 4)
        state["btc_candles"] = [round(float(c[4]),2) for c in candles[-30:]]
        state["momentum"]    = round(mom*100, 4)
        state["rsi"]         = round(r, 1)

        # ── Settle pending trades (5 minutes old) ────────────────────────
        now = time.time()
        for p in state["pending"][:]:
            if now - p["opened_at"] >= 300:
                went_up = price > p["btc_entry"]
                won = (p["side"] == "UP" and went_up) or \
                      (p["side"] == "DOWN" and not went_up)

                if won:
                    shares  = p["size"] / p["fill"]
                    payout  = shares * 1.0
                    bankroll += payout
                    pnl_val = round(payout - p["size"], 2)
                    pnl_str = f"+${pnl_val}"
                    state["wins"] += 1
                else:
                    pnl_val = -p["size"]
                    pnl_str = f"-${p['size']}"
                    state["losses"] += 1
                    last_loss_at = now

                for t in state["trades"]:
                    if t["id"] == p["id"]:
                        t["result"] = "WIN" if won else "LOSS"
                        t["pnl"]    = pnl_str
                        break

                state["log"].appendleft(
                    f"{time.strftime('%H:%M:%S')}  "
                    f"{'WIN' if won else 'LOSS'}  {pnl_str}  "
                    f"{p['side']}  BTC ${price:,.0f}"
                )
                state["pending"].remove(p)

        # ── Signal (momentum + tighter RSI + EMA-trend confirm) ──────────
        sig = "HOLD"
        skip_reason = None
        if mom > 0.0001 and r < 70 and ema9 > ema21:
            sig = "UP"
        elif mom < -0.0001 and r > 30 and ema9 < ema21:
            sig = "DOWN"

        # Volatility filter: skip chop (bottom 30%) and whipsaw (top 5%)
        if sig != "HOLD":
            if atr_pct < 0.30:
                skip_reason = f"low-vol (ATR pct {atr_pct:.2f})"
                sig = "HOLD"
            elif atr_pct > 0.95:
                skip_reason = f"whipsaw (ATR pct {atr_pct:.2f})"
                sig = "HOLD"

        # Cooldown after a recent loss
        if sig != "HOLD" and (now - last_loss_at) < LOSS_COOLDOWN_SEC:
            secs = int(LOSS_COOLDOWN_SEC - (now - last_loss_at))
            skip_reason = f"loss cooldown ({secs}s left)"
            sig = "HOLD"

        state["signal"] = sig
        if skip_reason:
            state["log"].appendleft(
                f"{time.strftime('%H:%M:%S')}  skip — {skip_reason}"
            )

        # ── Open trade (max 1 open at a time) ────────────────────────────
        if sig != "HOLD" and len(state["pending"]) == 0:
            kelly     = bankroll * min(abs(mom) * 5, 0.05)
            size      = round(max(min(kelly, 30.0), 5.0), 2)
            mkt_price = float(np.clip(0.5 + mom*3, 0.10, 0.90))
            fill      = round(mkt_price * 1.002, 4)
            trade_id  = int(now)
            settles   = time.strftime("%H:%M:%S", time.localtime(now+300))

            if size <= bankroll:
                bankroll -= size
                state["trades"].append({
                    "id":      trade_id,
                    "time":    time.strftime("%H:%M:%S"),
                    "side":    sig,
                    "size":    size,
                    "fill":    fill,
                    "btc":     f"${price:,.0f}",
                    "mom":     f"{mom*100:+.3f}%",
                    "rsi":     round(r, 1),
                    "settles": settles,
                    "result":  "open",
                    "pnl":     "—",
                })
                state["pending"].append({
                    "id":         trade_id,
                    "side":       sig,
                    "size":       size,
                    "fill":       fill,
                    "btc_entry":  price,
                    "opened_at":  now,
                })
                state["log"].appendleft(
                    f"{time.strftime('%H:%M:%S')}  OPEN {sig}  "
                    f"${size:.0f}  BTC ${price:,.0f}  settles {settles}"
                )
        elif sig != "HOLD" and len(state["pending"]) > 0:
            p = state["pending"][0]
            secs_left = max(0, int(300 - (now - p["opened_at"])))
            state["log"].appendleft(
                f"{time.strftime('%H:%M:%S')}  waiting...  "
                f"{secs_left//60}m {secs_left%60}s until settlement"
            )
        else:
            d = "up" if mom > 0 else "down"
            state["log"].appendleft(
                f"{time.strftime('%H:%M:%S')}  {d}  {mom*100:+.3f}%  RSI {r:.1f}"
            )

        state["bankroll"] = round(bankroll, 2)
        state["equity"].append(round(bankroll, 2))
        if len(state["equity"]) > 120:
            state["equity"].pop(0)

        save_state(bankroll, state["wins"], state["losses"], state["equity"])
        for t in state["trades"]:
            save_trade(t)

        time.sleep(60)



HTML = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta http-equiv="refresh" content="15">
<title>BTC Bot</title>
<link href="https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600&family=JetBrains+Mono:wght@400;500&display=swap" rel="stylesheet">
<style>
*{box-sizing:border-box;margin:0;padding:0}
body{font-family:'Inter',sans-serif;background:#0d0d0d;color:#e8e8e6;font-size:14px;padding:24px 28px;min-height:100vh}
header{display:flex;justify-content:space-between;align-items:center;margin-bottom:24px;padding-bottom:16px;border-bottom:1px solid #2a2a2a}
.logo{font-size:13px;font-weight:600}.logo span{color:#777;font-weight:400;margin-left:6px}
.btc{text-align:right}
.btc-price{font-family:'JetBrains Mono',monospace;font-size:22px;font-weight:500}
.btc-chg{font-family:'JetBrains Mono',monospace;font-size:12px;margin-top:2px}
.up{color:#4ade80}.down{color:#f87171}.neutral{color:#888}
.stats{display:grid;grid-template-columns:repeat(5,1fr);gap:10px;margin-bottom:18px}
.stat{background:#1a1a1a;border:1px solid #2a2a2a;border-radius:8px;padding:14px 16px}
.stat-label{font-size:11px;color:#888;text-transform:uppercase;letter-spacing:0.06em;margin-bottom:5px}
.stat-value{font-family:'JetBrains Mono',monospace;font-size:20px;font-weight:500}
.badge{display:inline-block;font-family:'JetBrains Mono',monospace;font-size:11px;font-weight:500;padding:2px 10px;border-radius:4px;margin-top:3px}
.badge.UP{background:#10331f;color:#4ade80;border:1px solid #1f6638}
.badge.DOWN{background:#3a1414;color:#f87171;border:1px solid #6b2222}
.badge.HOLD{background:#1f1f1f;color:#888;border:1px solid #2a2a2a}
.open-trade{background:#2a2010;border:1px solid #6b5a20;border-radius:8px;padding:12px 16px;margin-bottom:14px;font-size:13px;display:flex;align-items:center;gap:12px}
.open-label{font-size:11px;color:#d9a64a;text-transform:uppercase;letter-spacing:0.06em;margin-bottom:2px}
.grid{display:grid;grid-template-columns:1fr 280px;gap:14px}
.panel{background:#1a1a1a;border:1px solid #2a2a2a;border-radius:8px;padding:16px 18px;margin-bottom:14px}
.panel:last-child{margin-bottom:0}
.panel-title{font-size:11px;text-transform:uppercase;letter-spacing:0.06em;color:#888;margin-bottom:12px}
.ind{margin-bottom:14px}.ind:last-child{margin-bottom:0}
.ind-row{display:flex;justify-content:space-between;font-size:12px;color:#888;margin-bottom:4px}
.ind-row span:last-child{font-family:'JetBrains Mono',monospace;color:#e8e8e6;font-weight:500}
.track{height:3px;background:#2a2a2a;border-radius:2px}
.fill{height:100%;border-radius:2px}
.ind-sub{display:flex;justify-content:space-between;font-size:10px;color:#666;margin-top:3px}
table{width:100%;border-collapse:collapse;font-size:12px}
th{text-align:left;font-size:10px;text-transform:uppercase;letter-spacing:0.06em;color:#666;padding:0 6px 8px 6px;border-bottom:1px solid #2a2a2a;font-weight:400}
td{padding:7px 6px;border-bottom:1px solid #222;font-family:'JetBrains Mono',monospace;font-size:11px;color:#ccc}
tr:last-child td{border:none}
.empty{text-align:center;color:#666;padding:20px;font-size:12px}
.log-line{font-family:'JetBrains Mono',monospace;font-size:11px;color:#777;padding:4px 0;border-bottom:1px solid #222;white-space:nowrap;overflow:hidden;text-overflow:ellipsis}
.log-line:first-child{color:#e8e8e6}.log-line:last-child{border:none}
footer{font-size:11px;color:#666;margin-top:14px;display:flex;justify-content:space-between}
.warn{color:#d9a64a}
#cd{font-family:'JetBrains Mono',monospace;font-size:11px;color:#777;margin-top:3px}
.tv-frame{width:100%;height:380px;border:none;border-radius:6px;background:#0d0d0d}
</style>
</head>
<body>
<header>
  <div>
    <div class="logo">BTC 5m bot <span>paper trading</span></div>
    <div id="cd">next update in 15s</div>
  </div>
  <div class="btc">
    <div class="btc-price">${{"%.2f"|format(d.btc_price)}}</div>
    <div class="btc-chg {{d.chg_cls}}">{{"+" if d.btc_change>=0 else ""}}{{d.btc_change}}% (5m)</div>
  </div>
</header>

<div class="stats">
  <div class="stat"><div class="stat-label">Bankroll</div><div class="stat-value">${{"%.2f"|format(d.bankroll)}}</div></div>
  <div class="stat"><div class="stat-label">P&amp;L</div><div class="stat-value {{d.pnl_cls}}">{{"+" if d.pnl>=0 else ""}}${{d.pnl}}</div></div>
  <div class="stat"><div class="stat-label">Signal</div><span class="badge {{d.signal}}">{{d.signal}}</span></div>
  <div class="stat"><div class="stat-label">Trades</div><div class="stat-value">{{d.n_trades}}</div></div>
  <div class="stat"><div class="stat-label">Win / Loss</div><div class="stat-value"><span class="up">{{d.wins}}</span> / <span class="down">{{d.losses}}</span></div></div>
</div>

{% if d.pending %}
<div class="open-trade">
  <div>
    <div class="open-label">Open trade</div>
    <span class="badge {{d.pending[0].side}}">{{d.pending[0].side}}</span>
  </div>
  <div style="font-family:'JetBrains Mono',monospace;font-size:12px;color:#666">
    ${{d.pending[0].size}} &nbsp;·&nbsp; entry {{d.pending[0].btc_entry}} &nbsp;·&nbsp; settles at {{d.pending[0].settles}}
  </div>
</div>
{% endif %}

<div class="grid">
  <div>
    <div class="panel">
      <div class="panel-title">BTC live — TradingView</div>
      <iframe class="tv-frame"
        src="https://s.tradingview.com/widgetembed/?frameElementId=tv-btc&symbol=BINANCE%3ABTCUSDT&interval=1&hidesidetoolbar=1&hidetoptoolbar=0&saveimage=0&toolbarbg=0d0d0d&theme=dark&style=1&timezone=Etc%2FUTC&studies=%5B%5D&hideideas=1&showpopupbutton=0"
        scrolling="no" allowtransparency="true" frameborder="0"
        title="BTCUSDT live chart"></iframe>
    </div>
    <div class="panel">
      <div class="panel-title">Equity</div>
      <div style="position:relative;height:80px">
        <canvas id="eqChart" role="img" aria-label="Equity curve">Portfolio equity.</canvas>
      </div>
    </div>
    <div class="panel">
      <div class="panel-title">Trades</div>
      {% if d.trades %}
      <table>
        <tr><th>Time</th><th>Side</th><th>Size</th><th>BTC</th><th>Mom</th><th>RSI</th><th>Settles</th><th>Result</th><th>P&L</th></tr>
        {% for t in d.trades[-10:]|reverse %}
        <tr>
          <td>{{t.time}}</td>
          <td class="{{'up' if t.side=='UP' else 'down'}}" style="font-weight:500">{{t.side}}</td>
          <td>${{t.size}}</td>
          <td>{{t.btc}}</td>
          <td class="{{'up' if '+' in t.mom else 'down'}}">{{t.mom}}</td>
          <td>{{t.rsi}}</td>
          <td style="color:#a06000">{{t.settles}}</td>
          <td class="{{'up' if t.result=='WIN' else ('down' if t.result=='LOSS' else 'neutral')}}">{{t.result}}</td>
          <td class="{{'up' if '+' in t.pnl else ('down' if '-' in t.pnl else 'neutral')}}">{{t.pnl}}</td>
        </tr>
        {% endfor %}
      </table>
      {% else %}
      <div class="empty">Waiting for first signal...</div>
      {% endif %}
    </div>
  </div>
  <div>
    <div class="panel">
      <div class="panel-title">Indicators</div>
      <div class="ind">
        <div class="ind-row"><span>Momentum</span><span>{{"+" if d.momentum>=0 else ""}}{{d.momentum}}%</span></div>
        <div class="track"><div class="fill" style="width:{{d.mom_w}}%;background:{{'#1a7a4a' if d.momentum>=0 else '#b83232'}}"></div></div>
        <div class="ind-sub"><span>bear</span><span>neutral</span><span>bull</span></div>
      </div>
      <div class="ind">
        <div class="ind-row"><span>RSI</span><span>{{d.rsi}}</span></div>
        <div class="track"><div class="fill" style="width:{{d.rsi}}%;background:{{'#b83232' if d.rsi>70 else ('#1a7a4a' if d.rsi<30 else '#2563eb')}}"></div></div>
        <div class="ind-sub"><span>oversold</span><span>50</span><span>overbought</span></div>
      </div>
    </div>
    <div class="panel">
      <div class="panel-title">Log</div>
      {% for line in d.log %}
      <div class="log-line">{{line}}</div>
      {% else %}
      <div class="log-line">Starting...</div>
      {% endfor %}
    </div>
  </div>
</div>

<footer>
  <span>{{d.time}} &nbsp;·&nbsp; refresh 15s &nbsp;·&nbsp; {{d.n_trades}} trades</span>
  <span class="warn">VPN needed for live Polymarket</span>
</footer>

<script src="https://cdnjs.cloudflare.com/ajax/libs/Chart.js/4.4.1/chart.umd.js"></script>
<script>
let s=15;
setInterval(()=>{
  s--;
  if(s<=0)s=15;
  const pending = "{{d.pending[0].settles if d.pending else ''}}";
  if(pending) {
    document.getElementById('cd').textContent='trade settles at ' + pending + ' · refresh in ' + s + 's';
  } else {
    document.getElementById('cd').textContent='next update in ' + s + 's';
  }
},1000);
const eq={{d.equity|tojson}};
const eup=eq[eq.length-1]>=eq[0];
const ec=eup?'#4ade80':'#f87171';
const ef=eup?'rgba(74,222,128,0.10)':'rgba(248,113,113,0.10)';
const base={responsive:true,maintainAspectRatio:false,plugins:{legend:{display:false},tooltip:{callbacks:{label:c=>'$'+c.parsed.y.toLocaleString()}}}};
new Chart(document.getElementById('eqChart'),{type:'line',data:{labels:eq.map((_,i)=>i),datasets:[{data:eq,borderColor:ec,backgroundColor:ef,borderWidth:1.5,fill:true,tension:0.3,pointRadius:0}]},options:{...base,scales:{x:{display:false},y:{grid:{color:'#222'},border:{display:false},ticks:{font:{family:'JetBrains Mono',size:10},color:'#666',callback:v=>'$'+v.toFixed(0),maxTicksLimit:3}}}}});
</script>
</body>
</html>"""

@app.route("/")
def index():
    pnl = round(state["bankroll"] - 1000.0, 2)
    mom_w = min(max(round(state["momentum"] / 2 + 50), 2), 98)
    now = time.time()
    pending_display = [{
        "side":      p["side"],
        "size":      p["size"],
        "btc_entry": f"${p['btc_entry']:,.0f}",
        "settles":   time.strftime("%H:%M:%S", time.localtime(p["opened_at"]+300)),
    } for p in state["pending"]]

    d = type("D", (), {
        "bankroll":    state["bankroll"],
        "pnl":         pnl,
        "pnl_cls":     "up" if pnl >= 0 else "down",
        "n_trades":    len(state["trades"]),
        "trades":      state["trades"],
        "pending":     pending_display,
        "log":         list(state["log"]),
        "btc_price":   state["btc_price"],
        "btc_change":  state["btc_change"],
        "chg_cls":     "up" if state["btc_change"] >= 0 else "down",
        "btc_candles": state["btc_candles"],
        "equity":      state["equity"],
        "momentum":    state["momentum"],
        "rsi":         state["rsi"],
        "signal":      state["signal"],
        "mom_w":       mom_w,
        "wins":        state["wins"],
        "losses":      state["losses"],
        "time":        time.strftime("%H:%M:%S"),
    })()
    return render_template_string(HTML, d=d)

if __name__ == "__main__":
    threading.Thread(target=trade_loop, daemon=True).start()
    import os
    port = int(os.environ.get("PORT", 5000))
    print(f"\n  http://localhost:{port}\n")
    app.run(debug=False, host="0.0.0.0", port=port)
