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
    try:
        r = requests.get("https://api.binance.com/api/v3/klines",
            params={"symbol":"BTCUSDT","interval":"1m","limit":40}, timeout=5)
        return r.json()
    except:
        return []

def calc_rsi(closes, p=7):
    if len(closes) < p+1: return 50.0
    d = np.diff(closes)
    g = np.mean(np.where(d>0,d,0)[-p:])
    l = np.mean(np.where(d<0,-d,0)[-p:])
    return 100 if l==0 else 100-(100/(1+g/l))

def trade_loop():
    bankroll, wins, losses, equity, saved_trades = load_state()
    state["bankroll"] = bankroll
    state["wins"] = wins
    state["losses"] = losses
    state["equity"] = equity
    state["trades"] = saved_trades
   

    while True:
        candles = get_candles()
        if not candles:
            time.sleep(10)
            continue

        closes  = np.array([float(c[4]) for c in candles])
        price   = closes[-1]
        mom     = (closes[-1] - closes[-8]) / closes[-8]
        r       = calc_rsi(closes)
        chg     = (closes[-1] - closes[-6]) / closes[-6] * 100

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

        # ── Signal ────────────────────────────────────────────────────────
        sig = "HOLD"
        if mom >  0.0001 and r < 85: sig = "UP"
        if mom < -0.0001 and r > 15: sig = "DOWN"
        state["signal"] = sig

        # ── Open trade (max 1 open at a time) ────────────────────────────
        if sig != "HOLD" and len(state["pending"]) == 0:
            kelly     = bankroll * min(abs(mom) * 5, 0.05)
            size      = round(max(min(kelly, 30.0), 1.0), 2)
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
body{font-family:'Inter',sans-serif;background:#f5f5f3;color:#1c1c1a;font-size:14px;padding:24px 28px;min-height:100vh}
header{display:flex;justify-content:space-between;align-items:center;margin-bottom:24px;padding-bottom:16px;border-bottom:1px solid #e0e0dc}
.logo{font-size:13px;font-weight:600}.logo span{color:#888;font-weight:400;margin-left:6px}
.btc{text-align:right}
.btc-price{font-family:'JetBrains Mono',monospace;font-size:22px;font-weight:500}
.btc-chg{font-family:'JetBrains Mono',monospace;font-size:12px;margin-top:2px}
.up{color:#1a7a4a}.down{color:#b83232}.neutral{color:#888}
.stats{display:grid;grid-template-columns:repeat(5,1fr);gap:10px;margin-bottom:18px}
.stat{background:#fff;border:1px solid #e0e0dc;border-radius:8px;padding:14px 16px}
.stat-label{font-size:11px;color:#888;text-transform:uppercase;letter-spacing:0.06em;margin-bottom:5px}
.stat-value{font-family:'JetBrains Mono',monospace;font-size:20px;font-weight:500}
.badge{display:inline-block;font-family:'JetBrains Mono',monospace;font-size:11px;font-weight:500;padding:2px 10px;border-radius:4px;margin-top:3px}
.badge.UP{background:#edf7f2;color:#1a7a4a;border:1px solid #b8dfc9}
.badge.DOWN{background:#fdf1f1;color:#b83232;border:1px solid #f0b8b8}
.badge.HOLD{background:#f5f5f3;color:#888;border:1px solid #e0e0dc}
.open-trade{background:#fff8ed;border:1px solid #f0d090;border-radius:8px;padding:12px 16px;margin-bottom:14px;font-size:13px;display:flex;align-items:center;gap:12px}
.open-label{font-size:11px;color:#a06000;text-transform:uppercase;letter-spacing:0.06em;margin-bottom:2px}
.grid{display:grid;grid-template-columns:1fr 280px;gap:14px}
.panel{background:#fff;border:1px solid #e0e0dc;border-radius:8px;padding:16px 18px;margin-bottom:14px}
.panel:last-child{margin-bottom:0}
.panel-title{font-size:11px;text-transform:uppercase;letter-spacing:0.06em;color:#888;margin-bottom:12px}
.ind{margin-bottom:14px}.ind:last-child{margin-bottom:0}
.ind-row{display:flex;justify-content:space-between;font-size:12px;color:#888;margin-bottom:4px}
.ind-row span:last-child{font-family:'JetBrains Mono',monospace;color:#1c1c1a;font-weight:500}
.track{height:3px;background:#f0f0ec;border-radius:2px}
.fill{height:100%;border-radius:2px}
.ind-sub{display:flex;justify-content:space-between;font-size:10px;color:#bbb;margin-top:3px}
table{width:100%;border-collapse:collapse;font-size:12px}
th{text-align:left;font-size:10px;text-transform:uppercase;letter-spacing:0.06em;color:#aaa;padding:0 6px 8px 6px;border-bottom:1px solid #f0f0ec;font-weight:400}
td{padding:7px 6px;border-bottom:1px solid #f8f8f6;font-family:'JetBrains Mono',monospace;font-size:11px}
tr:last-child td{border:none}
.empty{text-align:center;color:#aaa;padding:20px;font-size:12px}
.log-line{font-family:'JetBrains Mono',monospace;font-size:11px;color:#aaa;padding:4px 0;border-bottom:1px solid #f8f8f6;white-space:nowrap;overflow:hidden;text-overflow:ellipsis}
.log-line:first-child{color:#1c1c1a}.log-line:last-child{border:none}
footer{font-size:11px;color:#aaa;margin-top:14px;display:flex;justify-content:space-between}
.warn{color:#a06000}
#cd{font-family:'JetBrains Mono',monospace;font-size:11px;color:#aaa;margin-top:3px}
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
      <div class="panel-title">BTC price — last 30 min</div>
      <div style="position:relative;height:150px">
        <canvas id="btcChart" role="img" aria-label="BTC price chart">BTC price over 30 minutes.</canvas>
      </div>
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
const up={{d.btc_change}}>=0;
const gc=up?'#1a7a4a':'#b83232';
const gf=up?'rgba(26,122,74,0.07)':'rgba(184,50,50,0.07)';
const eup=eq[eq.length-1]>=eq[0];
const ec=eup?'#1a7a4a':'#b83232';
const ef=eup?'rgba(26,122,74,0.07)':'rgba(184,50,50,0.07)';
const base={responsive:true,maintainAspectRatio:false,plugins:{legend:{display:false},tooltip:{callbacks:{label:c=>'$'+c.parsed.y.toLocaleString()}}}};
new Chart(document.getElementById('btcChart'),{type:'line',data:{labels:btc.map((_,i)=>i),datasets:[{data:btc,borderColor:gc,backgroundColor:gf,borderWidth:1.5,fill:true,tension:0.3,pointRadius:0}]},options:{...base,scales:{x:{display:false},y:{grid:{color:'#f5f5f3'},border:{display:false},ticks:{font:{family:'JetBrains Mono',size:10},color:'#aaa',callback:v=>'$'+v.toLocaleString(),maxTicksLimit:4}}}}});
new Chart(document.getElementById('eqChart'),{type:'line',data:{labels:eq.map((_,i)=>i),datasets:[{data:eq,borderColor:ec,backgroundColor:ef,borderWidth:1.5,fill:true,tension:0.3,pointRadius:0}]},options:{...base,scales:{x:{display:false},y:{grid:{color:'#f5f5f3'},border:{display:false},ticks:{font:{family:'JetBrains Mono',size:10},color:'#aaa',callback:v=>'$'+v.toFixed(0),maxTicksLimit:3}}}}});
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
    print("\n  http://localhost:5000\n")
import os
port = int(os.environ.get("PORT", 5000))
app.run(debug=False, host="0.0.0.0", port=port)