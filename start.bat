@echo off
REM Starts the BTC bot and a Cloudflare tunnel so you can open it on your phone.
REM Two windows will appear. Keep BOTH open while you're using the dashboard.
REM The phone URL appears in the "tunnel" window after a few seconds.

cd /d "%~dp0"

start "btc-bot"     cmd /k "python bot.py"
start "btc-tunnel"  cmd /k "cloudflared.exe tunnel --url http://localhost:5000"

exit
