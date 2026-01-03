# Divergence Sentinel (Multi‑TF Alerts)

**Purpose:** Run on a VM and send **Telegram alerts** when the Pivot Divergence @ Donchian Extremes signal appears on **any of 15m / 30m / 1h** across **a basket of symbols**.

This is an **alerts-only** bot (no trading). It mirrors the Pine logic:
- Donchian range location filter (extreme band)
- Confirmed pivots (pivotLen left/right) => **no repaint**
- Divergence vs the previous pivot of the same type using the oscillator:
  `osc = EMA( (close - open) * volume, oscLen )`

## What you get
- ✅ Monitors many symbols (default 20) and many timeframes (default 15m, 30m, 1h) in parallel
- ✅ Config-driven trading hours (Nairobi default: 08:00 → 01:00 next day, UTC+3)
- ✅ Telegram alerts with rich context (pivot price, entry close, slip(bps), Donchian %, divergence strength, vol/rsi/atr/trend)
- ✅ Reconnect + resubscribe robustness
- ✅ Warmup via REST klines on startup (so pivots/EMA are seeded properly)
- ✅ Extensible provider architecture (Binance now, Forex later)

---

## Quick start (Linux / VM)

```bash
# 1) Put the folder on the VM
unzip divergence-sentinel-alert-bot.zip -d /opt
cd /opt/divergence-sentinel-alert-bot

# 2) Create venv
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

# 3) Edit config
cp configs/default.yaml configs/my.yaml
nano configs/my.yaml   # set telegram.token + telegram.chat_ids

# 4) Run
python -m divergence_alert_bot.main --config configs/my.yaml
```

### Systemd service (recommended)
```bash
sudo cp systemd/divergence-sentinel.service /etc/systemd/system/
sudo systemctl daemon-reload
sudo systemctl enable divergence-sentinel
sudo systemctl start divergence-sentinel
sudo systemctl status divergence-sentinel -n 50
```

---

## Notes on “simultaneous”
Signals are evaluated **independently** per (symbol, timeframe) on each **closed candle** for that timeframe.
If BTCUSDT triggers on 15m and 1h around the same time, you will receive **two alerts**.

---

## Extending to Forex later
The strategy engine is provider-agnostic. To add Forex, implement another provider in:
`divergence_alert_bot/providers/` that yields closed candles in the same format (symbol, timeframe, OHLCV).

---

## Safety / Rate limits
- Startup warmup does ~ (symbols × timeframes) REST calls (default 60). This is within typical Binance limits.
- WebSocket uses a single connection and subscribes to all streams.

---

## License
MIT
