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

### TradingView parity mode (tv_parity)
- A new `strategy.mode: tv_parity` mirrors a simple TradingView entry engine (long-only by default) with optional confirm/RAW modes and an optional CVD proxy gate sampled from 1m signed volume.
- Mismatching filters are disabled in the provided example `configs/tv_parity.yaml` (trading_hours off, kill switches off, score=0). The example is set up for BTCUSDT on 15m and 1h; enable Telegram and tweak the CVD gate (fixed or percentile) before running.
- When `use_cvd_gate` is true, the runner auto-subscribes to 1m candles and aligns the CVD sample at or before each higher-timeframe close (optionally `close_time_ms - 1` for Pine parity).
- Signals formatted with `extra.mode == tv_parity` include entry mode (RAW/CONFIRM) plus CVD now/threshold for debugging parity.
- Webhook JSON is Pine-like: numeric `entry_price`, `confirm_time_ms`, and optional `tf` string. Configure in `webhook` section.

### Parity requirements
- Use the same market feed as TradingView (e.g., Binance futures symbols on TV should be the perpetual feed; mixing spot/futures will break parity).
- Ensure enough history for percentile CVD: the warmup planner pulls deep 1m history so thresholds are non-NA before trading.
- Trading hours, kill switches, and scoring are disabled in tv_parity configs; turn them on only if you accept divergence from Pine.

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

## Development
- Optional tests use `pytest` (see `requirements-dev.txt`): `pip install -r requirements-dev.txt && pytest -q`.

---

## License
MIT
