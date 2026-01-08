from __future__ import annotations

import asyncio
import hashlib
import json
import logging
import time
from typing import Dict, Tuple, Set, List, Optional

from .config import Config
from .models import Signal
from .timefilter import TradingHours
from .strategy import StrategyEngine
from .formatters import format_signal
from .notifier.telegram import TelegramNotifier
from .notifier.webhook import WebhookNotifier
from .time_parity import tv_confirm_ts
from .parity import assert_tv_parity_inputs, log_parity_signature
from .providers.binance import BinanceProvider, KlineEvent
from .cvd_store import CvdProxyStore

log = logging.getLogger("runner")


def _stable_strategy_signature(cfg: Config) -> str:
    sig = cfg.strategy.parity_signature()
    payload = json.dumps(sig, sort_keys=True, separators=(",", ":"), ensure_ascii=True)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


class AlertRunner:
    def __init__(self, cfg: Config):
        self.cfg = cfg
        self._validate_parity_strict()
        self.provider = BinanceProvider(
            market=cfg.provider.market,
            rest_timeout_s=cfg.provider.rest_timeout_s,
            ws_heartbeat_s=cfg.provider.ws_heartbeat_s,
        )
        self.th = TradingHours(
            enabled=cfg.trading_hours.enabled,
            timezone=cfg.trading_hours.timezone,
            start_hour=cfg.trading_hours.start_hour,
            end_hour=cfg.trading_hours.end_hour,
            days=cfg.trading_hours.days,
        )

        all_chat_ids = []
        for lst in (cfg.telegram.chat_ids, cfg.telegram.public_chat_ids, cfg.telegram.private_chat_ids):
            for cid in lst or []:
                scid = str(cid).strip()
                if scid and scid not in all_chat_ids:
                    all_chat_ids.append(scid)
        self.tg = TelegramNotifier(
            token=cfg.telegram.token,
            chat_ids=all_chat_ids,
            disable_web_page_preview=cfg.telegram.disable_web_page_preview,
        )
        self.webhook = WebhookNotifier(
            enabled=cfg.webhook.enabled,
            url=cfg.webhook.url,
            secret=cfg.webhook.secret,
            include_tf=cfg.webhook.include_tf,
            timeout_s=cfg.webhook.timeout_s,
            headers=cfg.webhook.headers or {},
        )

        self.engines: Dict[Tuple[str, str], StrategyEngine] = {}
        self._strategy_sig = _stable_strategy_signature(cfg)
        self._dedupe: Set[str] = set()
        self._metrics = {
            "bos_same_bar_entries_total": 0,
            "bos_same_bar_reject_total": 0,
        }
        self._cvd_store: Dict[str, CvdProxyStore] = {}
        self._pending: Dict[Tuple[str, str], List[Tuple[int, object]]] = {}
        self._need_cvd = (cfg.strategy.mode == "tv_parity" and cfg.strategy.use_cvd_gate)
        self._last_close: Dict[Tuple[str, str], int] = {}

    async def warmup(self) -> None:
        symbols = [s.upper() for s in (self.cfg.provider.symbols or [])]
        trade_tfs = list(self.cfg.provider.timeframes or [])
        if self.cfg.strategy.mode == "tv_parity":
            log_parity_signature(self.cfg.strategy)
            if getattr(self.cfg.strategy, "parity_strict", False):
                assert_tv_parity_inputs(self.cfg.strategy)
        if self.cfg.strategy.mode == "tv_parity":
            await self._warmup_tv_parity(symbols, trade_tfs)
        else:
            await self._warmup_sniper(symbols, trade_tfs)

    async def _warmup_sniper(self, symbols: List[str], tfs: List[str]) -> None:
        n = int(self.cfg.provider.warmup_candles)
        log.info("warmup_start(sniper) symbols=%d timeframes=%s candles=%d", len(symbols), tfs, n)

        # create engines
        for sym in symbols:
            for tf in tfs:
                self.engines[(sym, tf)] = StrategyEngine(
                    symbol=sym,
                    timeframe=tf,
                    don_len=self.cfg.strategy.don_len,
                    pivot_len=self.cfg.strategy.pivot_len,
                    osc_len=self.cfg.strategy.osc_len,
                    ext_band_pct=self.cfg.strategy.ext_band_pct,
                    mode=self.cfg.strategy.mode,
                    long_only=self.cfg.strategy.long_only,
                    trade_enabled=self.cfg.strategy.trade_enabled,
                    entry_wait_confirm=self.cfg.strategy.entry_wait_confirm,
                    use_bos_confirm=self.cfg.strategy.use_bos_confirm,
                    bos_atr_buffer=self.cfg.strategy.bos_atr_buffer,
                    max_wait_bars=self.cfg.strategy.max_wait_bars,
                    min_div_strength=self.cfg.strategy.min_div_strength,
                    cooldown_bars=self.cfg.strategy.cooldown_bars,
                    use_cvd_gate=self.cfg.strategy.use_cvd_gate,
                    use_dynamic_cvd_pct=self.cfg.strategy.use_dynamic_cvd_pct,
                    cvd_lookback_bars=self.cfg.strategy.cvd_lookback_bars,
                    cvd_pct=self.cfg.strategy.cvd_pct,
                    cvd_threshold=self.cfg.strategy.cvd_threshold,
                    use_close_minus_1ms=self.cfg.strategy.use_close_minus_1ms,
                    ks_short_enabled=self.cfg.strategy.ks_short_enabled,
                    ks_don_width_atr_max=self.cfg.strategy.ks_don_width_atr_max,
                    ks_long_enabled=self.cfg.strategy.ks_long_enabled,
                    ks_liq_percentile_min=self.cfg.strategy.ks_liq_percentile_min,
                    ks_liq_lookback=self.cfg.strategy.ks_liq_lookback,
                    min_score_to_trade=self.cfg.strategy.min_score_to_trade,
                    strategy_sig=self._strategy_sig,
                )

        sem = asyncio.Semaphore(max(1, int(getattr(self.cfg.provider, "warmup_concurrency", 5))))

        async def _one(sym: str, tf: str):
            try:
                async with sem:
                    candles = await self.provider.fetch_klines(sym, tf, n)
                eng = self.engines[(sym, tf)]
                for c in candles:
                    in_hours = self.th.within(c.close_time_ms)
                    eng.on_candle(c, allow_signals=False, in_trading_hours=in_hours)
                if candles:
                    self._last_close[(sym, tf)] = candles[-1].close_time_ms
                return None
            except Exception as e:
                return (sym, tf, repr(e))

        results = await asyncio.gather(*[_one(sym, tf) for sym in symbols for tf in tfs], return_exceptions=False)
        failures = [r for r in results if r is not None]
        if failures:
            for sym, tf, err in failures[:10]:
                log.warning("warmup_failed symbol=%s tf=%s err=%s", sym, tf, err)
            if len(failures) > 10:
                log.warning("warmup_failed_more count=%d", len(failures))
        log.info("warmup_done(sniper)")

        if self.tg.enabled():
            await self.tg.send(f"✅ {self.cfg.app.name}: warmup complete. Monitoring {len(symbols)} symbols × {len(tfs)} TFs.")

    async def _warmup_tv_parity(self, symbols: List[str], tfs: List[str]) -> None:
        log.info("warmup_start(tv_parity) symbols=%d timeframes=%s", len(symbols), tfs)

        for sym in symbols:
            if self.cfg.strategy.use_cvd_gate:
                self._cvd_store.setdefault(sym, CvdProxyStore(self.cfg.strategy.cvd_len_min))

        # create engines for trade timeframes only
        for sym in symbols:
            for tf in tfs:
                self.engines[(sym, tf)] = StrategyEngine(
                    symbol=sym,
                    timeframe=tf,
                    don_len=self.cfg.strategy.don_len,
                    pivot_len=self.cfg.strategy.pivot_len,
                    osc_len=self.cfg.strategy.osc_len,
                    ext_band_pct=self.cfg.strategy.ext_band_pct,
                    mode="tv_parity",
                    long_only=self.cfg.strategy.long_only,
                    trade_enabled=self.cfg.strategy.trade_enabled,
                    entry_wait_confirm=self.cfg.strategy.entry_wait_confirm,
                    use_bos_confirm=self.cfg.strategy.use_bos_confirm,
                    bos_atr_buffer=self.cfg.strategy.bos_atr_buffer,
                    max_wait_bars=self.cfg.strategy.max_wait_bars,
                    min_div_strength=self.cfg.strategy.min_div_strength,
                    cooldown_bars=self.cfg.strategy.cooldown_bars,
                    use_cvd_gate=self.cfg.strategy.use_cvd_gate,
                    use_dynamic_cvd_pct=self.cfg.strategy.use_dynamic_cvd_pct,
                    cvd_lookback_bars=self.cfg.strategy.cvd_lookback_bars,
                    cvd_pct=self.cfg.strategy.cvd_pct,
                    cvd_threshold=self.cfg.strategy.cvd_threshold,
                    use_close_minus_1ms=self.cfg.strategy.use_close_minus_1ms,
                    ks_short_enabled=False,
                    ks_don_width_atr_max=self.cfg.strategy.ks_don_width_atr_max,
                    ks_long_enabled=False,
                    ks_liq_percentile_min=self.cfg.strategy.ks_liq_percentile_min,
                    ks_liq_lookback=self.cfg.strategy.ks_liq_lookback,
                    min_score_to_trade=self.cfg.strategy.min_score_to_trade,
                    strategy_sig=self._strategy_sig,
                )

        # Warm up 1m CVD first
        if self.cfg.strategy.use_cvd_gate:
            tf_minutes = [self._tf_minutes(tf) for tf in tfs if tf]
            max_tf_min = max(tf_minutes) if tf_minutes else 0
            need_1m = max_tf_min * int(self.cfg.strategy.cvd_lookback_bars) + int(self.cfg.strategy.cvd_len_min) + 500

            for sym in symbols:
                try:
                    candles_1m = await self.provider.fetch_klines_history(sym, "1m", need_1m)
                    store = self._cvd_store[sym]
                    for c in candles_1m:
                        store.on_1m_candle(c, use_close_minus_1ms=self.cfg.strategy.use_close_minus_1ms)
                    if candles_1m:
                        self._last_close[(sym, "1m")] = candles_1m[-1].close_time_ms
                    log.info("warmup_cvd symbol=%s loaded_1m=%d need=%d", sym, len(candles_1m), need_1m)
                except Exception as e:
                    log.warning("warmup_cvd_failed symbol=%s err=%s", sym, e)

        # Warm up trade timeframes with aligned CVD samples
        safety = 10
        need_tf_bars = (
            int(self.cfg.strategy.cvd_lookback_bars)
            + int(self.cfg.strategy.don_len)
            + int(self.cfg.strategy.osc_len)
            + 2 * int(self.cfg.strategy.pivot_len)
            + safety
        )

        for sym in symbols:
            store = self._cvd_store.get(sym)
            for tf in tfs:
                try:
                    candles = await self.provider.fetch_klines_history(sym, tf, need_tf_bars)
                except Exception as e:
                    log.warning("warmup_tf_failed symbol=%s tf=%s err=%s", sym, tf, e)
                    continue
                eng = self.engines[(sym, tf)]
                for c in candles:
                    ts_adj = tv_confirm_ts(c.close_time_ms, self.cfg.strategy.use_close_minus_1ms)
                    cvd_now = store.value_at_or_before(ts_adj) if store else None
                    eng.on_candle(c, allow_signals=False, in_trading_hours=True, cvd_now=cvd_now)
                if candles:
                    self._last_close[(sym, tf)] = candles[-1].close_time_ms
                log.info("warmup_tf_done symbol=%s tf=%s bars=%d need=%d", sym, tf, len(candles), need_tf_bars)

        if self.tg.enabled():
            await self.tg.send(f"✅ {self.cfg.app.name}: warmup complete (tv_parity). Monitoring {len(symbols)} symbols × {len(tfs)} TFs.")

    def _tf_minutes(self, tf: str) -> int:
        tf = (tf or "").strip().lower()
        if tf.endswith("m"):
            return int(tf[:-1])
        if tf.endswith("h"):
            return int(tf[:-1]) * 60
        if tf.endswith("d"):
            return int(tf[:-1]) * 1440
        raise ValueError(f"Unsupported timeframe: {tf}")

    async def run_forever(self) -> None:
        symbols = [s.upper() for s in (self.cfg.provider.symbols or [])]
        trade_tfs = list(self.cfg.provider.timeframes or [])
        if not symbols or not trade_tfs:
            raise ValueError("No symbols/timeframes configured.")

        sub_tfs = list(trade_tfs)
        if self._need_cvd and "1m" not in sub_tfs:
            sub_tfs = ["1m"] + sub_tfs

        await self.warmup()

        async for evt in self.provider.stream_klines(symbols, sub_tfs):
            if self._is_duplicate(evt):
                continue
            if self._need_cvd and evt.timeframe == "1m":
                store = self._cvd_store.setdefault(evt.symbol, CvdProxyStore(self.cfg.strategy.cvd_len_min))
                store.on_1m_candle(evt.candle, use_close_minus_1ms=self.cfg.strategy.use_close_minus_1ms)
                await self._flush_pending(evt.symbol, store.latest_ts)
                continue

            if evt.timeframe not in trade_tfs:
                continue

            await self._process_trade_event(evt)

    def _add_pending(self, evt: KlineEvent, ts_adj: int) -> None:
        key = (evt.symbol, evt.timeframe)
        lst = self._pending.setdefault(key, [])
        lst.append((ts_adj, evt))
        lst.sort(key=lambda x: x[0])
        # cap to avoid unbounded growth
        if len(lst) > 5000:
            del lst[: len(lst) - 5000]

    async def _flush_pending(self, symbol: str, latest_ts: Optional[int]) -> None:
        if latest_ts is None:
            return
        for key, items in list(self._pending.items()):
            if key[0] != symbol:
                continue
            ready = [(ts, evt) for ts, evt in items if ts <= latest_ts]
            remaining = [(ts, evt) for ts, evt in items if ts > latest_ts]
            if ready:
                self._pending[key] = remaining
                for ts_adj, evt in ready:
                    store = self._cvd_store.get(symbol)
                    cvd_now = store.value_at_or_before(ts_adj) if store else None
                    await self._process_trade_event(evt, cvd_now=cvd_now, ts_adj=ts_adj)

    async def _process_trade_event(self, evt: KlineEvent, *, cvd_now: Optional[float] = None, ts_adj: Optional[int] = None) -> None:
        eng = self.engines.get((evt.symbol, evt.timeframe))
        if eng is None:
            return

        in_hours = True if self.cfg.strategy.mode == "tv_parity" else self.th.within(evt.candle.close_time_ms)
        if ts_adj is None:
            ts_adj = tv_confirm_ts(evt.candle.close_time_ms, self.cfg.strategy.use_close_minus_1ms)

        if self._need_cvd and cvd_now is None:
            store = self._cvd_store.setdefault(evt.symbol, CvdProxyStore(self.cfg.strategy.cvd_len_min))
            cvd_now = store.value_at_or_before(ts_adj)
            if cvd_now is None:
                self._add_pending(evt, ts_adj)
                return

        signals = eng.on_candle(evt.candle, allow_signals=True, in_trading_hours=in_hours, cvd_now=cvd_now)
        for sig in signals:
            await self._handle_signal(sig)

    def _is_duplicate(self, evt: KlineEvent) -> bool:
        key = (evt.symbol, evt.timeframe)
        last = self._last_close.get(key)
        ct = evt.candle.close_time_ms
        if last is not None and ct <= last:
            return True
        self._last_close[key] = ct
        return False

    def _validate_parity_strict(self) -> None:
        if self.cfg.strategy.mode != "tv_parity" or not getattr(self.cfg.strategy, "parity_strict", False):
            return
        errs = []
        if self.cfg.trading_hours.enabled:
            errs.append("trading_hours.enabled must be false in parity_strict")
        if self.cfg.strategy.ks_long_enabled or self.cfg.strategy.ks_short_enabled:
            errs.append("kill switches must be disabled in parity_strict")
        if int(self.cfg.strategy.min_score_to_trade) > 0:
            errs.append("min_score_to_trade must be 0 in parity_strict")
        if errs:
            raise ValueError("Parity strict config violation: " + "; ".join(errs))

    def _dedupe_key(self, sig: Signal) -> str:
        if sig.signal_id:
            return sig.signal_id
        return f"{sig.symbol}:{sig.timeframe}:{sig.side}:{sig.confirm_time_ms}"

    async def _handle_signal(self, sig: Signal) -> None:
        dedupe_key = self._dedupe_key(sig)
        if self.cfg.alerts.dedupe and dedupe_key in self._dedupe:
            self._metrics["bos_same_bar_reject_total"] += 1
            now_ms = int(time.time() * 1000)
            latency_ms = now_ms - int(sig.confirm_time_ms)
            log.info(
                "signal %s %s %s confirm_ts=%s confirm_close_ms=%s now_ms=%s latency_ms=%s entry_mode=BOS_CLOSE_SAME_BAR order_result=rejected signal_id=%s entries_total=%d reject_total=%d",
                sig.symbol,
                sig.timeframe,
                sig.side,
                sig.confirm_time_ms,
                sig.confirm_bar_close_ms,
                now_ms,
                latency_ms,
                sig.signal_id,
                self._metrics["bos_same_bar_entries_total"],
                self._metrics["bos_same_bar_reject_total"],
            )
            return
        self._dedupe.add(dedupe_key)
        self._metrics["bos_same_bar_entries_total"] += 1

        now_ms = int(time.time() * 1000)
        latency_ms = now_ms - int(sig.confirm_time_ms)
        log.info(
            "signal %s %s %s confirm_ts=%s confirm_close_ms=%s now_ms=%s latency_ms=%s entry_mode=BOS_CLOSE_SAME_BAR order_result=submitted signal_id=%s entries_total=%d reject_total=%d",
            sig.symbol,
            sig.timeframe,
            sig.side,
            sig.confirm_time_ms,
            sig.confirm_bar_close_ms,
            now_ms,
            latency_ms,
            sig.signal_id,
            self._metrics["bos_same_bar_entries_total"],
            self._metrics["bos_same_bar_reject_total"],
        )

        if self.webhook.enabled:
            try:
                await self.webhook.send_signal(sig)
            except Exception as e:
                log.warning("webhook_send_failed symbol=%s tf=%s err=%s", sig.symbol, sig.timeframe, e)

        if not self.tg.enabled():
            return

        alerts_cfg = self.cfg.alerts
        parse_mode = getattr(alerts_cfg, "parse_mode", "HTML") or "HTML"

        public_ids = self.cfg.telegram.public_chat_ids or []
        private_ids = self.cfg.telegram.private_chat_ids or []
        has_split = bool(public_ids or private_ids)

        if has_split:
            if public_ids:
                msg_public = format_signal(sig, alerts_cfg, detail_level="public")
                await self.tg.send(msg_public, chat_ids=public_ids, parse_mode=parse_mode)
            if private_ids:
                msg_internal = format_signal(sig, alerts_cfg, detail_level="internal")
                await self.tg.send(msg_internal, chat_ids=private_ids, parse_mode=parse_mode)
        else:
            msg = format_signal(sig, alerts_cfg, detail_level=alerts_cfg.detail_level)
            await self.tg.send(msg, parse_mode=parse_mode)
