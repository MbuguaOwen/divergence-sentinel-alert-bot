from __future__ import annotations

import asyncio
import logging
from typing import Dict, Tuple, Set

from .config import Config
from .models import Signal
from .timefilter import TradingHours
from .strategy import StrategyEngine
from .formatters import format_signal
from .notifier.telegram import TelegramNotifier
from .providers.binance import BinanceProvider

log = logging.getLogger("runner")


class AlertRunner:
    def __init__(self, cfg: Config):
        self.cfg = cfg
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

        self.engines: Dict[Tuple[str, str], StrategyEngine] = {}
        self._dedupe: Set[Tuple[str, str, str, int]] = set()  # (sym, tf, side, confirm_time)

    async def warmup(self) -> None:
        symbols = [s.upper() for s in (self.cfg.provider.symbols or [])]
        tfs = list(self.cfg.provider.timeframes or [])
        n = int(self.cfg.provider.warmup_candles)

        log.info("warmup_start symbols=%d timeframes=%s candles=%d", len(symbols), tfs, n)

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
                    ks_short_enabled=self.cfg.strategy.ks_short_enabled,
                    ks_don_width_atr_max=self.cfg.strategy.ks_don_width_atr_max,
                    ks_long_enabled=self.cfg.strategy.ks_long_enabled,
                    ks_liq_percentile_min=self.cfg.strategy.ks_liq_percentile_min,
                    ks_liq_lookback=self.cfg.strategy.ks_liq_lookback,
                    min_score_to_trade=self.cfg.strategy.min_score_to_trade,
                )

        # fetch klines concurrently but limited (avoid bursts / timeouts)
        sem = asyncio.Semaphore(max(1, int(getattr(self.cfg.provider, "warmup_concurrency", 5))))

        async def _one(sym: str, tf: str):
            try:
                async with sem:
                    candles = await self.provider.fetch_klines(sym, tf, n)
                eng = self.engines[(sym, tf)]
                for c in candles:
                    # warmup: do not alert
                    in_hours = self.th.within(c.close_time_ms)
                    eng.on_candle(c, allow_signals=False, in_trading_hours=in_hours)
                return None
            except Exception as e:
                return (sym, tf, repr(e))

        results = await asyncio.gather(*[_one(sym, tf) for sym in symbols for tf in tfs], return_exceptions=False)
        failures = [r for r in results if r is not None]
        if failures:
            # Do not crash the whole bot because one REST call timed out.
            for sym, tf, err in failures[:10]:
                log.warning("warmup_failed symbol=%s tf=%s err=%s", sym, tf, err)
            if len(failures) > 10:
                log.warning("warmup_failed_more count=%d", len(failures))
        log.info("warmup_done")

        if self.tg.enabled():
            await self.tg.send(f"✅ {self.cfg.app.name}: warmup complete. Monitoring {len(symbols)} symbols × {len(tfs)} TFs.")

    async def run_forever(self) -> None:
        symbols = [s.upper() for s in (self.cfg.provider.symbols or [])]
        tfs = list(self.cfg.provider.timeframes or [])
        if not symbols or not tfs:
            raise ValueError("No symbols/timeframes configured.")

        await self.warmup()

        async for evt in self.provider.stream_klines(symbols, tfs):
            eng = self.engines.get((evt.symbol, evt.timeframe))
            if eng is None:
                continue

            in_hours = self.th.within(evt.candle.close_time_ms)
            signals = eng.on_candle(evt.candle, allow_signals=True, in_trading_hours=in_hours)
            for sig in signals:
                await self._handle_signal(sig)

    async def _handle_signal(self, sig: Signal) -> None:
        key = (sig.symbol, sig.timeframe, sig.side, sig.confirm_time_ms)
        if self.cfg.alerts.dedupe and key in self._dedupe:
            return
        self._dedupe.add(key)

        log.info("signal %s %s %s confirm=%s", sig.symbol, sig.timeframe, sig.side, sig.confirm_time_ms)

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
