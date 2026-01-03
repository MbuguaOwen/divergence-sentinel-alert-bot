from __future__ import annotations

import asyncio
import json
import logging
from dataclasses import dataclass
from typing import AsyncIterator, Dict, List, Optional, Tuple

import aiohttp
import websockets

from ..models import Candle

log = logging.getLogger("binance")


def _rest_base(market: str) -> str:
    return "https://fapi.binance.com" if market == "futures" else "https://api.binance.com"


def _klines_path(market: str) -> str:
    return "/fapi/v1/klines" if market == "futures" else "/api/v3/klines"


def _ws_url(market: str) -> str:
    return "wss://fstream.binance.com/ws" if market == "futures" else "wss://stream.binance.com:9443/ws"


def _stream_name(symbol: str, tf: str) -> str:
    return f"{symbol.lower()}@kline_{tf}"


@dataclass(frozen=True)
class KlineEvent:
    symbol: str
    timeframe: str
    candle: Candle


class BinanceProvider:
    def __init__(
        self,
        market: str = "futures",
        *,
        rest_timeout_s: int = 20,
        ws_heartbeat_s: int = 20,
        rest_max_retries: int = 4,
        rest_backoff_s: float = 0.8,
        rest_conn_limit: int = 40,
        rest_conn_limit_per_host: int = 10,
    ):
        self.market = market
        self.rest_timeout_s = rest_timeout_s
        self.ws_heartbeat_s = ws_heartbeat_s

        # REST robustness
        self.rest_max_retries = rest_max_retries
        self.rest_backoff_s = rest_backoff_s
        self.rest_conn_limit = rest_conn_limit
        self.rest_conn_limit_per_host = rest_conn_limit_per_host

        self._session: Optional[aiohttp.ClientSession] = None

    async def close(self) -> None:
        """Close the shared aiohttp session (best-effort)."""
        if self._session is not None and not self._session.closed:
            await self._session.close()

    def _timeout(self) -> aiohttp.ClientTimeout:
        # Slightly more granular timeouts than total-only.
        return aiohttp.ClientTimeout(
            total=self.rest_timeout_s,
            connect=min(10, self.rest_timeout_s),
            sock_connect=min(10, self.rest_timeout_s),
            sock_read=max(10, int(self.rest_timeout_s * 0.75)),
        )

    def _connector(self) -> aiohttp.TCPConnector:
        return aiohttp.TCPConnector(
            limit=self.rest_conn_limit,
            limit_per_host=self.rest_conn_limit_per_host,
            ttl_dns_cache=300,
            enable_cleanup_closed=True,
        )

    async def _get_session(self) -> aiohttp.ClientSession:
        if self._session is None or self._session.closed:
            self._session = aiohttp.ClientSession(timeout=self._timeout(), connector=self._connector())
        return self._session

    async def fetch_klines(self, symbol: str, timeframe: str, limit: int) -> List[Candle]:
        base = _rest_base(self.market)
        path = _klines_path(self.market)
        url = base + path
        params = {"symbol": symbol.upper(), "interval": timeframe, "limit": int(limit)}

        sess = await self._get_session()

        backoff = float(self.rest_backoff_s)
        last_err: Optional[BaseException] = None
        for attempt in range(1, int(self.rest_max_retries) + 1):
            try:
                async with sess.get(url, params=params) as resp:
                    # Rate-limit / ban signals
                    if resp.status in (418, 429):
                        txt = await resp.text()
                        retry_after = resp.headers.get("Retry-After")
                        sleep_s = float(retry_after) if (retry_after and retry_after.isdigit()) else backoff
                        log.warning(
                            "rest_rate_limited status=%s symbol=%s tf=%s sleep=%.1fs body=%s",
                            resp.status,
                            symbol,
                            timeframe,
                            sleep_s,
                            txt[:200],
                        )
                        await asyncio.sleep(sleep_s)
                        backoff = min(backoff * 2.0, 20.0)
                        continue

                    if resp.status != 200:
                        txt = await resp.text()
                        raise RuntimeError(f"Binance klines failed: {resp.status} {txt[:500]}")

                    # Some proxies return a wrong content-type; be tolerant.
                    data = await resp.json(content_type=None)

                last_err = None
                break

            except (asyncio.TimeoutError, aiohttp.ClientError) as e:
                last_err = e
                if attempt >= int(self.rest_max_retries):
                    break
                log.warning(
                    "rest_timeout_or_client_err attempt=%d/%d symbol=%s tf=%s backoff=%.1fs err=%s",
                    attempt,
                    self.rest_max_retries,
                    symbol,
                    timeframe,
                    backoff,
                    e,
                )
                await asyncio.sleep(backoff)
                backoff = min(backoff * 2.0, 20.0)

        if last_err is not None:
            raise last_err

        out: List[Candle] = []
        for row in data:
            # [0]=open time, [6]=close time
            out.append(Candle(
                open_time_ms=int(row[0]),
                close_time_ms=int(row[6]),
                open=float(row[1]),
                high=float(row[2]),
                low=float(row[3]),
                close=float(row[4]),
                volume=float(row[5]),
            ))
        return out

    async def stream_klines(self, symbols: List[str], timeframes: List[str]) -> AsyncIterator[KlineEvent]:
        """Yields CLOSED klines for all (symbol, tf). Auto-reconnects."""
        streams = [_stream_name(sym, tf) for sym in symbols for tf in timeframes]
        ws_url = _ws_url(self.market)

        sub_msg = {"method": "SUBSCRIBE", "params": streams, "id": 1}

        backoff = 1
        while True:
            try:
                async with websockets.connect(
                    ws_url,
                    ping_interval=self.ws_heartbeat_s,
                    ping_timeout=self.ws_heartbeat_s,
                    close_timeout=5,
                    max_queue=5000,
                ) as ws:
                    backoff = 1
                    await ws.send(json.dumps(sub_msg))
                    log.info("ws_subscribed streams=%d market=%s", len(streams), self.market)

                    async for msg in ws:
                        try:
                            j = json.loads(msg)
                        except Exception:
                            continue
                        if "result" in j and j.get("id") == 1:
                            continue  # subscribe ack

                        data = j.get("data") or j
                        if not isinstance(data, dict):
                            continue
                        if data.get("e") != "kline":
                            continue

                        k = data.get("k", {})
                        if not k.get("x", False):
                            continue  # only closed candles

                        symbol = k.get("s", "").upper()
                        tf = k.get("i", "")
                        c = Candle(
                            open_time_ms=int(k.get("t")),
                            close_time_ms=int(k.get("T")),
                            open=float(k.get("o")),
                            high=float(k.get("h")),
                            low=float(k.get("l")),
                            close=float(k.get("c")),
                            volume=float(k.get("v")),
                        )
                        yield KlineEvent(symbol=symbol, timeframe=tf, candle=c)

            except Exception as e:
                log.warning("ws_error err=%s reconnect_in=%ss", e, backoff)
                await asyncio.sleep(backoff)
                backoff = min(backoff * 2, 60)
