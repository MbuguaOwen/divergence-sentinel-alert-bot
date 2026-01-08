from __future__ import annotations

import asyncio
import logging
from typing import Optional

import aiohttp

from ..models import Signal

log = logging.getLogger("webhook")


def format_price_pine(x: float) -> str:
    """Format like Pine's '#.########' then strip trailing zeros/dot."""
    s = f"{x:.8f}"
    s = s.rstrip("0").rstrip(".")
    return s or "0"


def tf_to_tv_period(tf: str) -> str:
    tf = (tf or "").lower()
    if tf.endswith("m"):
        return tf[:-1]
    if tf.endswith("h"):
        return str(int(tf[:-1]) * 60)
    if tf.endswith("d"):
        return str(int(tf[:-1]) * 1440)
    return tf


class WebhookNotifier:
    def __init__(self, *, enabled: bool, url: str, secret: str, include_tf: bool, timeout_s: int, headers: dict):
        self.enabled = bool(enabled)
        self.url = url or ""
        self.secret = secret or ""
        self.include_tf = bool(include_tf)
        self.timeout_s = int(timeout_s) if timeout_s is not None else 10
        self.headers = headers or {}

    async def send_signal(self, sig: Signal) -> None:
        if not self.enabled or not self.url:
            return

        fields = [
            f'"secret":"{self.secret}"',
            f'"symbol":"{sig.symbol}"',
            f'"side":"{sig.side}"',
            f'"entry_price":{format_price_pine(sig.entry_price)}',
            f'"confirm_time_ms":{int(sig.confirm_time_ms)}',
        ]
        if self.include_tf:
            fields.append(f'"tf":"{tf_to_tv_period(sig.timeframe)}"')
        if sig.signal_id:
            fields.append(f'"signal_id":"{sig.signal_id}"')
        if sig.confirm_bar_close_ms is not None:
            fields.append(f'"confirm_bar_close_ms":{int(sig.confirm_bar_close_ms)}')
        if sig.confirm_bar_index is not None:
            fields.append(f'"confirm_bar_index":{int(sig.confirm_bar_index)}')
        if sig.entry_intent:
            fields.append(f'"entry_intent":"{sig.entry_intent}"')
        if sig.entry_mode:
            fields.append(f'"entry_mode":"{sig.entry_mode}"')
        if sig.entry_price_reference is not None:
            fields.append(f'"entry_price_reference":{format_price_pine(sig.entry_price_reference)}')
        payload = "{" + ",".join(fields) + "}"
        try:
            timeout = aiohttp.ClientTimeout(total=self.timeout_s)
            async with aiohttp.ClientSession(timeout=timeout) as session:
                async with session.post(self.url, data=payload, headers={"Content-Type": "application/json", **self.headers}) as resp:
                    if resp.status >= 400:
                        text = await resp.text()
                        log.warning("webhook_bad_status status=%s body=%s", resp.status, text[:200])
        except Exception as e:
            # Log but do not crash
            log.warning("webhook_post_failed err=%s", e)
