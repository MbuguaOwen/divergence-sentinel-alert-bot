from __future__ import annotations

import aiohttp
from typing import List, Optional
import logging

log = logging.getLogger("telegram")


class TelegramNotifier:
    def __init__(self, token: str, chat_ids: List[str], *, disable_web_page_preview: bool = True):
        self.token = token.strip()
        self.chat_ids = [str(x).strip() for x in (chat_ids or []) if str(x).strip()]
        self.disable_web_page_preview = disable_web_page_preview

    def enabled(self) -> bool:
        return bool(self.token) and bool(self.chat_ids)

    async def send(self, text: str) -> None:
        if not self.enabled():
            return
        url = f"https://api.telegram.org/bot{self.token}/sendMessage"
        async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=15)) as sess:
            for chat_id in self.chat_ids:
                payload = {
                    "chat_id": chat_id,
                    "text": text,
                    "disable_web_page_preview": self.disable_web_page_preview,
                }
                try:
                    async with sess.post(url, json=payload) as resp:
                        if resp.status != 200:
                            body = await resp.text()
                            log.warning("telegram_send_failed chat_id=%s status=%s body=%s", chat_id, resp.status, body[:2000])
                except Exception as e:
                    log.exception("telegram_send_exception chat_id=%s err=%s", chat_id, e)
