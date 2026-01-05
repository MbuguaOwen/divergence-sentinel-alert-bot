from __future__ import annotations

from collections import deque
from typing import Deque, Optional, Tuple

from .models import Candle


class CvdProxyStore:
    """Rolling signed-volume proxy for CVD sampled from 1m candles."""

    def __init__(self, cvd_len_min: int, max_history: int = 250000) -> None:
        self.cvd_len_min = max(1, int(cvd_len_min))
        self.max_history = max(1, int(max_history))
        self._vol_deque: Deque[float] = deque()
        self._sum: float = 0.0
        self.history: Deque[Tuple[int, float]] = deque()

    def on_1m_candle(self, c: Candle, *, use_close_minus_1ms: bool) -> float:
        signed_vol = c.volume if c.close >= c.open else -c.volume
        self._vol_deque.append(signed_vol)
        self._sum += signed_vol
        if len(self._vol_deque) > self.cvd_len_min:
            old = self._vol_deque.popleft()
            self._sum -= old

        cvd_now = self._sum
        # Binance close_time_ms is already end-of-bar minus 1ms; adjust via parity rule.
        ts = c.close_time_ms if use_close_minus_1ms else (c.close_time_ms + 1)
        self.history.append((ts, cvd_now))
        while len(self.history) > self.max_history:
            self.history.popleft()
        return cvd_now

    def value_at_or_before(self, ts_ms: int) -> Optional[float]:
        for ts, val in reversed(self.history):
            if ts <= ts_ms:
                return val
        return None

    @property
    def latest_ts(self) -> Optional[int]:
        return self.history[-1][0] if self.history else None
