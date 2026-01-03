from __future__ import annotations
from typing import List, Optional, Tuple
import math


def ema_next(prev_ema: Optional[float], x: float, length: int) -> float:
    if length <= 1:
        return x
    alpha = 2.0 / (length + 1.0)
    return x if prev_ema is None else (alpha * x + (1.0 - alpha) * prev_ema)


def sma(values: List[float], length: int) -> Optional[float]:
    if length <= 0 or len(values) < length:
        return None
    return sum(values[-length:]) / float(length)


def rsi_wilder(closes: List[float], length: int = 14) -> Optional[float]:
    if length <= 0 or len(closes) < length + 1:
        return None
    # Wilder's smoothing
    gains = 0.0
    losses = 0.0
    for i in range(-length, 0):
        ch = closes[i] - closes[i - 1]
        if ch >= 0:
            gains += ch
        else:
            losses -= ch
    avg_gain = gains / length
    avg_loss = losses / length
    if avg_loss == 0:
        return 100.0
    rs = avg_gain / avg_loss
    return 100.0 - (100.0 / (1.0 + rs))


def true_range(high: float, low: float, prev_close: float) -> float:
    return max(high - low, abs(high - prev_close), abs(low - prev_close))


def atr_series(highs: List[float], lows: List[float], closes: List[float], length: int = 14) -> Optional[float]:
    if length <= 0 or len(closes) < length + 1:
        return None
    trs = []
    for i in range(-length, 0):
        tr = true_range(highs[i], lows[i], closes[i - 1])
        trs.append(tr)
    return sum(trs) / length


def pct_change(new: float, old: float) -> Optional[float]:
    if old == 0:
        return None
    return (new - old) / old * 100.0


def bps_diff(a: float, b: float) -> Optional[float]:
    if b == 0:
        return None
    return (a - b) / b * 10000.0
