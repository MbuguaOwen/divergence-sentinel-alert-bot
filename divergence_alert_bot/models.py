from __future__ import annotations
from dataclasses import dataclass
from typing import Optional


@dataclass(frozen=True)
class Candle:
    open_time_ms: int
    close_time_ms: int
    open: float
    high: float
    low: float
    close: float
    volume: float


@dataclass(frozen=True)
class Signal:
    symbol: str
    timeframe: str
    side: str  # LONG or SHORT
    pivot_time_ms: int
    confirm_time_ms: int
    pivot_price: float
    entry_price: float
    don_loc_pct: float
    price_change_pct: float
    osc_change_pct: float
    div_type: str  # Classic or Equal
    slip_bps: float
    structural_sl: Optional[float]
    structural_sl_distance_pct: Optional[float]
    prob_score: int
    bars_gap: int
    score_breakdown: str
    extra: dict
