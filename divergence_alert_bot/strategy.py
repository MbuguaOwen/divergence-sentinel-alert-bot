from __future__ import annotations

from dataclasses import dataclass
import hashlib
from typing import Dict, List, Optional, Tuple

from .models import Candle, Signal
from .indicators import ema_next, sma, pct_change, percentile_linear_interpolation
from .time_parity import tv_confirm_ts


def rma_next(prev: Optional[float], x: float, length: int) -> float:
    """Wilder's RMA (used by ta.atr / ta.rsi in Pine)."""
    if length <= 1:
        return x
    if prev is None:
        return x
    alpha = 1.0 / float(length)
    return prev + alpha * (x - prev)


def true_range(high: float, low: float, prev_close: float) -> float:
    return max(high - low, abs(high - prev_close), abs(low - prev_close))


@dataclass
class PivotMem:
    lastPL_price: Optional[float] = None
    lastPL_osc: Optional[float] = None
    lastPL_time_ms: Optional[int] = None
    lastPL_bar: Optional[int] = None  # pivot bar index

    lastPH_price: Optional[float] = None
    lastPH_osc: Optional[float] = None
    lastPH_time_ms: Optional[int] = None
    lastPH_bar: Optional[int] = None  # pivot bar index


class StrategyEngine:
    """Per (symbol, timeframe) stateful engine."""

    def __init__(
        self,
        symbol: str,
        timeframe: str,
        don_len: int,
        pivot_len: int,
        osc_len: int,
        ext_band_pct: float,
        *,
        mode: str = "sniper",
        long_only: bool = True,
        trade_enabled: bool = True,
        entry_wait_confirm: bool = True,
        use_bos_confirmation: bool = False,
        bos_atr_buffer: float = 0.0,
        max_wait_bars: int = 4,
        min_div_strength: float = 0.0,
        cooldown_bars: int = 0,
        use_cvd_gate: bool = False,
        use_dynamic_cvd_pct: bool = False,
        cvd_lookback_bars: int = 100,
        cvd_pct: int = 50,
        cvd_threshold: float = 0.0,
        use_close_minus_1ms: bool = False,
        strategy_sig: Optional[str] = None,
        # Kill switches
        ks_short_enabled: bool = True,
        ks_don_width_atr_max: float = 16.7,
        ks_long_enabled: bool = True,
        ks_liq_percentile_min: float = 0.25,
        ks_liq_lookback: int = 100,
        # Scoring
        min_score_to_trade: int = 6,
    ):
        self.symbol = symbol
        self.timeframe = timeframe
        self.don_len = don_len
        self.pivot_len = pivot_len
        self.osc_len = osc_len
        self.ext_band_pct = ext_band_pct
        self.mode = mode

        # TV parity toggles
        self.long_only = long_only
        self.trade_enabled = trade_enabled
        self.entry_wait_confirm = entry_wait_confirm
        self.use_bos_confirmation = use_bos_confirmation
        self.bos_atr_buffer = bos_atr_buffer
        self.max_wait_bars = max_wait_bars
        self.min_div_strength = min_div_strength
        self.cooldown_bars = cooldown_bars
        self.use_cvd_gate = use_cvd_gate
        self.use_dynamic_cvd_pct = use_dynamic_cvd_pct
        self.cvd_lookback_bars = cvd_lookback_bars
        self.cvd_pct = cvd_pct
        self.cvd_threshold = cvd_threshold
        self.use_close_minus_1ms = use_close_minus_1ms
        self.strategy_sig = strategy_sig

        self.ks_short_enabled = ks_short_enabled
        self.ks_don_width_atr_max = ks_don_width_atr_max
        self.ks_long_enabled = ks_long_enabled
        self.ks_liq_percentile_min = ks_liq_percentile_min
        self.ks_liq_lookback = ks_liq_lookback
        self.min_score_to_trade = min_score_to_trade

        self.candles: List[Candle] = []

        # stored series for feature parity
        self._osc_raw: List[float] = []   # (close-open)*vol
        self.osc: List[float] = []        # EMA of raw
        self.donHi: List[float] = []
        self.donLo: List[float] = []
        self.loc: List[float] = []        # 0..1 within don range
        self.hh_pivot: List[Optional[float]] = []
        self.cvd_tf: List[Optional[float]] = []  # per bar CVD sample for tv parity

        # Wilder ATR + RSI series (Pine ta.atr / ta.rsi parity)
        self._trs: List[float] = []
        self.atr14: List[Optional[float]] = []
        self.atr50: List[Optional[float]] = []
        self._atr14_prev: Optional[float] = None
        self._atr50_prev: Optional[float] = None

        self.rsi14: List[Optional[float]] = []
        self._rsi_avg_gain: Optional[float] = None
        self._rsi_avg_loss: Optional[float] = None
        self._rsi_gains: List[float] = []
        self._rsi_losses: List[float] = []

        # Volume / liquidity derived series
        self.avgVol20: List[Optional[float]] = []
        self.volRatio20: List[Optional[float]] = []
        self.don_width_atr: List[Optional[float]] = []
        self.liq_proxy: List[Optional[float]] = []
        self.liq_percentile: List[Optional[float]] = []

        # Trend context (close EMAs)
        self.ema20_close: List[Optional[float]] = []
        self.ema50_close: List[Optional[float]] = []

        self.piv = PivotMem()

        self.total_div = 0
        self.filtered_div = 0

        # TV parity state
        self.longSetup = False
        self.longTrig: Optional[float] = None
        self.longSetBar: Optional[int] = None
        self.lastEntryBar: Optional[int] = None
        self.longPivotIdx: Optional[int] = None
        self.longSetupNearLower: bool = False
        self.longSetupBullDiv: bool = False
        self.longSetupOscChange: float = 0.0

    def on_candle(self, c: Candle, *, allow_signals: bool, in_trading_hours: bool, cvd_now: Optional[float] = None) -> List[Signal]:
        """Consume a CLOSED candle. Returns zero or more signals for THIS close."""
        if self.mode == "tv_parity":
            return self._on_candle_tv_parity(c, allow_signals=allow_signals, cvd_now=cvd_now)
        return self._on_candle_sniper(c, allow_signals=allow_signals, in_trading_hours=in_trading_hours)

    def _on_candle_sniper(self, c: Candle, *, allow_signals: bool, in_trading_hours: bool) -> List[Signal]:
        self._append(c)
        signals: List[Signal] = []

        # Need enough bars to confirm a pivot
        if len(self.candles) < (2 * self.pivot_len + 1):
            return signals

        confirm_idx = len(self.candles) - 1
        pivot_idx = confirm_idx - self.pivot_len
        if pivot_idx - self.pivot_len < 0:
            return signals

        # Evaluate possible pivot(s) at pivot_idx
        lo_window = [self.candles[i].low for i in range(pivot_idx - self.pivot_len, pivot_idx + self.pivot_len + 1)]
        hi_window = [self.candles[i].high for i in range(pivot_idx - self.pivot_len, pivot_idx + self.pivot_len + 1)]
        pivot_low = self.candles[pivot_idx].low == min(lo_window)
        pivot_high = self.candles[pivot_idx].high == max(hi_window)

        if pivot_low:
            sig = self._handle_pivot_low(pivot_idx, confirm_idx, allow_signals, in_trading_hours)
            if sig:
                signals.append(sig)

        if pivot_high:
            sig = self._handle_pivot_high(pivot_idx, confirm_idx, allow_signals, in_trading_hours)
            if sig:
                signals.append(sig)

        return signals

    def _on_candle_tv_parity(self, c: Candle, *, allow_signals: bool, cvd_now: Optional[float]) -> List[Signal]:
        self._append(c, cvd_now=cvd_now)
        signals: List[Signal] = []

        # Need enough bars to confirm a pivot
        if len(self.candles) < (2 * self.pivot_len + 1):
            return signals

        confirm_idx = len(self.candles) - 1
        pivot_idx = confirm_idx - self.pivot_len
        if pivot_idx - self.pivot_len < 0:
            return signals

        # Detect strict pivots (ties rejected)
        pivot_low = self._is_strict_pivot_low(pivot_idx)
        if pivot_low:
            pl_price = self.candles[pivot_idx].low
            pl_osc = self.osc[pivot_idx]
            loc_p = self.loc[pivot_idx]

            near_lower = loc_p <= self.ext_band_pct
            has_prev = self.piv.lastPL_price is not None and self.piv.lastPL_osc is not None
            bull_div = has_prev and (pl_price <= self.piv.lastPL_price) and (pl_osc > self.piv.lastPL_osc)

            osc_change_pct = 0.0
            if has_prev and self.piv.lastPL_osc is not None:
                safe_prev = max(abs(float(self.piv.lastPL_osc)), 1e-9)
                osc_change_pct = ((pl_osc - float(self.piv.lastPL_osc)) / safe_prev) * 100.0

            strength_ok = (float(self.min_div_strength) <= 0.0) or (osc_change_pct >= float(self.min_div_strength))
            can_enter = self._can_enter(confirm_idx)

            if near_lower and bull_div and strength_ok and can_enter:
                self.longSetup = True
                self.longTrig = self.hh_pivot[confirm_idx] if confirm_idx < len(self.hh_pivot) else None
                self.longSetBar = confirm_idx
                self.longPivotIdx = pivot_idx
                self.longSetupNearLower = near_lower
                self.longSetupBullDiv = bull_div
                self.longSetupOscChange = osc_change_pct

            # Update pivot memory regardless (Pine behavior)
            self.piv.lastPL_price = pl_price
            self.piv.lastPL_osc = pl_osc
            self.piv.lastPL_time_ms = self.candles[pivot_idx].close_time_ms
            self.piv.lastPL_bar = pivot_idx

        # Confirm-mode trigger after setup creation (allows same-bar trigger)
        if self.longSetup and self.longSetBar is not None:
            if confirm_idx > (self.longSetBar + int(self.max_wait_bars)):
                self._clear_setup()
            else:
                cvd_pass, cvd_thr = self._cvd_gate_pass(cvd_now, confirm_idx)
                can_enter = self._can_enter(confirm_idx)

                bos_ok = True
                if self.use_bos_confirmation:
                    atr = self.atr14[confirm_idx] if confirm_idx < len(self.atr14) else None
                    if atr is None or self.longTrig is None:
                        bos_ok = False
                    else:
                        buf = (atr or 0.0) * float(self.bos_atr_buffer)
                        bos_ok = c.close > (self.longTrig + buf)

                if cvd_pass and bos_ok and can_enter and self.trade_enabled and allow_signals:
                    pivot_idx_for_entry = self.longPivotIdx if self.longPivotIdx is not None else (self.piv.lastPL_bar if self.piv.lastPL_bar is not None else confirm_idx)
                    osc_change = self.longSetupOscChange
                    near_lower = self.longSetupNearLower
                    bull_div = self.longSetupBullDiv
                    entry_mode = "BOS_CVD" if self.use_bos_confirmation else "CVD_WAIT"
                    sig = self._make_tv_signal(
                        side="LONG",
                        pivot_idx=pivot_idx_for_entry,
                        confirm_idx=confirm_idx,
                        entry_idx=confirm_idx,
                        cvd_now=cvd_now,
                        cvd_thr=cvd_thr,
                        cvd_gate_pass=cvd_pass,
                        entry_mode=entry_mode,
                        osc_change_pct=osc_change,
                        near_lower=near_lower,
                        bull_div=bull_div,
                    )
                    signals.append(sig)
                    self.lastEntryBar = confirm_idx
                    self._clear_setup()

        return signals

    def _clear_setup(self) -> None:
        self.longSetup = False
        self.longTrig = None
        self.longSetBar = None
        self.longPivotIdx = None
        self.longSetupNearLower = False
        self.longSetupBullDiv = False
        self.longSetupOscChange = 0.0

    def _signal_id(self, side: str, confirm_ts: int) -> Optional[str]:
        if not self.strategy_sig:
            return None
        base = f"{self.symbol}:{self.timeframe}:{side}:{confirm_ts}:{self.strategy_sig}"
        return hashlib.sha256(base.encode("utf-8")).hexdigest()

    def _cvd_gate_pass(self, cvd_now: Optional[float], bar_idx: int) -> Tuple[bool, Optional[float]]:
        if not self.use_cvd_gate:
            return True, None
        if self.use_dynamic_cvd_pct:
            thr = percentile_linear_interpolation(self.cvd_tf[: bar_idx + 1], int(self.cvd_lookback_bars), int(self.cvd_pct))
        else:
            thr = float(self.cvd_threshold)
        if thr is None or cvd_now is None:
            return False, thr
        return cvd_now >= thr, thr

    def _can_enter(self, bar_idx: int) -> bool:
        if self.cooldown_bars <= 0:
            return True
        if self.lastEntryBar is None:
            return True
        return (bar_idx - self.lastEntryBar) >= int(self.cooldown_bars)

    def _append(self, c: Candle, cvd_now: Optional[float] = None) -> None:
        self.candles.append(c)

        raw = (c.close - c.open) * c.volume
        self._osc_raw.append(raw)
        prev = self.osc[-1] if self.osc else None
        self.osc.append(ema_next(prev, raw, self.osc_len))

        # Donchian per bar
        w = self.don_len
        if len(self.candles) < w:
            highs = [x.high for x in self.candles]
            lows = [x.low for x in self.candles]
        else:
            highs = [x.high for x in self.candles[-w:]]
            lows = [x.low for x in self.candles[-w:]]
        dh = max(highs)
        dl = min(lows)
        self.donHi.append(dh)
        self.donLo.append(dl)

        rng = dh - dl
        loc = 0.5 if rng <= 0 else (c.close - dl) / rng
        self.loc.append(loc)

        # Rolling highest high over pivot_len bars (setup trigger reference)
        look = max(1, self.pivot_len)
        if len(self.candles) < look:
            hh = max(x.high for x in self.candles)
        else:
            hh = max(x.high for x in self.candles[-look:])
        self.hh_pivot.append(hh)

        # Close EMAs for trend context
        prev_e20 = self.ema20_close[-1] if self.ema20_close else None
        prev_e50 = self.ema50_close[-1] if self.ema50_close else None
        self.ema20_close.append(ema_next(prev_e20, c.close, 20))
        self.ema50_close.append(ema_next(prev_e50, c.close, 50))

        # ATR (Wilder)
        prev_close = self.candles[-2].close if len(self.candles) >= 2 else c.close
        tr = true_range(c.high, c.low, prev_close)
        self._trs.append(tr)

        self._atr14_prev = self._update_atr(tr, 14, self._atr14_prev)
        self._atr50_prev = self._update_atr(tr, 50, self._atr50_prev)
        self.atr14.append(self._atr14_prev)
        self.atr50.append(self._atr50_prev)

        # RSI (Wilder)
        if len(self.candles) >= 2:
            ch = c.close - prev_close
            gain = ch if ch > 0 else 0.0
            loss = -ch if ch < 0 else 0.0
        else:
            gain, loss = 0.0, 0.0

        self._rsi_gains.append(gain)
        self._rsi_losses.append(loss)

        rsi_len = 14
        rsi_val: Optional[float] = None
        if self._rsi_avg_gain is None or self._rsi_avg_loss is None:
            if len(self._rsi_gains) >= rsi_len:
                self._rsi_avg_gain = sum(self._rsi_gains[-rsi_len:]) / rsi_len
                self._rsi_avg_loss = sum(self._rsi_losses[-rsi_len:]) / rsi_len
        else:
            self._rsi_avg_gain = rma_next(self._rsi_avg_gain, gain, rsi_len)
            self._rsi_avg_loss = rma_next(self._rsi_avg_loss, loss, rsi_len)

        if self._rsi_avg_gain is not None and self._rsi_avg_loss is not None:
            if self._rsi_avg_loss == 0:
                rsi_val = 100.0
            else:
                rs = self._rsi_avg_gain / self._rsi_avg_loss
                rsi_val = 100.0 - (100.0 / (1.0 + rs))

        self.rsi14.append(rsi_val)

        # Volume ratios (SMA20)
        vols = [x.volume for x in self.candles]
        av20 = sma(vols, 20)
        self.avgVol20.append(av20)
        vr20 = (c.volume / av20) if (av20 is not None and av20 != 0) else None
        self.volRatio20.append(vr20)

        # Donchian width / ATR14
        a14 = self._atr14_prev
        if a14 is not None and a14 > 0:
            self.don_width_atr.append((dh - dl) / a14)
        else:
            self.don_width_atr.append(None)

        # Liquidity percentile proxy
        lp = c.volume * a14 if a14 is not None else None
        self.liq_proxy.append(lp)
        if lp is None:
            self.liq_percentile.append(None)
        else:
            look_liq = max(1, int(self.ks_liq_lookback))
            window = [x for x in (self.liq_proxy[-look_liq:]) if x is not None]
            if not window:
                self.liq_percentile.append(None)
            else:
                hi = max(window)
                lo = min(window)
                rng2 = hi - lo
                self.liq_percentile.append(0.5 if rng2 <= 0 else (lp - lo) / rng2)

        # Store CVD sample (tv parity only)
        self.cvd_tf.append(float(cvd_now) if cvd_now is not None else None)

    def _is_strict_pivot_low(self, pivot_idx: int) -> bool:
        if pivot_idx - self.pivot_len < 0 or pivot_idx + self.pivot_len >= len(self.candles):
            return False
        pivot_low = self.candles[pivot_idx].low
        for i in range(pivot_idx - self.pivot_len, pivot_idx + self.pivot_len + 1):
            if i == pivot_idx:
                continue
            if pivot_low >= self.candles[i].low:
                return False
        return True

    def _is_strict_pivot_high(self, pivot_idx: int) -> bool:
        if pivot_idx - self.pivot_len < 0 or pivot_idx + self.pivot_len >= len(self.candles):
            return False
        pivot_high = self.candles[pivot_idx].high
        for i in range(pivot_idx - self.pivot_len, pivot_idx + self.pivot_len + 1):
            if i == pivot_idx:
                continue
            if pivot_high <= self.candles[i].high:
                return False
        return True

        # Close EMAs for trend context
        prev_e20 = self.ema20_close[-1] if self.ema20_close else None
        prev_e50 = self.ema50_close[-1] if self.ema50_close else None
        self.ema20_close.append(ema_next(prev_e20, c.close, 20))
        self.ema50_close.append(ema_next(prev_e50, c.close, 50))

        # ATR (Wilder)
        prev_close = self.candles[-2].close if len(self.candles) >= 2 else c.close
        tr = true_range(c.high, c.low, prev_close)
        self._trs.append(tr)

        self._atr14_prev = self._update_atr(tr, 14, self._atr14_prev)
        self._atr50_prev = self._update_atr(tr, 50, self._atr50_prev)
        self.atr14.append(self._atr14_prev)
        self.atr50.append(self._atr50_prev)

        # RSI (Wilder)
        if len(self.candles) >= 2:
            ch = c.close - prev_close
            gain = ch if ch > 0 else 0.0
            loss = -ch if ch < 0 else 0.0
        else:
            gain, loss = 0.0, 0.0

        self._rsi_gains.append(gain)
        self._rsi_losses.append(loss)

        rsi_len = 14
        rsi_val: Optional[float] = None
        if self._rsi_avg_gain is None or self._rsi_avg_loss is None:
            if len(self._rsi_gains) >= rsi_len:
                self._rsi_avg_gain = sum(self._rsi_gains[-rsi_len:]) / rsi_len
                self._rsi_avg_loss = sum(self._rsi_losses[-rsi_len:]) / rsi_len
        else:
            self._rsi_avg_gain = rma_next(self._rsi_avg_gain, gain, rsi_len)
            self._rsi_avg_loss = rma_next(self._rsi_avg_loss, loss, rsi_len)

        if self._rsi_avg_gain is not None and self._rsi_avg_loss is not None:
            if self._rsi_avg_loss == 0:
                rsi_val = 100.0
            else:
                rs = self._rsi_avg_gain / self._rsi_avg_loss
                rsi_val = 100.0 - (100.0 / (1.0 + rs))

        self.rsi14.append(rsi_val)

        # Volume ratios (SMA20)
        vols = [x.volume for x in self.candles]
        av20 = sma(vols, 20)
        self.avgVol20.append(av20)
        vr20 = (c.volume / av20) if (av20 is not None and av20 != 0) else None
        self.volRatio20.append(vr20)

        # Donchian width / ATR14
        a14 = self._atr14_prev
        if a14 is not None and a14 > 0:
            self.don_width_atr.append((dh - dl) / a14)
        else:
            self.don_width_atr.append(None)

        # Liquidity percentile proxy
        lp = c.volume * a14 if a14 is not None else None
        self.liq_proxy.append(lp)
        if lp is None:
            self.liq_percentile.append(None)
        else:
            look = max(1, int(self.ks_liq_lookback))
            window = [x for x in (self.liq_proxy[-look:]) if x is not None]
            if not window:
                self.liq_percentile.append(None)
            else:
                hi = max(window)
                lo = min(window)
                rng2 = hi - lo
                self.liq_percentile.append(0.5 if rng2 <= 0 else (lp - lo) / rng2)

    def _update_atr(self, tr: float, length: int, prev_atr: Optional[float]) -> Optional[float]:
        """Wilder ATR with SMA seed at first full window."""
        if length <= 0:
            return None
        if len(self._trs) < length:
            return None
        if prev_atr is None:
            return sum(self._trs[-length:]) / float(length)
        return rma_next(prev_atr, tr, length)

    def _calc_long_score(
        self,
        *,
        osc_change_pct: float,
        loc_at_pivot_pct: float,
        vol_ratio_at_pivot: Optional[float],
        rsi_at_pivot: Optional[float],
        liq_pct: Optional[float],
        div_type: str,
        bars_gap: int,
    ) -> Tuple[int, str]:
        score = 0
        b = []

        if 15 <= osc_change_pct <= 40:
            score += 2
            b.append("Osc change 15-40% (+2)")
        elif osc_change_pct > 40:
            b.append("Osc change >40% (+0)")
        else:
            score += 1
            b.append("Osc change <15% (+1)")

        if loc_at_pivot_pct <= 5:
            score += 2
            b.append("Donchian loc <=5% (+2)")
        elif loc_at_pivot_pct <= 10:
            score += 1
            b.append("Donchian loc <=10% (+1)")
        else:
            b.append("Donchian loc mid (+0)")

        if liq_pct is None:
            b.append("Liquidity NA (+0)")
        elif liq_pct >= 0.50:
            score += 2
            b.append("Liquidity >=50% (+2)")
        elif liq_pct >= 0.25:
            score += 1
            b.append("Liquidity >=25% (+1)")
        else:
            b.append("Liquidity low (+0)")

        if vol_ratio_at_pivot is not None and vol_ratio_at_pivot >= 1.5:
            score += 1
            b.append("Volume ratio >=1.5x (+1)")
        else:
            b.append("Volume ratio <1.5x (+0)")

        if rsi_at_pivot is not None and rsi_at_pivot <= 35:
            score += 1
            b.append("RSI <=35 (+1)")
        else:
            b.append("RSI >35 (+0)")

        if div_type == "Classic":
            score += 1
            b.append("Classic divergence (+1)")
        else:
            b.append("Equal divergence (+0)")

        if bars_gap >= 15:
            score += 1
            b.append("Bars gap >=15 (+1)")
        else:
            b.append("Bars gap <15 (+0)")

        return score, "\n".join(b)

    def _calc_short_score(
        self,
        *,
        osc_change_pct: float,
        loc_at_pivot_pct: float,
        vol_ratio_at_pivot: Optional[float],
        rsi_at_pivot: Optional[float],
        don_w_atr: Optional[float],
        div_type: str,
        bars_gap: int,
    ) -> Tuple[int, str]:
        score = 0
        b = []

        if 15 <= osc_change_pct <= 40:
            score += 2
            b.append("Osc change 15-40% (+2)")
        elif osc_change_pct > 40:
            b.append("Osc change >40% (+0)")
        else:
            score += 1
            b.append("Osc change <15% (+1)")

        if loc_at_pivot_pct >= 95:
            score += 2
            b.append("Donchian loc >=95% (+2)")
        elif loc_at_pivot_pct >= 90:
            score += 1
            b.append("Donchian loc >=90% (+1)")
        else:
            b.append("Donchian loc mid (+0)")

        if don_w_atr is None:
            b.append("Don width/ATR NA (+0)")
        elif don_w_atr <= 12:
            score += 2
            b.append("Don width/ATR <=12 (+2)")
        elif don_w_atr <= 16.7:
            score += 1
            b.append("Don width/ATR <=16.7 (+1)")
        else:
            b.append("Don width/ATR wide (+0)")

        if vol_ratio_at_pivot is not None and vol_ratio_at_pivot >= 1.5:
            score += 1
            b.append("Volume ratio >=1.5x (+1)")
        else:
            b.append("Volume ratio <1.5x (+0)")

        if rsi_at_pivot is not None and rsi_at_pivot >= 65:
            score += 1
            b.append("RSI >=65 (+1)")
        else:
            b.append("RSI <65 (+0)")

        if div_type == "Classic":
            score += 1
            b.append("Classic divergence (+1)")
        else:
            b.append("Equal divergence (+0)")

        if bars_gap >= 15:
            score += 1
            b.append("Bars gap >=15 (+1)")
        else:
            b.append("Bars gap <15 (+0)")

        return score, "\n".join(b)

    def _handle_pivot_low(self, pivot_idx: int, confirm_idx: int, allow_signals: bool, in_trading_hours: bool) -> Optional[Signal]:
        """Bullish divergence at lower Donchian extreme + scoring + LONG liquidity kill switch."""
        pl_price = self.candles[pivot_idx].low
        pl_osc = self.osc[pivot_idx]
        loc_p = self.loc[pivot_idx]

        near_lower = loc_p <= self.ext_band_pct
        has_prev = self.piv.lastPL_price is not None and self.piv.lastPL_osc is not None
        bull_div = has_prev and (pl_price <= self.piv.lastPL_price) and (pl_osc > self.piv.lastPL_osc)

        sig: Optional[Signal] = None
        if near_lower and bull_div:
            self.total_div += 1
            if in_trading_hours:
                self.filtered_div += 1
                if allow_signals:
                    prev_price = float(self.piv.lastPL_price)
                    prev_osc = float(self.piv.lastPL_osc)

                    price_ch = float(pct_change(pl_price, prev_price) or 0.0)
                    osc_ch = 0.0
                    if abs(prev_osc) > 0:
                        osc_ch = ((pl_osc - prev_osc) / abs(prev_osc)) * 100.0

                    div_type = "Classic" if pl_price < prev_price else "Equal"
                    bars_gap = (confirm_idx - self.piv.lastPL_bar) if self.piv.lastPL_bar is not None else confirm_idx

                    loc_at_pivot_pct = loc_p * 100.0
                    vol_ratio_at_pivot = self.volRatio20[pivot_idx] if pivot_idx < len(self.volRatio20) else None
                    rsi_at_pivot = self.rsi14[pivot_idx] if pivot_idx < len(self.rsi14) else None
                    liq_at_pivot = self.liq_percentile[pivot_idx] if pivot_idx < len(self.liq_percentile) else None

                    blocked_by_liq = bool(
                        self.ks_long_enabled
                        and (liq_at_pivot is None or liq_at_pivot < self.ks_liq_percentile_min)
                    )

                    score, breakdown = self._calc_long_score(
                        osc_change_pct=osc_ch,
                        loc_at_pivot_pct=loc_at_pivot_pct,
                        vol_ratio_at_pivot=vol_ratio_at_pivot,
                        rsi_at_pivot=rsi_at_pivot,
                        liq_pct=liq_at_pivot,
                        div_type=div_type,
                        bars_gap=bars_gap,
                    )

                    is_high_prob = (score >= int(self.min_score_to_trade)) and (not blocked_by_liq)
                    if is_high_prob:
                        sig = self._make_signal(
                            side="LONG",
                            pivot_idx=pivot_idx,
                            confirm_idx=confirm_idx,
                            pivot_price=pl_price,
                            entry_price=self.candles[confirm_idx].close,
                            price_change_pct=price_ch,
                            osc_change_pct=osc_ch,
                            div_type=div_type,
                            slip_bps=((self.candles[confirm_idx].close - pl_price) / pl_price * 10000.0) if pl_price else 0.0,
                            structural_sl=prev_price,
                            prob_score=score,
                            bars_gap=bars_gap,
                            score_breakdown=breakdown,
                        )

        self.piv.lastPL_price = pl_price
        self.piv.lastPL_osc = pl_osc
        self.piv.lastPL_time_ms = self.candles[pivot_idx].close_time_ms
        self.piv.lastPL_bar = pivot_idx
        return sig

    def _handle_pivot_high(self, pivot_idx: int, confirm_idx: int, allow_signals: bool, in_trading_hours: bool) -> Optional[Signal]:
        """Bearish divergence at upper Donchian extreme + scoring + SHORT chaos kill switch."""
        ph_price = self.candles[pivot_idx].high
        ph_osc = self.osc[pivot_idx]
        loc_p = self.loc[pivot_idx]

        near_upper = loc_p >= (1.0 - self.ext_band_pct)
        has_prev = self.piv.lastPH_price is not None and self.piv.lastPH_osc is not None
        bear_div = has_prev and (ph_price >= self.piv.lastPH_price) and (ph_osc < self.piv.lastPH_osc)

        sig: Optional[Signal] = None
        if near_upper and bear_div:
            self.total_div += 1
            if in_trading_hours:
                self.filtered_div += 1
                if allow_signals:
                    prev_price = float(self.piv.lastPH_price)
                    prev_osc = float(self.piv.lastPH_osc)

                    price_ch = float(pct_change(ph_price, prev_price) or 0.0)
                    osc_ch = 0.0
                    if abs(prev_osc) > 0:
                        osc_ch = ((prev_osc - ph_osc) / abs(prev_osc)) * 100.0

                    div_type = "Classic" if ph_price > prev_price else "Equal"
                    bars_gap = (confirm_idx - self.piv.lastPH_bar) if self.piv.lastPH_bar is not None else confirm_idx

                    loc_at_pivot_pct = loc_p * 100.0
                    vol_ratio_at_pivot = self.volRatio20[pivot_idx] if pivot_idx < len(self.volRatio20) else None
                    rsi_at_pivot = self.rsi14[pivot_idx] if pivot_idx < len(self.rsi14) else None
                    don_w_atr = self.don_width_atr[pivot_idx] if pivot_idx < len(self.don_width_atr) else None

                    blocked_by_chaos = bool(
                        self.ks_short_enabled
                        and (don_w_atr is None or don_w_atr > self.ks_don_width_atr_max)
                    )

                    score, breakdown = self._calc_short_score(
                        osc_change_pct=osc_ch,
                        loc_at_pivot_pct=loc_at_pivot_pct,
                        vol_ratio_at_pivot=vol_ratio_at_pivot,
                        rsi_at_pivot=rsi_at_pivot,
                        don_w_atr=don_w_atr,
                        div_type=div_type,
                        bars_gap=bars_gap,
                    )

                    is_high_prob = (score >= int(self.min_score_to_trade)) and (not blocked_by_chaos)
                    if is_high_prob:
                        sig = self._make_signal(
                            side="SHORT",
                            pivot_idx=pivot_idx,
                            confirm_idx=confirm_idx,
                            pivot_price=ph_price,
                            entry_price=self.candles[confirm_idx].close,
                            price_change_pct=price_ch,
                            osc_change_pct=osc_ch,
                            div_type=div_type,
                            slip_bps=((ph_price - self.candles[confirm_idx].close) / ph_price * 10000.0) if ph_price else 0.0,
                            structural_sl=prev_price,
                            prob_score=score,
                            bars_gap=bars_gap,
                            score_breakdown=breakdown,
                        )

        self.piv.lastPH_price = ph_price
        self.piv.lastPH_osc = ph_osc
        self.piv.lastPH_time_ms = self.candles[pivot_idx].close_time_ms
        self.piv.lastPH_bar = pivot_idx
        return sig

    def _make_tv_signal(
        self,
        *,
        side: str,
        pivot_idx: int,
        confirm_idx: int,
        entry_idx: int,
        cvd_now: Optional[float],
        cvd_thr: Optional[float],
        cvd_gate_pass: bool,
        entry_mode: str,
        osc_change_pct: float,
        near_lower: bool,
        bull_div: bool,
    ) -> Signal:
        pivot_c = self.candles[pivot_idx]
        confirm_c = self.candles[confirm_idx]
        entry_c = self.candles[entry_idx]

        pivot_price = pivot_c.low if side == "LONG" else pivot_c.high
        entry_price = entry_c.close
        price_change_pct = pct_change(entry_price, pivot_price) or 0.0

        prev_price = self.piv.lastPL_price if side == "LONG" else self.piv.lastPH_price
        if prev_price is not None:
            if side == "LONG":
                div_type = "Classic" if pivot_price < prev_price else "Equal"
            else:
                div_type = "Classic" if pivot_price > prev_price else "Equal"
        else:
            div_type = "Classic"

        prev_bar = self.piv.lastPL_bar if side == "LONG" else self.piv.lastPH_bar
        bars_gap = confirm_idx - prev_bar if prev_bar is not None else confirm_idx
        slip_bps = ((entry_price - pivot_price) / pivot_price * 10000.0) if pivot_price else 0.0
        confirm_ts = tv_confirm_ts(confirm_c.close_time_ms, self.use_close_minus_1ms)
        signal_id = self._signal_id(side, confirm_ts)

        extra = {
            "mode": "tv_parity",
            "entry_mode": entry_mode,
            "entry_wait_confirm": bool(self.entry_wait_confirm),
            "use_bos_confirmation": bool(self.use_bos_confirmation),
            "bos_atr_buffer": float(self.bos_atr_buffer),
            "max_wait_bars": int(self.max_wait_bars),
            "min_div_strength": float(self.min_div_strength),
            "longTrig": self.longTrig,
            "cvd_now": cvd_now,
            "cvd_thr": cvd_thr,
            "cvd_gate_pass": cvd_gate_pass,
            "oscChangePct": osc_change_pct,
            "nearLower": near_lower,
            "bullDiv": bull_div,
        }

        return Signal(
            symbol=self.symbol,
            timeframe=self.timeframe,
            side=side,
            pivot_time_ms=pivot_c.close_time_ms,
            confirm_time_ms=confirm_ts,
            pivot_price=float(pivot_price),
            entry_price=float(entry_price),
            don_loc_pct=float(self.loc[pivot_idx] * 100.0),
            price_change_pct=float(price_change_pct),
            osc_change_pct=float(osc_change_pct),
            div_type=div_type,
            slip_bps=float(slip_bps),
            structural_sl=None,
            structural_sl_distance_pct=None,
            prob_score=0,
            bars_gap=int(bars_gap),
            score_breakdown="",
            extra=extra,
            confirm_bar_close_ms=confirm_c.close_time_ms,
            confirm_bar_index=int(confirm_idx),
            entry_price_reference=float(entry_price),
            entry_intent="IMMEDIATE_ON_CLOSE",
            entry_mode="BOS_CLOSE_SAME_BAR",
            signal_id=signal_id,
        )

    def _make_signal(
        self,
        *,
        side: str,
        pivot_idx: int,
        confirm_idx: int,
        pivot_price: float,
        entry_price: float,
        price_change_pct: float,
        osc_change_pct: float,
        div_type: str,
        slip_bps: float,
        structural_sl: Optional[float],
        prob_score: int,
        bars_gap: int,
        score_breakdown: str,
    ) -> Signal:
        don_loc_pct = float(self.loc[pivot_idx] * 100.0)
        extra = self._extra_features(confirm_idx, pivot_idx)
        structural_sl_distance_pct: Optional[float] = None
        if structural_sl is not None and entry_price:
            if side == "LONG":
                diff = entry_price - structural_sl
            else:
                diff = structural_sl - entry_price
            if diff > 0:
                structural_sl_distance_pct = (diff / entry_price) * 100.0
        confirm_c = self.candles[confirm_idx]
        confirm_ts = confirm_c.close_time_ms
        signal_id = self._signal_id(side, confirm_ts)
        return Signal(
            symbol=self.symbol,
            timeframe=self.timeframe,
            side=side,
            pivot_time_ms=self.candles[pivot_idx].close_time_ms,
            confirm_time_ms=confirm_ts,
            pivot_price=float(pivot_price),
            entry_price=float(entry_price),
            don_loc_pct=don_loc_pct,
            price_change_pct=float(price_change_pct),
            osc_change_pct=float(osc_change_pct),
            div_type=div_type,
            slip_bps=float(slip_bps),
            structural_sl=structural_sl,
            structural_sl_distance_pct=structural_sl_distance_pct,
            prob_score=int(prob_score),
            bars_gap=int(bars_gap),
            score_breakdown=score_breakdown or "",
            extra=extra,
            confirm_bar_close_ms=confirm_c.close_time_ms,
            confirm_bar_index=int(confirm_idx),
            entry_price_reference=float(entry_price),
            entry_intent="IMMEDIATE_ON_CLOSE",
            entry_mode="BOS_CLOSE_SAME_BAR",
            signal_id=signal_id,
        )

    def _extra_features(self, confirm_idx: int, pivot_idx: int) -> Dict:
        # A compact feature bundle for Telegram debugging and parity checks.
        vol_ratio_confirm = self.volRatio20[confirm_idx] if confirm_idx < len(self.volRatio20) else None
        rsi_confirm = self.rsi14[confirm_idx] if confirm_idx < len(self.rsi14) else None
        atr14_confirm = self.atr14[confirm_idx] if confirm_idx < len(self.atr14) else None
        atr50_confirm = self.atr50[confirm_idx] if confirm_idx < len(self.atr50) else None
        atr_ratio = None
        if atr14_confirm is not None and atr50_confirm is not None and atr50_confirm != 0:
            atr_ratio = atr14_confirm / atr50_confirm

        ema20 = self.ema20_close[confirm_idx] if confirm_idx < len(self.ema20_close) else None
        ema50 = self.ema50_close[confirm_idx] if confirm_idx < len(self.ema50_close) else None
        trend_state = None
        if ema20 is not None and ema50 is not None:
            trend_state = "Bullish" if ema20 > ema50 else "Bearish"

        bullish_bars3 = 0
        bearish_bars3 = 0
        for k in range(1, 4):
            if confirm_idx - k < 0:
                continue
            if self.candles[confirm_idx - k].close > self.candles[confirm_idx - k].open:
                bullish_bars3 += 1
            elif self.candles[confirm_idx - k].close < self.candles[confirm_idx - k].open:
                bearish_bars3 += 1
        recent_momentum = "Neutral"
        if bullish_bars3 > bearish_bars3:
            recent_momentum = "Bullish"
        elif bearish_bars3 > bullish_bars3:
            recent_momentum = "Bearish"

        return {
            "volRatio20_confirm": vol_ratio_confirm,
            "rsi14_confirm": rsi_confirm,
            "atr14_confirm": atr14_confirm,
            "atr50_confirm": atr50_confirm,
            "atrRatio_confirm": atr_ratio,
            "trendState": trend_state,
            "recentMomentum": recent_momentum,

            # Pivot bar features (used in the Pine score)
            "volRatio20_pivot": self.volRatio20[pivot_idx] if pivot_idx < len(self.volRatio20) else None,
            "rsi14_pivot": self.rsi14[pivot_idx] if pivot_idx < len(self.rsi14) else None,
            "liq_percentile_pivot": self.liq_percentile[pivot_idx] if pivot_idx < len(self.liq_percentile) else None,
            "don_width_atr_pivot": self.don_width_atr[pivot_idx] if pivot_idx < len(self.don_width_atr) else None,

            "pivot_osc": self.osc[pivot_idx],
            "entry_osc": self.osc[confirm_idx],
            "don_hi": self.donHi[confirm_idx],
            "don_lo": self.donLo[confirm_idx],
        }
