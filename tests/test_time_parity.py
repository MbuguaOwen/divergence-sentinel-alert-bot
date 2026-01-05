from divergence_alert_bot.strategy import StrategyEngine
from divergence_alert_bot.models import Candle
from divergence_alert_bot.time_parity import tv_confirm_ts


def _c(idx: int, o: float, h: float, l: float, c: float, v: float = 1.0) -> Candle:
    base = idx * 60_000
    return Candle(
        open_time_ms=base,
        close_time_ms=base + 60_000 - 1,
        open=o,
        high=h,
        low=l,
        close=c,
        volume=v,
    )


def test_tv_confirm_ts_helper():
    ct = 1_000
    assert tv_confirm_ts(ct, True) == ct
    assert tv_confirm_ts(ct, False) == ct + 1


def test_confirm_mode_without_bos_confirm_allows_trigger_without_atr():
    eng = StrategyEngine(
        symbol="BTCUSDT",
        timeframe="15m",
        don_len=2,
        pivot_len=1,
        osc_len=2,
        ext_band_pct=1.0,
        mode="tv_parity",
        use_cvd_gate=False,
        use_bos_confirm=False,  # should allow trig without ATR
        entry_wait_confirm=True,
        max_wait_bars=2,
        min_div_strength=0.0,
    )

    candles = [
        _c(0, 10, 11, 9, 9, 1),
        _c(1, 10, 10, 8, 8, 1),  # pivot 1 memory
        _c(2, 10, 11, 9, 9, 1),
        _c(3, 8, 10, 7, 10, 2),  # divergence pivot
        _c(4, 9, 10, 8, 9.5, 1),  # confirm bar, close>open
    ]

    signals = []
    for c in candles:
        signals.extend(eng.on_candle(c, allow_signals=True, in_trading_hours=True))

    assert any(sig.extra.get("mode") == "tv_parity" for sig in signals)
