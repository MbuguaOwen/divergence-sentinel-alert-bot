from divergence_alert_bot.strategy import StrategyEngine
from divergence_alert_bot.models import Candle
from divergence_alert_bot.indicators import percentile_linear_interpolation


def _candle(idx: int, open_p: float, high: float, low: float, close: float, vol: float) -> Candle:
    base = idx * 60_000
    return Candle(
        open_time_ms=base,
        close_time_ms=base + 60_000 - 1,
        open=open_p,
        high=high,
        low=low,
        close=close,
        volume=vol,
    )


def test_percentile_helper():
    vals = [1, 2, 3, 4]
    assert percentile_linear_interpolation(vals, 4, 50) == 2.5
    assert percentile_linear_interpolation(vals, 4, 75) == 3.25
    assert percentile_linear_interpolation(vals, 4, 25) == 1.75


def test_tv_parity_raw_mode_emits_and_respects_cooldown():
    eng = StrategyEngine(
        symbol="BTCUSDT",
        timeframe="15m",
        don_len=3,
        pivot_len=1,
        osc_len=2,
        ext_band_pct=1.0,
        mode="tv_parity",
        min_div_strength=0.0,
        entry_wait_confirm=False,
        cooldown_bars=3,
        use_cvd_gate=False,
    )

    candles = [
        _candle(0, 10, 11, 9, 9, 1),
        _candle(1, 10, 10, 8, 8, 1),  # first pivot sets memory
        _candle(2, 10, 11, 9, 9, 1),  # confirm for first pivot
        _candle(3, 8, 10, 7, 10, 2),  # second pivot bar (bullish osc)
        _candle(4, 9, 10, 8, 9, 1),   # confirm -> should signal
        _candle(5, 8, 10, 7, 10, 2),  # third pivot bar, within cooldown window
        _candle(6, 9, 10, 8, 9, 1),   # confirm, should be blocked by cooldown
    ]

    signals = []
    for c in candles:
        signals.extend(eng.on_candle(c, allow_signals=True, in_trading_hours=True))

    assert len(signals) == 1
    sig = signals[0]
    assert sig.side == "LONG"
    assert (sig.extra or {}).get("mode") == "tv_parity"


def test_tv_parity_confirm_mode_triggers_and_cancels():
    eng = StrategyEngine(
        symbol="BTCUSDT",
        timeframe="15m",
        don_len=3,
        pivot_len=1,
        osc_len=2,
        ext_band_pct=1.0,
        mode="tv_parity",
        min_div_strength=0.0,
        entry_wait_confirm=True,
        use_bos_confirm=False,
        max_wait_bars=1,
        use_cvd_gate=False,
    )

    # First divergence -> should trigger on next bar (close > open)
    seq1 = [
        _candle(0, 10, 11, 9, 9, 1),
        _candle(1, 10, 10, 8, 8, 1),
        _candle(2, 10, 11, 9, 9, 1),
        _candle(3, 8, 10, 7, 10, 2),
        _candle(4, 9, 10, 8, 11, 1),  # trigger bar (close>open)
    ]
    signals = []
    for c in seq1:
        signals.extend(eng.on_candle(c, allow_signals=True, in_trading_hours=True))
    assert any((s.extra or {}).get("mode") == "tv_parity" for s in signals)

    # Second divergence -> no trigger within window, should cancel
    eng = StrategyEngine(
        symbol="BTCUSDT",
        timeframe="15m",
        don_len=3,
        pivot_len=1,
        osc_len=2,
        ext_band_pct=1.0,
        mode="tv_parity",
        min_div_strength=0.0,
        entry_wait_confirm=True,
        use_bos_confirm=False,
        max_wait_bars=0,
        use_cvd_gate=False,
    )
    seq2 = [
        _candle(0, 10, 11, 9, 9, 1),
        _candle(1, 10, 10, 8, 8, 1),
        _candle(2, 10, 11, 9, 9, 1),
        _candle(3, 8, 10, 7, 10, 2),
        _candle(4, 9, 10, 8, 9, 1),  # trigger bar (close==open -> fail), window expires
    ]
    signals2 = []
    for c in seq2:
        signals2.extend(eng.on_candle(c, allow_signals=True, in_trading_hours=True))
    assert len(signals2) == 0
