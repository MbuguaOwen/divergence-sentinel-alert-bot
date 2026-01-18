from divergence_alert_bot.models import Candle
from divergence_alert_bot.strategy import StrategyEngine


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


def test_default_no_bos_waits_for_cvd_then_enters():
    eng = StrategyEngine(
        symbol="BTCUSDT",
        timeframe="15m",
        don_len=3,
        pivot_len=1,
        osc_len=2,
        ext_band_pct=1.0,
        mode="tv_parity",
        min_div_strength=0.0,
        use_cvd_gate=True,
        use_dynamic_cvd_pct=False,
        cvd_threshold=10.0,
        max_wait_bars=3,
    )
    assert eng.use_bos_confirmation is False

    candles = [
        _c(0, 10, 11, 9, 9, 1),
        _c(1, 10, 10, 8, 8, 1),
        _c(2, 10, 11, 9, 9, 1),
        _c(3, 8, 10, 7, 10, 2),
        _c(4, 9, 10, 8, 9, 1),   # setup created here
        _c(5, 9, 9.5, 8.5, 9, 1),  # CVD passes here, BOS still false
    ]
    cvd_vals = [0, 0, 0, 0, 0, 11]

    signals = []
    for c, cvd in zip(candles, cvd_vals):
        signals.extend(eng.on_candle(c, allow_signals=True, in_trading_hours=True, cvd_now=cvd))

    assert len(signals) == 1
    assert signals[0].confirm_bar_index == 5


def test_bos_confirmation_blocks_until_bos_true():
    warmup = [_c(i, 10, 10, 10, 10, 1) for i in range(14)]
    candles = warmup + [
        _c(14, 10, 10, 8, 8, 1),   # pivot 1
        _c(15, 10, 11, 9, 9, 1),
        _c(16, 8, 10, 7, 10, 2),   # pivot 2 (bull div)
        _c(17, 9, 10, 8, 9, 1),    # setup created here
        _c(18, 9, 9.5, 8.5, 9, 1),  # CVD passes, BOS still false
        _c(19, 10.2, 10.8, 10.1, 10.6, 1),  # BOS true
    ]
    cvd_vals = [0] * 18 + [10, 10]

    eng = StrategyEngine(
        symbol="BTCUSDT",
        timeframe="15m",
        don_len=3,
        pivot_len=1,
        osc_len=2,
        ext_band_pct=1.0,
        mode="tv_parity",
        min_div_strength=0.0,
        use_bos_confirmation=True,
        bos_atr_buffer=0.0,
        use_cvd_gate=True,
        use_dynamic_cvd_pct=False,
        cvd_threshold=5.0,
        max_wait_bars=5,
    )

    signals = []
    for c, cvd in zip(candles, cvd_vals):
        signals.extend(eng.on_candle(c, allow_signals=True, in_trading_hours=True, cvd_now=cvd))

    assert len(signals) == 1
    assert signals[0].confirm_bar_index == 19


def test_setup_expires_without_cvd():
    eng = StrategyEngine(
        symbol="BTCUSDT",
        timeframe="15m",
        don_len=3,
        pivot_len=1,
        osc_len=2,
        ext_band_pct=1.0,
        mode="tv_parity",
        min_div_strength=0.0,
        use_cvd_gate=True,
        use_dynamic_cvd_pct=False,
        cvd_threshold=10.0,
        max_wait_bars=1,
    )

    candles = [
        _c(0, 10, 11, 9, 9, 1),
        _c(1, 10, 10, 8, 8, 1),
        _c(2, 10, 11, 9, 9, 1),
        _c(3, 8, 10, 7, 10, 2),
        _c(4, 9, 10, 8, 9, 1),   # setup created here
        _c(5, 9, 10, 8.5, 9.2, 1),
        _c(6, 9, 10, 8.5, 9.1, 1),  # after expiry
    ]
    cvd_vals = [0, 0, 0, 0, 0, 0, 20]

    signals = []
    for c, cvd in zip(candles, cvd_vals):
        signals.extend(eng.on_candle(c, allow_signals=True, in_trading_hours=True, cvd_now=cvd))

    assert signals == []


def test_signals_only_emit_when_allowed():
    eng = StrategyEngine(
        symbol="BTCUSDT",
        timeframe="15m",
        don_len=3,
        pivot_len=1,
        osc_len=2,
        ext_band_pct=1.0,
        mode="tv_parity",
        min_div_strength=0.0,
        use_cvd_gate=True,
        use_dynamic_cvd_pct=False,
        cvd_threshold=10.0,
        max_wait_bars=3,
    )

    candles = [
        _c(0, 10, 11, 9, 9, 1),
        _c(1, 10, 10, 8, 8, 1),
        _c(2, 10, 11, 9, 9, 1),
        _c(3, 8, 10, 7, 10, 2),
        _c(4, 9, 10, 8, 9, 1),   # setup created here
        _c(5, 9, 9.5, 8.5, 9, 1),  # CVD passes here
        _c(6, 9, 9.5, 8.5, 9, 1),  # CVD passes here
    ]
    cvd_vals = [0, 0, 0, 0, 0, 11, 11]

    signals = []
    for idx, (c, cvd) in enumerate(zip(candles, cvd_vals)):
        allow = idx != 5
        signals.extend(eng.on_candle(c, allow_signals=allow, in_trading_hours=True, cvd_now=cvd))

    assert len(signals) == 1
    assert signals[0].confirm_bar_index == 6
