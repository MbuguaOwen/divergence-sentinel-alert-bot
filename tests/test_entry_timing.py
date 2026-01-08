import asyncio

from divergence_alert_bot.config import (
    AlertsConfig,
    AppConfig,
    Config,
    ProviderConfig,
    StrategyConfig,
    TelegramConfig,
    TradingHours,
    WebhookConfig,
)
from divergence_alert_bot.models import Candle, Signal
from divergence_alert_bot.runner import AlertRunner
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


def test_bos_same_bar_entry_long():
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
        use_bos_confirm=True,
        bos_atr_buffer=0.0,
        max_wait_bars=5,
        use_cvd_gate=False,
        use_close_minus_1ms=True,
    )

    candles = [_c(i, 10, 11, 9, 10, 1) for i in range(14)]
    candles.extend(
        [
            _c(14, 10, 10, 8, 9, 1),
            _c(15, 9, 11, 9, 10, 1),
            _c(16, 9, 12, 7, 11, 5),
            _c(17, 10, 11, 9, 10, 1),
            _c(18, 11, 13, 10, 12, 1),
        ]
    )

    signal_idx = None
    captured = []
    for i, c in enumerate(candles):
        out = eng.on_candle(c, allow_signals=True, in_trading_hours=True)
        if out and signal_idx is None:
            signal_idx = i
            captured.extend(out)

    assert captured
    sig = captured[0]
    assert signal_idx == 18
    assert sig.confirm_bar_close_ms == candles[18].close_time_ms
    assert sig.confirm_time_ms == candles[18].close_time_ms
    assert sig.confirm_bar_index == 18
    assert sig.entry_intent == "IMMEDIATE_ON_CLOSE"
    assert sig.entry_mode == "BOS_CLOSE_SAME_BAR"


def test_no_next_bar_delay():
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
        max_wait_bars=2,
        use_cvd_gate=False,
        use_close_minus_1ms=True,
    )

    candles = [
        _c(0, 10, 11, 9, 9, 1),
        _c(1, 10, 10, 8, 8, 1),
        _c(2, 10, 11, 9, 9, 1),
        _c(3, 8, 10, 7, 10, 2),
        _c(4, 9, 10, 8, 11, 1),
    ]

    signal_idx = None
    captured = []
    for i, c in enumerate(candles):
        out = eng.on_candle(c, allow_signals=True, in_trading_hours=True)
        if out and signal_idx is None:
            signal_idx = i
            captured.extend(out)

    assert captured
    sig = captured[0]
    assert signal_idx == 4
    assert sig.confirm_bar_index == 4
    assert sig.confirm_time_ms == candles[4].close_time_ms


def test_dedupe_same_signal():
    async def _run():
        cfg = Config(
            app=AppConfig(),
            provider=ProviderConfig(symbols=["BTCUSDT"], timeframes=["15m"]),
            strategy=StrategyConfig(use_cvd_gate=False),
            trading_hours=TradingHours(enabled=False),
            telegram=TelegramConfig(enabled=False, token="", chat_ids=[]),
            webhook=WebhookConfig(enabled=False),
            alerts=AlertsConfig(dedupe=True),
        )
        runner = AlertRunner(cfg)
        sig = Signal(
            symbol="BTCUSDT",
            timeframe="15m",
            side="LONG",
            pivot_time_ms=0,
            confirm_time_ms=1,
            pivot_price=1.0,
            entry_price=1.0,
            don_loc_pct=0.0,
            price_change_pct=0.0,
            osc_change_pct=0.0,
            div_type="Classic",
            slip_bps=0.0,
            structural_sl=None,
            structural_sl_distance_pct=None,
            prob_score=0,
            bars_gap=0,
            score_breakdown="",
            extra={},
            confirm_bar_close_ms=1,
            confirm_bar_index=0,
            entry_price_reference=1.0,
            entry_intent="IMMEDIATE_ON_CLOSE",
            signal_id="BTCUSDT:15m:LONG:1:test",
        )
        await runner._handle_signal(sig)
        await runner._handle_signal(sig)
        assert runner._metrics["bos_same_bar_entries_total"] == 1
        assert runner._metrics["bos_same_bar_reject_total"] == 1
        assert len(runner._dedupe) == 1

    asyncio.run(_run())
