from __future__ import annotations

from divergence_alert_bot.strategy import StrategyEngine
from divergence_alert_bot.models import Candle
from divergence_alert_bot.indicators import percentile_linear_interpolation


def candle(idx: int, open_p: float, high: float, low: float, close: float, vol: float = 1.0) -> Candle:
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


def divergence_sequence():
    """Sequence that creates a previous pivot then a bullish divergence pivot."""
    return [
        candle(0, 10, 11, 9, 9, 1),
        candle(1, 10, 10, 8, 8, 1),  # pivot 1 (confirmed at idx2)
        candle(2, 10, 11, 9, 9, 1),
        candle(3, 10, 10, 7, 10, 2),  # pivot 2 (confirmed at idx4)
        candle(4, 10, 11, 8, 11, 2),
    ]


def run_case(name: str, eng: StrategyEngine, cvd_values):
    sigs = []
    seq = divergence_sequence()
    for c, cvd in zip(seq, cvd_values):
        sigs.extend(eng.on_candle(c, allow_signals=True, in_trading_hours=True, cvd_now=cvd))
    print(f"{name}: signals={len(sigs)}", [s.confirm_time_ms for s in sigs])


def main():
    print("Percentile check (75% of [1,2,3,4]) =", percentile_linear_interpolation([1, 2, 3, 4], 4, 75))

    base_kwargs = dict(
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
        cvd_threshold=1.0,
    )

    # Case 1: Same-bar confirm (entry_wait_confirm True, trigger on creation bar)
    eng1 = StrategyEngine(**base_kwargs, entry_wait_confirm=True, use_bos_confirmation=False, max_wait_bars=1, cooldown_bars=0)
    run_case("same_bar_confirm", eng1, cvd_values=[2, 2, 2, 2, 2])

    # Case 2: Confirm within window (trigger on next bar)
    eng2 = StrategyEngine(**base_kwargs, entry_wait_confirm=True, use_bos_confirmation=False, max_wait_bars=2, cooldown_bars=0)
    run_case("confirm_next_bar", eng2, cvd_values=[2, 2, 2, 2, 2])

    # Case 3: Window expires, no signal
    eng3 = StrategyEngine(**base_kwargs, entry_wait_confirm=True, use_bos_confirmation=False, max_wait_bars=0, cooldown_bars=0)
    run_case("confirm_expired", eng3, cvd_values=[2, 2, 2, 2, 2])

    # Case 4: Cooldown blocks second entry
    eng4 = StrategyEngine(**base_kwargs, entry_wait_confirm=False, cooldown_bars=10, use_bos_confirmation=False)
    seq = divergence_sequence() + divergence_sequence()  # two divergences back-to-back
    sigs = []
    for i, c in enumerate(seq):
        cvd = 2
        sigs.extend(eng4.on_candle(c, allow_signals=True, in_trading_hours=True, cvd_now=cvd))
    print("cooldown_block", len(sigs), [s.confirm_time_ms for s in sigs])


if __name__ == "__main__":
    main()
