from __future__ import annotations

from datetime import datetime, timezone, timedelta

from .models import Signal


EAT = timezone(timedelta(hours=3))  # Nairobi (UTC+3)


def _fmt_ms(ts_ms: int, tz=EAT) -> str:
    dt = datetime.fromtimestamp(ts_ms / 1000.0, tz=timezone.utc).astimezone(tz)
    return dt.strftime("%Y-%m-%d %H:%M")


def format_signal(
    signal: Signal,
    *,
    include_features: bool = True,
    include_score_breakdown: bool = True,
) -> str:
    side = signal.side.upper()
    is_long = side == "LONG"

    header = f"HIGH PROB {side} ({signal.prob_score}/10)"
    core = f"{signal.symbol} | {signal.timeframe}"

    t_confirm = _fmt_ms(signal.confirm_time_ms)
    t_pivot = _fmt_ms(signal.pivot_time_ms)

    lines = [
        header,
        core,
        f"Confirm: {t_confirm} EAT | Pivot: {t_pivot} EAT",
        f"Pivot: {signal.pivot_price:g} -> Entry: {signal.entry_price:g} | Slip: {signal.slip_bps:.1f} bps",
        f"Div: {signal.div_type} | Price%: {signal.price_change_pct:+.3f}% | Osc%: {signal.osc_change_pct:+.1f}% | Gap: {signal.bars_gap} bars",
        f"Donchian loc: {signal.don_loc_pct:.1f}% (lower=0, upper=100)",
    ]

    if include_features and signal.extra:
        vr = signal.extra.get("volRatio20_pivot")
        rsi = signal.extra.get("rsi14_pivot")
        liq = signal.extra.get("liq_percentile_pivot")
        wid = signal.extra.get("don_width_atr_pivot")

        if vr is not None:
            lines.append(f"VolRatio20@pivot: {vr:.2f}x")
        if rsi is not None:
            lines.append(f"RSI14@pivot: {rsi:.1f}")

        if is_long and liq is not None:
            lines.append(f"Liquidity percentile@pivot: {liq*100:.0f}")
        if (not is_long) and wid is not None:
            lines.append(f"Donchian width / ATR14@pivot: {wid:.2f}")

    if include_score_breakdown and signal.score_breakdown:
        lines.append("Score breakdown:")
        lines.append(signal.score_breakdown)

    return "\n".join(lines)
