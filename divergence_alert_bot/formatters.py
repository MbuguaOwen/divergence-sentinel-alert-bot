from __future__ import annotations

import html
from datetime import datetime, timezone, timedelta
from typing import Optional

from .models import Signal


EAT = timezone(timedelta(hours=3))  # Nairobi (UTC+3)


def _fmt_ms(ts_ms: int, tz=EAT) -> str:
    dt = datetime.fromtimestamp(ts_ms / 1000.0, tz=timezone.utc).astimezone(tz)
    return dt.strftime("%Y-%m-%d %H:%M")


def _escape_markdown_v2(text: str) -> str:
    specials = r"\_*[]()~`>#+-=|{}.!"
    escaped = []
    for ch in str(text):
        if ch in specials:
            escaped.append("\\" + ch)
        else:
            escaped.append(ch)
    return "".join(escaped)


def _escape_text(text: str, parse_mode: str) -> str:
    if parse_mode == "MARKDOWNV2":
        return _escape_markdown_v2(text)
    return html.escape(str(text), quote=False)


def _bold(text: str, parse_mode: str) -> str:
    escaped = _escape_text(text, parse_mode)
    if parse_mode == "MARKDOWNV2":
        return f"**{escaped}**"
    return f"<b>{escaped}</b>"


def _fmt_price(val: Optional[float]) -> str:
    if val is None:
        return "-"
    return f"{val:g}"


def _safe_structural_sl(signal: Signal, max_distance_pct: Optional[float]) -> tuple[Optional[float], Optional[float]]:
    price = signal.structural_sl
    distance = signal.structural_sl_distance_pct
    if price is None or price <= 0 or distance is None:
        return None, None
    if max_distance_pct is not None and max_distance_pct > 0 and distance > max_distance_pct:
        return None, None
    return price, distance


def _compact_notes(signal: Signal) -> str:
    extra = signal.extra or {}
    parts = [
        f"Div: {signal.div_type}",
        f"Loc: {signal.don_loc_pct:.1f}%",
        f"Gap: {signal.bars_gap} bars",
        f"Osc%: {signal.osc_change_pct:+.1f}%",
    ]
    rsi_pivot = extra.get("rsi14_pivot")
    if rsi_pivot is not None:
        parts.append(f"RSI@pivot: {rsi_pivot:.1f}")
    vol_ratio = extra.get("volRatio20_pivot")
    if vol_ratio is not None:
        parts.append(f"Vol@pivot: {vol_ratio:.2f}x")
    return " | ".join(parts)


def _format_legacy(signal: Signal, parse_mode: str, *, include_features: bool, include_score_breakdown: bool) -> str:
    side = signal.side.upper()
    header = f"HIGH PROB {side} ({signal.prob_score}/10)"
    core = f"{_escape_text(signal.symbol, parse_mode)} | {_bold(signal.timeframe, parse_mode)}"

    t_confirm = _fmt_ms(signal.confirm_time_ms)
    t_pivot = _fmt_ms(signal.pivot_time_ms)

    lines = [
        _escape_text(header, parse_mode),
        core,
        _escape_text(f"Confirm: {t_confirm} EAT | Pivot: {t_pivot} EAT", parse_mode),
        _escape_text(f"Pivot: {_fmt_price(signal.pivot_price)} -> Entry: {_fmt_price(signal.entry_price)} | Slip: {signal.slip_bps:.1f} bps", parse_mode),
        _escape_text(f"Div: {signal.div_type} | Price%: {signal.price_change_pct:+.3f}% | Osc%: {signal.osc_change_pct:+.1f}% | Gap: {signal.bars_gap} bars", parse_mode),
        _escape_text(f"Donchian loc: {signal.don_loc_pct:.1f}% (lower=0, upper=100)", parse_mode),
    ]

    if include_features and signal.extra:
        vr = signal.extra.get("volRatio20_pivot")
        rsi = signal.extra.get("rsi14_pivot")
        liq = signal.extra.get("liq_percentile_pivot")
        wid = signal.extra.get("don_width_atr_pivot")

        if vr is not None:
            lines.append(_escape_text(f"VolRatio20@pivot: {vr:.2f}x", parse_mode))
        if rsi is not None:
            lines.append(_escape_text(f"RSI14@pivot: {rsi:.1f}", parse_mode))

        if side == "LONG" and liq is not None:
            lines.append(_escape_text(f"Liquidity percentile@pivot: {liq*100:.0f}", parse_mode))
        if side == "SHORT" and wid is not None:
            lines.append(_escape_text(f"Donchian width / ATR14@pivot: {wid:.2f}", parse_mode))

    if include_score_breakdown and signal.score_breakdown:
        lines.append(_escape_text("Score breakdown:", parse_mode))
        lines.append(_escape_text(signal.score_breakdown, parse_mode))

    return "\n".join(lines)


def _format_corporate(signal: Signal, parse_mode: str, detail_level: str, cfg) -> str:
    is_public = detail_level == "public"

    pipe = "\\|" if parse_mode == "MARKDOWNV2" else "|"
    lines = [
        f"{_bold(signal.symbol, parse_mode)}  {pipe}  {_bold(signal.timeframe, parse_mode)}",
        f"{_bold(f'HIGH PROB {signal.side.upper()}', parse_mode)} • Score: {_bold(f'{signal.prob_score}/10', parse_mode)}",
        "",
    ]

    lines.append(_escape_text(f"Time (EAT): {_fmt_ms(signal.confirm_time_ms)}", parse_mode))

    if getattr(cfg, "include_entry_reference", True):
        lines.append(_escape_text(f"Entry Reference: {_fmt_price(signal.entry_price)}", parse_mode))

    sl_price, sl_distance = _safe_structural_sl(signal, getattr(cfg, "structural_sl_max_distance_pct", None))
    if getattr(cfg, "include_structural_sl", True) and sl_price is not None:
        sl_line = f"Structural SL: {_fmt_price(sl_price)}"
        if sl_distance is not None:
            sl_line += f" ({sl_distance:.1f}% from entry)"
        lines.append(_escape_text(sl_line, parse_mode))

    if not is_public:
        if getattr(cfg, "include_pivot_reference", True):
            lines.append(_escape_text(f"Pivot Reference: {_fmt_price(signal.pivot_price)}", parse_mode))
        if getattr(cfg, "include_slippage_bps", True):
            lines.append(_escape_text(f"Slippage: {signal.slip_bps:.1f} bps", parse_mode))
        if getattr(cfg, "include_features", True):
            notes = _compact_notes(signal)
            if notes:
                lines.append(_escape_text(f"Notes: {notes}", parse_mode))
        if getattr(cfg, "include_score_breakdown", True) and signal.score_breakdown:
            lines.append(_escape_text("Score breakdown:", parse_mode))
            lines.append(_escape_text(signal.score_breakdown, parse_mode))

    footer = (getattr(cfg, "footer", "") or "").strip()
    if footer:
        lines.append("")
        lines.append(_escape_text(footer, parse_mode))

    return "\n".join(lines)


def _format_tv_parity(signal: Signal, parse_mode: str, cfg) -> str:
    extra = signal.extra or {}
    side = signal.side.upper()
    entry_mode = extra.get("entry_mode") or ("CONFIRM" if extra.get("entry_wait_confirm") else "RAW")
    header = f"TV PARITY {side} {entry_mode}"

    lines = [
        _escape_text(header, parse_mode),
        f"{_escape_text(signal.symbol, parse_mode)} | {_bold(signal.timeframe, parse_mode)}",
        _escape_text(f"Confirm: {_fmt_ms(signal.confirm_time_ms)}", parse_mode),
        _escape_text(f"Entry: {_fmt_price(signal.entry_price)} | Pivot: {_fmt_price(signal.pivot_price)}", parse_mode),
    ]

    cvd_now = extra.get("cvd_now")
    cvd_thr = extra.get("cvd_thr")
    if cvd_now is not None or cvd_thr is not None:
        gate_pass = extra.get("cvd_gate_pass")
        lines.append(_escape_text(f"CVD: {cvd_now} / Thr: {cvd_thr} | Pass: {gate_pass}", parse_mode))

    lines.append(_escape_text(f"Osc%: {signal.osc_change_pct:+.1f}% | Don loc: {signal.don_loc_pct:.1f}%", parse_mode))
    lines.append(_escape_text(f"Trigger: BOS={extra.get('use_bos_confirmation')} buf={extra.get('bos_atr_buffer')} wait={extra.get('max_wait_bars')} longTrig={extra.get('longTrig')}", parse_mode))

    if getattr(cfg, "include_structural_sl", True):
        sl_price, sl_distance = _safe_structural_sl(signal, getattr(cfg, "structural_sl_max_distance_pct", None))
        if sl_price is not None:
            sl_line = f"Structural SL: {_fmt_price(sl_price)}"
            if sl_distance is not None:
                sl_line += f" ({sl_distance:.1f}% from entry)"
            lines.append(_escape_text(sl_line, parse_mode))

    footer = (getattr(cfg, "footer", "") or "").strip()
    if footer:
        lines.append("")
        lines.append(_escape_text(footer, parse_mode))

    return "\n".join(lines)


def format_signal(signal: Signal, cfg, *, detail_level: Optional[str] = None) -> str:
    """Format a signal for Telegram alerts using the requested style/detail level."""
    if (signal.extra or {}).get("mode") == "tv_parity":
        parse_mode = (getattr(cfg, "parse_mode", "HTML") or "HTML").upper()
        return _format_tv_parity(signal, parse_mode, cfg)

    style = (getattr(cfg, "style", "corporate") or "corporate").lower()
    parse_mode = (getattr(cfg, "parse_mode", "HTML") or "HTML").upper()
    chosen_detail = (detail_level or getattr(cfg, "detail_level", "public") or "public").lower()

    include_features = getattr(cfg, "include_features", True)
    include_score_breakdown = getattr(cfg, "include_score_breakdown", True)

    if style != "corporate":
        return _format_legacy(
            signal,
            parse_mode,
            include_features=include_features and chosen_detail != "public",
            include_score_breakdown=include_score_breakdown and chosen_detail != "public",
        )

    return _format_corporate(signal, parse_mode, chosen_detail, cfg)
