def tv_confirm_ts(close_time_ms: int, use_close_minus_1ms: bool) -> int:
    """TradingView parity timestamp helper.

    Binance close_time_ms is *typically* end-of-bar minus 1ms.
    Pine can either use time_close - 1ms (useCloseMinus1ms=true) to match Binance,
    or raw time_close (useCloseMinus1ms=false).
    """
    return close_time_ms if use_close_minus_1ms else close_time_ms + 1


def canonical_close_ms(ms: int) -> int:
    """Canonicalize a kline close timestamp.

    Some feeds occasionally emit boundary-style timestamps (e.g., exactly on the minute)
    instead of the usual end-of-bar minus 1ms. This tiny drift can break CVD alignment
    and cause candle de-duplication/out-of-order checks to diverge between processes.

    We normalize to the end-of-minute minus 1ms convention.
    """
    ms = int(ms)
    mod = ms % 60_000
    if mod == 59_999:
        return ms          # already end-1ms (correct)
    if mod == 0:
        return ms - 1      # boundary -> end-1ms
    if mod == 59_998:
        return ms + 1      # rare drift -> normalize
    return ms
