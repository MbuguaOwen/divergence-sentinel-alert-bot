def tv_confirm_ts(close_time_ms: int, use_close_minus_1ms: bool) -> int:
    """TradingView parity timestamp helper.

    Binance close_time_ms is already time_close - 1. Pine uses time_close - 1 when useCloseMinus1ms=true.
    """
    return close_time_ms if use_close_minus_1ms else close_time_ms + 1
