from __future__ import annotations

import logging
from typing import Dict

from .config import StrategyConfig

log = logging.getLogger("parity")

PINE_DEFAULTS_REFERENCE: Dict[str, object] = {
    "don_len": 120,
    "pivot_len": 5,
    "osc_len": 14,
    "ext_band_pct": 0.15,
    "long_only": True,
    "trade_enabled": True,
    "entry_wait_confirm": True,
    "use_bos_confirm": True,
    "bos_atr_buffer": 0.10,
    "max_wait_bars": 30,
    "min_div_strength": 15.0,
    "cooldown_bars": 0,
    "use_cvd_gate": True,
    "cvd_len_min": 60,
    "use_dynamic_cvd_pct": True,
    "cvd_lookback_bars": 2880,
    "cvd_pct": 75,
    "cvd_threshold": 244.075,
    "use_close_minus_1ms": True,
}


def parity_signature(cfg: StrategyConfig) -> Dict[str, object]:
    return {
        "don_len": cfg.don_len,
        "pivot_len": cfg.pivot_len,
        "osc_len": cfg.osc_len,
        "ext_band_pct": cfg.ext_band_pct,
        "long_only": cfg.long_only,
        "trade_enabled": cfg.trade_enabled,
        "entry_wait_confirm": cfg.entry_wait_confirm,
        "use_bos_confirm": cfg.use_bos_confirm,
        "bos_atr_buffer": cfg.bos_atr_buffer,
        "max_wait_bars": cfg.max_wait_bars,
        "min_div_strength": cfg.min_div_strength,
        "cooldown_bars": cfg.cooldown_bars,
        "use_cvd_gate": cfg.use_cvd_gate,
        "cvd_len_min": cfg.cvd_len_min,
        "use_dynamic_cvd_pct": cfg.use_dynamic_cvd_pct,
        "cvd_lookback_bars": cfg.cvd_lookback_bars,
        "cvd_pct": cfg.cvd_pct,
        "cvd_threshold": cfg.cvd_threshold,
        "use_close_minus_1ms": cfg.use_close_minus_1ms,
    }


def assert_tv_parity_inputs(cfg: StrategyConfig) -> None:
    sig = parity_signature(cfg)
    mismatches = []
    for k, v in PINE_DEFAULTS_REFERENCE.items():
        if sig.get(k) != v:
            mismatches.append(f"{k}: cfg={sig.get(k)} expected={v}")
    if mismatches:
        raise ValueError("tv_parity inputs mismatch Pine defaults: " + "; ".join(mismatches))


def log_parity_signature(cfg: StrategyConfig) -> None:
    sig = parity_signature(cfg)
    log.info("TV PARITY INPUTS (parity_strict=%s): %s", getattr(cfg, "parity_strict", False), sig)
    if cfg.mode == "tv_parity" and not getattr(cfg, "parity_strict", False):
        log.warning("parity_strict=false; your config may not match Pine inputs; parity not guaranteed.")
