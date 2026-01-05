from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional
import os
import yaml


def _env_override(value: Any, env_key: str) -> Any:
    env_val = os.getenv(env_key)
    if env_val is None:
        return value
    # basic parsing
    if isinstance(value, bool):
        return env_val.strip().lower() in ("1", "true", "yes", "y", "on")
    if isinstance(value, int):
        try:
            return int(env_val)
        except ValueError:
            return value
    if isinstance(value, float):
        try:
            return float(env_val)
        except ValueError:
            return value
    return env_val


@dataclass
class TradingHours:
    enabled: bool = True
    timezone: str = "UTC+3"
    start_hour: int = 8
    end_hour: int = 1
    days: Optional[List[int]] = None  # 0=Mon .. 6=Sun


@dataclass
class StrategyConfig:
    # Mode selection
    mode: str = "sniper"  # sniper | tv_parity
    long_only: bool = True
    trade_enabled: bool = True
    parity_strict: bool = False

    # TV parity / entry engine toggles
    entry_wait_confirm: bool = True
    use_bos_confirm: bool = True
    bos_atr_buffer: float = 0.10
    max_wait_bars: int = 30
    min_div_strength: float = 15.0
    cooldown_bars: int = 0

    # TV parity CVD gate
    use_cvd_gate: bool = True
    cvd_len_min: int = 60  # minutes
    use_dynamic_cvd_pct: bool = True
    cvd_lookback_bars: int = 2880
    cvd_pct: int = 75
    cvd_threshold: float = 244.075
    use_close_minus_1ms: bool = True

    # Shared defaults (legacy sniper)
    don_len: int = 120
    pivot_len: int = 5
    osc_len: int = 14
    ext_band_pct: float = 0.15

    # High Probability Sniper inputs (Pine parity)
    ks_short_enabled: bool = True
    ks_don_width_atr_max: float = 16.7

    ks_long_enabled: bool = True
    ks_liq_percentile_min: float = 0.25
    ks_liq_lookback: int = 100

    min_score_to_trade: int = 6

    def parity_signature(self) -> Dict[str, object]:
        return {
            "don_len": self.don_len,
            "pivot_len": self.pivot_len,
            "osc_len": self.osc_len,
            "ext_band_pct": self.ext_band_pct,
            "long_only": self.long_only,
            "trade_enabled": self.trade_enabled,
            "entry_wait_confirm": self.entry_wait_confirm,
            "use_bos_confirm": self.use_bos_confirm,
            "bos_atr_buffer": self.bos_atr_buffer,
            "max_wait_bars": self.max_wait_bars,
            "min_div_strength": self.min_div_strength,
            "cooldown_bars": self.cooldown_bars,
            "use_cvd_gate": self.use_cvd_gate,
            "cvd_len_min": self.cvd_len_min,
            "use_dynamic_cvd_pct": self.use_dynamic_cvd_pct,
            "cvd_lookback_bars": self.cvd_lookback_bars,
            "cvd_pct": self.cvd_pct,
            "cvd_threshold": self.cvd_threshold,
            "use_close_minus_1ms": self.use_close_minus_1ms,
        }


@dataclass
class ProviderConfig:
    type: str = "binance"
    market: str = "futures"  # futures|spot
    symbols: List[str] = None
    timeframes: List[str] = None
    warmup_candles: int = 350
    rest_timeout_s: int = 20
    ws_heartbeat_s: int = 20
    warmup_concurrency: int = 5


@dataclass
class TelegramConfig:
    enabled: bool = True
    token: str = ""
    chat_ids: List[str] = None
    public_chat_ids: List[str] = None
    private_chat_ids: List[str] = None
    disable_web_page_preview: bool = True


@dataclass
class WebhookConfig:
    enabled: bool = False
    url: str = ""
    secret: str = ""
    include_tf: bool = True
    timeout_s: int = 10
    headers: Dict[str, str] = None


@dataclass
class AlertsConfig:
    parse_mode: str = "HTML"  # HTML | MarkdownV2
    style: str = "corporate"
    detail_level: str = "public"  # public | internal
    footer: str = ""
    include_entry_reference: bool = True
    include_pivot_reference: bool = True
    include_slippage_bps: bool = True
    include_structural_sl: bool = True
    structural_sl_max_distance_pct: float = 8.0  # percent distance from entry
    include_features: bool = True
    include_score_breakdown: bool = True
    dedupe: bool = True


@dataclass
class AppConfig:
    name: str = "Divergence Sentinel"
    log_level: str = "INFO"


@dataclass
class Config:
    app: AppConfig
    provider: ProviderConfig
    strategy: StrategyConfig
    trading_hours: TradingHours
    telegram: TelegramConfig
    webhook: WebhookConfig
    alerts: AlertsConfig


def load_config(path: str) -> Config:
    with open(path, "r", encoding="utf-8") as f:
        raw = yaml.safe_load(f) or {}

    app = raw.get("app", {})
    provider = raw.get("provider", {})
    strategy = raw.get("strategy", {})
    th = raw.get("trading_hours", {})
    tg = raw.get("telegram", {})
    wh = raw.get("webhook", {})
    alerts = raw.get("alerts", {})

    cfg = Config(
        app=AppConfig(**app),
        provider=ProviderConfig(**provider),
        strategy=StrategyConfig(**strategy),
        trading_hours=TradingHours(**th),
        telegram=TelegramConfig(**tg),
        webhook=WebhookConfig(**wh),
        alerts=AlertsConfig(**alerts),
    )

    # env overrides (useful on servers)
    cfg.telegram.token = _env_override(cfg.telegram.token, "TELEGRAM_TOKEN")
    if cfg.telegram.chat_ids is None:
        cfg.telegram.chat_ids = []
    if cfg.telegram.public_chat_ids is None:
        cfg.telegram.public_chat_ids = []
    if cfg.telegram.private_chat_ids is None:
        cfg.telegram.private_chat_ids = []

    # Allow TELEGRAM_CHAT_IDS="id1,id2"
    chat_env = os.getenv("TELEGRAM_CHAT_IDS")
    if chat_env:
        cfg.telegram.chat_ids = [x.strip() for x in chat_env.split(",") if x.strip()]

    public_env = os.getenv("TELEGRAM_PUBLIC_CHAT_IDS")
    if public_env:
        cfg.telegram.public_chat_ids = [x.strip() for x in public_env.split(",") if x.strip()]

    private_env = os.getenv("TELEGRAM_PRIVATE_CHAT_IDS")
    if private_env:
        cfg.telegram.private_chat_ids = [x.strip() for x in private_env.split(",") if x.strip()]

    # Webhook optional env override for secret/url
    cfg.webhook.secret = _env_override(cfg.webhook.secret, "WEBHOOK_SECRET")
    cfg.webhook.url = _env_override(cfg.webhook.url, "WEBHOOK_URL")
    if cfg.webhook.headers is None:
        cfg.webhook.headers = {}

    return cfg
