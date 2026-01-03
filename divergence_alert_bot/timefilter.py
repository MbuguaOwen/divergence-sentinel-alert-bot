from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone, timedelta
from typing import Optional, List
import re


_TZ_RE = re.compile(r"^UTC([+-])(\d{1,2})$")


def _parse_tz(tz_str: str) -> timezone:
    tz_str = (tz_str or "UTC").strip().upper()
    if tz_str == "UTC":
        return timezone.utc
    m = _TZ_RE.match(tz_str)
    if not m:
        raise ValueError(f"Unsupported timezone format: {tz_str} (use 'UTC' or 'UTC+3' etc.)")
    sign = 1 if m.group(1) == "+" else -1
    hours = int(m.group(2))
    return timezone(timedelta(hours=sign * hours))


@dataclass
class TradingHours:
    enabled: bool
    timezone: str
    start_hour: int
    end_hour: int
    days: Optional[List[int]] = None  # 0=Mon..6=Sun

    def within(self, ts_ms: int) -> bool:
        if not self.enabled:
            return True
        tz = _parse_tz(self.timezone)
        dt = datetime.fromtimestamp(ts_ms / 1000, tz=timezone.utc).astimezone(tz)
        if self.days is not None and dt.weekday() not in set(self.days):
            return False

        h = dt.hour
        s = self.start_hour
        e = self.end_hour

        # Interpret hours as inclusive hour buckets.
        # start == end -> always on
        if s == e:
            return True
        if s < e:
            return (h >= s) and (h <= e)
        # Cross-midnight window (e.g., 8 -> 1)
        return (h >= s) or (h <= e)
