from __future__ import annotations

import threading
import time

_LOCK = threading.Lock()
_NEXT_ALLOWED_AT: dict[str, float] = {}


def rate_limit_wait(key: str, rpm: int) -> None:
    if not rpm or rpm <= 0:
        return
    interval = 60.0 / float(rpm)
    sleep_s = 0.0
    with _LOCK:
        now = time.monotonic()
        next_allowed = _NEXT_ALLOWED_AT.get(key, 0.0)
        if now < next_allowed:
            sleep_s = next_allowed - now
            _NEXT_ALLOWED_AT[key] = next_allowed + interval
        else:
            _NEXT_ALLOWED_AT[key] = now + interval
    if sleep_s > 0:
        time.sleep(sleep_s)
