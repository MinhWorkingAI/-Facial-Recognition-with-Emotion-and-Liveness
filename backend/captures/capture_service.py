import json
import time
from datetime import datetime
from typing import Any

from app.config import settings

MIN_INTERVAL_SECONDS = 10  # 0 = save on every eligible request
ONLY_IF_FACES = True

_last_save_at = 0.0


def save(image_bytes: bytes, payload: dict[str, Any]) -> None:
    if not settings.captures_enabled:
        return
    if ONLY_IF_FACES and not payload.get("faces"):
        return

    global _last_save_at
    now = time.monotonic()
    if MIN_INTERVAL_SECONDS and now - _last_save_at < MIN_INTERVAL_SECONDS:
        return
    _last_save_at = now

    out_dir = settings.captures_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    (out_dir / f"{stamp}_frame.jpg").write_bytes(image_bytes)
    (out_dir / f"{stamp}_predictions.json").write_text(
        json.dumps(payload, indent=2),
        encoding="utf-8",
    )
