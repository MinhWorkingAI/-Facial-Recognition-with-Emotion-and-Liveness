import json
from datetime import datetime
from typing import Any, Mapping

from app.config import settings


def save(image_bytes: bytes, payload: Mapping[str, Any]) -> None:
    """Write frame JPEG + predictions JSON when captures are enabled."""
    if not settings.captures_enabled:
        return

    out_dir = settings.captures_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    stamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    (out_dir / f"{stamp}_frame.jpg").write_bytes(image_bytes)
    (out_dir / f"{stamp}_predictions.json").write_text(
        json.dumps(dict(payload), indent=2),
        encoding="utf-8",
    )
