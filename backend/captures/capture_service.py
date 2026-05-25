from __future__ import annotations

import json
from datetime import datetime, timezone

from app.config import settings
from app.schemas.pipeline_schema import FrameAnalysisResponse


def save(image_bytes: bytes, response: FrameAnalysisResponse) -> None:
    """Write frame JPEG + predictions JSON when captures are enabled."""
    if not settings.captures_enabled:
        return

    out_dir = settings.captures_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    stamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S_%f")
    (out_dir / f"{stamp}_frame.jpg").write_bytes(image_bytes)
    (out_dir / f"{stamp}_predictions.json").write_text(
        json.dumps(response.model_dump(), indent=2),
        encoding="utf-8",
    )
