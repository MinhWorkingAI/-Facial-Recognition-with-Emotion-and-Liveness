from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from pathlib import Path

from app.config import settings
from app.schemas.pipeline_schema import FrameAnalysisResponse

logger = logging.getLogger(__name__)


class CaptureService:
    """Persist incoming frames and their pipeline predictions to disk."""

    def __init__(self, output_dir: Path | None = None, enabled: bool | None = None) -> None:
        self.output_dir = output_dir or settings.captures_dir
        self.enabled = settings.captures_enabled if enabled is None else enabled

    def save(self, image_bytes: bytes, response: FrameAnalysisResponse) -> Path | None:
        if not self.enabled:
            return None

        self.output_dir.mkdir(parents=True, exist_ok=True)

        timestamp = datetime.now(timezone.utc)
        stamp = timestamp.strftime("%Y%m%d_%H%M%S_%f")

        image_path = self.output_dir / f"{stamp}_frame.jpg"
        json_path = self.output_dir / f"{stamp}_predictions.json"

        image_path.write_bytes(image_bytes)

        payload = {
            "saved_at": timestamp.isoformat(),
            "image_file": image_path.name,
            **response.model_dump(),
        }
        json_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

        logger.info("Saved capture to %s", self.output_dir)
        return image_path
