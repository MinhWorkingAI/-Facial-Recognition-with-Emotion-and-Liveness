from typing import Literal

from pydantic import BaseModel


class AntiSpoofingResult(BaseModel):
    label: Literal["real", "spoof"]
    confidence: float


class AntiSpoofingResponse(BaseModel):
    anti_spoofing: AntiSpoofingResult
