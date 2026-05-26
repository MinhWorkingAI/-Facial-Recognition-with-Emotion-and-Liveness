from pydantic import BaseModel, Field


class NormalizedBox(BaseModel):
    x: float
    y: float
    w: float
    h: float


class NormalizedPoint(BaseModel):
    x: float
    y: float


class DetectedFace(BaseModel):
    bbox: NormalizedBox
    detection_confidence: float
    crop_width: int
    crop_height: int
    keypoints: list[NormalizedPoint] = Field(default_factory=list)
