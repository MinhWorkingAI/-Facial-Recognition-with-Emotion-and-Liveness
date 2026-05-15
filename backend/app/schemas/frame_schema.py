from typing import Literal, Optional

from pydantic import BaseModel


class NormalizedBox(BaseModel):
    x: float
    y: float
    w: float
    h: float


class DetectedFace(BaseModel):
    bbox: NormalizedBox
    detection_confidence: float
    crop_width: int
    crop_height: int


class EmotionResult(BaseModel):
    label: str
    confidence: float


class AntiSpoofingResult(BaseModel):
    label: Literal["real", "fake"]
    confidence: float


class RecognitionResult(BaseModel):
    label: str
    confidence: float
    matched: bool


class FaceAnalysis(BaseModel):
    face: DetectedFace
    emotion: EmotionResult
    anti_spoofing: AntiSpoofingResult
    recognition: RecognitionResult


class FrameAnalysisResponse(BaseModel):
    image_width: int
    image_height: int
    faces: list[FaceAnalysis]


class RegisterResponse(BaseModel):
    person_id: str
    status: str
    message: str


class DetectionResponse(BaseModel):
    image_width: int
    image_height: int
    faces: list[DetectedFace]


class EmotionResponse(BaseModel):
    emotion: EmotionResult


class AntiSpoofingResponse(BaseModel):
    anti_spoofing: AntiSpoofingResult


class RecognitionResponse(BaseModel):
    recognition: RecognitionResult
