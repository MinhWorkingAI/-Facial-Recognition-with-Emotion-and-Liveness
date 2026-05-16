from pydantic import BaseModel
from dataclasses import dataclass, field

from app.schemas.anti_spoofing_schema import AntiSpoofingResult
from app.schemas.common_schema import DetectedFace
from app.schemas.emotion_schema import EmotionResult
from app.schemas.verification_schema import RecognitionResult


class FaceAnalysis(BaseModel):
    face: DetectedFace
    emotion: EmotionResult
    anti_spoofing: AntiSpoofingResult
    recognition: RecognitionResult


class FrameAnalysisResponse(BaseModel):
    image_width: int
    image_height: int
    faces: list[FaceAnalysis]

@dataclass
class FaceResult:
    bbox: tuple                  # (x, y, w, h) normalised
    detection_score: float
    is_live: bool = False
    liveness_score: float = 0.0
    emotion: str = ""
    emotion_score: float = 0.0
    employee_id: str | None = None
    employee_name: str | None = None
    similarity: float = 0.0
    verified: bool = False


@dataclass
class InferenceResult:
    faces: list[FaceResult] = field(default_factory=list)
    attendance_triggered: list[str] = field(default_factory=list)
